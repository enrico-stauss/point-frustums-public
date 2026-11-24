import torch
from torch import nn
from torchvision.models.resnet import Bottleneck, conv1x1

from point_frustums.config_dataclasses.fpn import ConfigFPNExtraLayer, ConfigFPNLayer
from point_frustums.ops.spherical_coos_convolutions import Conv2dSpherical, UpsampleSpherical


class BottleneckRangeImage(Bottleneck):
    """
    Modifies the standard Bottleneck block to use range image padding with the 3x3 convolution
    """

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        **kwargs,
    ):
        """
        Overwrite the conv2 attribute to a custom convolution with circular padding along the azimuthal (horizontal)
        dimension and zero-padding along the polar (vertical) dimension.
        :param planes:
        :param stride:
        :param groups:
        :param base_width:
        :param dilation:
        :param kwargs:
        """
        super().__init__(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            groups=groups,
            base_width=base_width,
            dilation=dilation,
            **kwargs,
        )
        width = int(planes * (base_width / 64.0)) * groups
        self.conv2 = Conv2dSpherical(
            in_channels=width,
            out_channels=width,
            kernel_size=3,
            stride=stride,
            groups=groups,
            dilation=dilation,
            bias=False,
        )


class FPN(nn.Module):
    block = BottleneckRangeImage

    @staticmethod
    def norm_layer(num_channels):
        return nn.GroupNorm(num_groups=8, num_channels=num_channels)

    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        layers: dict[str, ConfigFPNLayer],
        extra_layers: dict[str, ConfigFPNExtraLayer] | None = None,
        dropout: float = 0.0,
        upsampling_mode: str = "bilinear",
    ):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.dropout = dropout
        self.upsampling_mode = upsampling_mode
        self.layers = dict(sorted(layers.items()))
        self.names_layers = tuple(self.layers.keys())
        self.names_layers_reversed = tuple(reversed(self.names_layers))
        self.first_layer, self.last_layer = self.names_layers[0], self.names_layers[-1]
        self.extra_layers = dict(sorted(extra_layers.items())) if extra_layers is not None else {}
        self.names_extra_layers = tuple(self.extra_layers.keys())

        self.strides = {}
        self.up, self.lateral, self.down, self.post = self.build_layers()
        self.extra, self.extra_post = self.build_extra_layers()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_first_block(self, n_channels_in, n_channels, n_channels_out, stride):
        """
        The first block of each layer optionally performs downsampling.
        - If the layer is configured with a stride > 1
        - Or if the number of input channels is not equal to the number of output channels (expanded working planes)
        """
        downsample = None
        if stride != 1 or n_channels_in != n_channels_out:
            downsample = nn.Sequential(
                conv1x1(n_channels_in, n_channels_out, stride),
                self.norm_layer(num_channels=n_channels_out),
            )
        block = self.block(
            inplanes=n_channels_in,
            planes=n_channels,
            stride=stride,
            downsample=downsample,
            norm_layer=self.norm_layer,
        )
        return block

    def build_layers(self) -> tuple[nn.ModuleList, nn.ModuleList, nn.ModuleList, nn.ModuleList]:
        """Build the layers of the FPN.
        A FPN layer consists of several components:
            1. The top-down path is made up of a series of blocks (the first sets the size and dimension for the layer)
            2. The bottom-up path is made up of a simple upsampling layer
            3. The lateral connection maps the output of the bottom-up to the output dimension
            4. The post step applies a 3x3 convolution to the sum of (bottom-up[i] + top_down[i]) to mitigate artifacts
        and the procedure is:
        1. Evaluate entire bottom-up path and the lateral connection and store each output in the results dictionary
        2. Propagate from the highest level down to evaluate the top-down path, at each layer add to the results and
           apply the post-step

        Dropout is applied throughout the network:
        - Geometric 2d dropout before every block in the bottom-up path (and before the conv of extra layers)
        - Standard dropout is applied in the beginning of the lateral connection
        - Geometric 2d dropout before the post layer
        :return:
        """
        n_channels_in = self.n_channels_in

        up, lateral, down, post = {}, {}, {}, {}
        prev = None
        for name, layer in self.layers.items():
            self.strides[name] = (
                layer.stride[0] * self.strides.get(prev, (1, 1))[0],
                layer.stride[1] * self.strides.get(prev, (1, 1))[1],
            )
            n_channels_out = layer.n_channels * self.block.expansion
            # Step 1: The first block of each layer is a special case as it optionally performs the downsampling
            up[name] = nn.Sequential()
            up[name].append(nn.Dropout2d(self.dropout))
            up[name].append(self.norm_layer(num_channels=n_channels_in))
            up[name].append(self._get_first_block(n_channels_in, layer.n_channels, n_channels_out, layer.stride))
            # Step 2: Add the remaining blocks (no stride/downsampling)
            for _ in range(1, layer.n_blocks):
                up[name].append(nn.Dropout2d(self.dropout))
                up[name].append(self.norm_layer(num_channels=n_channels_out))
                up[name].append(self.block(n_channels_out, layer.n_channels, norm_layer=self.norm_layer))

            # Step 3: Build the lateral connection (Conv2D) and the top-down-path upsampling layer (interpolate)
            lateral[name] = nn.Sequential(
                nn.Dropout(self.dropout),
                self.norm_layer(num_channels=n_channels_out),
                conv1x1(n_channels_out, self.n_channels_out),
                self.norm_layer(num_channels=self.n_channels_out),
            )
            n_channels_in = n_channels_out

            if prev is not None:
                # Delay initialization of the bottom-up pathway by one layer as it is dependent on the specification of
                # the consecutive layer
                if self.upsampling_mode == "aligned":
                    # Consistent with the sampling positions of the strided convolutions (see UpsampleSpherical)
                    down[prev] = UpsampleSpherical(scale_factor=layer.stride)
                else:
                    down[prev] = nn.Upsample(scale_factor=layer.stride, mode=self.upsampling_mode)
                post[prev] = nn.Sequential(
                    nn.Dropout2d(self.dropout),
                    self.norm_layer(num_channels=self.n_channels_out),
                    Conv2dSpherical(self.n_channels_out, self.n_channels_out, kernel_size=3, bias=False),
                    self.norm_layer(num_channels=self.n_channels_out),
                )
            prev = name

        up = nn.ModuleList([up[layer] for layer in self.names_layers])
        lateral = nn.ModuleList([lateral[layer] for layer in self.names_layers])
        down = nn.ModuleList([down[layer] for layer in self.names_layers_reversed if layer in down])
        post = nn.ModuleList([post[layer] for layer in self.names_layers_reversed if layer in post])

        return up, lateral, down, post

    def build_extra_layers(self) -> tuple[nn.ModuleList, nn.ModuleList]:
        prev = list(self.strides.keys())[-1]
        extra_layers = nn.ModuleList()
        extra_layers_post = nn.ModuleList()
        n_channels_in = self.layers[self.last_layer].n_channels * self.block.expansion
        for name, layer in self.extra_layers.items():
            self.strides[name] = (layer.stride[0] * self.strides[prev][0], layer.stride[1] * self.strides[prev][1])
            extra_layers.append(
                nn.Sequential(
                    nn.Dropout(self.dropout),
                    Conv2dSpherical(n_channels_in, n_channels_in, kernel_size=3, stride=layer.stride),
                    self.norm_layer(num_channels=n_channels_in),
                )
            )
            extra_layers_post.append(
                nn.Sequential(
                    nn.Dropout2d(self.dropout),
                    Conv2dSpherical(n_channels_in, self.n_channels_out, kernel_size=3, bias=False),
                    self.norm_layer(num_channels=self.n_channels_out),
                )
            )
            prev = name
        return extra_layers, extra_layers_post

    def forward(self, input: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass of the FPN"""
        results = {}

        # Walk through the bottom-up path and directly apply the lateral FC layer
        for layer, up, lateral in zip(self.names_layers, self.up, self.lateral):
            input = up(input)
            results[layer] = lateral(input)

        # Evaluate extra layers, building on the result of the last bottom-up layer
        for layer, extra, extra_post in zip(self.names_extra_layers, self.extra, self.extra_post):
            input = extra(input)
            results[layer] = extra_post(input)

        # Walk through the top-down path, add the up-sampled featuremaps to the intermediate result, apply the FC layer
        input = results[self.names_layers_reversed[0]]
        for layer, down, post in zip(self.names_layers_reversed[1:], self.down, self.post):
            results[layer] += down(input)
            input = results[layer]
            results[layer] = post(results[layer]) + results[layer]

        return results
