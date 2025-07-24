from point_frustums.dataloaders.nuscenes import NuScenesStreamingDataset
from point_frustums.utils.statistics import WelfordsAlgorithm
from tqdm import tqdm
import torch
from typing import Optional


MAX_SAMPLES = 2000 # total: 28130 samples in train split


# First pass: Evaluate the statistics of the raw pointclouds
ds = NuScenesStreamingDataset("/teamspace/studios/nuscenes/datasets/lit_nuscenes/train")
wa_mean = WelfordsAlgorithm()
wa_std = WelfordsAlgorithm()


for i, sample in tqdm(enumerate(ds)):
     if i > 2:
             break
     wa_mean.update(sample["lidar"]["LIDAR_TOP"].mean(dim=0))
     wa_std.update(sample["lidar"]["LIDAR_TOP"].std(dim=0))

mean = wa_mean.finalize()[0]
std = wa_std.finalize()[0]

print(f"{mean=}")
print(f"{std=}")



# Second pass: Evaluate the statistics of the decorator function outputs
from point_frustums.augmentations.augmentations_other import Normalize, NormalizationParameters
from point_frustums.config_dataclasses.point_frustums import ConfigDiscretize, ConfigDecorate, DecoratorFunction
ds = NuScenesStreamingDataset(
     "/teamspace/studios/nuscenes/datasets/lit_nuscenes/train", 
     augmentations=[
          Normalize(
               apply_to = ["lidar"],
               lidar=NormalizationParameters(
                    channels=[2, 3, 4, 5], 
                    mean=[1.1974e+00,  1.8337e+01,  2.4331e-01, 1.3075e+01], 
                    std=[2.0488, 19.5468,  0.1407, 12.8591]
               )
          )
     ]
)

discretize = ConfigDiscretize(n_splits_azi=480, n_splits_pol= 40, fov_azi_deg= (-180, 180), fov_pol_deg= (60, 100))
channels_in = ["x", "y", "z", "intensity", "timestamp", "radial", "polar", "azimuthal"]
decorate = ConfigDecorate(
     functions=[
          DecoratorFunction(id = "relative_angle", channel = "azimuthal", std = 1.0 ),
          DecoratorFunction(id = "relative_angle", channel = "polar", std= 1.0 ),
          DecoratorFunction(id = "distance_to_mean", channel = "radial", std= 1.0 ),
          DecoratorFunction(id = "distance_to_mean", channel= "x", std= 1.0 ),
          DecoratorFunction(id = "distance_to_mean", channel= "y", std= 1.0 ),
          DecoratorFunction(id = "distance_to_mean", channel= "z", std= 1.0 )
          ],
     channels_out = [ "z", "intensity", "timestamp", "radial", "delta_azi", "delta_pol", "delta_r", "delta_x", "delta_y", "delta_z" ])

# Copy out encoder methods and remove references to class
def filter_fov(pc: torch.Tensor) -> torch.Tensor:
     channel_index_azi, channel_index_pol = channels_in.index("azimuthal"), channels_in.index("polar")
     mask = (
          (pc[:, channel_index_azi] >= discretize.fov_azi[0])
          & (pc[:, channel_index_azi] < discretize.fov_azi[1])
          & (pc[:, channel_index_pol] >= discretize.fov_pol[0])
          & (pc[:, channel_index_pol] < discretize.fov_pol[1])
     )
     return pc[mask, ...]

def get_frustum_index(pc: torch.Tensor):
     channel_index_azi, channel_index_pol = channels_in.index("azimuthal"), channels_in.index("polar")
     i_azi = ((pc[:, channel_index_azi] - discretize.fov_azi[0]) // discretize.delta_azi).to(torch.int64)
     i_pol = ((pc[:, channel_index_pol] - discretize.fov_pol[0]) // discretize.delta_pol).to(torch.int64)
     return i_azi + i_pol * discretize.n_splits_azi

def sort_by_frustum_index(pc: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
     indices, sorting_indices = torch.sort(indices)
     pc = pc[sorting_indices, :]
     return pc, indices
     
def get_frustum_stats(indices):
     i_unique, i_inv, counts = indices.unique_consecutive(return_inverse=True, return_counts=True)
     counts_padded = counts.new_zeros((discretize.n_splits,))
     counts_padded[i_unique] = counts
     return i_unique, i_inv, counts, counts_padded


def decorator_distance_to_mean(
    pc_channel: torch.Tensor,
    n_frustums: int,
    counts_padded: torch.Tensor,
    i_frustum: torch.Tensor,
    i_unique: torch.Tensor,
    i_inv: torch.Tensor,
    std: Optional[float] = 1.0,
):
    """
    :param pc_channel:
    :param n_frustums:
    :param counts_padded:
    :param i_frustum:
    :param i_unique:
    :param i_inv:
    :param std: The normalization constant that scales the standard deviation to 1, determined empirically
    :return:
    """
    mean = torch.scatter_add(pc_channel.new_zeros((n_frustums,)), dim=0, index=i_frustum, src=pc_channel)
    mask = counts_padded != 0
    mean[mask] = mean[mask].div(counts_padded[mask])
    # Subset to the number of non-empty frustums (i_frustum) and then map the frustum mean value to the points (i_inv)
    pc_channel = pc_channel - mean[i_unique][i_inv]
    return (pc_channel / std)[:, None]


def decorator_relative_angle(pc_channel: torch.Tensor, delta_rad: float, std: Optional[float] = 1.0):
    """
    Returns the (rad) angle between the center of the frustum and the point in the respective dimension.
    :param pc_channel:
    :param delta_rad:
    :param std: The normalization constant that scales the standard deviation to 1, determined empirically
    :return:
    """
    # Normalize output by shifting to the frustum center and dividing by an (empirically estimated) standard deviation
    angle = (pc_channel % delta_rad - (delta_rad / 2)) / delta_rad
    angle = angle / std
    return angle[:, None]

def get_decorator_keep_channels():
     """Evaluates the indices of the input channels that are kept after applying the decorators in correct order."""
     channels = []
     keep_channels = set(channels_in).intersection(decorate.channels_out)
     for channel in channels_in:
          if channel in keep_channels:
               channels.append(channels_in.index(channel))
     return channels

decorator_keep_channels = get_decorator_keep_channels()

def apply_decorators(
     pc: torch.Tensor,
     counts_padded: torch.Tensor,
     i_frustum: torch.Tensor,
     i_unique: torch.Tensor,
     i_inv: torch.Tensor,
) -> torch.Tensor:
     """
     Apply the configured decorator functions to obtain invariant features.
     :param pc:
     :param counts_padded:
     :param i_frustum:
     :param i_unique:
     :param i_inv:
     :return:
     """
     decorated_channels = []
     for fn in decorate.functions:
          pc_channel = pc[:, channels_in.index(fn.channel)]
          match fn.id:
               case "relative_angle":
                    delta_rad = {"azimuthal": discretize.delta_azi, "polar": discretize.delta_pol}[fn.channel]
                    decorated_channels.append(
                         decorator_relative_angle(
                              pc_channel=pc_channel,
                              delta_rad=delta_rad,
                              std=fn.std,
                         )
                    )
               case "distance_to_mean":
                    decorated_channels.append(
                         decorator_distance_to_mean(
                              pc_channel=pc_channel,
                              n_frustums=discretize.n_splits,
                              counts_padded=counts_padded,
                              i_frustum=i_frustum,
                              i_unique=i_unique,
                              i_inv=i_inv,
                              std=fn.std,
                         )
                    )
               case _:
                    raise NotImplementedError(f"Decorator {fn.id} not implemented.")
     return torch.cat((pc[:, decorator_keep_channels], *decorated_channels), dim=1)


wa_decorated_mean = WelfordsAlgorithm()
wa_decorated_std = WelfordsAlgorithm()

for i, sample in tqdm(enumerate(ds)):
     if i > MAX_SAMPLES:
             break
     pc = sample["lidar"]["LIDAR_TOP"]
     pc = filter_fov(pc)
     i_frustum = get_frustum_index(pc)
     pc, i_frustum = sort_by_frustum_index(pc, i_frustum)
     i_unique, i_inv, counts, counts_padded = get_frustum_stats(i_frustum)
     # Apply decorators to entire PC
     decorated = apply_decorators(pc, counts_padded=counts_padded, i_frustum=i_frustum, i_unique=i_unique, i_inv=i_inv)
     wa_decorated_mean.update(decorated.mean(dim=0))
     wa_decorated_std.update(decorated.std(dim=0))

decorated_mean = wa_decorated_mean.finalize()[0]
decorated_std = wa_decorated_std.finalize()[0]

print(f"Decorated columns: {decorate.channels_out}")
print(f"{decorated_mean=}")
print(f"{decorated_std=}")

for channel, mean, std in zip(decorate.channels_out, decorated_mean.tolist(), decorated_std.tolist()):
     print(f"{channel=} | {mean=} | {std=}")