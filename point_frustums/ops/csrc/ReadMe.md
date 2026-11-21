# 3D Non-Maximum Suppression for Oriented Boxes (CPU · CUDA)

This directory contains the C++/CUDA extension of point-frustums. Its centerpiece is a **class-aware non-maximum
suppression (NMS) for arbitrarily oriented 3D bounding boxes**, built on an exact 3D intersection-over-union (IoU).
It is used to deduplicate the detections of the network before evaluation.

```python
import point_frustums  # registers torch.ops.point_frustums
from point_frustums.ops.nms import nms_3d

# labels: (N,) int, scores: (N,) float, boxes: (N, 8, 3) float corners
keep = nms_3d(labels, scores, boxes, iou_threshold=0.4, distance_threshold=0.1)  # (N,) bool
```

`nms_3d` dispatches on the device of the inputs to the CPU or CUDA kernel of the compiled extension
(`torch.ops.point_frustums.nms`).

## Algorithm

For every box `i`, the kernel checks whether there is a box `j` that

1. has a **higher score**,
2. has the **same class label**,
3. is **close enough**: the center distance scaled by the volume of box `i` is below `distance_threshold`. This cheap
   pre-check skips the expensive IoU for most pairs, and
4. overlaps box `i` with a **3D IoU** of at least `iou_threshold`.

If such a box exists, box `i` is a duplicate and is removed.

A box is removed as soon as any higher-scoring box of its class overlaps it, even if that box is itself removed. This
is a deliberate design decision. Classic greedy NMS only lets kept boxes suppress others, which makes it inherently
sequential. This formulation is fully parallel instead: every pair `(i, j)` can be evaluated independently, which is
what makes the GPU kernels possible. In crowded scenes it can remove slightly more boxes than greedy NMS.

The IoU of two oriented boxes is computed exactly. Their intersection is a convex polyhedron, whose volume is computed
from its faces.

## Original Implementation (January to April 2024)

The first version of the NMS was written between January and April 2024. It consists of the following parts:

- **3D IoU, reused and modified from PyTorch3D** ([`383b80e`](https://github.com/enrico-stauss/point-frustums-public/commit/383b80e)).
  I copied the CPU and CUDA implementation of PyTorch3D's `box3d_overlap` and changed it in two ways:
  - **Pairwise variant:** I added a variant that evaluates only `IoU(boxes1[n], boxes2[n])` instead of all `N x M`
    combinations (`IoUBox3DPairwise`, CPU and CUDA). The target assignment uses it to compare predictions with their
    matched targets without computing the full matrix.
  - **Reusable functions:** I restructured the IoU kernels into functions that other code can call (`GetBoxTris`,
    `GetBoxPlanes` and `BoxVolume` writing into pre-allocated memory, and `GetIntersectionAndUnion`). This lets the
    NMS compute the IoU of individual box pairs inside its own kernels.
- **CPU NMS** ([`451acb3`](https://github.com/enrico-stauss/point-frustums-public/commit/451acb3)): the boxes are
  sorted by score, and the triangles, planes, center and volume of every box are precomputed. A nested loop then runs
  the label, distance and IoU checks described above.
- **CUDA NMS** ([`1579ca2`](https://github.com/enrico-stauss/point-frustums-public/commit/1579ca2),
  [`5d220eb`](https://github.com/enrico-stauss/point-frustums-public/commit/5d220eb),
  [`0aa1c52`](https://github.com/enrico-stauss/point-frustums-public/commit/0aa1c52),
  [`f62ea9b`](https://github.com/enrico-stauss/point-frustums-public/commit/f62ea9b)):
  - **Pairwise duplicate test:** the sequential loop was reformulated as a pairwise duplicate test, launched as a 2D
    grid of threads over all (lower score, higher score) pairs.
  - **Precomputed box geometry:** a separate `BoxesMetaKernel` precomputes the geometry of every box once. It is
    stored in a packed `BoxParameters` struct (center, 12 triangles, 6 planes, volume) that is mapped onto the rows of
    a tensor with `reinterpret_cast`. Each thread can then load all parameters of a box from contiguous memory.
- **Fix of a pre-allocation bound in the PyTorch3D IoU** ([`04371f7`](https://github.com/enrico-stauss/point-frustums-public/commit/04371f7)):
  the fixed-size buffer for the faces of the intersection polyhedron (`MAX_TRIS`) was too small for some box
  configurations. This occasionally produced illegal memory accesses on the GPU (see
  [pytorch3d#1777](https://github.com/facebookresearch/pytorch3d/issues/1777)).
- **Integration:** both backends are registered as a PyTorch operator (`TORCH_LIBRARY`, [`ext.cpp`](ext.cpp)) and
  selected by the device of the inputs ([`nms/nms.h`](nms/nms.h)).

## Layout

```
csrc/
├── ext.cpp                  # Operator registration (torch.ops.point_frustums.{iou_box3d, iou_box3d_pairwise, nms})
├── cuda_version.cu          # CUDA version query
├── iou_box3d/
│   ├── iou_utils.h          # Exact 3D IoU, CPU (PyTorch3D, modified)
│   ├── iou_utils.cuh        # Exact 3D IoU, CUDA (PyTorch3D, modified)
│   ├── iou_box3d.h          # Device dispatch
│   ├── iou_box3d_cpu.cpp    # CPU: N x M and pairwise IoU
│   └── iou_box3d.cu         # CUDA: N x M and pairwise IoU
├── nms/
│   ├── nms_utils.h/.cuh     # Packed per-box parameters (BoxParameters)
│   ├── nms.h                # Device dispatch
│   ├── nms.cpp              # CPU NMS
│   └── nms_cuda.cu          # CUDA NMS
└── utils/
    ├── float_math.cuh       # float3 arithmetic for CUDA
    ├── vec3.h               # 3D vector type for CPU
    └── pytorch3d_cutils.h   # Input checks
```

## Credits

The 3D IoU builds on PyTorch3D's `box3d_overlap` (BSD license, (c) Meta Platforms, Inc.).
