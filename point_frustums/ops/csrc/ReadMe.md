# 3D Non-Maximum Suppression for Oriented Boxes (CPU · CUDA · Metal)

This directory contains the C++/CUDA extension of point-frustums. Its centerpiece is a **class-aware non-maximum
suppression (NMS) for arbitrarily oriented 3D bounding boxes**, built on an exact 3D intersection-over-union (IoU).
It is used to deduplicate the detections of the network before evaluation.

```python
import point_frustums  # registers torch.ops.point_frustums
from point_frustums.ops.nms import nms_3d

# labels: (N,) int, scores: (N,) float, boxes: (N, 8, 3) float corners
keep = nms_3d(labels, scores, boxes, iou_threshold=0.4, distance_threshold=0.1)  # (N,) bool
```

`nms_3d` dispatches on the device of the inputs: CPU and CUDA tensors go to the compiled extension
(`torch.ops.point_frustums.nms`), MPS tensors (Apple GPUs) to the Metal kernels in
[`point_frustums/ops/metal.py`](../metal.py).

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

## Port to the Rewritten PyTorch3D IoU (2026)

PyTorch3D later rewrote `box3d_overlap`
([facebookresearch/pytorch3d@31debbd](https://github.com/facebookresearch/pytorch3d/commit/31debbdc7a9a7521b3ce5cfabfe21992a987a094)).
The new version clips box 1 by the six half-spaces of box 2 one after another. This is exact, handles boxes that share
a face, and no longer needs the fixed-size face buffer. It is a single header that compiles for both host and device.
The NMS was ported onto it (commit "Port the 3D IoU and NMS kernels to pytorch3d's sequential polyhedron clipping"):

- **Shared IoU header:** [`iou_box3d/iou_utils.h`](iou_box3d/iou_utils.h) is now the upstream header and is used by
  the CPU, CUDA and Metal builds. All three call `iou_box3d::BoxOverlap` on the raw corners.
- **Less precomputed data:** per box, the NMS now only stores the center and volume (`ComputeBoxMeta` in
  [`nms/nms_utils.h`](nms/nms_utils.h)) for the distance pre-check, instead of triangles and planes.
- **Parallel CPU loop:** the CPU outer loop runs in parallel (`at::parallel_for`). This is possible because a box only
  depends on the boxes with a higher score, not on whether those are kept.
- **CUDA grid and memory:** the CUDA grid is now sized to the input. The original launch used a fixed 256 x 256 grid
  of 16 x 16 blocks, so boxes beyond index 4096 were never compared. The `N x N` duplicate matrix was also replaced by
  an `N`-sized mask, and threads exit early once their box is already known to be a duplicate.

Results on CPU (Apple M1 Pro):

|                                           | Original (2024) | Ported (2026) |
|-------------------------------------------|-----------------|---------------|
| Max. IoU error vs. exact float64 reference | 7.8e-4          | 7.7e-8        |
| IoU, 1000 x 1000 boxes                     | 12.1 s          | 0.019 s       |
| NMS, 1000 boxes                            | 1.1 s           | 0.005 s       |

The keep masks of the original and the ported NMS are identical.

## Metal Backend for Apple GPUs (2026)

The extension is compiled for CPU and CUDA only. Before this backend, IoU and NMS on Apple GPUs had to copy every input
to the CPU. [`point_frustums/ops/metal.py`](../metal.py) adds Metal kernels
(commit "Add Metal kernels for the 3D IoU and NMS ops"):

- **Runtime compilation from the same headers:** the kernels are compiled at runtime with `torch.mps.compile_shader`
  from **the same headers** as the CPU and CUDA build ([`iou_box3d/iou_utils.h`](iou_box3d/iou_utils.h),
  [`nms/nms_utils.h`](nms/nms_utils.h)). All three backends therefore run one implementation of the geometry.
- **Small Metal-specific layer:** only the kernel entry points in `metal.py` (four short functions) are written in
  Metal itself. They do the same as the CUDA kernels: one thread per box for the per-box meta data, and one thread per
  (lower score, higher score) pair for the duplicate test.
- **Address space macros:** the shared headers mark pointers and references with `IOU_THREAD`, `IOU_GLOBAL` and
  `IOU_CONSTANT`. These expand to Metal's address spaces (`thread`, `device`, `constant`) when compiled as Metal and
  to nothing for C++/CUDA. Code that Metal does not support (`double` overloads, standard library includes) is guarded
  with `__METAL_VERSION__`.
- **Results:** IoU, intersection volumes and NMS keep masks are bitwise identical to the CPU extension. A 1000 x 1000
  IoU takes 5.6 ms on the GPU of an M1 Pro, compared with 30 ms for the previous round trip through the CPU.

## Tests

[`test/ops/test_metal.py`](../../../test/ops/test_metal.py) checks the Metal backend against the CPU extension:

- **Full IoU matrix:** volumes and IoUs of random boxes, with yaw-only and with arbitrary 3D rotations.
- **Pairwise IoU:** random pairs plus two edge cases, identical boxes (IoU 1) and boxes that only touch along a face
  (IoU 0).
- **Dispatch:** `iou_vol_3d` routes MPS tensors to Metal.
- **NMS:** keep masks on 400 boxes with planted near-duplicates match exactly, with and without the distance pre-check,
  and duplicates are actually removed.

```bash
pytest test/ops
```

## Status

- **CPU and Metal:** tested and used for training on macOS.
- **CUDA:** the ported CUDA implementation ([`nms/nms_cuda.cu`](nms/nms_cuda.cu),
  [`iou_box3d/iou_box3d.cu`](iou_box3d/iou_box3d.cu)) has not yet been compiled and tested on a CUDA machine since the
  2026 port.

## Layout

```
csrc/
├── ext.cpp                  # Operator registration (torch.ops.point_frustums.{iou_box3d, iou_box3d_pairwise, nms})
├── cuda_version.cu          # CUDA version query
├── iou_box3d/
│   ├── iou_utils.h          # Exact 3D IoU (PyTorch3D, header-only, CPU/CUDA/Metal)
│   ├── iou_box3d.h          # Device dispatch
│   ├── iou_box3d_cpu.cpp    # CPU: N x M and pairwise IoU
│   └── iou_box3d.cu         # CUDA: N x M and pairwise IoU
├── nms/
│   ├── nms_utils.h          # Per-box meta data and distance pre-check (CPU/CUDA/Metal)
│   ├── nms.h                # Device dispatch
│   ├── nms.cpp              # CPU NMS
│   └── nms_cuda.cu          # CUDA NMS
└── utils/
    └── pytorch3d_cutils.h   # Input checks
```

## Credits

The 3D IoU builds on PyTorch3D's `box3d_overlap` (BSD license, (c) Meta Platforms, Inc.). The 2026 port and the Metal
backend were developed with the help of an AI coding assistant.
