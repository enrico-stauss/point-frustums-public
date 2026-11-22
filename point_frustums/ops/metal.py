"""
Metal (MPS) implementations of the 3D box IoU and NMS ops.

The kernels are compiled at runtime via torch.mps.compile_shader from the same headers that are used for the CPU and
CUDA builds (csrc/iou_box3d/iou_utils.h and csrc/nms/nms_utils.h), so all backends share one implementation of the
algorithm. The headers annotate pointers and references with address space macros that only expand on Metal.
"""

import re
from functools import cache
from pathlib import Path

import torch

_CSRC = Path(__file__).parent / "csrc"
_HEADERS = ("iou_box3d/iou_utils.h", "nms/nms_utils.h")

_KERNELS = """
// One thread per (n, m) pair
kernel void iou_box3d_kernel(
    device const float* boxes1 [[buffer(0)]],  // (N, 8, 3)
    device const float* boxes2 [[buffer(1)]],  // (M, 8, 3)
    device float* vols [[buffer(2)]],          // (N, M)
    device float* ious [[buffer(3)]],          // (N, M)
    constant long& N [[buffer(4)]],
    constant long& M [[buffer(5)]],
    uint2 idx [[thread_position_in_grid]]) {
  const long m = idx.x;
  const long n = idx.y;
  if (n >= N || m >= M) {
    return;
  }
  float vol, iou;
  iou_box3d::BoxOverlap(boxes1 + 24 * n, boxes2 + 24 * m, vol, iou);
  vols[n * M + m] = vol;
  ious[n * M + m] = iou;
}

// One thread per n, evaluates boxes1[n] against boxes2[n]
kernel void iou_box3d_pairwise_kernel(
    device const float* boxes1 [[buffer(0)]],  // (N, 8, 3)
    device const float* boxes2 [[buffer(1)]],  // (N, 8, 3)
    device float* vols [[buffer(2)]],          // (N,)
    device float* ious [[buffer(3)]],          // (N,)
    constant long& N [[buffer(4)]],
    uint n [[thread_position_in_grid]]) {
  if (n >= N) {
    return;
  }
  float vol, iou;
  iou_box3d::BoxOverlap(boxes1 + 24 * n, boxes2 + 24 * n, vol, iou);
  vols[n] = vol;
  ious[n] = iou;
}

kernel void nms_box_meta_kernel(
    device const float* boxes [[buffer(0)]],  // (N, 8, 3)
    device float* meta [[buffer(1)]],         // (N, kBoxMetaSize)
    constant long& N [[buffer(2)]],
    uint n [[thread_position_in_grid]]) {
  if (n >= N) {
    return;
  }
  ComputeBoxMeta(boxes + 24 * n, meta + kBoxMetaSize * n);
}

// One thread per (lower score, higher score) pair, equivalent to the CPU/CUDA implementation: box i is marked as
// duplicate if a box j of the same class with a higher score is close enough and the IoU reaches the threshold.
kernel void nms_duplicates_kernel(
    device const float* boxes [[buffer(0)]],      // (N, 8, 3)
    device const float* meta [[buffer(1)]],       // (N, kBoxMetaSize)
    device const int* labels [[buffer(2)]],       // (N,)
    device const long* sort_idx [[buffer(3)]],    // (N,) ascending score
    device bool* is_duplicate [[buffer(4)]],      // (N,)
    constant long& N [[buffer(5)]],
    constant float& iou_threshold [[buffer(6)]],
    constant float& distance_threshold [[buffer(7)]],
    uint2 idx [[thread_position_in_grid]]) {
  const long tid_i = idx.x;
  const long tid_j = idx.y;
  if (tid_i >= N || tid_j >= N || tid_i >= tid_j) {
    return;
  }
  const long i = sort_idx[tid_i];
  const long j = sort_idx[tid_j];
  if (labels[i] != labels[j] || is_duplicate[i]) {
    return;
  }
  if (ExceedsDistance(meta + kBoxMetaSize * i, meta + kBoxMetaSize * j, distance_threshold)) {
    return;
  }
  float vol, iou;
  iou_box3d::BoxOverlap(boxes + 24 * i, boxes + 24 * j, vol, iou);
  if (iou >= iou_threshold) {
    // Several threads may write here concurrently, but all of them write the same value.
    is_duplicate[i] = true;
  }
}
"""


def _read_header(name: str) -> str:
    source = (_CSRC / name).read_text(encoding="utf-8")
    # The headers are concatenated in dependency order, local includes are resolved by that.
    return re.sub(r'^#include ".*"$', "", source, flags=re.M)


@cache
def _library():
    source = "#include <metal_stdlib>\nusing namespace metal;\n"
    source += "\n".join(_read_header(h) for h in _HEADERS)
    return torch.mps.compile_shader(source + _KERNELS)


def _check_boxes(*boxes: torch.Tensor):
    for b in boxes:
        if b.device.type != "mps" or b.dtype != torch.float32 or b.dim() != 3 or b.shape[1:] != (8, 3):
            raise ValueError(f"Expected float32 MPS boxes of shape (N, 8, 3), got {b.dtype} {b.device} {b.shape}")


def iou_box3d(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Intersection volume and IoU between all boxes, boxes1: (N, 8, 3), boxes2: (M, 8, 3) -> (N, M), (N, M)."""
    _check_boxes(boxes1, boxes2)
    boxes1, boxes2 = boxes1.contiguous(), boxes2.contiguous()
    n, m = boxes1.shape[0], boxes2.shape[0]
    vols = torch.empty((n, m), dtype=torch.float32, device=boxes1.device)
    ious = torch.empty_like(vols)
    if n * m > 0:
        _library().iou_box3d_kernel(boxes1, boxes2, vols, ious, n, m, threads=(m, n))
    return vols, ious


def iou_box3d_pairwise(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Intersection volume and IoU between boxes1[n] and boxes2[n], both (N, 8, 3) -> (N,), (N,)."""
    _check_boxes(boxes1, boxes2)
    if boxes1.shape[0] != boxes2.shape[0]:
        raise ValueError("The pairwise IoU3D requires boxes to be of the same shape (N, 8, 3)")
    boxes1, boxes2 = boxes1.contiguous(), boxes2.contiguous()
    n = boxes1.shape[0]
    vols = torch.empty((n,), dtype=torch.float32, device=boxes1.device)
    ious = torch.empty_like(vols)
    if n > 0:
        _library().iou_box3d_pairwise_kernel(boxes1, boxes2, vols, ious, n, threads=n)
    return vols, ious


def nms(
    labels: torch.Tensor,
    scores: torch.Tensor,
    boxes: torch.Tensor,
    iou_threshold: float,
    distance_threshold: float,
) -> torch.Tensor:
    """Boolean mask of the boxes to keep, equivalent to torch.ops.point_frustums.nms."""
    _check_boxes(boxes)
    n = boxes.shape[0]
    if not labels.shape[0] == scores.shape[0] == n:
        raise ValueError("The number of labels, scores and boxes must be equal")
    if n == 0:
        return torch.empty((0,), dtype=torch.bool, device=boxes.device)
    boxes = boxes.contiguous()
    labels = labels.to(torch.int32).contiguous()
    sort_idx = scores.argsort().to(torch.int64).contiguous()

    lib = _library()
    meta = torch.empty((n, 4), dtype=torch.float32, device=boxes.device)
    lib.nms_box_meta_kernel(boxes, meta, n, threads=n)
    is_duplicate = torch.zeros((n,), dtype=torch.bool, device=boxes.device)
    lib.nms_duplicates_kernel(
        boxes, meta, labels, sort_idx, is_duplicate, n, float(iou_threshold), float(distance_threshold), threads=(n, n)
    )
    return is_duplicate.logical_not()


_SEGMENT_ATTENTION_HEADER = "segment_attention/segment_attention_utils.h"

# One thread per (head, point) row, see csrc/segment_attention/segment_attention_utils.h. The rows are ordered
# head-major, so the threads of a SIMD group mostly belong to the same segment and read the same keys.
_SEGMENT_ATTENTION_KERNELS = """
kernel void segment_attention_forward(
    device const float* q [[buffer(0)]],             // (N, H, D)
    device const float* k [[buffer(1)]],             // (N, H, D)
    device const float* v [[buffer(2)]],             // (N, H, D)
    device const long* segment_ids [[buffer(3)]],    // (N,)
    device const long* cu_seqlens [[buffer(4)]],     // (S + 1,)
    device float* out [[buffer(5)]],                 // (N, H, D)
    device float* lse [[buffer(6)]],                 // (N, H)
    constant long& N [[buffer(7)]],
    constant long& H [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant float& p_drop [[buffer(10)]],
    constant long& seed [[buffer(11)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= N * H) {
    return;
  }
  const long h = row / N, i = row % N;
  const long s = segment_ids[i];
  segment_attention::ForwardRow<HEAD_DIM>(q, k, v, out, lse, i, h, cu_seqlens[s], cu_seqlens[s + 1], H, scale, p_drop,
                                          static_cast<uint>(seed));
}

kernel void segment_attention_backward_query(
    device const float* grad_out [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const float* k [[buffer(2)]],
    device const float* v [[buffer(3)]],
    device const float* lse [[buffer(4)]],
    device const float* delta [[buffer(5)]],
    device const long* segment_ids [[buffer(6)]],
    device const long* cu_seqlens [[buffer(7)]],
    device float* grad_q [[buffer(8)]],
    constant long& N [[buffer(9)]],
    constant long& H [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    constant float& p_drop [[buffer(12)]],
    constant long& seed [[buffer(13)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= N * H) {
    return;
  }
  const long h = row / N, i = row % N;
  const long s = segment_ids[i];
  segment_attention::BackwardQueryRow<HEAD_DIM>(grad_out, q, k, v, lse, delta, grad_q, i, h, cu_seqlens[s],
                                                cu_seqlens[s + 1], H, scale, p_drop, static_cast<uint>(seed));
}

kernel void segment_attention_backward_key_value(
    device const float* grad_out [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const float* k [[buffer(2)]],
    device const float* v [[buffer(3)]],
    device const float* lse [[buffer(4)]],
    device const float* delta [[buffer(5)]],
    device const long* segment_ids [[buffer(6)]],
    device const long* cu_seqlens [[buffer(7)]],
    device float* grad_k [[buffer(8)]],
    device float* grad_v [[buffer(9)]],
    constant long& N [[buffer(10)]],
    constant long& H [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    constant float& p_drop [[buffer(13)]],
    constant long& seed [[buffer(14)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= N * H) {
    return;
  }
  const long h = row / N, j = row % N;
  const long s = segment_ids[j];
  segment_attention::BackwardKeyValueRow<HEAD_DIM>(grad_out, q, k, v, lse, delta, grad_k, grad_v, j, h,
                                                   cu_seqlens[s], cu_seqlens[s + 1], H, scale, p_drop,
                                                   static_cast<uint>(seed));
}
"""


@cache
def _segment_attention_library(head_dim: int):
    """The kernels are compiled for every head dimension (a template parameter, so the rows fit in registers)."""
    source = "#include <metal_stdlib>\nusing namespace metal;\n"
    source += f"#define HEAD_DIM {head_dim}\n" + _read_header(_SEGMENT_ATTENTION_HEADER)
    return torch.mps.compile_shader(source + _SEGMENT_ATTENTION_KERNELS)


def segment_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    segment_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    dropout_p: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equivalent to torch.ops.point_frustums.segment_attention_forward, returns out (N, H, D) and lse (N, H)."""
    n, h, d = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((n, h), dtype=q.dtype, device=q.device)
    if n > 0:
        _segment_attention_library(d).segment_attention_forward(
            q, k, v, segment_ids, cu_seqlens, out, lse, n, h, float(scale), float(dropout_p), int(seed), threads=n * h
        )
    return out, lse


def segment_attention_backward(  # pylint: disable=too-many-arguments
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    segment_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    dropout_p: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equivalent to torch.ops.point_frustums.segment_attention_backward, returns grad_q, grad_k, grad_v."""
    n, h, d = q.shape
    grad_q, grad_k, grad_v = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    if n > 0:
        delta = (grad_out * out).sum(-1).contiguous()
        lib = _segment_attention_library(d)
        args = (grad_out, q, k, v, lse, delta, segment_ids, cu_seqlens)
        hyper = (n, h, float(scale), float(dropout_p), int(seed))
        lib.segment_attention_backward_query(*args, grad_q, *hyper, threads=n * h)
        lib.segment_attention_backward_key_value(*args, grad_k, grad_v, *hyper, threads=n * h)
    return grad_q, grad_k, grad_v
