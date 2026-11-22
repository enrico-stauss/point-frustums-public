# Segment attention (block-diagonal attention without padding)

This is exact attention restricted to segments of a packed sequence. In the transformer frustum encoder (TFE), a segment is
the set of points of one frustum. It computes the same thing as xFormers `memory_efficient_attention` with a
`BlockDiagonalMask`, or flash attention "varlen". Unlike those, it runs on CPU, CUDA and MPS from one shared
implementation.

## Why a custom kernel

The points per frustum are very uneven. On nuScenes mini at 360x40, a sample has 4–9k non-empty frustums with
32–52 points on average and up to ~650. The alternatives on the Mac all fall short:

| Approach | Memory | Exact | CPU | CUDA | MPS |
|---|---|---|---|---|---|
| Padded SDPA (pad every frustum to the largest) | O(frustums × max²) | yes | ✓ | ✓ | ✓ (doesn't fit) |
| Padded SDPA with a cap of K points | O(frustums × K²) | no, drops points | ✓ | ✓ | ✓ |
| xFormers `BlockDiagonalMask` | O(N) | yes | – | ✓ | – |
| `torch.nn.attention.varlen.varlen_attn`, nested (jagged) tensors | O(N) | yes | – | ✓ (fp16/bf16) | – |
| FlexAttention with a document mask | O(N) | yes | – | ✓ (Triton) | – |
| **This kernel** | **O(N)** | **yes** | ✓ | ✓ (untested) | ✓ |

## Design

- **Layout:**
  - The points are sorted by segment and packed without padding.
  - `q`, `k`, `v` are `[N, H, D]`.
  - The segments are given by `cu_seqlens` `[S + 1]` (the same convention as flash attention varlen) and
    `segment_ids` `[N]`.
- **Forward:**
  - One thread per (head, point) row loops over the keys of its segment with an online softmax, as flash attention
    does.
  - It stores only the output and the log-sum-exp (LSE) per row.
- **Backward:**
  - The probabilities are recomputed from the LSE; the attention matrix is never stored.
  - Two passes without atomics: one thread per query row computes `dQ`, and one thread per key row computes `dK`
    and `dV`.
  - `delta = rowsum(dO * O)` is precomputed.
- **Dropout** on the attention probabilities uses a counter-based hash of `(seed, head, i, j)`. The backward pass
  reproduces the mask without storing it.
- **Rows are ordered head-major,** so the threads of a SIMD group or warp mostly belong to the same segment. They
  read the same keys at the same time, which gives broadcast loads.
- **One algorithm for three backends:**
  - The algorithm lives in `segment_attention_utils.h`, templated on the head dimension (`D` ∈ {4, 8, 16, 32, 64}).
  - It is compiled by the C++ compiler (CPU, `segment_attention_cpu.cpp`) and by nvcc (`segment_attention_cuda.cu`).
  - On MPS it is compiled at runtime with `torch.mps.compile_shader` (`ops/metal.py`), with the same address-space
    macros as the IoU and NMS headers.
- **Python entry point:** `point_frustums.ops.segment_attention.segment_attention`, an autograd function.
  `segment_softmax_pooling` handles the cross attention (one query per segment) exactly with scatter operations; it
  needs no kernel.

## Status

**Correctness** (`test/ops/test_segment_attention.py`):
- Forward and all gradients match a dense float64 reference to ~1e-6 on CPU and MPS, including with dropout (the
  mask is reproduced by a Python port of the hash).
- The CUDA kernel compiles from the same header but has not been built or run (no CUDA machine available).

**Performance:**
- MPS, synthetic layout: 232k points, 7000 segments up to 650 points (Σn² = 22M), H = 2, D = 16.
  - Forward: 6.6 ms.
  - Forward + backward: 24 ms.
  - The capped, padded SDPA with K = 64 takes 60 ms for forward + backward.
  - There is no extra memory beyond the inputs and outputs.
- In the full TFE (2 encoders, E = 32, ~265k points per sample), the kernel accounts for ~50 of ~400 ms per sample.
  The rest is the dense per-point layers (LayerNorm, qkv, FFN), which are bandwidth-bound. Optimizing the kernel further
  (threadgroup tiling of the keys) would gain little on the Mac.
- Training on nuScenes mini at 360x40, bs 4, MPS: ~11 GB driver memory (peak 12) and ~3.9 s/step, against
  6–8 GB and ~1.6 s/step for the PointNet-like encoder.
- CPU: 0.4 s forward, 2.5 s forward + backward for the synthetic layout. The extension is built without OpenMP on
  macOS, so `at::parallel_for` probably runs single-threaded; the CPU path is meant for testing.

**Open points:**
- Build and test on CUDA.
  - Compare against `varlen_attn` (needs fp16/bf16) and decide whether to use the native path there.
  - For long segments, a tiled version (keys in shared memory, one warp per query block) would be faster.
- Half precision inputs are converted to float32; the kernels compute in float32.
