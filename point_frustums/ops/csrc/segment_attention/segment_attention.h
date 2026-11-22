#pragma once
#include "utils/pytorch3d_cutils.h"
#include <torch/extension.h>
#include <tuple>

// Exact attention within segments of a packed sequence (see segment_attention_utils.h)
//
// Args:
//     q, k, v: float32 tensors of shape (N, H, D), sorted by segment, D in {4, 8, 16, 32, 64}
//     segment_ids: int64 tensor of shape (N,), the segment of each point
//     cu_seqlens: int64 tensor of shape (S + 1,), the start of each segment and N
//     scale: the scale of the scores, usually 1 / sqrt(D)
//     dropout_p, seed: dropout on the attention probabilities, reproduced in the backward pass by the seed
// Returns:
//     out: (N, H, D)
//     lse: (N, H) the log-sum-exp of the scores, required by the backward pass

std::tuple<at::Tensor, at::Tensor> SegmentAttentionForwardCpu(const at::Tensor &q, const at::Tensor &k,
                                                              const at::Tensor &v, const at::Tensor &segment_ids,
                                                              const at::Tensor &cu_seqlens, double scale,
                                                              double dropout_p, int64_t seed);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
SegmentAttentionBackwardCpu(const at::Tensor &grad_out, const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                            const at::Tensor &out, const at::Tensor &lse, const at::Tensor &segment_ids,
                            const at::Tensor &cu_seqlens, double scale, double dropout_p, int64_t seed);

std::tuple<at::Tensor, at::Tensor> SegmentAttentionForwardCuda(const at::Tensor &q, const at::Tensor &k,
                                                               const at::Tensor &v, const at::Tensor &segment_ids,
                                                               const at::Tensor &cu_seqlens, double scale,
                                                               double dropout_p, int64_t seed);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
SegmentAttentionBackwardCuda(const at::Tensor &grad_out, const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                             const at::Tensor &out, const at::Tensor &lse, const at::Tensor &segment_ids,
                             const at::Tensor &cu_seqlens, double scale, double dropout_p, int64_t seed);

inline void SegmentAttentionCheckInputs(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                                        const at::Tensor &segment_ids, const at::Tensor &cu_seqlens) {
  TORCH_CHECK(q.dim() == 3 && q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q, k, v must be of shape (N, H, D)");
  TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat && v.scalar_type() == at::kFloat,
              "q, k, v must be float32");
  TORCH_CHECK(segment_ids.dim() == 1 && segment_ids.size(0) == q.size(0), "segment_ids must be of shape (N,)");
  TORCH_CHECK(segment_ids.scalar_type() == at::kLong && cu_seqlens.scalar_type() == at::kLong,
              "segment_ids and cu_seqlens must be int64");
  const int64_t D = q.size(2);
  TORCH_CHECK(D == 4 || D == 8 || D == 16 || D == 32 || D == 64, "The head dimension must be one of 4, 8, 16, 32, 64");
}

// Calls FN<D>() for the runtime head dimension
#define SEGMENT_ATTENTION_DISPATCH_HEAD_DIM(HEAD_DIM, ...)                                                             \
  [&] {                                                                                                                \
    switch (HEAD_DIM) {                                                                                                \
    case 4: {                                                                                                          \
      constexpr int kD = 4;                                                                                            \
      return __VA_ARGS__();                                                                                            \
    }                                                                                                                  \
    case 8: {                                                                                                          \
      constexpr int kD = 8;                                                                                            \
      return __VA_ARGS__();                                                                                            \
    }                                                                                                                  \
    case 16: {                                                                                                         \
      constexpr int kD = 16;                                                                                           \
      return __VA_ARGS__();                                                                                            \
    }                                                                                                                  \
    case 32: {                                                                                                         \
      constexpr int kD = 32;                                                                                           \
      return __VA_ARGS__();                                                                                            \
    }                                                                                                                  \
    case 64: {                                                                                                         \
      constexpr int kD = 64;                                                                                           \
      return __VA_ARGS__();                                                                                            \
    }                                                                                                                  \
    default:                                                                                                           \
      TORCH_CHECK(false, "Unsupported head dimension ", HEAD_DIM);                                                    \
    }                                                                                                                  \
  }()

inline std::tuple<at::Tensor, at::Tensor> SegmentAttentionForward(const at::Tensor &q, const at::Tensor &k,
                                                                  const at::Tensor &v, const at::Tensor &segment_ids,
                                                                  const at::Tensor &cu_seqlens, double scale,
                                                                  double dropout_p, int64_t seed) {
  SegmentAttentionCheckInputs(q, k, v, segment_ids, cu_seqlens);
  if (q.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(k);
    CHECK_CUDA(v);
    CHECK_CUDA(segment_ids);
    CHECK_CUDA(cu_seqlens);
    return SegmentAttentionForwardCuda(q.contiguous(), k.contiguous(), v.contiguous(), segment_ids.contiguous(),
                                       cu_seqlens.contiguous(), scale, dropout_p, seed);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return SegmentAttentionForwardCpu(q.contiguous(), k.contiguous(), v.contiguous(), segment_ids.contiguous(),
                                    cu_seqlens.contiguous(), scale, dropout_p, seed);
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
SegmentAttentionBackward(const at::Tensor &grad_out, const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                         const at::Tensor &out, const at::Tensor &lse, const at::Tensor &segment_ids,
                         const at::Tensor &cu_seqlens, double scale, double dropout_p, int64_t seed) {
  SegmentAttentionCheckInputs(q, k, v, segment_ids, cu_seqlens);
  if (q.is_cuda()) {
#ifdef WITH_CUDA
    return SegmentAttentionBackwardCuda(grad_out.contiguous(), q.contiguous(), k.contiguous(), v.contiguous(),
                                        out.contiguous(), lse.contiguous(), segment_ids.contiguous(),
                                        cu_seqlens.contiguous(), scale, dropout_p, seed);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return SegmentAttentionBackwardCpu(grad_out.contiguous(), q.contiguous(), k.contiguous(), v.contiguous(),
                                     out.contiguous(), lse.contiguous(), segment_ids.contiguous(),
                                     cu_seqlens.contiguous(), scale, dropout_p, seed);
}
