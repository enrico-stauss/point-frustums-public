#include "segment_attention/segment_attention.h"
#include "segment_attention/segment_attention_utils.h"
#include <ATen/Parallel.h>
#include <torch/extension.h>

// One task per (head, point) row, rows of the same head and segment are adjacent

std::tuple<at::Tensor, at::Tensor> SegmentAttentionForwardCpu(const at::Tensor &q, const at::Tensor &k,
                                                              const at::Tensor &v, const at::Tensor &segment_ids,
                                                              const at::Tensor &cu_seqlens, const double scale,
                                                              const double dropout_p, const int64_t seed) {
  const int64_t N = q.size(0), H = q.size(1), D = q.size(2);
  at::Tensor out = at::empty_like(q);
  at::Tensor lse = at::empty({N, H}, q.options());
  if (N == 0) {
    return {out, lse};
  }
  const float *q_ = q.data_ptr<float>(), *k_ = k.data_ptr<float>(), *v_ = v.data_ptr<float>();
  const int64_t *segment = segment_ids.data_ptr<int64_t>(), *cu = cu_seqlens.data_ptr<int64_t>();
  float *out_ = out.data_ptr<float>(), *lse_ = lse.data_ptr<float>();
  const uint32_t seed_ = static_cast<uint32_t>(seed);
  SEGMENT_ATTENTION_DISPATCH_HEAD_DIM(D, [&] {
    at::parallel_for(0, N * H, 64, [&](int64_t begin, int64_t end) {
      for (int64_t row = begin; row < end; ++row) {
        const int64_t h = row / N, i = row % N;
        const int64_t s = segment[i];
        segment_attention::ForwardRow<kD>(q_, k_, v_, out_, lse_, i, h, cu[s], cu[s + 1], H, scale, dropout_p,
                                          seed_);
      }
    });
  });
  return {out, lse};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
SegmentAttentionBackwardCpu(const at::Tensor &grad_out, const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                            const at::Tensor &out, const at::Tensor &lse, const at::Tensor &segment_ids,
                            const at::Tensor &cu_seqlens, const double scale, const double dropout_p,
                            const int64_t seed) {
  const int64_t N = q.size(0), H = q.size(1), D = q.size(2);
  at::Tensor grad_q = at::empty_like(q), grad_k = at::empty_like(k), grad_v = at::empty_like(v);
  if (N == 0) {
    return {grad_q, grad_k, grad_v};
  }
  const at::Tensor delta = (grad_out * out).sum(-1).contiguous();
  const float *do_ = grad_out.data_ptr<float>(), *q_ = q.data_ptr<float>(), *k_ = k.data_ptr<float>(),
              *v_ = v.data_ptr<float>(), *lse_ = lse.data_ptr<float>(), *delta_ = delta.data_ptr<float>();
  const int64_t *segment = segment_ids.data_ptr<int64_t>(), *cu = cu_seqlens.data_ptr<int64_t>();
  float *dq_ = grad_q.data_ptr<float>(), *dk_ = grad_k.data_ptr<float>(), *dv_ = grad_v.data_ptr<float>();
  const uint32_t seed_ = static_cast<uint32_t>(seed);
  SEGMENT_ATTENTION_DISPATCH_HEAD_DIM(D, [&] {
    at::parallel_for(0, N * H, 64, [&](int64_t begin, int64_t end) {
      for (int64_t row = begin; row < end; ++row) {
        const int64_t h = row / N, i = row % N;
        const int64_t s = segment[i];
        segment_attention::BackwardQueryRow<kD>(do_, q_, k_, v_, lse_, delta_, dq_, i, h, cu[s], cu[s + 1], H, scale,
                                                dropout_p, seed_);
        segment_attention::BackwardKeyValueRow<kD>(do_, q_, k_, v_, lse_, delta_, dk_, dv_, i, h, cu[s], cu[s + 1], H,
                                                   scale, dropout_p, seed_);
      }
    });
  });
  return {grad_q, grad_k, grad_v};
}
