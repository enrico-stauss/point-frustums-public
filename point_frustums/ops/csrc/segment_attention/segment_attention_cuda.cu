#include "segment_attention/segment_attention_utils.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>

// One thread per (head, point) row. The rows are ordered head-major, so the threads of a warp mostly belong to the
// same segment and read the same keys at the same time (broadcast loads).

template <int D>
__global__ void SegmentAttentionForwardKernel(const float *__restrict__ q, const float *__restrict__ k,
                                              const float *__restrict__ v, const int64_t *__restrict__ segment,
                                              const int64_t *__restrict__ cu, const int64_t N, const int64_t H,
                                              const float scale, const float p_drop, const uint32_t seed,
                                              float *__restrict__ out, float *__restrict__ lse) {
  const int64_t row = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (row >= N * H) {
    return;
  }
  const int64_t h = row / N, i = row % N;
  const int64_t s = segment[i];
  segment_attention::ForwardRow<D>(q, k, v, out, lse, i, h, cu[s], cu[s + 1], H, scale, p_drop, seed);
}

template <int D>
__global__ void SegmentAttentionBackwardQueryKernel(const float *__restrict__ grad_out, const float *__restrict__ q,
                                                    const float *__restrict__ k, const float *__restrict__ v,
                                                    const float *__restrict__ lse, const float *__restrict__ delta,
                                                    const int64_t *__restrict__ segment,
                                                    const int64_t *__restrict__ cu, const int64_t N, const int64_t H,
                                                    const float scale, const float p_drop, const uint32_t seed,
                                                    float *__restrict__ grad_q) {
  const int64_t row = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (row >= N * H) {
    return;
  }
  const int64_t h = row / N, i = row % N;
  const int64_t s = segment[i];
  segment_attention::BackwardQueryRow<D>(grad_out, q, k, v, lse, delta, grad_q, i, h, cu[s], cu[s + 1], H, scale,
                                         p_drop, seed);
}

template <int D>
__global__ void SegmentAttentionBackwardKeyValueKernel(
    const float *__restrict__ grad_out, const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, const float *__restrict__ lse, const float *__restrict__ delta,
    const int64_t *__restrict__ segment, const int64_t *__restrict__ cu, const int64_t N, const int64_t H,
    const float scale, const float p_drop, const uint32_t seed, float *__restrict__ grad_k,
    float *__restrict__ grad_v) {
  const int64_t row = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (row >= N * H) {
    return;
  }
  const int64_t h = row / N, j = row % N;
  const int64_t s = segment[j];
  segment_attention::BackwardKeyValueRow<D>(grad_out, q, k, v, lse, delta, grad_k, grad_v, j, h, cu[s], cu[s + 1], H,
                                            scale, p_drop, seed);
}

#define SEGMENT_ATTENTION_CUDA_DISPATCH(HEAD_DIM, KERNEL, GRID, BLOCK, STREAM, ...)                                    \
  switch (HEAD_DIM) {                                                                                                  \
  case 4:                                                                                                              \
    KERNEL<4><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__);                                                                \
    break;                                                                                                             \
  case 8:                                                                                                              \
    KERNEL<8><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__);                                                                \
    break;                                                                                                             \
  case 16:                                                                                                             \
    KERNEL<16><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__);                                                               \
    break;                                                                                                             \
  case 32:                                                                                                             \
    KERNEL<32><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__);                                                               \
    break;                                                                                                             \
  case 64:                                                                                                             \
    KERNEL<64><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__);                                                               \
    break;                                                                                                             \
  default:                                                                                                             \
    TORCH_CHECK(false, "Unsupported head dimension ", HEAD_DIM);                                                      \
  }

std::tuple<at::Tensor, at::Tensor> SegmentAttentionForwardCuda(const at::Tensor &q, const at::Tensor &k,
                                                               const at::Tensor &v, const at::Tensor &segment_ids,
                                                               const at::Tensor &cu_seqlens, const double scale,
                                                               const double dropout_p, const int64_t seed) {
  at::cuda::CUDAGuard device_guard(q.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t N = q.size(0), H = q.size(1), D = q.size(2);
  at::Tensor out = at::empty_like(q);
  at::Tensor lse = at::empty({N, H}, q.options());
  if (N == 0) {
    return {out, lse};
  }
  const int threads = 128;
  const int64_t blocks = (N * H + threads - 1) / threads;
  SEGMENT_ATTENTION_CUDA_DISPATCH(D, SegmentAttentionForwardKernel, blocks, threads, stream, q.data_ptr<float>(),
                                  k.data_ptr<float>(), v.data_ptr<float>(), segment_ids.data_ptr<int64_t>(),
                                  cu_seqlens.data_ptr<int64_t>(), N, H, static_cast<float>(scale),
                                  static_cast<float>(dropout_p), static_cast<uint32_t>(seed), out.data_ptr<float>(),
                                  lse.data_ptr<float>());
  AT_CUDA_CHECK(cudaGetLastError());
  return {out, lse};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
SegmentAttentionBackwardCuda(const at::Tensor &grad_out, const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                             const at::Tensor &out, const at::Tensor &lse, const at::Tensor &segment_ids,
                             const at::Tensor &cu_seqlens, const double scale, const double dropout_p,
                             const int64_t seed) {
  at::cuda::CUDAGuard device_guard(q.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t N = q.size(0), H = q.size(1), D = q.size(2);
  at::Tensor grad_q = at::empty_like(q), grad_k = at::empty_like(k), grad_v = at::empty_like(v);
  if (N == 0) {
    return {grad_q, grad_k, grad_v};
  }
  const at::Tensor delta = (grad_out * out).sum(-1).contiguous();
  const int threads = 128;
  const int64_t blocks = (N * H + threads - 1) / threads;
  const float scale_ = static_cast<float>(scale), p_drop = static_cast<float>(dropout_p);
  const uint32_t seed_ = static_cast<uint32_t>(seed);
  SEGMENT_ATTENTION_CUDA_DISPATCH(D, SegmentAttentionBackwardQueryKernel, blocks, threads, stream,
                                  grad_out.data_ptr<float>(), q.data_ptr<float>(), k.data_ptr<float>(),
                                  v.data_ptr<float>(), lse.data_ptr<float>(), delta.data_ptr<float>(),
                                  segment_ids.data_ptr<int64_t>(), cu_seqlens.data_ptr<int64_t>(), N, H, scale_, p_drop,
                                  seed_, grad_q.data_ptr<float>());
  SEGMENT_ATTENTION_CUDA_DISPATCH(D, SegmentAttentionBackwardKeyValueKernel, blocks, threads, stream,
                                  grad_out.data_ptr<float>(), q.data_ptr<float>(), k.data_ptr<float>(),
                                  v.data_ptr<float>(), lse.data_ptr<float>(), delta.data_ptr<float>(),
                                  segment_ids.data_ptr<int64_t>(), cu_seqlens.data_ptr<int64_t>(), N, H, scale_, p_drop,
                                  seed_, grad_k.data_ptr<float>(), grad_v.data_ptr<float>());
  AT_CUDA_CHECK(cudaGetLastError());
  return {grad_q, grad_k, grad_v};
}
