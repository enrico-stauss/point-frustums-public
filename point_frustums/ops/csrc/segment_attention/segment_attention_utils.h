#pragma once
#ifndef __METAL_VERSION__
#include <math.h>
#include <stdint.h>
#endif

// Exact attention within segments of a packed sequence (block-diagonal attention, as with the xFormers
// BlockDiagonalMask or flash attention "varlen"), shared by the CPU, CUDA and Metal implementations.
//
// Layout: the points are sorted by segment (frustum) and packed without padding; q, k, v, out and the gradients are
// [N, H, D], lse and delta are [N, H]. Point i attends to all points j of its segment [cu_seqlens[s], cu_seqlens[s+1]).
//
// Each thread processes one (point, head) row and loops over the keys of its segment with an online softmax (as flash
// attention does), so neither the forward nor the backward pass stores the attention matrix: the memory is O(N * H * D)
// and independent of the segment sizes. The backward pass recomputes the probabilities from the log-sum-exp (lse) of
// the forward pass:
//   dQ row i:  dq_i = scale * sum_j P_ij * (Z_ij * dO_i . v_j - delta_i) * k_j
//   dK/dV row j: dv_j = sum_i P_ij * Z_ij * dO_i,  dk_j = scale * sum_i P_ij * (Z_ij * dO_i . v_j - delta_i) * q_i
// with delta_i = dO_i . O_i and Z_ij the (rescaled) dropout mask. Both run without atomics.
//
// Dropout on the attention probabilities uses a counter-based hash of (seed, head, i, j), so the backward pass
// reproduces the mask of the forward pass without storing it.

#ifdef __CUDACC__
#define SA_HOST_DEVICE __host__ __device__
#else
#define SA_HOST_DEVICE
#endif

#ifdef __METAL_VERSION__
#define SA_GLOBAL device
typedef long sa_long;
typedef uint sa_uint;
#else
#define SA_GLOBAL
typedef int64_t sa_long;
typedef uint32_t sa_uint;
#endif

namespace segment_attention {

// Integer hash with good avalanche properties (lowbias32 by Chris Wellons)
SA_HOST_DEVICE inline sa_uint Hash(sa_uint x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

// Dropout multiplier Z_ij of the probability P_ij: 0 if dropped, else 1 / (1 - p_drop)
SA_HOST_DEVICE inline float DropoutScale(const sa_uint seed, const sa_long h, const sa_long i, const sa_long j,
                                         const float p_drop) {
  if (p_drop <= 0.0f) {
    return 1.0f;
  }
  sa_uint x = Hash(seed ^ Hash(static_cast<sa_uint>(h) + 0x9e3779b9U));
  x = Hash(x ^ static_cast<sa_uint>(i));
  x = Hash(x + static_cast<sa_uint>(j) * 0x85ebca6bU);
  // Uniform in [0, 1) with 24 bits
  const float u = static_cast<float>(x >> 8) * (1.0f / 16777216.0f);
  return u < p_drop ? 0.0f : 1.0f / (1.0f - p_drop);
}

// Forward pass of the row (i, h): out[i, h] and lse[i, h]
template <int D>
SA_HOST_DEVICE inline void ForwardRow(const SA_GLOBAL float *q, const SA_GLOBAL float *k, const SA_GLOBAL float *v,
                                      SA_GLOBAL float *out, SA_GLOBAL float *lse, const sa_long i, const sa_long h,
                                      const sa_long start, const sa_long end, const sa_long H, const float scale,
                                      const float p_drop, const sa_uint seed) {
  float q_i[D];
  float acc[D];
  for (int d = 0; d < D; ++d) {
    q_i[d] = q[(i * H + h) * D + d] * scale;
    acc[d] = 0.0f;
  }
  // Running maximum and normalizer of the online softmax; the maximum is initialized with the first score instead of
  // -inf as Metal compiles with fast math, where infinities are not guaranteed to be handled.
  float m = 0.0f;
  float l = 0.0f;
  for (sa_long j = start; j < end; ++j) {
    const SA_GLOBAL float *k_j = k + (j * H + h) * D;
    float s = 0.0f;
    for (int d = 0; d < D; ++d) {
      s += q_i[d] * k_j[d];
    }
    if (j == start) {
      m = s;
    } else if (s > m) {
      const float correction = exp(m - s);
      l *= correction;
      for (int d = 0; d < D; ++d) {
        acc[d] *= correction;
      }
      m = s;
    }
    const float p = exp(s - m);
    l += p;
    const float w = p * DropoutScale(seed, h, i, j, p_drop);
    if (w != 0.0f) {
      const SA_GLOBAL float *v_j = v + (j * H + h) * D;
      for (int d = 0; d < D; ++d) {
        acc[d] += w * v_j[d];
      }
    }
  }
  const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
  for (int d = 0; d < D; ++d) {
    out[(i * H + h) * D + d] = acc[d] * inv_l;
  }
  lse[i * H + h] = l > 0.0f ? m + log(l) : 0.0f;
}

// Backward pass of the query row (i, h): dq[i, h]
template <int D>
SA_HOST_DEVICE inline void BackwardQueryRow(const SA_GLOBAL float *grad_out, const SA_GLOBAL float *q,
                                            const SA_GLOBAL float *k, const SA_GLOBAL float *v,
                                            const SA_GLOBAL float *lse, const SA_GLOBAL float *delta,
                                            SA_GLOBAL float *grad_q, const sa_long i, const sa_long h,
                                            const sa_long start, const sa_long end, const sa_long H, const float scale,
                                            const float p_drop, const sa_uint seed) {
  float q_i[D];
  float do_i[D];
  float dq[D];
  for (int d = 0; d < D; ++d) {
    q_i[d] = q[(i * H + h) * D + d] * scale;
    do_i[d] = grad_out[(i * H + h) * D + d];
    dq[d] = 0.0f;
  }
  const float lse_i = lse[i * H + h];
  const float delta_i = delta[i * H + h];
  for (sa_long j = start; j < end; ++j) {
    const SA_GLOBAL float *k_j = k + (j * H + h) * D;
    const SA_GLOBAL float *v_j = v + (j * H + h) * D;
    float s = 0.0f;
    float dp = 0.0f;
    for (int d = 0; d < D; ++d) {
      s += q_i[d] * k_j[d];
      dp += do_i[d] * v_j[d];
    }
    const float p = exp(s - lse_i);
    const float ds = p * (dp * DropoutScale(seed, h, i, j, p_drop) - delta_i);
    for (int d = 0; d < D; ++d) {
      dq[d] += ds * k_j[d];
    }
  }
  for (int d = 0; d < D; ++d) {
    grad_q[(i * H + h) * D + d] = dq[d] * scale;
  }
}

// Backward pass of the key/value row (j, h): dk[j, h] and dv[j, h]
template <int D>
SA_HOST_DEVICE inline void BackwardKeyValueRow(const SA_GLOBAL float *grad_out, const SA_GLOBAL float *q,
                                               const SA_GLOBAL float *k, const SA_GLOBAL float *v,
                                               const SA_GLOBAL float *lse, const SA_GLOBAL float *delta,
                                               SA_GLOBAL float *grad_k, SA_GLOBAL float *grad_v, const sa_long j,
                                               const sa_long h, const sa_long start, const sa_long end,
                                               const sa_long H, const float scale, const float p_drop,
                                               const sa_uint seed) {
  float k_j[D];
  float v_j[D];
  float dk[D];
  float dv[D];
  for (int d = 0; d < D; ++d) {
    k_j[d] = k[(j * H + h) * D + d] * scale;
    v_j[d] = v[(j * H + h) * D + d];
    dk[d] = 0.0f;
    dv[d] = 0.0f;
  }
  for (sa_long i = start; i < end; ++i) {
    const SA_GLOBAL float *q_i = q + (i * H + h) * D;
    const SA_GLOBAL float *do_i = grad_out + (i * H + h) * D;
    float s = 0.0f;
    float dp = 0.0f;
    for (int d = 0; d < D; ++d) {
      s += q_i[d] * k_j[d];
      dp += do_i[d] * v_j[d];
    }
    const float p = exp(s - lse[i * H + h]);
    const float z = DropoutScale(seed, h, i, j, p_drop);
    const float ds = p * (dp * z - delta[i * H + h]);
    const float pz = p * z;
    for (int d = 0; d < D; ++d) {
      dv[d] += pz * do_i[d];
      dk[d] += ds * q_i[d];
    }
  }
  for (int d = 0; d < D; ++d) {
    grad_k[(j * H + h) * D + d] = dk[d] * scale;
    grad_v[(j * H + h) * D + d] = dv[d];
  }
}

} // namespace segment_attention
