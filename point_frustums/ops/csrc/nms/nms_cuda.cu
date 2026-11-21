//
// Created by Enrico Stauss on 19.01.24.
//
#include "iou_box3d/iou_utils.h"
#include "nms/nms_utils.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAGuard.h>

// The CUDA implementation evaluates the IoU for all pairs of boxes (i, j) where box j has the same class, a higher
// score and is close enough. If the IoU reaches the threshold, box i is marked as duplicate (it has a better scoring
// alternative) and is removed. This is equivalent to the CPU implementation.

__global__ void BoxesMetaKernel(const float *__restrict__ boxes, // (N, 8, 3)
                                const int64_t N,
                                float *__restrict__ meta) { // (N, kBoxMetaSize)
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) {
    return;
  }
  ComputeBoxMeta(boxes + 24 * tid, meta + kBoxMetaSize * tid);
}

__global__ void NMSDuplicateKernel(const float *__restrict__ boxes,      // (N, 8, 3)
                                   const float *__restrict__ meta,       // (N, kBoxMetaSize)
                                   const int32_t *__restrict__ labels,   // (N,)
                                   const int64_t *__restrict__ sort_idx, // (N,) ascending score
                                   const int64_t N, const double iou_threshold, const double distance_threshold,
                                   bool *__restrict__ is_duplicate) { // (N,)
  // Parallelize over all combinations of (lower score, higher score) with a 2D grid
  const int64_t tid_i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int64_t tid_j = blockIdx.y * static_cast<int64_t>(blockDim.y) + threadIdx.y;
  if (tid_i >= N || tid_j >= N || tid_i >= tid_j) {
    return;
  }

  const int64_t i = sort_idx[tid_i];
  const int64_t j = sort_idx[tid_j];
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

at::Tensor NMSCuda(const at::Tensor &labels, const at::Tensor &scores, const at::Tensor &boxes,
                   const double iou_threshold, const double distance_threshold) {
  // Check inputs are on the same device
  at::TensorArg labels_t{labels, "labels", 1}, scores_t{scores, "scores", 2}, boxes_t{boxes, "boxes", 3};
  at::CheckedFrom c = "NMSCuda";
  at::checkAllSameGPU(c, {boxes_t, scores_t, labels_t});
  at::checkAllSameType(c, {boxes_t, scores_t});

  TORCH_CHECK(boxes.dim() == 3 && boxes.size(1) == 8 && boxes.size(2) == 3, "Boxes must be of shape (N, 8, 3)");
  TORCH_CHECK(boxes.size(0) == scores.size(0), "The number of boxes must equal the number of scores");
  TORCH_CHECK(labels.size(0) == scores.size(0), "The number of labels must equal the number of scores");

  const int64_t N = boxes.size(0);
  if (N == 0) {
    return at::empty({0}, boxes.options().dtype(at::kBool));
  }

  // Set the device for the kernel launch based on the device of the boxes
  at::cuda::CUDAGuard device_guard(boxes.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // 0. Sort by ascending score
  const auto sort_idx = scores.argsort().to(at::kLong).contiguous();
  const auto labels_int = labels.to(at::kInt).contiguous();

  // 1. Evaluate the center and volume of all boxes
  auto meta = at::empty({N, kBoxMetaSize}, boxes.options().dtype(at::kFloat));
  const int64_t threads_meta = 256;
  BoxesMetaKernel<<<(N + threads_meta - 1) / threads_meta, threads_meta, 0, stream>>>(boxes.data_ptr<float>(), N,
                                                                                         meta.data_ptr<float>());
  AT_CUDA_CHECK(cudaGetLastError());

  // 2. Mark every box that has a better scoring duplicate
  auto is_duplicate = at::zeros({N}, boxes.options().dtype(at::kBool));
  const dim3 threads(16, 16);
  const int64_t blocks_per_dim = (N + threads.x - 1) / threads.x;
  TORCH_CHECK(blocks_per_dim <= 65535, "NMSCuda supports at most ", 65535 * threads.x, " boxes");
  const dim3 blocks(blocks_per_dim, blocks_per_dim);
  NMSDuplicateKernel<<<blocks, threads, 0, stream>>>(
      boxes.data_ptr<float>(), meta.data_ptr<float>(), labels_int.data_ptr<int32_t>(), sort_idx.data_ptr<int64_t>(), N,
      iou_threshold, distance_threshold, is_duplicate.data_ptr<bool>());
  AT_CUDA_CHECK(cudaGetLastError());

  // 3. Keep all boxes that are no duplicate
  return is_duplicate.logical_not();
}
