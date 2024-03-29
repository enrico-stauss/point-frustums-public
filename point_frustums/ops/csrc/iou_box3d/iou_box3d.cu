/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "iou_box3d/iou_utils.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <math.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>

// Parallelize over N*M computations which can each be done independently
__global__ void IoUBox3DKernel(
    const at::PackedTensorAccessor64<float, 3, at::RestrictPtrTraits> boxes1,
    const at::PackedTensorAccessor64<float, 3, at::RestrictPtrTraits> boxes2,
    at::PackedTensorAccessor64<float, 2, at::RestrictPtrTraits> vols,
    at::PackedTensorAccessor64<float, 2, at::RestrictPtrTraits> ious) {
  const size_t N = boxes1.size(0);
  const size_t M = boxes2.size(0);

  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;

  FaceVerts box1_tris[NUM_TRIS];
  FaceVerts box2_tris[NUM_TRIS];
  FaceVerts box1_planes[NUM_PLANES];
  FaceVerts box2_planes[NUM_PLANES];

  for (size_t i = tid; i < N * M; i += stride) {
    const size_t n = i / M; // box1 index
    const size_t m = i % M; // box2 index

    // Convert to array of structs of face vertices i.e. effectively (F, 3, 3)
    // FaceVerts is a data type defined in iou_utils.cuh
    GetBoxTris(boxes1[n], box1_tris);
    GetBoxTris(boxes2[m], box2_tris);

    // Calculate the position of the center of the box which is used in
    // several calculations. This requires a tensor as input.
    const float3 box1_center = BoxCenter(boxes1[n]);
    const float3 box2_center = BoxCenter(boxes2[m]);

    // Convert to an array of face vertices
    GetBoxPlanes(boxes1[n], box1_planes);
    GetBoxPlanes(boxes2[m], box2_planes);

    // Get Box Volumes
    const float box1_vol = BoxVolume(box1_tris, box1_center, NUM_TRIS);
    const float box2_vol = BoxVolume(box2_tris, box2_center, NUM_TRIS);

    thrust::tuple<float, float> result =
        GetIntersectionAndUnion(box1_center, box2_center, box1_tris, box2_tris,
                                box1_planes, box2_planes, box1_vol, box2_vol);

    // Write the volume and IoU to global memory
    vols[n][m] = thrust::get<0>(result);
    ious[n][m] = thrust::get<1>(result);
  }
}

std::tuple<at::Tensor, at::Tensor>
IoUBox3DCuda(const at::Tensor &boxes1,   // (N, 8, 3)
             const at::Tensor &boxes2) { // (M, 8, 3)
  // Check inputs are on the same device
  at::TensorArg boxes1_t{boxes1, "boxes1", 1}, boxes2_t{boxes2, "boxes2", 2};
  at::CheckedFrom c = "IoUBox3DCuda";
  at::checkAllSameGPU(c, {boxes1_t, boxes2_t});
  at::checkAllSameType(c, {boxes1_t, boxes2_t});

  // Set the device for the kernel launch based on the device of boxes1
  at::cuda::CUDAGuard device_guard(boxes1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(boxes2.size(2) == boxes1.size(2), "Boxes must have shape (8, 3)");

  TORCH_CHECK((boxes2.size(1) == 8) && (boxes1.size(1) == 8),
              "Boxes must have shape (8, 3)");

  const int64_t N = boxes1.size(0);
  const int64_t M = boxes2.size(0);

  auto vols = at::zeros({N, M}, boxes1.options());
  auto ious = at::zeros({N, M}, boxes1.options());

  if (vols.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(vols, ious);
  }

  const size_t blocks = 512;
  const size_t threads = 256;

  IoUBox3DKernel<<<blocks, threads, 0, stream>>>(
      boxes1.packed_accessor64<float, 3, at::RestrictPtrTraits>(),
      boxes2.packed_accessor64<float, 3, at::RestrictPtrTraits>(),
      vols.packed_accessor64<float, 2, at::RestrictPtrTraits>(),
      ious.packed_accessor64<float, 2, at::RestrictPtrTraits>());

  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(vols, ious);
}

// Modification that only computes the pairwise IoU
__global__ void IoUBox3DKernelPairwise(
    const at::PackedTensorAccessor64<float, 3, at::RestrictPtrTraits> boxes1,
    const at::PackedTensorAccessor64<float, 3, at::RestrictPtrTraits> boxes2,
    at::PackedTensorAccessor64<float, 1, at::RestrictPtrTraits> vols,
    at::PackedTensorAccessor64<float, 1, at::RestrictPtrTraits> ious) {
  const size_t N = boxes1.size(0);

  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;

  FaceVerts box1_tris[NUM_TRIS];
  FaceVerts box2_tris[NUM_TRIS];
  FaceVerts box1_planes[NUM_PLANES];
  FaceVerts box2_planes[NUM_PLANES];

  for (size_t i = tid; i < N; i += stride) {
    const size_t n = i; // box index

    // Convert to array of structs of face vertices i.e. effectively (F, 3, 3)
    // FaceVerts is a data type defined in iou_utils.cuh
    GetBoxTris(boxes1[n], box1_tris);
    GetBoxTris(boxes2[n], box2_tris);

    // Calculate the position of the center of the box which is used in
    // several calculations. This requires a tensor as input.
    const float3 box1_center = BoxCenter(boxes1[n]);
    const float3 box2_center = BoxCenter(boxes2[n]);

    // Convert to an array of face vertices
    GetBoxPlanes(boxes1[n], box1_planes);
    GetBoxPlanes(boxes2[n], box2_planes);

    // Get Box Volumes
    const float box1_vol = BoxVolume(box1_tris, box1_center, NUM_TRIS);
    const float box2_vol = BoxVolume(box2_tris, box2_center, NUM_TRIS);

    thrust::tuple<float, float> result =
        GetIntersectionAndUnion(box1_center, box2_center, box1_tris, box2_tris,
                                box1_planes, box2_planes, box1_vol, box2_vol);

    // Write the volume and IoU to global memory
    vols[n] = thrust::get<0>(result);
    ious[n] = thrust::get<1>(result);
  }
}

std::tuple<at::Tensor, at::Tensor>
IoUBox3DCudaPairwise(const at::Tensor &boxes1,   // (N, 8, 3)
                     const at::Tensor &boxes2) { // (N, 8, 3)
  // Check inputs are on the same device
  at::TensorArg boxes1_t{boxes1, "boxes1", 1}, boxes2_t{boxes2, "boxes2", 2};
  at::CheckedFrom c = "IoUBox3DCudaPairwise";
  at::checkAllSameGPU(c, {boxes1_t, boxes2_t});
  at::checkAllSameType(c, {boxes1_t, boxes2_t});

  // Set the device for the kernel launch based on the device of boxes1
  at::cuda::CUDAGuard device_guard(boxes1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(
      boxes2.size(0) == boxes1.size(0),
      "The pairwise IoU3D requires boxes to be of the same shape (N, 8, 3)");
  TORCH_CHECK((boxes2.size(1) == 8) && (boxes1.size(1) == 8),
              "Boxes must have shape (N, 8, 3)");
  TORCH_CHECK(boxes2.size(2) == boxes1.size(2),
              "Boxes must have shape (N, 8, 3)");

  const int64_t N = boxes1.size(0);

  auto vols = at::zeros({N}, boxes1.options());
  auto ious = at::zeros({N}, boxes1.options());

  if (vols.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(vols, ious);
  }

  const size_t blocks = 512;
  const size_t threads = 256;

  IoUBox3DKernelPairwise<<<blocks, threads, 0, stream>>>(
      boxes1.packed_accessor64<float, 3, at::RestrictPtrTraits>(),
      boxes2.packed_accessor64<float, 3, at::RestrictPtrTraits>(),
      vols.packed_accessor64<float, 1, at::RestrictPtrTraits>(),
      ious.packed_accessor64<float, 1, at::RestrictPtrTraits>());

  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(vols, ious);
}