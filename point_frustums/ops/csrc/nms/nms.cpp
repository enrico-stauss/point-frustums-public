//
// Created by Enrico Stauss on 18.01.24.
//
#include "nms/nms_utils.h"
#include "iou_box3d/iou_utils.h"
#include <ATen/Parallel.h>
#include <torch/extension.h>

// Strategy:
// 1. Get the index that sorts the scores in ascending order
// 2. Precalculate the center and volume of each box
// 3. For each box i (parallelized): iterate over all boxes j with a higher score
//  - Skip if the label differs
//  - Skip if the center distance scaled by the volume of box i exceeds the distance threshold
//  - Evaluate the IoU; if it reaches the threshold, box i is removed (a better scoring duplicate exists)
at::Tensor NMSCpu(const at::Tensor &labels_t, const at::Tensor &scores, const at::Tensor &boxes,
                  const double iou_threshold, const double distance_threshold) {

  if (labels_t.numel() == 0) {
    return at::empty({0}, boxes.options().dtype(at::kBool));
  }
  TORCH_CHECK(boxes.dim() == 3 && boxes.size(1) == 8 && boxes.size(2) == 3, "Boxes must be of shape (N, 8, 3)");
  TORCH_CHECK(boxes.size(0) == scores.size(0), "The number of boxes must equal the number of scores");
  TORCH_CHECK(labels_t.size(0) == scores.size(0), "The number of labels must equal the number of scores");

  const int64_t N = labels_t.size(0);
  const float *boxes_data = boxes.data_ptr<float>();
  const int32_t *labels = labels_t.data_ptr<int32_t>();

  std::vector<float> meta(N * kBoxMetaSize);
  for (int64_t i = 0; i < N; ++i) {
    ComputeBoxMeta(boxes_data + 24 * i, meta.data() + kBoxMetaSize * i);
  }

  const at::Tensor sorted_idx_t = std::get<1>(scores.sort(0)).to(at::kLong);
  const int64_t *sorted_idx = sorted_idx_t.data_ptr<int64_t>();

  // Mask of the boxes that are kept
  at::Tensor remains_t = torch::ones({N}, labels_t.options().dtype(at::kBool));
  bool *remains = remains_t.data_ptr<bool>();

  // Each box only depends on the boxes with a higher score, not on whether those are kept, so the outer loop can be
  // parallelized.
  at::parallel_for(0, N, 16, [&](int64_t begin, int64_t end) {
    for (int64_t _i = begin; _i < end; ++_i) {
      const int64_t i = sorted_idx[_i];
      const float *meta_i = meta.data() + kBoxMetaSize * i;
      for (int64_t _j = _i + 1; _j < N; ++_j) {
        const int64_t j = sorted_idx[_j];
        if (labels[i] != labels[j]) {
          continue;
        }
        if (ExceedsDistance(meta_i, meta.data() + kBoxMetaSize * j, distance_threshold)) {
          continue;
        }
        float vol, iou;
        iou_box3d::BoxOverlap(boxes_data + 24 * i, boxes_data + 24 * j, vol, iou);
        if (iou >= iou_threshold) {
          remains[i] = false;
          break;
        }
      }
    }
  });

  return remains_t;
}
