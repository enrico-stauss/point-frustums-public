//
// Created by Enrico Stauss on 15.03.24.
//

#pragma once
#include "iou_box3d/iou_utils.h"

// Per-box quantities used by the NMS pre-check (center distance scaled by volume), shared by the CPU and CUDA
// implementations. The IoU itself is evaluated by iou_box3d::BoxOverlap on the raw corners.
constexpr int kBoxMetaSize = 4; // [center_x, center_y, center_z, volume]

IOU_HOST_DEVICE inline void ComputeBoxMeta(const float *box, float *meta) {
  iou_box3d::Point<float> corners[8];
  iou_box3d::LoadBox(box, corners);
  const iou_box3d::Point<float> center = iou_box3d::BoxCenter(corners);
  meta[0] = center.x;
  meta[1] = center.y;
  meta[2] = center.z;
  meta[3] = fabsf(iou_box3d::BoxSignedVolume(corners, center));
}

// Returns true if the center distance between the boxes, scaled by the volume of box i, exceeds the threshold
IOU_HOST_DEVICE inline bool ExceedsDistance(const float *meta_i, const float *meta_j, const double distance_threshold) {
  const float dx = meta_i[0] - meta_j[0];
  const float dy = meta_i[1] - meta_j[1];
  const float dz = meta_i[2] - meta_j[2];
  return sqrtf(dx * dx + dy * dy + dz * dz) / meta_i[3] > distance_threshold;
}
