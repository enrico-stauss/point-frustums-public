#include "iou_box3d/iou_box3d.h"
#include "nms/nms.h"
#include "segment_attention/segment_attention.h"
#include <torch/extension.h>

namespace point_frustums {

#if defined(WITH_CUDA) || defined(WITH_HIP)
extern int get_cudart_version();
#endif

std::string get_cuda_version() {
#if defined(WITH_CUDA) || defined(WITH_HIP)
  std::ostringstream oss;

#if defined(WITH_CUDA)
  oss << "CUDA ";
#else
  oss << "HIP ";
#endif

  // copied from
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/CUDAHooks.cpp#L231
  auto printCudaStyleVersion = [&](int v) {
    oss << (v / 1000) << "." << (v / 10 % 100);
    if (v % 10 != 0) {
      oss << "." << (v % 10);
    }
  };
  printCudaStyleVersion(get_cudart_version());
  return oss.str();
#else // neither CUDA nor HIP
  return std::string("not available");
#endif
}

bool has_cuda() {
#if defined(WITH_CUDA)
  return true;
#else
  return false;
#endif
}

// similar to
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Version.cpp
std::string get_compiler_version() {
  std::ostringstream ss;
#if defined(__GNUC__)
#ifndef __clang__

#if ((__GNUC__ <= 4) && (__GNUC_MINOR__ <= 8))
#error "GCC >= 4.9 is required!"
#endif

  { ss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__; }
#endif
#endif

#if defined(__clang_major__)
  { ss << "clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__; }
#endif

#if defined(_MSC_VER)
  { ss << "MSVC " << _MSC_FULL_VER; }
#endif
  return ss.str();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_compiler_version", &get_compiler_version, "get_compiler_version");
  m.def("get_cuda_version", &get_cuda_version, "get_cuda_version");
  m.def("has_cuda", &has_cuda, "has_cuda");
}

TORCH_LIBRARY(point_frustums, m) {
  m.def("iou_box3d", &IoUBox3D);
  m.def("iou_box3d_pairwise", &IoUBox3DPairwise);
  m.def("nms", &NMS);
  m.def("segment_attention_forward", &SegmentAttentionForward);
  m.def("segment_attention_backward", &SegmentAttentionBackward);
}
} // namespace point_frustums