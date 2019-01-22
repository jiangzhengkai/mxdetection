#include "./roi_align_fpn_inv-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../common/cuda_utils.h"
#include "mxnet_op.h"

namespace mshadow {
namespace cuda {

template <typename Dtype>
__device__ void ROIAlignFPNInv_bilinear_interp(const Dtype val_bottom_data,
                                               Dtype* top_data,
                                               const Dtype x,
                                               const Dtype y,
                                               const int width) {
  int x1 = floor(x);
  int x2 = ceil(x);
  int y1 = floor(y);
  int y2 = ceil(y);
  Dtype dist_x = static_cast<Dtype>(x - x1);
  Dtype dist_y = static_cast<Dtype>(y - y1);
  Dtype value_x = (2 * dist_x * dist_x - 2 * dist_x + 1);
  Dtype value_y = (2 * dist_y * dist_y - 2 * dist_y + 1);
  Dtype value = val_bottom_data / (value_x * value_y);

  atomicAdd(top_data + y1 * width + x1, (1 - dist_x) * (1 - dist_y) * value);
  atomicAdd(top_data + y2 * width + x1, (1 - dist_x) * dist_y * value);
  atomicAdd(top_data + y1 * width + x2, dist_x * (1 - dist_y) * value);
  atomicAdd(top_data + y2 * width + x2, dist_x * dist_y * value);
}

template <typename Dtype>
__device__ Dtype ROIAlignFPNInv_bilinear_interp_backward(const Dtype* top_diff,
                                                         const Dtype x,
                                                         const Dtype y,
                                                         const int width) {
  int x1 = floor(x);
  int x2 = ceil(x);
  int y1 = floor(y);
  int y2 = ceil(y);
  Dtype dist_x = static_cast<Dtype>(x - x1);
  Dtype dist_y = static_cast<Dtype>(y - y1);
  Dtype value_x = (2 * dist_x * dist_x - 2 * dist_x + 1);
  Dtype value_y = (2 * dist_y * dist_y - 2 * dist_y + 1);
  Dtype value_xy = value_x * value_y;
  Dtype value11 = top_diff[y1*width + x1] / value_xy;
  Dtype value12 = top_diff[y2*width + x1] / value_xy;
  Dtype value21 = top_diff[y1*width + x2] / value_xy;
  Dtype value22 = top_diff[y2*width + x2] / value_xy;

  Dtype value = (1 - dist_x)*(1 - dist_y)*value11 + (1 - dist_x)*dist_y*value12 +
                 dist_x*(1 - dist_y)*value21 + dist_x*dist_y*value22;
  return value;
}

template <typename Dtype>
__global__ void ROIAlignFPNInvForwardKernel(const int count,
                                            const int sample_per_part,
                                            const Dtype* bottom_data,
                                            const Dtype* bottom_rois,
                                            const int channels,
                                            const int pooled_height,
                                            const int pooled_width,
                                            const int height_res2,
                                            const int width_res2,
                                            const int height_res3,
                                            const int width_res3,
                                            const int height_res4,
                                            const int width_res4,
                                            const int height_res5,
                                            const int width_res5,
                                            const int min_layer_ind,
                                            const int max_layer_ind,
                                            const bool has_area,
                                            Dtype* top_data_res2,
                                            Dtype* top_data_res3,
                                            Dtype* top_data_res4,
                                            Dtype* top_data_res5) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const Dtype* offset_bottom_rois;
    if (has_area) {
      offset_bottom_rois = bottom_rois + n * 6;
    } else {
      offset_bottom_rois = bottom_rois + n * 5;
    }
    int roi_batch_ind = offset_bottom_rois[0];
    if (roi_batch_ind < 0) {
      continue;
    }

    Dtype* top_data;
    Dtype spatial_scale;
    int height;
    int width;

    Dtype area;
    if (has_area) {
      area = offset_bottom_rois[5];
    } else {
      area = (offset_bottom_rois[4] - offset_bottom_rois[2] + 1) * (offset_bottom_rois[3] - offset_bottom_rois[1] + 1);
    }
    if (area <= 0) {
      area = static_cast<Dtype>(1e-6);
    }
    int layer_ind = floor(4 + log2(sqrt(area) / 224));
    if (layer_ind < min_layer_ind) {
      layer_ind = min_layer_ind;
    }
    if (layer_ind > max_layer_ind) {
      layer_ind = max_layer_ind;
    }

    if (layer_ind <= 2) {
      top_data = top_data_res2;
      spatial_scale = static_cast<Dtype>(1.0 / 4);
      height = height_res2;
      width = width_res2;
    } else if (layer_ind == 3) {
      top_data = top_data_res3;
      spatial_scale = static_cast<Dtype>(1.0 / 8);
      height = height_res3;
      width = width_res3;
    } else if (layer_ind == 4) {
      top_data = top_data_res4;
      spatial_scale = static_cast<Dtype>(1.0 / 16);
      height = height_res4;
      width = width_res4;
    } else {
      top_data = top_data_res5;
      spatial_scale = static_cast<Dtype>(1.0 / 32);
      height = height_res5;
      width = width_res5;
    }

    Dtype* offset_top_data = top_data + (roi_batch_ind * channels + c) * height * width;

    Dtype roi_start_w = offset_bottom_rois[1] * spatial_scale;
    Dtype roi_start_h = offset_bottom_rois[2] * spatial_scale;
    Dtype roi_end_w = (offset_bottom_rois[3] + 1.) * spatial_scale;
    Dtype roi_end_h = (offset_bottom_rois[4] + 1.) * spatial_scale;

    Dtype roi_width = max(roi_end_w - roi_start_w, 0.1);
    Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);
    Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);
    Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);

    Dtype xstart = static_cast<Dtype>(pw) * bin_size_w + roi_start_w;
    Dtype ystart = static_cast<Dtype>(ph) * bin_size_h + roi_start_h;
    Dtype xend = static_cast<Dtype>(pw + 1) * bin_size_w + roi_start_w;
    Dtype yend = static_cast<Dtype>(ph + 1) * bin_size_h + roi_start_h;

    xstart = min(max(xstart, static_cast<Dtype>(0)), static_cast<Dtype>(width));
    ystart = min(max(ystart, static_cast<Dtype>(0)), static_cast<Dtype>(height));
    xend = min(max(xend, static_cast<Dtype>(0)), static_cast<Dtype>(width));
    yend = min(max(yend, static_cast<Dtype>(0)), static_cast<Dtype>(height));

    if (xend <= xstart || yend <= ystart) {
      continue;
    }

    Dtype val_bottom_data = bottom_data[index] / (sample_per_part * sample_per_part);

    Dtype xstride = (xend - xstart) / static_cast<Dtype>(sample_per_part + 1);
    Dtype ystride = (yend - ystart) / static_cast<Dtype>(sample_per_part + 1);

    for (int i = 1; i <= sample_per_part; ++i) {
      Dtype y = ystart + ystride * i;
      for (int j = 1; j <= sample_per_part; ++j) {
        Dtype x = xstart + xstride * j;
        ROIAlignFPNInv_bilinear_interp(val_bottom_data, offset_top_data, x, y, width);
      }
    }
  }
}

template<typename Dtype>
inline void ROIAlignFPNInvForward(const Tensor<gpu, 4, Dtype> &out_data_res2,
                                  const Tensor<gpu, 4, Dtype> &out_data_res3,
                                  const Tensor<gpu, 4, Dtype> &out_data_res4,
                                  const Tensor<gpu, 4, Dtype> &out_data_res5,
                                  const Tensor<gpu, 4, Dtype> &in_data,
                                  const Tensor<gpu, 2, Dtype> &bbox,
                                  const int sample_per_part) {

  Dtype *top_data_res2 = out_data_res2.dptr_;
  Dtype *top_data_res3 = out_data_res3.dptr_;
  Dtype *top_data_res4 = out_data_res4.dptr_;
  Dtype *top_data_res5 = out_data_res5.dptr_;
  const Dtype *bottom_data = in_data.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;

  int height_res2;
  int width_res2;
  int height_res3;
  int width_res3;
  int height_res4;
  int width_res4;
  int height_res5;
  int width_res5;
  int min_layer_ind = 1000;
  int max_layer_ind = 0;
  if (top_data_res2 != NULL) {
    height_res2 = out_data_res2.size(2);
    width_res2 = out_data_res2.size(3);
    min_layer_ind = min_layer_ind > 2 ? 2 : min_layer_ind;
    max_layer_ind = max_layer_ind < 2 ? 2 : max_layer_ind;
  }
  if (top_data_res3 != NULL) {
    height_res3 = out_data_res3.size(2);
    width_res3 = out_data_res3.size(3);
    min_layer_ind = min_layer_ind > 3 ? 3 : min_layer_ind;
    max_layer_ind = max_layer_ind < 3 ? 3 : max_layer_ind;
  }
  if (top_data_res4 != NULL) {
    height_res4 = out_data_res4.size(2);
    width_res4 = out_data_res4.size(3);
    min_layer_ind = min_layer_ind > 4 ? 4 : min_layer_ind;
    max_layer_ind = max_layer_ind < 4 ? 4 : max_layer_ind;
  }
  if (top_data_res5 != NULL) {
    height_res5 = out_data_res5.size(2);
    width_res5 = out_data_res5.size(3);
    min_layer_ind = min_layer_ind > 5 ? 5 : min_layer_ind;
    max_layer_ind = max_layer_ind < 5 ? 5 : max_layer_ind;
  }

  const int count = in_data.shape_.Size();
  const int channels = in_data.size(1);
  const int pooled_height = in_data.size(2);
  const int pooled_width = in_data.size(3);
  const bool has_area = bbox.size(1) > 5 ? true : false;
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "ROIAlignFPNInv Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(in_data.stream_);
  ROIAlignFPNInvForwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, sample_per_part, bottom_data, bottom_rois, channels, pooled_height, pooled_width,
      height_res2, width_res2, height_res3, width_res3, height_res4, width_res4, height_res5, width_res5,
      min_layer_ind, max_layer_ind, has_area, top_data_res2, top_data_res3, top_data_res4, top_data_res5);
}

template <typename Dtype>
__global__ void ROIAlignFPNInvBackwardKernel(const int count,
                                             const int sample_per_part,
                                             const Dtype* top_diff_res2,
                                             const Dtype* top_diff_res3,
                                             const Dtype* top_diff_res4,
                                             const Dtype* top_diff_res5,
                                             const Dtype* bottom_rois,
                                             const int channels,
                                             const int pooled_height,
                                             const int pooled_width,
                                             const int height_res2,
                                             const int width_res2,
                                             const int height_res3,
                                             const int width_res3,
                                             const int height_res4,
                                             const int width_res4,
                                             const int height_res5,
                                             const int width_res5,
                                             const int min_layer_ind,
                                             const int max_layer_ind,
                                             const bool has_area,
                                             Dtype* bottom_diff) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const Dtype* offset_bottom_rois;
    if (has_area) {
      offset_bottom_rois = bottom_rois + n * 6;
    } else {
      offset_bottom_rois = bottom_rois + n * 5;
    }
    int roi_batch_ind = offset_bottom_rois[0];
    if (roi_batch_ind < 0) {
      bottom_diff[index] = 0;
      continue;
    }

    const Dtype* top_diff;
    Dtype spatial_scale;
    int height;
    int width;

    Dtype area;
    if (has_area) {
      area = offset_bottom_rois[5];
    } else {
      area = (offset_bottom_rois[4] - offset_bottom_rois[2] + 1) * (offset_bottom_rois[3] - offset_bottom_rois[1] + 1);
    }
    if (area <= 0) {
      area = static_cast<Dtype>(1e-6);
    }
    int layer_ind = floor(4 + log2(sqrt(area) / 224));
    if (layer_ind < min_layer_ind) {
      layer_ind = min_layer_ind;
    }
    if (layer_ind > max_layer_ind) {
      layer_ind = max_layer_ind;
    }

    if (layer_ind <= 2) {
      top_diff = top_diff_res2;
      spatial_scale = static_cast<Dtype>(1.0 / 4);
      height = height_res2;
      width = width_res2;
    } else if (layer_ind == 3) {
      top_diff = top_diff_res3;
      spatial_scale = static_cast<Dtype>(1.0 / 8);
      height = height_res3;
      width = width_res3;
    } else if (layer_ind == 4) {
      top_diff = top_diff_res4;
      spatial_scale = static_cast<Dtype>(1.0 / 16);
      height = height_res4;
      width = width_res4;
    } else {
      top_diff = top_diff_res5;
      spatial_scale = static_cast<Dtype>(1.0 / 32);
      height = height_res5;
      width = width_res5;
    }

    const Dtype* offset_top_diff = top_diff + (roi_batch_ind * channels + c) * height * width;

    Dtype roi_start_w = offset_bottom_rois[1] * spatial_scale;
    Dtype roi_start_h = offset_bottom_rois[2] * spatial_scale;
    Dtype roi_end_w = (offset_bottom_rois[3] + 1.) * spatial_scale;
    Dtype roi_end_h = (offset_bottom_rois[4] + 1.) * spatial_scale;

    Dtype roi_width = max(roi_end_w - roi_start_w, 0.1);
    Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);
    Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);
    Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);

    Dtype xstart = static_cast<Dtype>(pw) * bin_size_w + roi_start_w;
    Dtype ystart = static_cast<Dtype>(ph) * bin_size_h + roi_start_h;
    Dtype xend = static_cast<Dtype>(pw + 1) * bin_size_w + roi_start_w;
    Dtype yend = static_cast<Dtype>(ph + 1) * bin_size_h + roi_start_h;

    xstart = min(max(xstart, static_cast<Dtype>(0)), static_cast<Dtype>(width));
    ystart = min(max(ystart, static_cast<Dtype>(0)), static_cast<Dtype>(height));
    xend = min(max(xend, static_cast<Dtype>(0)), static_cast<Dtype>(width));
    yend = min(max(yend, static_cast<Dtype>(0)), static_cast<Dtype>(height));

    if (xend <= xstart || yend <= ystart) {
      bottom_diff[index] = 0;
      continue;
    }

    Dtype xstride = (xend - xstart) / static_cast<Dtype>(sample_per_part + 1);
    Dtype ystride = (yend - ystart) / static_cast<Dtype>(sample_per_part + 1);

    Dtype val = 0;
    for (int i = 1; i <= sample_per_part; ++i) {
      Dtype y = ystart + ystride * i;
      for (int j = 1; j <= sample_per_part; ++j) {
        Dtype x = xstart + xstride * j;
        val += ROIAlignFPNInv_bilinear_interp_backward(offset_top_diff, x, y, width);
      }
    }
    bottom_diff[index] = val / (sample_per_part * sample_per_part);
  }
}


template<typename Dtype>
inline void ROIAlignFPNInvBackward(const Tensor<gpu, 4, Dtype> &in_grad,
                                   const Tensor<gpu, 4, Dtype> &out_grad_res2,
                                   const Tensor<gpu, 4, Dtype> &out_grad_res3,
                                   const Tensor<gpu, 4, Dtype> &out_grad_res4,
                                   const Tensor<gpu, 4, Dtype> &out_grad_res5,
                                   const Tensor<gpu, 2, Dtype> &bbox,
                                   const int sample_per_part) {
  Dtype *bottom_diff = in_grad.dptr_;
  const Dtype *top_diff_res2 = out_grad_res2.dptr_;
  const Dtype *top_diff_res3 = out_grad_res3.dptr_;
  const Dtype *top_diff_res4 = out_grad_res4.dptr_;
  const Dtype *top_diff_res5 = out_grad_res5.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;

  int height_res2;
  int width_res2;
  int height_res3;
  int width_res3;
  int height_res4;
  int width_res4;
  int height_res5;
  int width_res5;
  int min_layer_ind = 1000;
  int max_layer_ind = 0;
  if (top_diff_res2 != NULL) {
    height_res2 = out_grad_res2.size(2);
    width_res2 = out_grad_res2.size(3);
    min_layer_ind = min_layer_ind > 2 ? 2 : min_layer_ind;
    max_layer_ind = max_layer_ind < 2 ? 2 : max_layer_ind;
  }
  if (top_diff_res3 != NULL) {
    height_res3 = out_grad_res3.size(2);
    width_res3 = out_grad_res3.size(3);
    min_layer_ind = min_layer_ind > 3 ? 3 : min_layer_ind;
    max_layer_ind = max_layer_ind < 3 ? 3 : max_layer_ind;
  }
  if (top_diff_res4 != NULL) {
    height_res4 = out_grad_res4.size(2);
    width_res4 = out_grad_res4.size(3);
    min_layer_ind = min_layer_ind > 4 ? 4 : min_layer_ind;
    max_layer_ind = max_layer_ind < 4 ? 4 : max_layer_ind;
  }
  if (top_diff_res5 != NULL) {
    height_res5 = out_grad_res5.size(2);
    width_res5 = out_grad_res5.size(3);
    min_layer_ind = min_layer_ind > 5 ? 5 : min_layer_ind;
    max_layer_ind = max_layer_ind < 5 ? 5 : max_layer_ind;
  }
  const int count = in_grad.shape_.Size();
  const int channels = in_grad.size(1);
  const int pooled_height = in_grad.size(2);
  const int pooled_width = in_grad.size(3);
  const bool has_area = bbox.size(1) > 5 ? true : false;
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "ROIAlignFPNInv Backward");
  cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
  ROIAlignFPNInvBackwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, sample_per_part, top_diff_res2, top_diff_res3, top_diff_res4, top_diff_res5, bottom_rois,
      channels, pooled_height, pooled_width, height_res2, width_res2, height_res3, width_res3, height_res4,
      width_res4, height_res5, width_res5, min_layer_ind, max_layer_ind, has_area, bottom_diff);
}

}  // namespace cuda

template<typename Dtype>
inline void ROIAlignFPNInvForward(const Tensor<gpu, 4, Dtype> &out_data_res2,
                                  const Tensor<gpu, 4, Dtype> &out_data_res3,
                                  const Tensor<gpu, 4, Dtype> &out_data_res4,
                                  const Tensor<gpu, 4, Dtype> &out_data_res5,
                                  const Tensor<gpu, 4, Dtype> &in_data,
                                  const Tensor<gpu, 2, Dtype> &bbox,
                                  const int sample_per_part) {
  cuda::ROIAlignFPNInvForward(out_data_res2, out_data_res3, out_data_res4, out_data_res5, in_data, bbox, sample_per_part);
}


template<typename Dtype>
inline void ROIAlignFPNInvBackward(const Tensor<gpu, 4, Dtype> &in_grad,
                                   const Tensor<gpu, 4, Dtype> &out_grad_res2,
                                   const Tensor<gpu, 4, Dtype> &out_grad_res3,
                                   const Tensor<gpu, 4, Dtype> &out_grad_res4,
                                   const Tensor<gpu, 4, Dtype> &out_grad_res5,
                                   const Tensor<gpu, 2, Dtype> &bbox,
                                   const int sample_per_part) {
  cuda::ROIAlignFPNInvBackward(in_grad, out_grad_res2, out_grad_res3, out_grad_res4, out_grad_res5, bbox, sample_per_part);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(ROIAlignFPNInvParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ROIAlignFPNInvOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
