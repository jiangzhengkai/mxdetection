#include "./roi_align_fpn-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../common/cuda_utils.h"
#include "mxnet_op.h"

namespace mshadow {
namespace cuda {

template <typename Dtype>
__device__ Dtype ROIAlignFPN_bilinear_interp(const Dtype* data,
                                             const Dtype x,
                                             const Dtype y,
                                             const int width) {
  int x1 = floor(x);
  int x2 = ceil(x);
  int y1 = floor(y);
  int y2 = ceil(y);
  Dtype dist_x = static_cast<Dtype>(x - x1);
  Dtype dist_y = static_cast<Dtype>(y - y1);
  Dtype value11 = data[y1*width + x1];
  Dtype value12 = data[y2*width + x1];
  Dtype value21 = data[y1*width + x2];
  Dtype value22 = data[y2*width + x2];
  Dtype value = (1 - dist_x)*(1 - dist_y)*value11 + (1 - dist_x)*dist_y*value12 + 
                 dist_x*(1 - dist_y)*value21 + dist_x*dist_y*value22;
  return value;
}

template <typename Dtype>
__device__ void ROIAlignFPN_bilinear_interp_backward(Dtype* data_diff,
                                                     Dtype val_top_diff, 
                                                     const Dtype x,
                                                     const Dtype y,
                                                     const int width) {
  int x1 = floor(x);
  int x2 = ceil(x);
  int y1 = floor(y);
  int y2 = ceil(y);
  Dtype dist_x = static_cast<Dtype>(x - x1);
  Dtype dist_y = static_cast<Dtype>(y - y1);
  atomicAdd(data_diff + y1 * width + x1, (1 - dist_x) * (1 - dist_y) * val_top_diff);
  atomicAdd(data_diff + y2 * width + x1, (1 - dist_x) * dist_y * val_top_diff);
  atomicAdd(data_diff + y1 * width + x2, dist_x * (1 - dist_y) * val_top_diff);
  atomicAdd(data_diff + y2 * width + x2, dist_x * dist_y * val_top_diff);
}

template <typename Dtype>
__global__ void ROIAlignFPNForwardKernel(const int count, 
                                         const int sample_per_part, 
                                         const Dtype* bottom_data_res2, 
                                         const Dtype* bottom_data_res3,
                                         const Dtype* bottom_data_res4,
                                         const Dtype* bottom_data_res5,
                                         const Dtype* bottom_rois,  
                                         const int channels,
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
                                         const int pooled_height, 
                                         const int pooled_width,
                                         const bool has_area,
                                         Dtype* top_data) {
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
      top_data[index] = 0;
      continue;
    }

    const Dtype* bottom_data;
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
      bottom_data = bottom_data_res2;
      spatial_scale = static_cast<Dtype>(1.0 / 4);
      height = height_res2;
      width = width_res2;
    } else if (layer_ind == 3) {
      bottom_data = bottom_data_res3;
      spatial_scale = static_cast<Dtype>(1.0 / 8);
      height = height_res3;
      width = width_res3;
    } else if (layer_ind == 4) {
      bottom_data = bottom_data_res4;
      spatial_scale = static_cast<Dtype>(1.0 / 16);
      height = height_res4;
      width = width_res4;
    } else {
      bottom_data = bottom_data_res5;
      spatial_scale = static_cast<Dtype>(1.0 / 32);
      height = height_res5;
      width = width_res5;
    }

    const Dtype* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

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
      top_data[index] = 0;
      continue;
    }

    Dtype xstride = (xend - xstart) / static_cast<Dtype>(sample_per_part + 1);
    Dtype ystride = (yend - ystart) / static_cast<Dtype>(sample_per_part + 1);

    Dtype val = 0;
    for (int i = 1; i <= sample_per_part; ++i) {
      Dtype y = ystart + ystride * i;
      for (int j = 1; j <= sample_per_part; ++j) {
        Dtype x = xstart + xstride * j;
        val += ROIAlignFPN_bilinear_interp(offset_bottom_data, x, y, width);
      }
    }
    top_data[index] = val / (sample_per_part * sample_per_part);
  }
}

template<typename Dtype>
inline void ROIAlignFPNForward(const Tensor<gpu, 4, Dtype> &out,
                               const Tensor<gpu, 4, Dtype> &data_res2,
                               const Tensor<gpu, 4, Dtype> &data_res3,
                               const Tensor<gpu, 4, Dtype> &data_res4,
                               const Tensor<gpu, 4, Dtype> &data_res5, 
                               const Tensor<gpu, 2, Dtype> &bbox,
                               const int sample_per_part) {
  const Dtype *bottom_data_res2 = data_res2.dptr_;
  const Dtype *bottom_data_res3 = data_res3.dptr_;
  const Dtype *bottom_data_res4 = data_res4.dptr_;
  const Dtype *bottom_data_res5 = data_res5.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *top_data = out.dptr_;
  int channels;
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
  if (bottom_data_res2 != NULL) {
    channels = data_res2.size(1);
    height_res2 = data_res2.size(2);
    width_res2 = data_res2.size(3);
    min_layer_ind = min_layer_ind > 2 ? 2 : min_layer_ind;
    max_layer_ind = max_layer_ind < 2 ? 2 : max_layer_ind;
  }
  if (bottom_data_res3 != NULL) {
    channels = data_res3.size(1);
    height_res3 = data_res3.size(2);
    width_res3 = data_res3.size(3);
    min_layer_ind = min_layer_ind > 3 ? 3 : min_layer_ind;
    max_layer_ind = max_layer_ind < 3 ? 3 : max_layer_ind;    
  }
  if (bottom_data_res4 != NULL) {
    channels = data_res4.size(1);
    height_res4 = data_res4.size(2);
    width_res4 = data_res4.size(3);
    min_layer_ind = min_layer_ind > 4 ? 4 : min_layer_ind;
    max_layer_ind = max_layer_ind < 4 ? 4 : max_layer_ind;
  }
  if (bottom_data_res5 != NULL) {
    channels = data_res5.size(1);
    height_res5 = data_res5.size(2);
    width_res5 = data_res5.size(3);
    min_layer_ind = min_layer_ind > 5 ? 5 : min_layer_ind;
    max_layer_ind = max_layer_ind < 5 ? 5 : max_layer_ind;
  }
  const int count = out.shape_.Size();
  const int pooled_height = out.size(2);
  const int pooled_width = out.size(3);
  const bool has_area = bbox.size(1) > 5 ? true : false;
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "ROIAlignFPN Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  ROIAlignFPNForwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, sample_per_part, bottom_data_res2, bottom_data_res3, bottom_data_res4, bottom_data_res5, bottom_rois, 
      channels, height_res2, width_res2, height_res3, width_res3, height_res4, width_res4, height_res5, width_res5, 
      min_layer_ind, max_layer_ind, pooled_height, pooled_width, has_area, top_data);
}

template<typename Dtype>
__global__ void ROIAlignFPNBackwardKernel(const int count,
                                          const int sample_per_part,
                                          const Dtype* top_diff, 
                                          const Dtype* bottom_rois, 
                                          const int num_rois, 
                                          const int channels, 
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
                                          const int pooled_height, 
                                          const int pooled_width,
                                          const bool has_area,
                                          Dtype* bottom_diff_res2,
                                          Dtype* bottom_diff_res3,
                                          Dtype* bottom_diff_res4,
                                          Dtype* bottom_diff_res5) {
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

    Dtype* bottom_diff;
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
      bottom_diff = bottom_diff_res2;
      spatial_scale = static_cast<Dtype>(1.0 / 4);
      height = height_res2;
      width = width_res2;
    } else if (layer_ind == 3) {
      bottom_diff = bottom_diff_res3;
      spatial_scale = static_cast<Dtype>(1.0 / 8);
      height = height_res3;
      width = width_res3;
    } else if (layer_ind == 4) {
      bottom_diff = bottom_diff_res4;
      spatial_scale = static_cast<Dtype>(1.0 / 16);
      height = height_res4;
      width = width_res4;
    } else {
      bottom_diff = bottom_diff_res5;
      spatial_scale = static_cast<Dtype>(1.0 / 32);
      height = height_res5;
      width = width_res5;
    }

    Dtype* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;

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
    Dtype val_top_diff = top_diff[index] / (sample_per_part * sample_per_part);

    Dtype xstride = (xend - xstart) / static_cast<Dtype>(sample_per_part + 1);
    Dtype ystride = (yend - ystart) / static_cast<Dtype>(sample_per_part + 1);

    for (int i = 1; i <= sample_per_part; ++i) {
      Dtype y = ystart + ystride * i;
      for (int j = 1; j <= sample_per_part; ++j) {
        Dtype x = xstart + xstride * j;
        ROIAlignFPN_bilinear_interp_backward(offset_bottom_diff, val_top_diff, x, y, width);
      }
    }
  }
}

template<typename Dtype>
inline void ROIAlignFPNBackward(const Tensor<gpu, 4, Dtype> &in_grad_res2,
                                const Tensor<gpu, 4, Dtype> &in_grad_res3,
                                const Tensor<gpu, 4, Dtype> &in_grad_res4,
                                const Tensor<gpu, 4, Dtype> &in_grad_res5, 
                                const Tensor<gpu, 4, Dtype> &out_grad,
                                const Tensor<gpu, 2, Dtype> &bbox,
                                const int sample_per_part) {
  Dtype *bottom_diff_res2 = in_grad_res2.dptr_;
  Dtype *bottom_diff_res3 = in_grad_res3.dptr_;
  Dtype *bottom_diff_res4 = in_grad_res4.dptr_;
  Dtype *bottom_diff_res5 = in_grad_res5.dptr_;  
  const Dtype *top_diff = out_grad.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;

  int channels;
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
  if (bottom_diff_res2 != NULL) {
    channels = in_grad_res2.size(1);
    height_res2 = in_grad_res2.size(2);
    width_res2 = in_grad_res2.size(3);
    min_layer_ind = min_layer_ind > 2 ? 2 : min_layer_ind;
    max_layer_ind = max_layer_ind < 2 ? 2 : max_layer_ind;
  }
  if (bottom_diff_res3 != NULL) {
    channels = in_grad_res3.size(1);
    height_res3 = in_grad_res3.size(2);
    width_res3 = in_grad_res3.size(3);
    min_layer_ind = min_layer_ind > 3 ? 3 : min_layer_ind;
    max_layer_ind = max_layer_ind < 3 ? 3 : max_layer_ind;    
  }
  if (bottom_diff_res4 != NULL) {
    channels = in_grad_res4.size(1);
    height_res4 = in_grad_res4.size(2);
    width_res4 = in_grad_res4.size(3);
    min_layer_ind = min_layer_ind > 4 ? 4 : min_layer_ind;
    max_layer_ind = max_layer_ind < 4 ? 4 : max_layer_ind;
  }
  if (bottom_diff_res5 != NULL) {
    channels = in_grad_res5.size(1);
    height_res5 = in_grad_res5.size(2);
    width_res5 = in_grad_res5.size(3);
    min_layer_ind = min_layer_ind > 5 ? 5 : min_layer_ind;
    max_layer_ind = max_layer_ind < 5 ? 5 : max_layer_ind;
  }

  const int count = out_grad.shape_.Size();
  const int pooled_height = out_grad.size(2);
  const int pooled_width = out_grad.size(3);
  const int num_rois = bbox.size(0);
  const bool has_area = bbox.size(1) > 5 ? true : false;
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "ROIAlignFPN Backward");
  cudaStream_t stream = Stream<gpu>::GetStream(out_grad.stream_);
  ROIAlignFPNBackwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, sample_per_part, top_diff, bottom_rois, num_rois, channels, height_res2, width_res2,
      height_res3, width_res3, height_res4, width_res4, height_res5, width_res5, min_layer_ind, max_layer_ind,
      pooled_height, pooled_width, has_area, bottom_diff_res2, bottom_diff_res3, bottom_diff_res4, bottom_diff_res5);
}

}  // namespace cuda


template<typename Dtype>
inline void ROIAlignFPNForward(const Tensor<gpu, 4, Dtype> &out,
                               const Tensor<gpu, 4, Dtype> &data_res2,
                               const Tensor<gpu, 4, Dtype> &data_res3,
                               const Tensor<gpu, 4, Dtype> &data_res4,
                               const Tensor<gpu, 4, Dtype> &data_res5, 
                               const Tensor<gpu, 2, Dtype> &bbox,
                               const int sample_per_part) {
  cuda::ROIAlignFPNForward(out, data_res2, data_res3, data_res4, data_res5, bbox, sample_per_part);
}


template<typename Dtype>
inline void ROIAlignFPNBackward(const Tensor<gpu, 4, Dtype> &in_grad_res2,
                                const Tensor<gpu, 4, Dtype> &in_grad_res3,
                                const Tensor<gpu, 4, Dtype> &in_grad_res4,
                                const Tensor<gpu, 4, Dtype> &in_grad_res5, 
                                const Tensor<gpu, 4, Dtype> &out_grad,
                                const Tensor<gpu, 2, Dtype> &bbox,
                                const int sample_per_part) {
  cuda::ROIAlignFPNBackward(in_grad_res2, in_grad_res3, in_grad_res4, in_grad_res5, out_grad, bbox, sample_per_part);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(ROIAlignFPNParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ROIAlignFPNOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
