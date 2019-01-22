#include "./precise_roi_pooling-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../common/cuda_utils.h"
#include "mxnet_op.h"

namespace mshadow {
namespace cuda {

// x - floor(x1): integration between lb and ub
// template<typename Dtype>
// __device__ DType integral_func_1(const Dtype x, const Dtype lb, const Dtype ub) {
//   int x1 = floor(x);
//   Dtype val = ub * (ub / 2 - x1) - lb * (lb /2 - x1);
//   return val;
// }

// // ceil(x) - x: integration between lb and ub
// template <typename Dtype>
// __device__ DType integral_func_2(const Dtype x, const Dtype lb, const Dtype ub) {
//   int x2 = ceil(x);
//   Dtype val = ub * (x2 - ub / 2) - lb * (x2 - lb / 2);
//   return val;
// }

template <typename Dtype>
__device__ Dtype integral_bilinear_interp(const Dtype* data,
                                          const int x1,
                                          const int y1,
                                          const Dtype x_lb,
                                          const Dtype y_lb,
                                          const Dtype x_ub,
                                          const Dtype y_ub,
                                          const int width) {
  int x2 = x1 + 1;
  int y2 = y1 + 1;
  Dtype val_lt = data[y1 * width + x1];
  Dtype val_rt = data[y1 * width + x2];
  Dtype val_lb = data[y2 * width + x1];
  Dtype val_rb = data[y2 * width + x2];    
  Dtype val_x1 = x_ub * (x_ub / 2 - x1) - x_lb * (x_lb /2 - x1);
  Dtype val_x2 = x_ub * (x2 - x_ub / 2) - x_lb * (x2 - x_lb / 2);
  Dtype val_y1 = y_ub * (y_ub / 2 - y1) - y_lb * (y_lb /2 - y1);
  Dtype val_y2 = y_ub * (y2 - y_ub / 2) - y_lb * (y2 - y_lb / 2);
  Dtype val = val_y2 * val_x2 * val_lt + val_y2 * val_x1 * val_rt + 
              val_y1 * val_x2 * val_lb + val_y1 * val_x1 * val_rb;
  return val;
}

template <typename Dtype>
__device__ void integral_bilinear_interp_backward(Dtype* data_diff,
                                                  Dtype val_top_diff,
                                                  const int x1,
                                                  const int y1,
                                                  const Dtype x_lb,
                                                  const Dtype y_lb,
                                                  const Dtype x_ub,
                                                  const Dtype y_ub,
                                                  const int width) {
  int x2 = x1 + 1;
  int y2 = y1 + 1; 
  Dtype val_x1 = x_ub * (x_ub / 2 - x1) - x_lb * (x_lb /2 - x1);
  Dtype val_x2 = x_ub * (x2 - x_ub / 2) - x_lb * (x2 - x_lb / 2);
  Dtype val_y1 = y_ub * (y_ub / 2 - y1) - y_lb * (y_lb /2 - y1);
  Dtype val_y2 = y_ub * (y2 - y_ub / 2) - y_lb * (y2 - y_lb / 2);
  atomicAdd(data_diff + y1 * width + x1, val_y2 * val_x2 * val_top_diff);
  atomicAdd(data_diff + y1 * width + x2, val_y2 * val_x1 * val_top_diff);
  atomicAdd(data_diff + y2 * width + x1, val_y1 * val_x2 * val_top_diff);
  atomicAdd(data_diff + y2 * width + x2, val_y1 * val_x1 * val_top_diff);
}

template <typename Dtype>
__global__ void PreciseROIPoolForwardKernel(const int count, const Dtype* bottom_data,
                                            const float spatial_scale, const int channels,
                                            const int height, const int width,
                                            const int pooled_height, const int pooled_width,
                                            const Dtype* bottom_rois, Dtype* top_data) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const Dtype* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    if (roi_batch_ind < 0) {
      top_data[index] = 0;
      continue;
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
    Dtype area = (xend - xstart) * (yend - ystart);
    if (area <= 0) {
      top_data[index] = 0;
      continue;    
    }

    int xstart_f = floor(xstart);
    int ystart_f = floor(ystart);  
    int xend_c = ceil(xend);
    int yend_c = ceil(yend);

    Dtype val = 0;
    Dtype y_lb;
    Dtype y_ub;
    Dtype x_lb;
    Dtype x_ub;
    for (int y = ystart_f; y < yend_c; ++y) {
      y_lb = max(static_cast<Dtype>(y), ystart);
      y_ub = max(min(static_cast<Dtype>(y+1), yend), ystart);
      for (int x = xstart_f; x < xend_c; ++x) {
        x_lb = max(static_cast<Dtype>(x), xstart);    
        x_ub = max(min(static_cast<Dtype>(x+1), xend), xstart);        
        val += integral_bilinear_interp(offset_bottom_data, x, y, x_lb, y_lb, x_ub, y_ub, width);
      }
    }
    top_data[index] = val / area;
  }
}

template<typename Dtype>
inline void PreciseROIPoolForward(const Tensor<gpu, 4, Dtype> &out,
                                  const Tensor<gpu, 4, Dtype> &data,
                                  const Tensor<gpu, 2, Dtype> &bbox,
                                  const float spatial_scale) {
  const Dtype *bottom_data = data.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *top_data = out.dptr_;
  const int count = out.shape_.Size();
  const int channels = data.size(1);
  const int height = data.size(2);
  const int width = data.size(3);
  const int pooled_height = out.size(2);
  const int pooled_width = out.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "PreciseROIPooling Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  PreciseROIPoolForwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, bottom_data, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_rois, top_data);
}

template<typename Dtype>
__global__ void PreciseROIPoolBackwardKernel(const int count, const Dtype* top_diff, const int num_rois,
                                             const float spatial_scale, const int channels,
                                             const int height, const int width,
                                             const int pooled_height, const int pooled_width,
                                             Dtype* bottom_diff, const Dtype* bottom_rois) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const Dtype* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    if (roi_batch_ind < 0) {
      continue;
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
    Dtype area = (xend - xstart) * (yend - ystart);
    if (area <= 0) {
      continue;    
    }
    Dtype val_top_diff = top_diff[index] / area;

    int xstart_f = floor(xstart);
    int ystart_f = floor(ystart);  
    int xend_c = ceil(xend);
    int yend_c = ceil(yend);
           
    Dtype y_lb;
    Dtype y_ub;
    Dtype x_lb;
    Dtype x_ub;
    for (int y = ystart_f; y < yend_c; ++y) {
      y_lb = max(static_cast<Dtype>(y), ystart);
      y_ub = max(min(static_cast<Dtype>(y+1), yend), ystart);
      for (int x = xstart_f; x < xend_c; ++x) {
        x_lb = max(static_cast<Dtype>(x), xstart);    
        x_ub = max(min(static_cast<Dtype>(x+1), xend), xstart);         
        integral_bilinear_interp_backward(offset_bottom_diff, val_top_diff, x, y, x_lb, y_lb, x_ub, y_ub, width);
      }
    }
  }
}

template<typename Dtype>
inline void PreciseROIPoolBackward(const Tensor<gpu, 4, Dtype> &in_grad,
                                   const Tensor<gpu, 4, Dtype> &out_grad,
                                   const Tensor<gpu, 2, Dtype> &bbox,
                                   const float spatial_scale) {
  const Dtype *top_diff = out_grad.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *bottom_diff = in_grad.dptr_;
  const int count = out_grad.shape_.Size();
  const int num_rois = bbox.size(0);
  const int channels = in_grad.size(1);
  const int height = in_grad.size(2);
  const int width = in_grad.size(3);
  const int pooled_height = out_grad.size(2);
  const int pooled_width = out_grad.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "PreciseROIPooling Backward");
  cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
  PreciseROIPoolBackwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, top_diff, num_rois, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_diff, bottom_rois);
}

}  // namespace cuda


template<typename Dtype>
inline void PreciseROIPoolForward(const Tensor<gpu, 4, Dtype> &out,
                                  const Tensor<gpu, 4, Dtype> &data,
                                  const Tensor<gpu, 2, Dtype> &bbox,
                                  const float spatial_scale) {
  cuda::PreciseROIPoolForward(out, data, bbox, spatial_scale);
}

template<typename Dtype>
inline void PreciseROIPoolBackward(const Tensor<gpu, 4, Dtype> &in_grad,
                                   const Tensor<gpu, 4, Dtype> &out_grad,
                                   const Tensor<gpu, 2, Dtype> &bbox,
                                   const float spatial_scale) {
  cuda::PreciseROIPoolBackward(in_grad, out_grad, bbox, spatial_scale);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(PreciseROIPoolingParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new PreciseROIPoolingOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
