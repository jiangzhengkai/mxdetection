#include "./roi_align-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../common/cuda_utils.h"
#include "mxnet_op.h"

namespace mshadow {
namespace cuda {

template <typename Dtype>
__device__ Dtype roialign_bilinear_interp(const Dtype* data,
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
__device__ void roialign_bilinear_interp_backward(Dtype* data_diff,
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
__global__ void ROIAlignForwardKernel(const int count, const int sample_per_part, 
                                      const Dtype* bottom_data, const Dtype* bottom_rois,  
                                      const float spatial_scale, const int channels,
                                      const int height, const int width,
                                      const int pooled_height, const int pooled_width,
                                      Dtype* top_data) {
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
        val += roialign_bilinear_interp(offset_bottom_data, x, y, width);
      }
    }
    top_data[index] = val / (sample_per_part * sample_per_part);
  }
}

template<typename Dtype>
inline void ROIAlignForward(const Tensor<gpu, 4, Dtype> &out,
                            const Tensor<gpu, 4, Dtype> &data,
                            const Tensor<gpu, 2, Dtype> &bbox,
                            const int sample_per_part,
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
  CheckLaunchParam(dimGrid, dimBlock, "ROIAlign Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  ROIAlignForwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, sample_per_part, bottom_data, bottom_rois, spatial_scale, 
      channels, height, width, pooled_height, pooled_width, top_data);
}

template<typename Dtype>
__global__ void ROIAlignBackwardKernel(const int count, const int sample_per_part,
                                       const Dtype* top_diff, const Dtype* bottom_rois, 
                                       const int num_rois, const float spatial_scale, 
                                       const int channels, const int height, const int width,
                                       const int pooled_height, const int pooled_width,
                                       Dtype* bottom_diff) {
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
        roialign_bilinear_interp_backward(offset_bottom_diff, val_top_diff, x, y, width);
      }
    }
  }
}

template<typename Dtype>
inline void ROIAlignBackward(const Tensor<gpu, 4, Dtype> &in_grad,
                             const Tensor<gpu, 4, Dtype> &out_grad,
                             const Tensor<gpu, 2, Dtype> &bbox,
                             const int sample_per_part,
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
  CheckLaunchParam(dimGrid, dimBlock, "ROIAlign Backward");
  cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
  ROIAlignBackwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, sample_per_part, top_diff, bottom_rois, num_rois, spatial_scale, 
      channels, height, width, pooled_height, pooled_width, bottom_diff);
}

}  // namespace cuda


template<typename Dtype>
inline void ROIAlignForward(const Tensor<gpu, 4, Dtype> &out,
                            const Tensor<gpu, 4, Dtype> &data,
                            const Tensor<gpu, 2, Dtype> &bbox,
                            const int sample_per_part,
                            const float spatial_scale) {
  cuda::ROIAlignForward(out, data, bbox, sample_per_part, spatial_scale);
}

template<typename Dtype>
inline void ROIAlignBackward(const Tensor<gpu, 4, Dtype> &in_grad,
                             const Tensor<gpu, 4, Dtype> &out_grad,
                             const Tensor<gpu, 2, Dtype> &bbox,
                             const int sample_per_part,
                             const float spatial_scale) {
  cuda::ROIAlignBackward(in_grad, out_grad, bbox, sample_per_part, spatial_scale);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(ROIAlignParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ROIAlignOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
