#include "./crop_and_resize-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include "../common/cuda_utils.h"
#include <algorithm>
#include <vector>

namespace mshadow {
namespace cuda {

template<typename Dtype>
__global__ void CropAndResizeForwardKernel(const int count, const Dtype* bottom_data,
                                           const float spatial_scale, const int channels,
                                           const int image_height, const int image_width,
                                           const int crop_height, const int crop_width,
                                           const Dtype* bottom_rois, Dtype* top_data) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, ph, pw) is an element in the pooled output
    int x = index % crop_width;
    int y = (index / crop_width) % crop_height;
    int c = (index / crop_width / crop_height) % channels;
    int n = index / crop_width / crop_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];

    if (roi_batch_ind < 0) {
      top_data[index] = 0;
      continue;
    }

    Dtype x1 = bottom_rois[1] * spatial_scale;
    Dtype y1 = bottom_rois[2] * spatial_scale;
    Dtype x2 = bottom_rois[3] * spatial_scale;
    Dtype y2 = bottom_rois[4] * spatial_scale;

    Dtype roi_height = max(y2 - y1 + 1., 1.);
    Dtype roi_width = max(x2 - x1 + 1., 1.);  

    Dtype height_scale = static_cast<Dtype>(roi_height - 1) / static_cast<Dtype>(crop_height - 1);
    Dtype width_scale = static_cast<Dtype>(roi_width - 1) / static_cast<Dtype>(crop_width - 1);

    const Dtype in_y = y1 + y * height_scale;
    if (in_y < 0 || in_y > image_height - 1) {
      top_data[index] = 0;
      continue;
    }
    const Dtype in_x = x1 + x * width_scale;
    if (in_x < 0 || in_x > image_width - 1) {
      top_data[index] = 0;
      continue;
    }    

    const int top_y_index = floor(in_y);
    const int bottom_y_index = ceil(in_y);
    const Dtype y_lerp = in_y - top_y_index;

    const int left_x_index = floor(in_x);
    const int right_x_index = ceil(in_x);
    const Dtype x_lerp = in_x - left_x_index;

    const Dtype* bottom_data_n = bottom_data + (roi_batch_ind * channels + c) * image_height * image_width;
    const Dtype top_left = bottom_data_n[top_y_index * image_width + left_x_index];
    const Dtype top_right = bottom_data_n[top_y_index * image_width + right_x_index];
    const Dtype bottom_left = bottom_data_n[bottom_y_index * image_width + left_x_index];
    const Dtype bottom_right = bottom_data_n[bottom_y_index * image_width + right_x_index];

    const Dtype top = top_left + (top_right - top_left) * x_lerp;
    const Dtype bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
    top_data[index] = top + (bottom - top) * y_lerp;
  }
}

template<typename Dtype>
inline void CropAndResizeForward(const Tensor<gpu, 4, Dtype> &out,
                                 const Tensor<gpu, 4, Dtype> &data,
                                 const Tensor<gpu, 2, Dtype> &bbox,
                                 const float spatial_scale) {
  const Dtype *bottom_data = data.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *top_data = out.dptr_;
  const int count = out.shape_.Size();
  const int channels = data.size(1);
  const int image_height = data.size(2);
  const int image_width = data.size(3);
  const int crop_height = out.size(2);
  const int crop_width = out.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "CropAndResize Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  CropAndResizeForwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, bottom_data, spatial_scale, channels, image_height, image_width,
      crop_height, crop_width, bottom_rois, top_data);
}

template<typename Dtype>
__global__ void CropAndResizeBackwardKernel(const int count, const Dtype* top_diff,                        
                                            const float spatial_scale, const int channels,
                                            const int image_height, const int image_width,
                                            const int crop_height, const int crop_width,
                                            Dtype* bottom_diff, const Dtype* bottom_rois) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, ph, pw) is an element in the pooled output
    int x = index % crop_width;
    int y = (index / crop_width) % crop_height;
    int c = (index / crop_width / crop_height) % channels;
    int n = index / crop_width / crop_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];

    if (roi_batch_ind < 0) {
      continue;
    }

    Dtype x1 = bottom_rois[1] * spatial_scale;
    Dtype y1 = bottom_rois[2] * spatial_scale;
    Dtype x2 = bottom_rois[3] * spatial_scale;
    Dtype y2 = bottom_rois[4] * spatial_scale;

    Dtype roi_height = max(y2 - y1 + 1., 1.);
    Dtype roi_width = max(x2 - x1 + 1., 1.);  

    Dtype height_scale = static_cast<Dtype>(roi_height - 1) / static_cast<Dtype>(crop_height - 1);
    Dtype width_scale = static_cast<Dtype>(roi_width - 1) / static_cast<Dtype>(crop_width - 1);

    const Dtype in_y = y1 + y * height_scale;
    if (in_y < 0 || in_y > image_height - 1) {
      continue;
    }

    const Dtype in_x = x1 + x * width_scale;
    if (in_x < 0 || in_x > image_width - 1) {
      continue;
    }    

    const int top_y_index = floor(in_y);
    const int bottom_y_index = ceil(in_y);
    const Dtype y_lerp = in_y - top_y_index;

    const int left_x_index = floor(in_x);
    const int right_x_index = ceil(in_x);
    const Dtype x_lerp = in_x - left_x_index;

    Dtype* bottom_diff_n = bottom_diff + (roi_batch_ind * channels + c) * image_height * image_width;
    atomicAdd(bottom_diff_n + top_y_index * image_width + left_x_index, (1 - x_lerp) * (1 - y_lerp) * top_diff[index]);
    atomicAdd(bottom_diff_n + top_y_index * image_width + right_x_index, x_lerp * (1 - y_lerp) * top_diff[index]);
    atomicAdd(bottom_diff_n + bottom_y_index * image_width + left_x_index, (1 - x_lerp) * y_lerp * top_diff[index]);
    atomicAdd(bottom_diff_n + bottom_y_index * image_width + right_x_index, x_lerp * y_lerp * top_diff[index]);

  }
}

template<typename Dtype>
inline void CropAndResizeBackward(const Tensor<gpu, 4, Dtype> &in_grad,
                                  const Tensor<gpu, 4, Dtype> &out_grad,
                                  const Tensor<gpu, 2, Dtype> &bbox,
                                  const float spatial_scale) {
  const Dtype *top_diff = out_grad.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *bottom_diff = in_grad.dptr_;
  const int count = out_grad.shape_.Size();
  const int channels = in_grad.size(1);
  const int image_height = in_grad.size(2);
  const int image_width = in_grad.size(3);
  const int crop_height = out_grad.size(2);
  const int crop_width = out_grad.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "CropAndResize Backward");
  cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
  CropAndResizeBackwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, top_diff, spatial_scale, channels, image_height, image_width,
      crop_height, crop_width, bottom_diff, bottom_rois);
}

}  // namespace cuda

template<typename Dtype>
inline void CropAndResizeForward(const Tensor<gpu, 4, Dtype> &out,
                                 const Tensor<gpu, 4, Dtype> &data,
                                 const Tensor<gpu, 2, Dtype> &bbox,
                                 const float spatial_scale) {
  cuda::CropAndResizeForward(out, data, bbox, spatial_scale);
}

template<typename Dtype>
inline void CropAndResizeBackward(const Tensor<gpu, 4, Dtype> &in_grad,
                                  const Tensor<gpu, 4, Dtype> &out_grad,
                                  const Tensor<gpu, 2, Dtype> &bbox,
                                  const float spatial_scale) {
  cuda::CropAndResizeBackward(in_grad, out_grad, bbox, spatial_scale);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(CropAndResizeParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new CropAndResizeOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
