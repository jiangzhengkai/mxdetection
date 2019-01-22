#include "./pool_points_interp-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../common/cuda_utils.h"
#include "mxnet_op.h"


namespace mshadow {
namespace cuda {

template <typename Dtype>
__device__ Dtype bilinear_interpolate(const Dtype* bottom_data, 
  const int height, const int width, Dtype y, Dtype x) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int) y;
  int x_low = (int) x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (Dtype) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (Dtype) x_low;
  } else {
    x_high = x_low + 1;
  }

  Dtype ly = y - y_low;
  Dtype lx = x - x_low;
  Dtype hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  Dtype v1 = bottom_data[y_low * width + x_low];
  Dtype v2 = bottom_data[y_low * width + x_high];
  Dtype v3 = bottom_data[y_high * width + x_low];
  Dtype v4 = bottom_data[y_high * width + x_high];
  Dtype w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  Dtype val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}


template <typename Dtype>
__global__ void PoolPointsInterpForwardKernel(const int count,
                                              const Dtype* bottom_data, 
                                              const Dtype* bottom_points,
                                              const Dtype spatial_scale, 
                                              const int channels,
                                              const int height,
                                              const int width,
                                              const int num_points,
                                              Dtype* top_data) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    int c = index % channels;
    int n = index / channels;
    
    const Dtype* offset_bottom_points = bottom_points + n * 2;
    Dtype X_point = offset_bottom_points[0] * spatial_scale;
    Dtype Y_point = offset_bottom_points[1] * spatial_scale;

    int roi_batch_ind = n / num_points; 
    const Dtype* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

    Dtype val = bilinear_interpolate(offset_bottom_data, height, width, Y_point, X_point);
    top_data[index] = val;
  }
}

template<typename Dtype>
inline void PoolPointsInterpForward(const Tensor<gpu, 2, Dtype> &out_data,
                                    const Tensor<gpu, 4, Dtype> &in_data,
                                    const Tensor<gpu, 2, Dtype> &points,
                                    const float spatial_scale) {
  const Dtype *bottom_data = in_data.dptr_;
  const Dtype *bottom_points = points.dptr_;
  Dtype *top_data = out_data.dptr_;

  const int count = out_data.shape_.Size();
  const int batch_size = in_data.size(0);
  const int channels = in_data.size(1);
  const int height = in_data.size(2);
  const int width = in_data.size(3);
  const int num_points = points.size(0) / batch_size;

  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "PoolPointsInterp Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out_data.stream_);

  PoolPointsInterpForwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, bottom_data, bottom_points, spatial_scale, channels, height, width, num_points, top_data);
}


template <typename Dtype>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width, Dtype y, Dtype x,
    Dtype& w1, Dtype& w2, Dtype& w3, Dtype& w4,
    int& x_low, int& x_high, int& y_low, int& y_high) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (Dtype) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (Dtype) x_low;
  } else {
    x_high = x_low + 1;
  }

  Dtype ly = y - y_low;
  Dtype lx = x - x_low;
  Dtype hy = 1. - ly, hx = 1. - lx;

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}


template <typename Dtype>
__global__ void PoolPointsInterpBackwardKernel(const int count, 
                                               const Dtype* top_data_diff, 
                                               const Dtype* bottom_points,
                                               const Dtype spatial_scale, 
                                               const int channels, 
                                               const int height, 
                                               const int width, 
                                               const int num_points, 
                                               Dtype* bottom_data_diff) {

  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    int c = index % channels;
    int n = index / channels;

    const Dtype* offset_bottom_points = bottom_points + n * 2;
    Dtype X_point = offset_bottom_points[0] * spatial_scale;
    Dtype Y_point = offset_bottom_points[1] * spatial_scale;

    int roi_batch_ind = n / num_points; 
    Dtype* offset_bottom_data_diff = bottom_data_diff + (roi_batch_ind * channels + c) * height * width;

    Dtype w1, w2, w3, w4;
    int x_low, x_high, y_low, y_high;
    bilinear_interpolate_gradient(height, width, Y_point, X_point, 
                                  w1, w2, w3, w4, 
                                  x_low, x_high, y_low, y_high);

    if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
      atomicAdd(offset_bottom_data_diff + y_low * width + x_low, top_data_diff[index] * w1);
      atomicAdd(offset_bottom_data_diff + y_low * width + x_high, top_data_diff[index] * w2);
      atomicAdd(offset_bottom_data_diff + y_high * width + x_low, top_data_diff[index] * w3);
      atomicAdd(offset_bottom_data_diff + y_high * width + x_high, top_data_diff[index] * w4);
    } 
  } 
} 


template<typename Dtype>
inline void PoolPointsInterpBackward(const Tensor<gpu, 4, Dtype> &in_data_grad,
                                     const Tensor<gpu, 2, Dtype> &out_data_grad,
                                     const Tensor<gpu, 2, Dtype> &points,
                                     const float spatial_scale) {
  const Dtype *top_data_diff = out_data_grad.dptr_;
  const Dtype* bottom_points = points.dptr_;
  Dtype *bottom_data_diff = in_data_grad.dptr_;
  
  const int count = out_data_grad.shape_.Size();
  const int batch_size = in_data_grad.size(0);
  const int channels = in_data_grad.size(1);
  const int height = in_data_grad.size(2);
  const int width = in_data_grad.size(3);
  const int num_points = points.size(0) / batch_size;
  
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "PoolPointsInterp Backward");
  cudaStream_t stream = Stream<gpu>::GetStream(in_data_grad.stream_);

  PoolPointsInterpBackwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, top_data_diff, bottom_points, spatial_scale, channels, height, width, num_points, bottom_data_diff);
}

}  // namespace cuda

template<typename Dtype>
inline void PoolPointsInterpForward(const Tensor<gpu, 2, Dtype> &out_data,
                                    const Tensor<gpu, 4, Dtype> &in_data,
                                    const Tensor<gpu, 2, Dtype> &points,
                                    const float spatial_scale) {
  cuda::PoolPointsInterpForward(out_data, in_data, points, spatial_scale);
}

template<typename Dtype>
inline void PoolPointsInterpBackward(const Tensor<gpu, 4, Dtype> &in_data_grad,
                                     const Tensor<gpu, 2, Dtype> &out_data_grad,
                                     const Tensor<gpu, 2, Dtype> &points,
                                     const float spatial_scale) {
  cuda::PoolPointsInterpBackward(in_data_grad, out_data_grad, points, spatial_scale);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(PoolPointsInterpParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new PoolPointsInterpOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
