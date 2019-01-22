#include "./sigmoid_focal_loss-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../common/cuda_utils.h"
#include "mxnet_op.h"


namespace mshadow {
namespace cuda {

template<typename Dtype>
__global__ void SigmoidFocalLossKernel(
    const Dtype* bottom_data, const Dtype* bottom_label, Dtype* top_prob_data, Dtype* top_loss_data,
    const int count, const int channel, const int spatial_dim,
    const float ignore_label, const float alpha, const float gamma) {

  for (int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       i < count; i += blockDim.x * gridDim.x * gridDim.y) {

    int s = i % spatial_dim;
    int c = (i / spatial_dim) % channel; 
    int n = i / (spatial_dim * channel);    
 
    int t = bottom_label[n * spatial_dim + s]; 

    // p = 1. / 1. + expf(-x)
    float p = 1. / (1. + expf(-bottom_data[i]));

    top_prob_data[i] = p;

    if (t == ignore_label) {
      top_loss_data[i] = 0.0;
    } else if (t == c + 1) {     
      // (1 - p)**gamma * log(p) where
      float term1 = powf((1. - p), gamma) * logf(max(p, FLT_MIN));
      top_loss_data[i] = -alpha * term1;
    } else { // t != c + 1
      // p**gamma * log(1 - p)
      float term2 = powf(p, gamma) * (-1. * bottom_data[i] * (bottom_data[i] >= 0) -
           logf(1. + expf(bottom_data[i] - 2. * bottom_data[i] * (bottom_data[i] >= 0))));      
      top_loss_data[i] = -(1.0 - alpha) * term2;
    }
  }
}


template<typename Dtype>
inline void SigmoidFocalLossForward(const Tensor<gpu, 3, Dtype> &out_prob,
                                    const Tensor<gpu, 3, Dtype> &out_loss,
                                    const Tensor<gpu, 3, Dtype> &in_data,
                                    const Tensor<gpu, 2, Dtype> &in_label,
                                    const float ignore_label,
                                    const float alpha,
                                    const float gamma) {
  const Dtype *bottom_data = in_data.dptr_;
  const Dtype *bottom_label = in_label.dptr_;
  Dtype *top_prob_data = out_prob.dptr_;
  Dtype *top_loss_data = out_loss.dptr_;

  const int count = out_prob.shape_.Size();
  const int channel = in_data.size(1);
  const int spatial_dim = in_data.size(2);

  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "SigmoidFocalLoss Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out_prob.stream_);

  SigmoidFocalLossKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      bottom_data, bottom_label, top_prob_data, top_loss_data,
      count, channel, spatial_dim, ignore_label, alpha, gamma);
}

template<typename Dtype>
__global__ void SigmoidFocalLossGradientKernel(
    const Dtype* bottom_data, const Dtype* bottom_label, Dtype* bottom_data_diff,
    const int count, const int channel, const int spatial_dim,
    const float ignore_label, const float alpha, const float gamma) {

  for (int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       i < count; i += blockDim.x * gridDim.x * gridDim.y) {

    int s = i % spatial_dim;
    int c = (i / spatial_dim) % channel; 
    int n = i / (spatial_dim * channel);    
 
    int t = bottom_label[n * spatial_dim + s]; 

    // p = 1. / 1. + expf(-x)
    float p = 1. / (1. + expf(-bottom_data[i]));

    if (t == ignore_label) {
      bottom_data_diff[i] = 0.0;
    } else if (t == c + 1) {     
      // (1-p)**g * (1 - p - g*p*log(p))
      float term1 =
          powf((1. - p), gamma) *
          (1. - p - (p * gamma * logf(max(p, FLT_MIN))));
      bottom_data_diff[i] = -alpha * term1;
    } else { // t != c + 1
      // (p**g) * (g*(1-p)*log(1-p) - p)
      float term2 =
          powf(p, gamma) *
          ((-1. * bottom_data[i] * (bottom_data[i] >= 0) -
           logf(1. + expf(bottom_data[i] - 2. * bottom_data[i] * (bottom_data[i] >= 0)))) *
           (1. - p) * gamma - p);     
      bottom_data_diff[i] = -(1.0 - alpha) * term2;
    }
  }
}

template<typename Dtype>
inline void SigmoidFocalLossBackward(const Tensor<gpu, 3, Dtype> &in_data_grad,
                                     const Tensor<gpu, 3, Dtype> &in_data,
                                     const Tensor<gpu, 2, Dtype> &in_label,
                                     const float ignore_label,
                                     const float alpha,
                                     const float gamma) {
  const Dtype *bottom_data = in_data.dptr_;
  const Dtype* bottom_label = in_label.dptr_;
  Dtype *bottom_data_diff = in_data_grad.dptr_;
  
  const int count = in_data_grad.shape_.Size();
  const int channel = in_data.size(1);
  const int spatial_dim = in_data.size(2);
  
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "SigmoidFocalLoss Backward");
  cudaStream_t stream = Stream<gpu>::GetStream(in_data_grad.stream_);

  SigmoidFocalLossGradientKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      bottom_data, bottom_label, bottom_data_diff, count, channel, spatial_dim, 
      ignore_label, alpha, gamma);
}

}  // namespace cuda

template<typename Dtype>
inline void SigmoidFocalLossForward(const Tensor<gpu, 3, Dtype> &out_prob,
                                    const Tensor<gpu, 3, Dtype> &out_loss,
                                    const Tensor<gpu, 3, Dtype> &in_data,
                                    const Tensor<gpu, 2, Dtype> &in_label,
                                    const float ignore_label,
                                    const float alpha,
                                    const float gamma) {
  cuda::SigmoidFocalLossForward(out_prob, out_loss, in_data, in_label, ignore_label, alpha, gamma);
}

template<typename Dtype>
inline void SigmoidFocalLossBackward(const Tensor<gpu, 3, Dtype> &in_data_grad,
                                     const Tensor<gpu, 3, Dtype> &in_data,
                                     const Tensor<gpu, 2, Dtype> &in_label,
                                     const float ignore_label,
                                     const float alpha,
                                     const float gamma) {
  cuda::SigmoidFocalLossBackward(in_data_grad, in_data, in_label, ignore_label, alpha, gamma);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(SigmoidFocalLossParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SigmoidFocalLossOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
