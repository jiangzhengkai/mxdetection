#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>


#include <algorithm>
#include <functional>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <iostream>

#include "../operator_common.h"
#include "../mshadow_op.h"
#include "./proposal_fpn-inl.h"

#define ProposalFPN_DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define ProposalFPN_FRCNN_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
} while (0)

namespace mshadow {
namespace cuda {

// scores are (b, anchor, h, w)
// workspace_proposalfpns are (h * w * anchor, 5)
// w defines "x" and h defines "y"
// count should be total anchors numbers, h * w * anchors
template<typename Dtype>
__global__ void ProposalFPN_GridKernel(const int count,
                                       const int num_anchors,
                                       const int height,
                                       const int width,
                                       const int feature_stride,
                                       const Dtype* scores,
                                       Dtype* workspace_proposalfpns) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int a = index % num_anchors;
    int w = (index / num_anchors) % width;
    int h = index / num_anchors / width;

    workspace_proposalfpns[index * 5 + 0] = workspace_proposalfpns[a * 5 + 0] + w * feature_stride;
    workspace_proposalfpns[index * 5 + 1] = workspace_proposalfpns[a * 5 + 1] + h * feature_stride;
    workspace_proposalfpns[index * 5 + 2] = workspace_proposalfpns[a * 5 + 2] + w * feature_stride;
    workspace_proposalfpns[index * 5 + 3] = workspace_proposalfpns[a * 5 + 3] + h * feature_stride;
    workspace_proposalfpns[index * 5 + 4] = scores[(a * height + h) * width + w];
  }
}

// boxes are (h * w * anchor, 5)
// deltas are (b, 4 * anchor, h, w)
// out_pred_boxes are (h * w * anchor, 5)
// count should be total anchors numbers, h * w * anchors
// in-place write: boxes and out_pred_boxes are the same location
template<typename Dtype>
__global__ void ProposalFPN_BBoxPredKernel(const int count,
                                           const int num_anchors,
                                           const int feat_height,
                                           const int feat_width,
                                           const int real_height,
                                           const int real_width,
                                           const float im_height,
                                           const float im_width,
                                           const Dtype* boxes,
                                           const Dtype* deltas,
                                           Dtype* out_pred_boxes) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int a = index % num_anchors;
    int w = (index / num_anchors) % feat_width;
    int h = index / num_anchors / feat_width;

    float width = boxes[index * 5 + 2] - boxes[index * 5 + 0] + 1.0f;
    float height = boxes[index * 5 + 3] - boxes[index * 5 + 1] + 1.0f;
    float ctr_x = boxes[index * 5 + 0] + 0.5f * (width - 1.0f);
    float ctr_y = boxes[index * 5 + 1] + 0.5f * (height - 1.0f);

    float dx = deltas[((a * 4) * feat_height + h) * feat_width + w];
    float dy = deltas[((a * 4 + 1) * feat_height + h) * feat_width + w];
    float dw = deltas[((a * 4 + 2) * feat_height + h) * feat_width + w];
    float dh = deltas[((a * 4 + 3) * feat_height + h) * feat_width + w];

    float pred_ctr_x = dx * width + ctr_x;
    float pred_ctr_y = dy * height + ctr_y;
    float pred_w = exp(dw) * width;
    float pred_h = exp(dh) * height;

    float pred_x1 = pred_ctr_x - 0.5f * (pred_w - 1.0f);
    float pred_y1 = pred_ctr_y - 0.5f * (pred_h - 1.0f);
    float pred_x2 = pred_ctr_x + 0.5f * (pred_w - 1.0f);
    float pred_y2 = pred_ctr_y + 0.5f * (pred_h - 1.0f);

    pred_x1 = max(min(pred_x1, im_width - 1.0f), 0.0f);
    pred_y1 = max(min(pred_y1, im_height - 1.0f), 0.0f);
    pred_x2 = max(min(pred_x2, im_width - 1.0f), 0.0f);
    pred_y2 = max(min(pred_y2, im_height - 1.0f), 0.0f);

    out_pred_boxes[index * 5 + 0] = pred_x1;
    out_pred_boxes[index * 5 + 1] = pred_y1;
    out_pred_boxes[index * 5 + 2] = pred_x2;
    out_pred_boxes[index * 5 + 3] = pred_y2;

    if (h >= real_height || w >= real_width) {
      out_pred_boxes[index * 5 + 4] = -1.0f;
    }
  }
}

// boxes are (h * w * anchor, 5)
// deltas are (b, 4 * anchor, h, w)
// out_pred_boxes are (h * w * anchor, 5)
// count should be total anchors numbers, h * w * anchors
// in-place write: boxes and out_pred_boxes are the same location
template<typename Dtype>
__global__ void ProposalFPN_IoUPredKernel(const int count,
                                          const int num_anchors,
                                          const int feat_height,
                                          const int feat_width,
                                          const int real_height,
                                          const int real_width,
                                          const float im_height,
                                          const float im_width,
                                          const Dtype* boxes,
                                          const Dtype* deltas,
                                          Dtype* out_pred_boxes) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int a = index % num_anchors;
    int w = (index / num_anchors) % feat_width;
    int h = index / num_anchors / feat_width;

    float x1 = boxes[index * 5 + 0];
    float y1 = boxes[index * 5 + 1];
    float x2 = boxes[index * 5 + 2];
    float y2 = boxes[index * 5 + 3];

    float dx1 = deltas[((a * 4) * feat_height + h) * feat_width + w];
    float dy1 = deltas[((a * 4 + 1) * feat_height + h) * feat_width + w];
    float dx2 = deltas[((a * 4 + 2) * feat_height + h) * feat_width + w];
    float dy2 = deltas[((a * 4 + 3) * feat_height + h) * feat_width + w];

    float pred_x1 = max(min(x1 + dx1, im_width - 1.0f), 0.0f);
    float pred_y1 = max(min(y1 + dy1, im_height - 1.0f), 0.0f);
    float pred_x2 = max(min(x2 + dx2, im_width - 1.0f), 0.0f);
    float pred_y2 = max(min(y2 + dy2, im_height - 1.0f), 0.0f);

    out_pred_boxes[index * 5 + 0] = pred_x1;
    out_pred_boxes[index * 5 + 1] = pred_y1;
    out_pred_boxes[index * 5 + 2] = pred_x2;
    out_pred_boxes[index * 5 + 3] = pred_y2;

    if (h >= real_height || w >= real_width) {
      out_pred_boxes[index * 5 + 4] = -1.0f;
    }
  }
}

// filter box with stride less than rpn_min_size
// filter: set score to zero
// dets (n, 5)
template<typename Dtype>
__global__ void ProposalFPN_FilterBoxKernel(const int count,
                                            const float min_size,
                                            Dtype* dets) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    float iw = dets[index * 5 + 2] - dets[index * 5 + 0] + 1.0f;
    float ih = dets[index * 5 + 3] - dets[index * 5 + 1] + 1.0f;
    if (iw < min_size || ih < min_size) {
      dets[index * 5 + 0] -= min_size / 2;
      dets[index * 5 + 1] -= min_size / 2;
      dets[index * 5 + 2] += min_size / 2;
      dets[index * 5 + 3] += min_size / 2;
      dets[index * 5 + 4] = -1.0f;
    }
  }
}

// copy score and init order
// dets (n, 5); score (n, ); order (n, )
// count should be n (total anchors or ProposalFPNs)
template<typename Dtype>
__global__ void ProposalFPN_CopyScoreKernel(const int count,
                                            const Dtype* dets,
                                            Dtype* score,
                                            Dtype* order) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    score[index] = dets[index * 5 + 4];
    order[index] = index;
  }
}

// reorder ProposalFPNs according to order and keep the top_n ProposalFPNs
// prev_dets (n, 5); order (n, ); dets (n, 5)
// count should be output anchor numbers (top_n)
template<typename Dtype>
__global__ void ProposalFPN_ReorderKernel(const int count,
                                          const Dtype* prev_dets,
                                          const Dtype* order,
                                          Dtype* dets) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    const int order_i = order[index];
    for (int j = 0; j < 5; ++j) {
      dets[index * 5 + j] = prev_dets[order_i * 5 + j];
    }
  }
}

__device__ inline float ProposalFPN_devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__global__ void ProposalFPN_nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                                       const float *dev_boxes, unsigned long long *dev_mask) {
  const int threadsPerBlock = sizeof(unsigned long long) * 8;
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; ++i) {
      if (ProposalFPN_devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = ProposalFPN_DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void ProposalFPN_nms(const mshadow::Tensor<gpu, 2>& boxes,
                     const float nms_overlap_thresh,
                     int *keep,
                     int *num_out,
                     unsigned long long* mask_dev) {
  const int threadsPerBlock = sizeof(unsigned long long) * 8;
  const int boxes_num = boxes.size(0);
  const int boxes_dim = boxes.size(1);
  const int col_blocks = ProposalFPN_DIVUP(boxes_num, threadsPerBlock);

  float* boxes_dev = boxes.dptr_;

  dim3 blocks(ProposalFPN_DIVUP(boxes_num, threadsPerBlock),
              ProposalFPN_DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  ProposalFPN_nms_kernel<<<blocks, threads>>>(boxes_num,
                                              nms_overlap_thresh,
                                              boxes_dev,
                                              mask_dev);
  ProposalFPN_FRCNN_CUDA_CHECK(cudaPeekAtLastError());
  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  ProposalFPN_FRCNN_CUDA_CHECK(cudaMemcpy(&mask_host[0],
                              mask_dev,
                              sizeof(unsigned long long) * boxes_num * col_blocks,
                              cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; ++i) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; ++j) {
        remv[j] |= p[j];
      }
    }
  }
  *num_out = num_to_keep;
}

// copy ProposalFPNs to output
// dets (top_n, 5); keep (top_n, ); out (top_n, )
// count should be top_n (total anchors or ProposalFPNs)
template<typename Dtype>
__global__ void ProposalFPN_PrepareOutput(const int count,
                                          const Dtype* dets,
                                          const int* keep,
                                          const int out_size,
                                          const int image_ind,
                                          Dtype* out,
                                          Dtype* score) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    out[index * 5] = image_ind;
    if (index < out_size) {
      int keep_i = keep[index];
      for (int j = 0; j < 4; ++j) {
        out[index * 5 + j + 1] = dets[keep_i * 5 + j];
      }
      score[index] = dets[keep_i * 5 + 4];
    } else {
      int keep_i = keep[index % out_size];
      for (int j = 0; j < 4; ++j) {
        out[index * 5 + j + 1] = dets[keep_i * 5 + j];
      }
      score[index] = dets[keep_i * 5 + 4];
    }
  }
}

}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu, typename Dtype>
class ProposalFPNGPUOp : public Operator{
 public:
  explicit ProposalFPNGPUOp(ProposalFPNParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda;
    int num_stride = param_.feature_strides.ndim();
    CHECK_EQ(in_data.size(), 2 * num_stride + 1);
    CHECK_EQ(out_data.size(), 2);
    CHECK_EQ(req[proposal_fpn::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();
   
    int num_images = in_data[0].shape_[0];

    std::vector<std::vector<float>> scales;
    std::vector<float> scale_vec;
    for (size_t k = 0; k < param_.scales.ndim(); ++k){
      if (param_.scales[k] == -1){
        scales.push_back(scale_vec);
        scale_vec.clear();
      } else {
        scale_vec.push_back(param_.scales[k]);
      }
    }
    scales.push_back(scale_vec);
    CHECK_EQ(scales.size(), num_stride);

    int count = 0;
    for (size_t k = 0; k < num_stride; ++k) {
      CHECK_EQ(in_data[k].shape_[0], num_images);
      int num_anchors = param_.ratios.ndim() * scales[k].size();
      CHECK_EQ(num_anchors, in_data[k].shape_[1] / 2);
      int height = in_data[k].shape_[2];
      int width = in_data[k].shape_[3];
      count += num_anchors * height * width; 
    }

    // set to -1 for max
    int rpn_pre_nms_top_n = (param_.rpn_pre_nms_top_n > 0) ? param_.rpn_pre_nms_top_n : count;
    rpn_pre_nms_top_n = std::min(rpn_pre_nms_top_n, count);
    int rpn_post_nms_top_n = std::min(param_.rpn_post_nms_top_n, rpn_pre_nms_top_n);

    // get workspace
    ResourceSession rs(ctx.requested[proposal_fpn::kTempResource]);
    int workspace_size = count * 5 + 2 * count + rpn_pre_nms_top_n * 5;
    Tensor<xpu, 1> workspace = ctx.requested[proposal_fpn::kTempResource].get_space<xpu>(Shape1(workspace_size), s);
    int start = 0;
    Tensor<xpu, 2> workspace_proposalfpns(workspace.dptr_ + start, Shape2(count, 5));
    start += count * 5;
    Tensor<xpu, 2> workspace_pre_nms(workspace.dptr_ + start, Shape2(2, count));
    start += 2 * count;
    Tensor<xpu, 2> workspace_ordered_proposalfpns(workspace.dptr_ + start, Shape2(rpn_pre_nms_top_n, 5));
    start += rpn_pre_nms_top_n * 5;
    CHECK_EQ(workspace_size, start) << workspace_size << " " << start << std::endl;

    const int threadsPerBlock = sizeof(unsigned long long) * 8;
    const int boxes_num = workspace_ordered_proposalfpns.size(0);
    const int col_blocks = ProposalFPN_DIVUP(boxes_num, threadsPerBlock);
    Tensor<xpu, 1, unsigned long long> mask_dev = 
          ctx.requested[proposal_fpn::kTempResource].get_space_typed<xpu, 1, unsigned long long>(Shape1(boxes_num * col_blocks), s);  

    std::vector<int> _keep(rpn_pre_nms_top_n);
    int* keep = ctx.requested[proposal_fpn::kTempResource].get_space_typed<xpu, 1, int>(Shape1(rpn_pre_nms_top_n), s).dptr_; 

    Tensor<xpu, 2> im_info = in_data[2 * num_stride].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> out = out_data[proposal_fpn::kOut].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> out_score = out_data[proposal_fpn::kScore].get<xpu, 2, real_t>(s);

    dim3 dimGrid((count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock);
    dim3 dimBlock(kMaxThreadsPerBlock);

    for (size_t n = 0; n < num_images; ++n) {
      // im_info is small, we want to copy them to cpu
      std::vector<float> cpu_im_info(3);
      ProposalFPN_FRCNN_CUDA_CHECK(cudaMemcpy(&cpu_im_info[0], im_info.dptr_ + n * 3,
                                   sizeof(float) * cpu_im_info.size(),
                                   cudaMemcpyDeviceToHost));

      auto workspace_proposalfpns_ptr = workspace_proposalfpns.dptr_;
      for (size_t k = 0; k < num_stride; ++k) {
        int feature_stride = param_.feature_strides[k];
        int num_anchors = param_.ratios.ndim() * scales[k].size();
        int height = in_data[k].shape_[2];
        int width = in_data[k].shape_[3];
        int count_k = num_anchors * height * width; 

        Shape<4> fg_scores_shape = Shape4(1, num_anchors, height, width);
        real_t* foreground_score_ptr = in_data[k].dptr<real_t>() + n * 2 * count_k + fg_scores_shape.Size();

        Tensor<xpu, 4> scores = Tensor<xpu, 4>(foreground_score_ptr, fg_scores_shape);
        Tensor<xpu, 4> bbox_deltas = in_data[k + num_stride].get<xpu, 4, real_t>(s);    

        // Generate first anchors based on base anchor
        std::vector<float> base_anchor(4);
        base_anchor[0] = 0.0;
        base_anchor[1] = 0.0;
        base_anchor[2] = feature_stride - 1.0;
        base_anchor[3] = feature_stride - 1.0;  
        std::vector<float> anchors;
        utils::ProposalFPN_GenerateAnchors(base_anchor, param_.ratios, scales[k], &anchors);

        // Copy generated anchors to GPU
        ProposalFPN_FRCNN_CUDA_CHECK(cudaMemcpy(workspace_proposalfpns_ptr, &anchors[0], sizeof(float) * anchors.size(), cudaMemcpyHostToDevice));

        // Copy ProposalFPNs to a mesh grid
        dimGrid.x = (count_k + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
        CheckLaunchParam(dimGrid, dimBlock, "ProposalFPNGrid");
        ProposalFPN_GridKernel<<<dimGrid, dimBlock>>>(
          count_k, num_anchors, height, width, feature_stride,
          scores.dptr_, workspace_proposalfpns_ptr);
        ProposalFPN_FRCNN_CUDA_CHECK(cudaPeekAtLastError());

        // prevent padded predictions
        int real_height = static_cast<int>(cpu_im_info[0] / feature_stride);
        int real_width = static_cast<int>(cpu_im_info[1] / feature_stride);
        CHECK_GE(height, real_height) << height << " " << real_height << std::endl;
        CHECK_GE(width, real_width) << width << " " << real_width << std::endl;

        // Transform anchors and bbox_deltas into bboxes
        CheckLaunchParam(dimGrid, dimBlock, "BBoxPred");
        if (param_.iou_loss) {
          ProposalFPN_IoUPredKernel<<<dimGrid, dimBlock>>>(
            count_k, num_anchors, height, width, real_height, real_width,
            cpu_im_info[0], cpu_im_info[1],
            workspace_proposalfpns_ptr, bbox_deltas.dptr_ + n * 4 * count_k, workspace_proposalfpns_ptr);
        } else {
          ProposalFPN_BBoxPredKernel<<<dimGrid, dimBlock>>>(
            count_k, num_anchors, height, width, real_height, real_width,
            cpu_im_info[0], cpu_im_info[1],
            workspace_proposalfpns_ptr, bbox_deltas.dptr_ + n * 4 * count_k, workspace_proposalfpns_ptr);
        }
        ProposalFPN_FRCNN_CUDA_CHECK(cudaPeekAtLastError());

        // filter boxes with less than rpn_min_size
        CheckLaunchParam(dimGrid, dimBlock, "FilterBox");
        ProposalFPN_FilterBoxKernel<<<dimGrid, dimBlock>>>(count_k, param_.rpn_min_size[k] * cpu_im_info[2], workspace_proposalfpns_ptr);
        ProposalFPN_FRCNN_CUDA_CHECK(cudaPeekAtLastError());

        workspace_proposalfpns_ptr += count_k * 5; 
      }

      dimGrid.x = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;

      // Copy score to a continuous memory
      Tensor<xpu, 1> score = workspace_pre_nms[0];
      Tensor<xpu, 1> order = workspace_pre_nms[1];

      CheckLaunchParam(dimGrid, dimBlock, "CopyScore");
      ProposalFPN_CopyScoreKernel<<<dimGrid, dimBlock>>>(
        count, workspace_proposalfpns.dptr_, score.dptr_, order.dptr_);
      ProposalFPN_FRCNN_CUDA_CHECK(cudaPeekAtLastError());

      // argsort score, save order
      thrust::stable_sort_by_key(thrust::device,
                                score.dptr_,
                                score.dptr_ + score.size(0),
                                order.dptr_,
                                thrust::greater<real_t>());
      ProposalFPN_FRCNN_CUDA_CHECK(cudaPeekAtLastError());

      // Reorder ProposalFPNs according to order
      dimGrid.x = (rpn_pre_nms_top_n + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
      CheckLaunchParam(dimGrid, dimBlock, "ReorderProposalFPNs");
      ProposalFPN_ReorderKernel<<<dimGrid, dimBlock>>>(
        rpn_pre_nms_top_n, workspace_proposalfpns.dptr_, order.dptr_, workspace_ordered_proposalfpns.dptr_);
      ProposalFPN_FRCNN_CUDA_CHECK(cudaPeekAtLastError());

      // perform nms
      mask_dev = 0ULL;
      int out_size = 0;
      ProposalFPN_nms(workspace_ordered_proposalfpns, param_.threshold, &_keep[0], &out_size, mask_dev.dptr_);

      // copy nms result to gpu  
      ProposalFPN_FRCNN_CUDA_CHECK(cudaMemcpy(keep, &_keep[0], sizeof(int) * _keep.size(), cudaMemcpyHostToDevice));

      // copy results after nms
      dimGrid.x = (rpn_post_nms_top_n + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
      CheckLaunchParam(dimGrid, dimBlock, "PrepareOutput");
      ProposalFPN_PrepareOutput<<<dimGrid, dimBlock>>>(
        rpn_post_nms_top_n, workspace_ordered_proposalfpns.dptr_, keep, out_size, n,
        out.dptr_ + n * rpn_post_nms_top_n * 5, out_score.dptr_ + n * rpn_post_nms_top_n);
      ProposalFPN_FRCNN_CUDA_CHECK(cudaPeekAtLastError());
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    int num_stride = param_.feature_strides.ndim();
    CHECK_EQ(in_grad.size(), 2 * num_stride + 1);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    for (size_t k = 0; k < num_stride; ++k) {
      Tensor<xpu, 4> gscores = in_grad[k].get<xpu, 4, real_t>(s);
      Tensor<xpu, 4> gbbox = in_grad[k + num_stride].get<xpu, 4, real_t>(s);
      Assign(gscores, req[k], 0);
      Assign(gbbox, req[k + num_stride], 0);
    }

    Tensor<xpu, 2> ginfo = in_grad[2 * num_stride].get<xpu, 2, real_t>(s);
    Assign(ginfo, req[2 * num_stride], 0);
  }

 private:
  ProposalFPNParam param_;
};  // class ProposalFPNGPUOp


template<>
Operator* CreateOp<gpu>(ProposalFPNParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ProposalFPNGPUOp<gpu, DType>(param);
  });
  return op;
}



}  // namespace op
}  // namespace mxnet
