#ifndef MXNET_OPERATOR_CONTRIB_PROPOSALFPN_INL_H_
#define MXNET_OPERATOR_CONTRIB_PROPOSALFPN_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <cstring>
#include <iostream>
#include "../operator_common.h"
#include "../mshadow_op.h"


namespace mxnet {
namespace op {

namespace proposal_fpn {
enum ProposalFPNOpInputs {kClsProb, kBBoxPred, kImInfo};
enum ProposalFPNOpOutputs {kOut, kScore};
enum ProposalFPNForwardResource {kTempResource};
}  // ProposalFPN

struct ProposalFPNParam : public dmlc::Parameter<ProposalFPNParam> {
  int rpn_pre_nms_top_n;
  int rpn_post_nms_top_n;
  float threshold;
  nnvm::Tuple<int> rpn_min_size;
  nnvm::Tuple<float> scales;
  nnvm::Tuple<float> ratios;
  nnvm::Tuple<int> feature_strides;
  bool output_score;
  bool iou_loss;
  DMLC_DECLARE_PARAMETER(ProposalFPNParam) {
    float tmp[] = {0, 0, 0, 0};
    int tmp1[] = {0, 0, 0, 0};
    DMLC_DECLARE_FIELD(rpn_pre_nms_top_n).set_default(6000)
    .describe("Number of top scoring boxes to keep after applying NMS to RPN ProposalFPNs");
    DMLC_DECLARE_FIELD(rpn_post_nms_top_n).set_default(300)
    .describe("Overlap threshold used for non-maximum"
              "suppresion(suppress boxes with IoU >= this threshold");
    DMLC_DECLARE_FIELD(threshold).set_default(0.7)
    .describe("NMS value, below which to suppress.");
    tmp1[0] = 0; tmp1[1] = 0; tmp1[2] = 0; tmp1[3] = 0;
    DMLC_DECLARE_FIELD(rpn_min_size).set_default(nnvm::Tuple<int>(tmp1, tmp1 + 4))
    .describe("Minimum height or width in ProposalFPN");
    tmp[0] = 4.0f; tmp[1] = 8.0f; tmp[2] = 16.0f; tmp[3] = 32.0f;
    DMLC_DECLARE_FIELD(scales).set_default(nnvm::Tuple<float>(tmp, tmp + 4))
    .describe("Used to generate anchor windows by enumerating scales");
    tmp[0] = 0.5f; tmp[1] = 1.0f; tmp[2] = 2.0f;
    DMLC_DECLARE_FIELD(ratios).set_default(nnvm::Tuple<float>(tmp, tmp + 3))
    .describe("Used to generate anchor windows by enumerating ratios");
    tmp1[0] = 32; tmp1[1] = 16; tmp1[2] = 8; tmp1[3] = 4;
    DMLC_DECLARE_FIELD(feature_strides).set_default(nnvm::Tuple<int>(tmp1, tmp1 + 4))
    .describe("The size of the receptive field each unit in the convolution layer of the rpn,"
              "for example the product of all stride's prior to this layer.");
    DMLC_DECLARE_FIELD(output_score).set_default(false)
    .describe("Add score to outputs");
    DMLC_DECLARE_FIELD(iou_loss).set_default(false)
    .describe("Usage of IoU Loss");
  }
};

template<typename xpu>
Operator* CreateOp(ProposalFPNParam param, int dtype);

#if DMLC_USE_CXX11
class ProposalFPNProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    std::vector<std::string> ret;
    for (size_t k = 0; k < param_.feature_strides.ndim(); ++k) {
      ret.push_back(std::string("cls_prob_stride") + std::to_string(param_.feature_strides[k]));
    }
    for (size_t k = 0; k < param_.feature_strides.ndim(); ++k) {
      ret.push_back(std::string("bbox_pred_stride") + std::to_string(param_.feature_strides[k]));
    }
    ret.push_back("im_info");
    return ret;
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "score"};
  }

  int NumVisibleOutputs() const override {
    if (param_.output_score) {
      return 2;
    } else {
      return 1;
    }
  }

  int NumOutputs() const override {
    return 2;
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2 * param_.feature_strides.ndim() + 1);
    const TShape &dshape = in_shape->at(proposal_fpn::kClsProb);
    if (dshape.ndim() == 0) return false;
    Shape<2> im_info_shape;
    im_info_shape = Shape2(dshape[0], 3);
    SHAPE_ASSIGN_CHECK(*in_shape, 2 * param_.feature_strides.ndim(), im_info_shape);
    out_shape->clear();
    // output
    out_shape->push_back(Shape2(param_.rpn_post_nms_top_n * dshape[0], 5));
    // score
    out_shape->push_back(Shape2(param_.rpn_post_nms_top_n * dshape[0], 1));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ProposalFPNProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_ProposalFPN";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  ProposalFPNParam param_;
};  // class ProposalFPNProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

//========================
// Anchor Generation Utils
//========================
namespace mxnet {
namespace op {
namespace utils {

inline void ProposalFPN_MakeAnchor(float w,
                                   float h,
                                   float x_ctr,
                                   float y_ctr,
                                   std::vector<float> *out_anchors) {
  out_anchors->push_back(x_ctr - 0.5f * (w - 1.0f));
  out_anchors->push_back(y_ctr - 0.5f * (h - 1.0f));
  out_anchors->push_back(x_ctr + 0.5f * (w - 1.0f));
  out_anchors->push_back(y_ctr + 0.5f * (h - 1.0f));
  out_anchors->push_back(0.0f);
}

inline void ProposalFPN_Transform(float scale, 
                                  float ratio,
                                  const std::vector<float>& base_anchor,
                                  std::vector<float>  *out_anchors) {
  float w = base_anchor[2] - base_anchor[0] + 1.0f;
  float h = base_anchor[3] - base_anchor[1] + 1.0f;
  float x_ctr = base_anchor[0] + 0.5 * (w - 1.0f);
  float y_ctr = base_anchor[1] + 0.5 * (h - 1.0f);
  float size = w * h;
  float size_ratios = std::floor(size / ratio);
  float new_w = std::floor(std::sqrt(size_ratios) + 0.5f) * scale;
  float new_h = std::floor((new_w / scale * ratio) + 0.5f) * scale;

  ProposalFPN_MakeAnchor(new_w, new_h, x_ctr, y_ctr, out_anchors);
}

// out_anchors must have shape (n, 5), where n is ratios.size() * scales.size()
inline void ProposalFPN_GenerateAnchors(const std::vector<float>& base_anchor,
                                        const nnvm::Tuple<float>& ratios,
                                        const std::vector<float>& scales,
                                        std::vector<float> *out_anchors) {
  for (size_t j = 0; j < ratios.ndim(); ++j) {
    for (size_t k = 0; k < scales.size(); ++k) {
      ProposalFPN_Transform(scales[k], ratios[j], base_anchor, out_anchors);
    }
  }
}

}  // namespace utils
}  // namespace op
}  // namespace mxnet

#endif  //  MXNET_OPERATOR_CONTRIB_PROPOSALFPN_INL_H_
