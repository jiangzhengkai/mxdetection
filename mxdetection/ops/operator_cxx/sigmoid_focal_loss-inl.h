#ifndef MXNET_OPERATOR_SIGMOID_FOCAL_LOSS_INL_H_
#define MXNET_OPERATOR_SIGMOID_FOCAL_LOSS_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./mshadow_op.h"
#include "./operator_common.h"


namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace SigmoidFocalLoss {
enum SigmoidFocalLossOpInputs {kData, kLabel};
enum SigmoidFocalLossOpOutputs {kProb, kLoss};
enum SigmoidFocalLossOpResource {kTempSpace};
}  // SigmoidFocalLoss

struct SigmoidFocalLossParam : public dmlc::Parameter<SigmoidFocalLossParam> {
  float grad_scale;
  float ignore_label;
  float alpha;
  float gamma;
  DMLC_DECLARE_PARAMETER(SigmoidFocalLossParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scales the gradient by a float factor.");
    DMLC_DECLARE_FIELD(ignore_label).set_default(-1.0f)
    .describe("The instances whose `labels` == `ignore_label` will be ignored "
              "during backward, if `use_ignore` is set to ``true``).");  
    DMLC_DECLARE_FIELD(alpha).set_default(1.0f)
    .describe("Focal Loss's alpha hyper-parameter.");
    DMLC_DECLARE_FIELD(gamma).set_default(1.0f)
    .describe("Focal Loss's gamma hyper-parameter.");    
  }
};

template<typename xpu, typename DType>
class SigmoidFocalLossOp : public Operator {
 public:
  explicit SigmoidFocalLossOp(SigmoidFocalLossParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 2);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    int n = in_data[SigmoidFocalLoss::kData].size(0);
    int k = in_data[SigmoidFocalLoss::kData].size(1);
    Shape<3> s3 = Shape3(n, k, static_cast<int>(in_data[SigmoidFocalLoss::kData].Size()/n/k));
    Shape<2> s2 = Shape2(s3[0], s3[2]);

    Tensor<xpu, 3, DType> data = in_data[SigmoidFocalLoss::kData].get_with_shape<xpu, 3, DType>(s3, s);
    Tensor<xpu, 2, DType> label = in_data[SigmoidFocalLoss::kLabel].get_with_shape<xpu, 2, DType>(s2, s);
    Tensor<xpu, 3, DType> out_prob = out_data[SigmoidFocalLoss::kProb].get_with_shape<xpu, 3, DType>(s3, s);
    Tensor<xpu, 3, DType> out_loss = out_data[SigmoidFocalLoss::kLoss].get_with_shape<xpu, 3, DType>(s3, s);
    
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(label.CheckContiguous(), true);
    CHECK_EQ(out_prob.CheckContiguous(), true);
    CHECK_EQ(out_loss.CheckContiguous(), true);

    SigmoidFocalLossForward(out_prob, out_loss, data, label, param_.ignore_label, param_.alpha, param_.gamma);

    int num_pos_label = 0;
    Tensor<cpu, 2, DType> workspace = ctx.requested[SigmoidFocalLoss::kTempSpace].get_host_space_typed<2, DType>(label.shape_);
    Copy(workspace, label, label.stream_);
    for (index_t i = 0; i < workspace.size(0); ++i) {
      for (index_t j = 0; j < workspace.size(1); ++j) {
        if (static_cast<int>(workspace[i][j]) > 0) {
          ++num_pos_label;
        }
      }
    }
    num_pos_label = num_pos_label == 0 ? 1 : num_pos_label;
    out_loss *= DType(1.0 / num_pos_label);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 2U);
    CHECK_GE(in_grad.size(), 1U);
    CHECK_GE(req.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    int n = in_data[SigmoidFocalLoss::kData].size(0);
    int k = in_data[SigmoidFocalLoss::kData].size(1);
    Shape<3> s3 = Shape3(n, k, static_cast<int>(in_data[SigmoidFocalLoss::kData].Size()/n/k));
    Shape<2> s2 = Shape2(s3[0], s3[2]);

    Tensor<xpu, 3, DType> data = in_data[SigmoidFocalLoss::kData].get_with_shape<xpu, 3, DType>(s3, s);
    Tensor<xpu, 2, DType> label = in_data[SigmoidFocalLoss::kLabel].get_with_shape<xpu, 2, DType>(s2, s);
    Tensor<xpu, 3, DType> in_data_grad = in_grad[SigmoidFocalLoss::kData].get_with_shape<xpu, 3, DType>(s3, s);

    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(label.CheckContiguous(), true);
    CHECK_EQ(in_data_grad.CheckContiguous(), true);

    if (kAddTo == req[SigmoidFocalLoss::kData] || kWriteTo == req[SigmoidFocalLoss::kData]) {
      if (kWriteTo == req[SigmoidFocalLoss::kData]) {
        in_data_grad = 0.0f;
      }

      SigmoidFocalLossBackward(in_data_grad, data, label, param_.ignore_label, param_.alpha, param_.gamma);

      int num_pos_label = 0;
      Tensor<cpu, 2, DType> workspace = ctx.requested[SigmoidFocalLoss::kTempSpace].get_host_space_typed<2, DType>(label.shape_);
      Copy(workspace, label, label.stream_);
      for (index_t i = 0; i < workspace.size(0); ++i) {
        for (index_t j = 0; j < workspace.size(1); ++j) {
          if (static_cast<int>(workspace[i][j]) > 0) {
            ++num_pos_label;
          }
        }
      }
      num_pos_label = num_pos_label == 0 ? 1 : num_pos_label;
      in_data_grad *= DType(param_.grad_scale / num_pos_label);
    }
    
    if (kWriteTo == req[SigmoidFocalLoss::kLabel]) {
      Tensor<xpu, 2, DType> in_label_grad = in_grad[SigmoidFocalLoss::kLabel].get_with_shape<xpu, 2, DType>(s2, s);
      CHECK_EQ(in_label_grad.CheckContiguous(), true);
      in_label_grad = 0.0f;
    }
  }

 private:
  SigmoidFocalLossParam param_;
};  // class SigmoidFocalLossOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(SigmoidFocalLossParam param, int dtype);

#if DMLC_USE_CXX11
class SigmoidFocalLossProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output_prob", "output_loss"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, label]";
    TShape &dshape = in_shape->at(SigmoidFocalLoss::kData);
    if (dshape.ndim() == 0) return false;

    TShape lshape1 = Shape2(dshape[0], dshape.Size()/dshape[0]/dshape[1]);
    TShape lshape2(dshape.ndim() - 1);
    lshape2[0] = dshape[0];
    for (index_t i = 2; i < dshape.ndim(); ++i)
      lshape2[i-1] = dshape[i];
    TShape lshape3 = dshape;
    lshape3[1] = 1;
    if (in_shape->at(SigmoidFocalLoss::kLabel).ndim() == 0) {
      in_shape->at(SigmoidFocalLoss::kLabel) = lshape1;
    } else if (in_shape->at(SigmoidFocalLoss::kLabel) == lshape1) {
    } else if (in_shape->at(SigmoidFocalLoss::kLabel) == lshape2) {
    } else if (in_shape->at(SigmoidFocalLoss::kLabel) == lshape3) {
    } else {
      std::ostringstream os;
      os << "Expecting " << lshape1 << " or " << lshape2
         << ". But got " << in_shape->at(SigmoidFocalLoss::kLabel);
      throw InferShapeError(os.str(), SigmoidFocalLoss::kLabel);
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(dshape);
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
    SigmoidFocalLossProp* sigmoid_focal_loss_sym = new SigmoidFocalLossProp();
    sigmoid_focal_loss_sym->param_ = this->param_;
    return sigmoid_focal_loss_sym;
  }

  std::string TypeString() const override {
    return "SigmoidFocalLoss";
  }

  virtual std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[SigmoidFocalLoss::kData], in_data[SigmoidFocalLoss::kLabel]};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  SigmoidFocalLossParam param_;
};  // class SigmoidFocalLossProp
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SIGMOID_FOCAL_LOSS_INL_H_
