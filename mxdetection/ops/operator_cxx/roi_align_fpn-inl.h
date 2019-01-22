#ifndef MXNET_OPERATOR_ROI_ALIGN_FPN_INL_H_
#define MXNET_OPERATOR_ROI_ALIGN_FPN_INL_H_

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
namespace roialignfpn {
enum ROIAlignFPNOpInputs {kDataRes, kBox};
enum ROIAlignFPNOpOutputs {kOut};
}  // ROIAlignFPN

struct ROIAlignFPNParam : public dmlc::Parameter<ROIAlignFPNParam> {
  TShape pooled_size;
  int sample_per_part;
  nnvm::Tuple<int> feature_strides;
  DMLC_DECLARE_PARAMETER(ROIAlignFPNParam) {
    int tmp[] = {0, 0, 0, 0};
    DMLC_DECLARE_FIELD(pooled_size)
    .set_expect_ndim(2).enforce_nonzero()
    .describe("ROI pooling output shape (h,w) "); 
    DMLC_DECLARE_FIELD(sample_per_part).set_default(1).describe("fix samples per part");
    tmp[0] = 32; tmp[1] = 16; tmp[2] = 8; tmp[3] = 4;
    DMLC_DECLARE_FIELD(feature_strides).set_default(nnvm::Tuple<int>(tmp, tmp + 4)).describe("");
  }
};

template<typename xpu, typename DType>
class ROIAlignFPNOp : public Operator {
 public:
  explicit ROIAlignFPNOp(ROIAlignFPNParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data_res2; data_res2.dptr_ = NULL;
    Tensor<xpu, 4, DType> data_res3; data_res3.dptr_ = NULL;
    Tensor<xpu, 4, DType> data_res4; data_res4.dptr_ = NULL;
    Tensor<xpu, 4, DType> data_res5; data_res5.dptr_ = NULL;
    for (size_t k = 0; k < param_.feature_strides.ndim(); ++k) {
      int feature_stride = param_.feature_strides[k];
      if (feature_stride == 4) {
        data_res2 = in_data[k].get<xpu, 4, DType>(s);
        CHECK_EQ(data_res2.CheckContiguous(), true);
      } 
      else if (feature_stride == 8) {
        data_res3 = in_data[k].get<xpu, 4, DType>(s);
        CHECK_EQ(data_res3.CheckContiguous(), true);
      } 
      else if (feature_stride == 16) {
        data_res4 = in_data[k].get<xpu, 4, DType>(s);
        CHECK_EQ(data_res4.CheckContiguous(), true);
      } 
      else if (feature_stride == 32) {
        data_res5 = in_data[k].get<xpu, 4, DType>(s);
        CHECK_EQ(data_res5.CheckContiguous(), true);
      }
       else {
        CHECK_EQ(0, 1);
      }
    }
    Tensor<xpu, 2, DType> bbox = in_data[param_.feature_strides.ndim()].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[roialignfpn::kOut].get<xpu, 4, DType>(s);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    CHECK_EQ(req[roialignfpn::kOut], kWriteTo);
    ROIAlignFPNForward(out, data_res2, data_res3, data_res4, data_res5, bbox, param_.sample_per_part);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> grad_in_res2; grad_in_res2.dptr_ = NULL;
    Tensor<xpu, 4, DType> grad_in_res3; grad_in_res3.dptr_ = NULL;
    Tensor<xpu, 4, DType> grad_in_res4; grad_in_res4.dptr_ = NULL;
    Tensor<xpu, 4, DType> grad_in_res5; grad_in_res5.dptr_ = NULL;
    for (size_t k = 0; k < param_.feature_strides.ndim(); ++k) {
      int feature_stride = param_.feature_strides[k];
      if (feature_stride == 4) {
        grad_in_res2 = in_grad[k].get<xpu, 4, DType>(s);
        CHECK_EQ(grad_in_res2.CheckContiguous(), true);
        if (req[k] == kWriteTo) {
          grad_in_res2 = 0.0f;
        }
      } 
      else if (feature_stride == 8) {
        grad_in_res3 = in_grad[k].get<xpu, 4, DType>(s);
        CHECK_EQ(grad_in_res3.CheckContiguous(), true);
        if (req[k] == kWriteTo) {
          grad_in_res3 = 0.0f;
        }
      } 
      else if (feature_stride == 16) {
        grad_in_res4 = in_grad[k].get<xpu, 4, DType>(s);
        CHECK_EQ(grad_in_res4.CheckContiguous(), true);
        if (req[k] == kWriteTo) {
          grad_in_res4 = 0.0f;
        }
      } 
      else if (feature_stride == 32) {
        grad_in_res5 = in_grad[k].get<xpu, 4, DType>(s);
        CHECK_EQ(grad_in_res5.CheckContiguous(), true);
        if (req[k] == kWriteTo) {
          grad_in_res5 = 0.0f;
        }
      } 
      else {
        CHECK_EQ(0, 1);
      }
    }
    Tensor<xpu, 2, DType> grad_roi = in_grad[param_.feature_strides.ndim()].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> grad_out = out_grad[roialignfpn::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> bbox = in_data[param_.feature_strides.ndim()].get<xpu, 2, DType>(s);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(grad_out.CheckContiguous(), true);
    ROIAlignFPNBackward(grad_in_res2, grad_in_res3, grad_in_res4, grad_in_res5, grad_out, bbox, param_.sample_per_part);
    grad_roi = 0.0f;
  }

 private:
  ROIAlignFPNParam param_;
};  // class ROIAlignFPNOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(ROIAlignFPNParam param, int dtype);

#if DMLC_USE_CXX11
class ROIAlignFPNProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    std::vector<std::string> ret;
    for (size_t k = 0; k < param_.feature_strides.ndim(); ++k) {
      ret.push_back(std::string("data_res") + std::to_string(param_.feature_strides[k]));
    }
    ret.push_back("rois");
    return ret;
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
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
    CHECK_EQ(in_shape->size(), param_.feature_strides.ndim() + 1);

    // data: [batch_size, c, h, w]
    TShape dshape_res0 = in_shape->at(0);
    for (size_t k = 0; k < param_.feature_strides.ndim(); ++k) {
      TShape dshape_res = in_shape->at(k);
      CHECK_EQ(dshape_res.ndim(), 4U) << "data should be a 4D tensor";
      CHECK_EQ(dshape_res[1], dshape_res0[1]);
    }

    // bbox: [num_rois, 5]
    TShape bshape = in_shape->at(param_.feature_strides.ndim());
    CHECK_EQ(bshape.ndim(), 2U) << "bbox should be a 2D tensor of shape [batch, 5]";

    // out: [num_rois, c, pooled_h, pooled_w]
    out_shape->clear();
    out_shape->push_back(Shape4(bshape[0], dshape_res0[1], param_.pooled_size[0], param_.pooled_size[1]));
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
    return true;
  }

  OperatorProperty* Copy() const override {
    ROIAlignFPNProp* roi_align_fpn_sym = new ROIAlignFPNProp();
    roi_align_fpn_sym->param_ = this->param_;
    return roi_align_fpn_sym;
  }

  std::string TypeString() const override {
    return "ROIAlignFPN";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[roialignfpn::kOut], in_data[param_.feature_strides.ndim()]};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  ROIAlignFPNParam param_;
};  // class ROIAlignFPNProp
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ROI_ALIGN_FPN_INL_H_
