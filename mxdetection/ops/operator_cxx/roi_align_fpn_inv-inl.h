#ifndef MXNET_OPERATOR_ROI_ALIGN_FPN_INV_INL_H_
#define MXNET_OPERATOR_ROI_ALIGN_FPN_INV_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
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
namespace ROIAlignFPNInvinv {
enum ROIAlignFPNInvInvOpInputs {kData, kBox, kDataRes};
enum ROIAlignFPNInvInvOpOutputs {kOutRes};
}  // ROIAlignFPNInv

struct ROIAlignFPNInvParam : public dmlc::Parameter<ROIAlignFPNInvParam> {
  int sample_per_part;
  nnvm::Tuple<int> feature_strides;
  DMLC_DECLARE_PARAMETER(ROIAlignFPNInvParam) {
    DMLC_DECLARE_FIELD(sample_per_part).set_default(1).describe("fix samples per part");
    int tmp[] = {4, 8, 16, 32};
    DMLC_DECLARE_FIELD(feature_strides).set_default(nnvm::Tuple<int>(tmp, tmp + 4)).describe("");
  }
};

template<typename xpu, typename DType>
class ROIAlignFPNInvOp : public Operator {
 public:
  explicit ROIAlignFPNInvOp(ROIAlignFPNInvParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> out_data_res2; out_data_res2.dptr_ = NULL;
    Tensor<xpu, 4, DType> out_data_res3; out_data_res3.dptr_ = NULL;
    Tensor<xpu, 4, DType> out_data_res4; out_data_res4.dptr_ = NULL;
    Tensor<xpu, 4, DType> out_data_res5; out_data_res5.dptr_ = NULL;
    for (size_t k = 0; k < param_.feature_strides.ndim(); ++k) {
      int feature_stride = param_.feature_strides[k];
      if (feature_stride == 4) {
        out_data_res2 = out_data[k].get<xpu, 4, DType>(s);
        CHECK_EQ(out_data_res2.CheckContiguous(), true);
        if (req[k] == kWriteTo) {
          out_data_res2 = 0.0f;
        }
      } 
      else if (feature_stride == 8) {
        out_data_res3 = out_data[k].get<xpu, 4, DType>(s);
        CHECK_EQ(out_data_res3.CheckContiguous(), true);
        if (req[k] == kWriteTo) {
          out_data_res3 = 0.0f;
        }
      } 
      else if (feature_stride == 16) {
        out_data_res4 = out_data[k].get<xpu, 4, DType>(s);
        CHECK_EQ(out_data_res4.CheckContiguous(), true);
        if (req[k] == kWriteTo) {
          out_data_res4 = 0.0f;
        }
      } 
      else if (feature_stride == 32) {
        out_data_res5 = out_data[k].get<xpu, 4, DType>(s);
        CHECK_EQ(out_data_res5.CheckContiguous(), true);
        if (req[k] == kWriteTo) {
          out_data_res5 = 0.0f;
        }
      }
       else {
        CHECK_EQ(0, 1);
      }
    }
    Tensor<xpu, 4, DType> data = in_data[0].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> bbox = in_data[1].get<xpu, 2, DType>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(bbox.CheckContiguous(), true);

    ROIAlignFPNInvForward(out_data_res2, out_data_res3, out_data_res4, out_data_res5, data, bbox, param_.sample_per_part);

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
    Tensor<xpu, 4, DType> out_grad_res2; out_grad_res2.dptr_ = NULL;
    Tensor<xpu, 4, DType> out_grad_res3; out_grad_res3.dptr_ = NULL;
    Tensor<xpu, 4, DType> out_grad_res4; out_grad_res4.dptr_ = NULL;
    Tensor<xpu, 4, DType> out_grad_res5; out_grad_res5.dptr_ = NULL;
    for (size_t k = 0; k < param_.feature_strides.ndim(); ++k) {
      int feature_stride = param_.feature_strides[k];
      if (feature_stride == 4) {
        out_grad_res2 = out_grad[k].get<xpu, 4, DType>(s);
        CHECK_EQ(out_grad_res2.CheckContiguous(), true);
      } 
      else if (feature_stride == 8) {
        out_grad_res3 = out_grad[k].get<xpu, 4, DType>(s);
        CHECK_EQ(out_grad_res3.CheckContiguous(), true);
      } 
      else if (feature_stride == 16) {
        out_grad_res4 = out_grad[k].get<xpu, 4, DType>(s);
        CHECK_EQ(out_grad_res4.CheckContiguous(), true);
      } 
      else if (feature_stride == 32) {
        out_grad_res5 = out_grad[k].get<xpu, 4, DType>(s);
        CHECK_EQ(out_grad_res5.CheckContiguous(), true);
      } 
      else {
        CHECK_EQ(0, 1);
      }
    }
    Tensor<xpu, 4, DType> in_grad_diff = in_grad[0].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> bbox = in_data[1].get<xpu, 2, DType>(s);
    CHECK_EQ(in_grad_diff.CheckContiguous(), true);
    CHECK_EQ(bbox.CheckContiguous(), true);
    if (req[0] == kWriteTo) {
      in_grad_diff = 0.0f;
    }
    ROIAlignFPNInvBackward(in_grad_diff, out_grad_res2, out_grad_res3, out_grad_res4, out_grad_res5, bbox, param_.sample_per_part);

    Tensor<xpu, 2, DType> in_grad_bbox = in_grad[1].get<xpu, 2, DType>(s);
    ASSIGN_DISPATCH(in_grad_bbox, req[1], 0);
    for (size_t k = 0; k < param_.feature_strides.ndim(); ++k) {
        Tensor<xpu, 4, DType> in_grad_res = in_grad[2 + k].get<xpu, 4, DType>(s);
        ASSIGN_DISPATCH(in_grad_res, req[2 + k], 0);
    }
  }

 private:
  ROIAlignFPNInvParam param_;
};  // class ROIAlignFPNInvOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(ROIAlignFPNInvParam param, int dtype);

#if DMLC_USE_CXX11
class ROIAlignFPNInvProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    std::vector<std::string> ret;
    ret.push_back("data");
    ret.push_back("rois");
    for (size_t k = 0; k < param_.feature_strides.ndim(); ++k) {
      ret.push_back(std::string("data_res") + std::to_string(param_.feature_strides[k]));
    }
    return ret;
  }

  std::vector<std::string> ListOutputs() const override {
    std::vector<std::string> ret;
    for (size_t k = 0; k < param_.feature_strides.ndim(); ++k) {
      ret.push_back(std::string("output_res") + std::to_string(param_.feature_strides[k]));
    }
    return ret;
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
    CHECK_GE(in_shape->size(), 2);

    // data: [num_rois, c, pooled_h, pooled_w]
    TShape dshape = in_shape->at(0);
    CHECK_EQ(dshape.ndim(), 4U) << "data should be a 4D tensor";
    // bbox: [num_rois, 5]
    TShape bshape = in_shape->at(1);
    CHECK_EQ(bshape.ndim(), 2U) << "bbox should be a 2D tensor of shape [num_rois, 5]";
    // data_res: [batch_size, c, h, w]
    for (size_t k = 0; k < param_.feature_strides.ndim(); ++k) {
      TShape dshape_res = in_shape->at(2 + k);
      CHECK_EQ(dshape_res.ndim(), 4U) << "data should be a 4D tensor";
      CHECK_EQ(dshape_res[1], dshape[1]);
    }

    // out: [batch_size, c, h, w]
    out_shape->clear();
    for (size_t k = 0; k < param_.feature_strides.ndim(); ++k) {
      TShape dshape_res = in_shape->at(2 + k);
      out_shape->push_back(dshape_res);
    }
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
    for (size_t k = 0; k < param_.feature_strides.ndim(); ++k) {
        out_type->push_back(dtype);
    }
    return true;
  }

  OperatorProperty* Copy() const override {
    ROIAlignFPNInvProp* roi_align_fpn_inv_sym = new ROIAlignFPNInvProp();
    roi_align_fpn_inv_sym->param_ = this->param_;
    return roi_align_fpn_inv_sym;
  }

  std::string TypeString() const override {
    return "ROIAlignFPNInv";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    std::vector<int> ret;
    for (size_t k = 0; k < param_.feature_strides.ndim(); ++k) {
      ret.push_back(out_grad[k]);
    }
    ret.push_back(in_data[1]);
    return ret;
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  ROIAlignFPNInvParam param_;
};  // class ROIAlignFPNInvProp
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ROI_ALIGN_FPN_INV_INL_H_
