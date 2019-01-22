#ifndef MXNET_OPERATOR_CROP_AND_RESIZE_INL_H_
#define MXNET_OPERATOR_CROP_AND_RESIZE_INL_H_

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
namespace crop_and_resize {
enum CropAndResizeOpInputs {kData, kBox};
enum CropAndResizeOpOutputs { kOut };
}  // crop_and_resize

struct CropAndResizeParam : public dmlc::Parameter<CropAndResizeParam> {
  TShape crop_size;
  float spatial_scale;
  DMLC_DECLARE_PARAMETER(CropAndResizeParam) {
    DMLC_DECLARE_FIELD(crop_size)
    .set_expect_ndim(2).enforce_nonzero()
    .describe("crop shape (h,w) ");
    DMLC_DECLARE_FIELD(spatial_scale).set_range(0.0, 1.0)
    .describe("Ratio of input feature map height (or w) to raw image height (or w). "
    "Equals the reciprocal of total stride in convolutional layers");
  }
};

template<typename xpu, typename DType>
class CropAndResizeOp : public Operator {
 public:
  explicit CropAndResizeOp(CropAndResizeParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(out_data[crop_and_resize::kOut].shape_[0], in_data[crop_and_resize::kBox].shape_[0]);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[crop_and_resize::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> bbox = in_data[crop_and_resize::kBox].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[crop_and_resize::kOut].get<xpu, 4, DType>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    out = -FLT_MAX;
    CropAndResizeForward(out, data, bbox, param_.spatial_scale);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(out_grad[crop_and_resize::kOut].shape_[0], in_data[crop_and_resize::kBox].shape_[0]);
    CHECK_NE(req[crop_and_resize::kData], kWriteInplace) <<
      "CropAndResize: Backward doesn't support kWriteInplace.";
    CHECK_NE(req[crop_and_resize::kBox], kWriteInplace) <<
      "CropAndResize: Backward doesn't support kWriteInplace.";
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> grad_out = out_grad[crop_and_resize::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> bbox = in_data[crop_and_resize::kBox].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> grad_in = in_grad[crop_and_resize::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> grad_roi = in_grad[crop_and_resize::kBox].get<xpu, 2, DType>(s);
    CHECK_EQ(grad_out.CheckContiguous(), true);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(grad_in.CheckContiguous(), true);
    if (kAddTo == req[crop_and_resize::kData] || kWriteTo == req[crop_and_resize::kData]) {
      if (kWriteTo == req[crop_and_resize::kData]) {
        grad_in = 0.0f;
      }
      CropAndResizeBackward(grad_in, grad_out, bbox, param_.spatial_scale);
    }
    if (kWriteTo == req[crop_and_resize::kBox]) {
      grad_roi = 0.0f;
    }
  }

 private:
  CropAndResizeParam param_;
};  // class CropAndResizeOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(CropAndResizeParam param, int dtype);

#if DMLC_USE_CXX11
class CropAndResizeProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "rois"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  int NumOutputs() const override {
    return 1;
  }

  int NumVisibleOutputs() const override {
    return 1;
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
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, rois]";

    // data: [batch_size, c, h, w]
    TShape dshape = in_shape->at(crop_and_resize::kData);
    CHECK_EQ(dshape.ndim(), 4U) << "data should be a 4D tensor";

    // bbox: [num_rois, 5]
    TShape bshape = in_shape->at(crop_and_resize::kBox);
    CHECK_EQ(bshape.ndim(), 2U) << "bbox should be a 2D tensor of shape [batch, 5]";
    CHECK_EQ(bshape[1], 5U) << "bbox should be a 2D tensor of shape [batch, 5]";

    // out: [num_rois, c, crop_h, crop_w]
    out_shape->clear();
    out_shape->push_back(
         Shape4(bshape[0], dshape[1], param_.crop_size[0], param_.crop_size[1]));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 2U);
    int dtype = (*in_type)[0];
    CHECK_EQ(dtype, (*in_type)[1]);
    CHECK_NE(dtype, -1) << "Input must have specified type";

    out_type->clear();
    out_type->push_back(dtype);
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    CropAndResizeProp* crop_and_resize_sym = new CropAndResizeProp();
    crop_and_resize_sym->param_ = this->param_;
    return crop_and_resize_sym;
  }

  std::string TypeString() const override {
    return "CropAndResize";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[crop_and_resize::kOut], in_data[crop_and_resize::kBox]};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  CropAndResizeParam param_;
};  // class CropAndResizeProp
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CROP_AND_RESIZE_INL_H_
