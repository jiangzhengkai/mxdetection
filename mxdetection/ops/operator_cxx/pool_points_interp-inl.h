#ifndef MXNET_OPERATOR_POOL_POINTS_INTERP_INL_H_
#define MXNET_OPERATOR_POOL_POINTS_INTERP_INL_H_

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
namespace PoolPointsInterp {
enum PoolPointsInterpOpInputs {kData, kPoints};
enum PoolPointsInterpOpOutputs {kOut};
}  // PoolPointsInterp

struct PoolPointsInterpParam : public dmlc::Parameter<PoolPointsInterpParam> {
  float spatial_scale;
  DMLC_DECLARE_PARAMETER(PoolPointsInterpParam) {
    DMLC_DECLARE_FIELD(spatial_scale).set_default(1.0).describe("");
  }
};

template<typename xpu, typename DType>
class PoolPointsInterpOp : public Operator {
 public:
  explicit PoolPointsInterpOp(PoolPointsInterpParam p) {
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
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[PoolPointsInterp::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> points = in_data[PoolPointsInterp::kPoints].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[PoolPointsInterp::kOut].get<xpu, 2, DType>(s);

    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(points.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);

    PoolPointsInterpForward(out, data, points, param_.spatial_scale);
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
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> in_data_grad = in_grad[PoolPointsInterp::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> points = in_data[PoolPointsInterp::kPoints].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> out_data_grad = out_grad[PoolPointsInterp::kOut].get<xpu, 2, DType>(s);

    CHECK_EQ(in_data_grad.CheckContiguous(), true);
    CHECK_EQ(points.CheckContiguous(), true);
    CHECK_EQ(out_data_grad.CheckContiguous(), true);

    if (kAddTo == req[PoolPointsInterp::kData] || kWriteTo == req[PoolPointsInterp::kData]) {
      if (kWriteTo == req[PoolPointsInterp::kData]) {
        in_data_grad = 0.0f;
      }
      PoolPointsInterpBackward(in_data_grad, out_data_grad, points, param_.spatial_scale);
    }
    
    if (kWriteTo == req[PoolPointsInterp::kPoints]) {
      Tensor<xpu, 2, DType> in_points_grad = in_grad[PoolPointsInterp::kPoints].get<xpu, 2, DType>(s);
      CHECK_EQ(in_points_grad.CheckContiguous(), true);
      in_points_grad = 0.0f;
    }
  }

 private:
  PoolPointsInterpParam param_;
};  // class PoolPointsInterpOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(PoolPointsInterpParam param, int dtype);

#if DMLC_USE_CXX11
class PoolPointsInterpProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "points"};
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
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, points]";


    // data: [batch_size, c, h, w]
    TShape dshape = in_shape->at(PoolPointsInterp::kData);
    CHECK_EQ(dshape.ndim(), 4U) << "data should be a 4D tensor";

    // points: [batch_size * num_points, 2]
    TShape pshape = in_shape->at(PoolPointsInterp::kPoints);
    CHECK_EQ(pshape.ndim(), 2U) << "points should be a 2D tensor of shape [batch_size * num_points, 2]";
    CHECK_EQ(pshape[1], 2U) << "points should be a 2D tensor of shape [batch_size * num_points, 2]";

    out_shape->clear();
    out_shape->push_back(Shape2(pshape[0], dshape[1]));
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
    PoolPointsInterpProp* pool_points_interp_sym = new PoolPointsInterpProp();
    pool_points_interp_sym->param_ = this->param_;
    return pool_points_interp_sym;
  }

  std::string TypeString() const override {
    return "PoolPointsInterp";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[PoolPointsInterp::kOut], in_data[PoolPointsInterp::kPoints]};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  PoolPointsInterpParam param_;
};  // class PoolPointsInterpProp
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_POOL_POINTS_INTERP_INL_H_
