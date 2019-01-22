#include "./pool_points_interp-inl.h"


namespace mshadow {
template<typename Dtype>
inline void PoolPointsInterpForward(const Tensor<cpu, 2, Dtype> &out_data,
                                    const Tensor<cpu, 4, Dtype> &in_data,
                                    const Tensor<cpu, 2, Dtype> &points,
                                    const float spatial_scale) {
  return;
}

template<typename Dtype>
inline void PoolPointsInterpBackward(const Tensor<cpu, 4, Dtype> &in_data_grad,
                                     const Tensor<cpu, 2, Dtype> &out_data_grad,
                                     const Tensor<cpu, 2, Dtype> &points,
                                     const float spatial_scale) {
  return;  
}
}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(PoolPointsInterpParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new PoolPointsInterpOp<cpu, DType>(param);
  });
  return op;
}

Operator *PoolPointsInterpProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape, 
                                               std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(PoolPointsInterpParam);

MXNET_REGISTER_OP_PROPERTY(PoolPointsInterp, PoolPointsInterpProp)
.describe(R"code(PoolPointsInterp)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "The input array to the pooling operator, "
                                            " a 4D Feature maps ")
.add_argument("points", "NDArray-or-Symbol", "points matrix.")
.add_arguments(PoolPointsInterpParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
