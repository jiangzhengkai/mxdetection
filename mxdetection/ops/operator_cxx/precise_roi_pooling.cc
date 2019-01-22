#include "./precise_roi_pooling-inl.h"

namespace mshadow {
template<typename Dtype>
inline void PreciseROIPoolForward(const Tensor<cpu, 4, Dtype> &out,
                                  const Tensor<cpu, 4, Dtype> &data,
                                  const Tensor<cpu, 2, Dtype> &bbox,
                                  const float spatial_scale_) {
  return;
}

template<typename Dtype>
inline void PreciseROIPoolBackward(const Tensor<cpu, 4, Dtype> &in_grad,
                                   const Tensor<cpu, 4, Dtype> &out_grad,
                                   const Tensor<cpu, 2, Dtype> &bbox,
                                   const float spatial_scale_) {
  return;
}
}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(PreciseROIPoolingParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new PreciseROIPoolingOp<cpu, DType>(param);
  });
  return op;
}

Operator *PreciseROIPoolingProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(PreciseROIPoolingParam);

MXNET_REGISTER_OP_PROPERTY(PreciseROIPooling, PreciseROIPoolingProp)
.describe(R"code(ROIConvolution)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "The input array to the pooling operator, "
                                            " a 4D Feature maps ")
.add_argument("rois", "NDArray-or-Symbol", "Bounding box coordinates, a 2D array of "
"[[batch_index, x1, y1, x2, y2]], where (x1, y1) and (x2, y2) are top left and bottom right "
"corners of designated region of interest. `batch_index` indicates the index of corresponding "
"image in the input array")
.add_arguments(PreciseROIPoolingParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
