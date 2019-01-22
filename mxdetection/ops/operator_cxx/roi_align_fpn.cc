#include "./roi_align_fpn-inl.h"

namespace mshadow {


template<typename Dtype>
inline void ROIAlignFPNForward(const Tensor<cpu, 4, Dtype> &out,
                               const Tensor<cpu, 4, Dtype> &data_res2,
                               const Tensor<cpu, 4, Dtype> &data_res3,
                               const Tensor<cpu, 4, Dtype> &data_res4,
                               const Tensor<cpu, 4, Dtype> &data_res5, 
                               const Tensor<cpu, 2, Dtype> &bbox,
                               const int sample_per_part) {
  return;
}


template<typename Dtype>
inline void ROIAlignFPNBackward(const Tensor<cpu, 4, Dtype> &in_grad_res2,
                                const Tensor<cpu, 4, Dtype> &in_grad_res3,
                                const Tensor<cpu, 4, Dtype> &in_grad_res4,
                                const Tensor<cpu, 4, Dtype> &in_grad_res5, 
                                const Tensor<cpu, 4, Dtype> &out_grad,
                                const Tensor<cpu, 2, Dtype> &bbox,
                                const int sample_per_part) {
  return;
}
}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(ROIAlignFPNParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ROIAlignFPNOp<cpu, DType>(param);
  });
  return op;
}

Operator *ROIAlignFPNProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                         std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(ROIAlignFPNParam);

MXNET_REGISTER_OP_PROPERTY(ROIAlignFPN, ROIAlignFPNProp)
.describe(R"code(ROIAlignFPN)code" ADD_FILELINE)
.add_argument("data_res", "NDArray-or-Symbol", "The input array to the pooling operator, a 4D Feature maps ")
.add_argument("rois", "NDArray-or-Symbol", "Bounding box coordinates, a 2D array of "
"[[batch_index, x1, y1, x2, y2]], where (x1, y1) and (x2, y2) are top left and bottom right "
"corners of designated region of interest. `batch_index` indicates the index of corresponding "
"image in the input array")
.add_arguments(ROIAlignFPNParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
