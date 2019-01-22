#include "./crop_and_resize-inl.h"
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace mshadow {
template<typename Dtype>
inline void CropAndResizeForward(const Tensor<cpu, 4, Dtype> &out,
                                 const Tensor<cpu, 4, Dtype> &data,
                                 const Tensor<cpu, 2, Dtype> &bbox,
                                 const float spatial_scale_) {
  // NOT_IMPLEMENTED;
  return;
}

template<typename Dtype>
inline void CropAndResizeBackward(const Tensor<cpu, 4, Dtype> &in_grad,
                                  const Tensor<cpu, 4, Dtype> &out_grad,
                                  const Tensor<cpu, 2, Dtype> &bbox,
                                  const float spatial_scale_) {
  // NOT_IMPLEMENTED;
  return;
}
}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(CropAndResizeParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new CropAndResizeOp<cpu, DType>(param);
  });
  return op;
}

Operator *CropAndResizeProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                              std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(CropAndResizeParam);

MXNET_REGISTER_OP_PROPERTY(CropAndResize, CropAndResizeProp)
.describe(R"code(Performs region of interest(ROI) pooling on the input array.

ROI pooling is a variant of a max pooling layer, in which the output size is fixed and
region of interest is a parameter. Its purpose is to perform max pooling on the inputs
of non-uniform sizes to obtain fixed-size feature maps. ROI pooling is a neural-net
layer mostly used in training a `Fast R-CNN` network for object detection.

This operator takes a 4D feature map as an input array and region proposals as `rois`,
then it pools over sub-regions of input and produces a fixed-sized output array
regardless of the ROI size.

To crop the feature map accordingly, you can resize the bounding box coordinates
by changing the parameters `rois` and `spatial_scale`.

The cropped feature maps are pooled by standard max pooling operation to a fixed size output
indicated by a `pooled_size` parameter. batch_size will change to the number of region
bounding boxes after `ROIPooling`.

The size of each region of interest doesn't have to be perfectly divisible by
the number of pooling sections(`pooled_size`).

Example::

  x = [[[[  0.,   1.,   2.,   3.,   4.,   5.],
         [  6.,   7.,   8.,   9.,  10.,  11.],
         [ 12.,  13.,  14.,  15.,  16.,  17.],
         [ 18.,  19.,  20.,  21.,  22.,  23.],
         [ 24.,  25.,  26.,  27.,  28.,  29.],
         [ 30.,  31.,  32.,  33.,  34.,  35.],
         [ 36.,  37.,  38.,  39.,  40.,  41.],
         [ 42.,  43.,  44.,  45.,  46.,  47.]]]]

  // region of interest i.e. bounding box coordinates.
  y = [[0,0,0,4,4]]

  // returns array of shape (2,2) according to the given roi with max pooling.
  ROIPooling(x, y, (2,2), 1.0) = [[[[ 14.,  16.],
                                    [ 26.,  28.]]]]

  // region of interest is changed due to the change in `spacial_scale` parameter.
  ROIPooling(x, y, (2,2), 0.7) = [[[[  7.,   9.],
                                    [ 19.,  21.]]]]

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "The input array to the pooling operator, "
                                            " a 4D Feature maps ")
.add_argument("rois", "NDArray-or-Symbol", "Bounding box coordinates, a 2D array of "
"[[batch_index, x1, y1, x2, y2]], where (x1, y1) and (x2, y2) are top left and bottom right "
"corners of designated region of interest. `batch_index` indicates the index of corresponding "
"image in the input array")
.add_arguments(CropAndResizeParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
