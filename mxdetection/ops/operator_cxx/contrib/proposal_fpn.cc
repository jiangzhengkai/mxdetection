#include "./proposal_fpn-inl.h"


namespace mxnet {
namespace op {

template<typename xpu, typename Dtype>
class ProposalFPNOp : public Operator{
 public:
  explicit ProposalFPNOp(ProposalFPNParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    return;
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    return;
  }

 private:
  ProposalFPNParam param_;
};  // class ProposalFPNOp


template<>
Operator *CreateOp<cpu>(ProposalFPNParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ProposalFPNOp<cpu, DType>(param);
  });
  return op;
}


Operator *ProposalFPNProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}


DMLC_REGISTER_PARAMETER(ProposalFPNParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_ProposalFPN, ProposalFPNProp)
.describe("Generate region ProposalFPNs via RPN")
.add_argument("cls_score", "NDArray-or-Symbol", "Score of how likely ProposalFPN is object.")
.add_argument("bbox_pred", "NDArray-or-Symbol", "BBox Predicted deltas from anchors for ProposalFPNs")
.add_argument("im_info", "NDArray-or-Symbol", "Image size and scale.")
.add_arguments(ProposalFPNParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
