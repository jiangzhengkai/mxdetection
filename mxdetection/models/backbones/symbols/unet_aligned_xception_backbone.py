from xception_utils import *
from unet_config import *


def get_backbone_symbol(data,
                        bypass_type='norm',
                        repeat=16,
                        use_deformable_conv=True,
                        use_global_stats_stage1=True,
                        use_global_stats_stage2=False,
                        use_sync_bn=False
                        ):
    conv0_data = mx.sym.Convolution(data=data,
                                    num_filter=32,
                                    kernel=(3, 3),
                                    stride=(2, 2),
                                    pad=(1, 1),
                                    no_bias=True,
                                    lr_mult=lr_mult_stage1,
                                    workspace=512,
                                    name='aligned_xception_conv0')

    conv0_data = mx.sym.BatchNorm(data=conv0_data,
                                  fix_gamma=fix_gamma,
                                  eps=eps,
                                  momentum=bn_mom_stage1,
                                  use_global_stats=use_global_stats_stage1,
                                  sync=True if use_sync_bn else False,
                                  attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                  if mirroring_level >= 2 else {},
                                  lr_mult=lr_mult_stage1,
                                  name='aligned_xception_conv0_bn')

    conv0_data = mx.sym.Activation(data=conv0_data,
                                   act_type='relu',
                                   name='aligned_xception_conv0_relu')
    if mirroring_level >= 1:
        conv0_data._set_attr(force_mirroring='True')

    conv1_data = mx.sym.Convolution(data=conv0_data,
                                    num_filter=64,
                                    kernel=(3, 3),
                                    stride=(1, 1),
                                    pad=(1, 1),
                                    no_bias=True,
                                    lr_mult=lr_mult_stage1,
                                    workspace=512,
                                    name='aligned_xception_conv1')

    conv1_data = mx.sym.BatchNorm(data=conv1_data,
                                  fix_gamma=fix_gamma,
                                  eps=eps,
                                  momentum=bn_mom_stage1,
                                  use_global_stats=use_global_stats_stage1,
                                  sync=True if use_sync_bn else False,
                                  attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                  if mirroring_level >= 2 else {},
                                  lr_mult=lr_mult_stage1,
                                  name='aligned_xception_conv1_bn')

    conv1_data = mx.sym.Activation(data=conv1_data,
                                   act_type='relu',
                                   name='aligned_xception_conv1_relu')
    if mirroring_level >= 1:
        conv1_data._set_attr(force_mirroring='True')

    # downscale 1/2
    stem_res1 = xception_residual_reduction(data=conv1_data,
                                            in_channels=64,
                                            out_channels=128,
                                            kernel=(3, 3),
                                            stride=(2, 2),
                                            pad=(1, 1),
                                            bias=False,
                                            bypass_type=bypass_type,
                                            first_act=False,
                                            use_global_stats=use_global_stats_stage1,
                                            bn_mom=bn_mom_stage1,
                                            lr_mult=lr_mult_stage1,
                                            name='aligned_xception_stem_res1')
    # arg_shapes, out_shapes, aux_shapes = \
    #   stem_res.infer_shape(**{"data": (256, 3, 224, 224)})
    # print out_shapes

    stem_res2 = xception_residual_norm(data=stem_res1,
                                       in_channels=128,
                                       out_channels=256,
                                       kernel=(3, 3),
                                       stride=(1, 1),
                                       pad=(1, 1),
                                       bias=False,
                                       bypass_type=bypass_type,
                                       use_global_stats=use_global_stats_stage1,
                                       bn_mom=bn_mom_stage1,
                                       lr_mult=lr_mult_stage1,
                                       name='aligned_xception_stem_res2')
    # downscale 1/4
    stem_res3 = xception_residual_reduction(data=stem_res2,
                                            in_channels=256,
                                            out_channels=256,
                                            kernel=(3, 3),
                                            stride=(2, 2),
                                            pad=(1, 1),
                                            bias=False,
                                            bypass_type=bypass_type,
                                            first_act=True,
                                            use_global_stats=use_global_stats_stage1,
                                            bn_mom=bn_mom_stage1,
                                            lr_mult=lr_mult_stage1,
                                            name='aligned_xception_stem_res3')
    stem_res4 = xception_residual_norm(data=stem_res3,
                                       in_channels=256,
                                       out_channels=728,
                                       kernel=(3, 3),
                                       stride=(1, 1),
                                       pad=(1, 1),
                                       bias=False,
                                       bypass_type=bypass_type,
                                       use_global_stats=use_global_stats_stage1,
                                       bn_mom=bn_mom_stage1,
                                       lr_mult=lr_mult_stage1,
                                       deformable_conv=True if use_deformable_conv else False,
                                       name='aligned_xception_stem_res4')
    # downscale 1/8

    stem_res5 = xception_residual_reduction(data=stem_res4,
                                            in_channels=728,
                                            out_channels=728,
                                            kernel=(3, 3),
                                            stride=(2, 2),
                                            pad=(1, 1),
                                            bias=False,
                                            bypass_type=bypass_type,
                                            first_act=True,
                                            use_global_stats=use_global_stats_stage1,
                                            bn_mom=bn_mom_stage1,
                                            lr_mult=lr_mult_stage1,
                                            name='aligned_xception_stem_res5')
    stem_res = stem_res5
    for i in range(repeat):
        index = i + 6
        stem_res = xception_residual_norm(data=stem_res,
                                          in_channels=728,
                                          out_channels=728,
                                          kernel=(3, 3),
                                          stride=(1, 1),
                                          pad=(1, 1),
                                          bias=False,
                                          bypass_type=bypass_type,
                                          use_global_stats=use_global_stats_stage1,
                                          bn_mom=bn_mom_stage1,
                                          deformable_conv=True if (use_deformable_conv
                                                                   and (i == (repeat - 1))) else False,
                                          lr_mult=lr_mult_stage1,
                                          name='aligned_xception_stem_res{}'.format(index))
    # downscale 1/16

    index = repeat + 6
    stem_res18 = xception_residual_reductionbranch(data=stem_res,
                                                   in_channels=728,
                                                   out_channels=1024,
                                                   kernel=(3, 3),
                                                   stride=(2, 2),
                                                   pad=(1, 1),
                                                   bias=False,
                                                   bypass_type=bypass_type,
                                                   first_act=True,
                                                   use_global_stats=use_global_stats_stage1,
                                                   bn_mom=bn_mom_stage1,
                                                   deformable_conv=True if use_deformable_conv else False,
                                                   lr_mult=lr_mult_stage1,
                                                   name='aligned_xception_stem_res'.format(index))
    # downscale 1/32
    index = repeat + 7
    stem_res19 = xception_residual_reductionbranch(data=stem_res18,
                                                   in_channels=1024,
                                                   out_channels=2048,
                                                   kernel=(3, 3),
                                                   stride=(2, 2),
                                                   pad=(1, 1),
                                                   bias=False,
                                                   bypass_type=bypass_type,
                                                   first_act=True,
                                                   use_global_stats=use_global_stats_stage2,
                                                   bn_mom=bn_mom_stage2,
                                                   deformable_conv=True if use_deformable_conv else False,
                                                   lr_mult=lr_mult_stage2,
                                                   name='aligned_xception_stem_res{}'.format(index))

    # downscale 1/64

    # output 64
    stem_output_64 = mx.sym.Activation(data=stem_res19,
                                       act_type='relu',
                                       name='aligned_xception_output_64_relu')
    if mirroring_level >= 1:
        stem_output_64._set_attr(force_mirroring='True')

    # output 32

    stem_output_32 = mx.sym.Activation(data=stem_res18,
                                       act_type='relu',
                                       name='aligned_xception_output_32_relu')
    if mirroring_level >= 1:
        stem_output_32._set_attr(force_mirroring='True')

    # output_16

    stem_output_16 = mx.sym.Activation(data=stem_res,
                                       act_type='relu',
                                       name='aligned_xception_output_16_relu')
    if mirroring_level >= 1:
        stem_output_16._set_attr(force_mirroring='True')

    # output_8
    stem_output_8 = mx.sym.Activation(data=stem_res4,
                                      act_type='relu',
                                      name='aligned_xception_output_8_relu')
    if mirroring_level >= 1:
        stem_output_8._set_attr(force_mirroring='True')

    # output_4
    stem_output_4 = mx.sym.Activation(data=stem_res2,
                                      act_type='relu',
                                      name='aligned_xception_output_4_relu')
    if mirroring_level >= 1:
        stem_output_4._set_attr(force_mirroring='True')

    return [stem_output_4, stem_output_8, stem_output_16, stem_output_32]