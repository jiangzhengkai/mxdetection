import mxnet as mx
from unet_config import *
def Separable_conv2d(data,
                     in_channels,
                     out_channels,
                     kernel,
                     pad,
                     stride=(1,1),
                     bias=False,
                     bn_out=False,
                     act_out=False,
                     name=None,
                     use_global_stats=False,
                     bn_mom=0.9,
                     lr_mult=1,
                     workspace=512):
	#depthwise
    dw_out = mx.sym.Convolution(data=data,
                                num_filter=in_channels,
                                kernel=kernel,
                                pad=pad,
                                stride=stride,
                                no_bias=False if bias else True,
                                num_group=in_channels,
                                workspace=workspace,
                                lr_mult=lr_mult,
                                name=name +'_conv2d_depthwise')
    if bn_out:
        dw_out = mx.sym.BatchNorm(data=dw_out,
                                  fix_gamma=fix_gamma,
                                  eps=eps,
                                  momentum=bn_mom,
                                  use_global_stats=use_global_stats,
                                  sync=True if use_sync_bn else False,
                                  attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                  if mirroring_level >= 2 else {},
                                  lr_mult=lr_mult,
                                  name=name+'_conv2d_depthwise_bn')
    if act_out:
        dw_out = mx.sym.Activation(data=dw_out,
                                   act_type='relu',
                                   name=name+'_conv2d_depthwise_relu')
        if mirroring_level >= 1:
            dw_out._set_attr(force_mirroring='True')
    #pointwise
    pw_out = mx.sym.Convolution(data=dw_out,
                                num_filter=out_channels,
                                kernel=(1, 1),
                                stride=(1, 1),
                                pad=(0, 0),
                                num_group=1,
                                no_bias=False if bias else True,
                                lr_mult=lr_mult,
                                workspace=workspace,
                                name=name+'_conv2d_pointwise')
    return pw_out

def Deformable_separable_conv2d(data,
                                in_channels,
                                out_channels,
                                kernel,
                                pad,
                                stride=(1,1),
                                bias=False,
                                bn_out=False,
                                act_out=False,
                                name=None,
                                use_global_stats=False,
                                bn_mom=0.9,
                                num_deformable_group=1,
                                lr_mult=1,
                                workspace=512):

    #offset
    init_zero=mx.init.Zero()
    conv_offset=mx.sym.Convolution(data=data,
                                   num_filter=2*kernel[0]*kernel[1]*num_deformable_group,
                                   kernel=kernel,
                                   stride=stride,
                                   pad=pad,
                                   num_group=1,
                                   no_bias=False,
                                   attr={'__init__':init_zero.dumps()},
                                   lr_mult=lr_mult_offset,
                                   name=name+'_conv2d_depthwise_offset')
	#depthwise
    dw_out = mx.contrib.sym.DeformableConvolution(data=data,
                                                  offset=conv_offset,
                                                  num_deformable_group=num_deformable_group,
                                                  num_filter=in_channels,
                                                  kernel=kernel,
                                                  pad=pad,
                                                  stride=stride,
                                                  no_bias=False if bias else True,
                                                  num_group=in_channels,
                                                  lr_mult=lr_mult,
                                                  workspace=workspace,
                                                  name=name +'_conv2d_depthwise')
    if bn_out:
        dw_out = mx.sym.BatchNorm(data=dw_out,
                                  fix_gamma=fix_gamma,
                                  eps=eps,
                                  momentum=bn_mom,
                                  use_global_stats=use_global_stats,
                                  sync=True if use_sync_bn else False,
                                  attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                  if mirroring_level >= 2 else {},
                                  lr_mult=lr_mult,
                                  name=name + '_conv2d_depthwise_bn')


    if act_out:
        dw_out = mx.sym.Activation(data=dw_out,
                                   act_type='relu',
                                   name=name+'_conv2d_depthwise_relu')
        if mirroring_level >= 1:
            dw_out._set_attr(force_mirroring='True')
    #pointwise
    pw_out = mx.sym.Convolution(data=dw_out,
                                num_filter=out_channels,
                                kernel=(1, 1),
                                stride=(1, 1),
                                pad=(0, 0),
                                num_group=1,
                                no_bias=False if bias else True,
                                workspace=workspace,
                                lr_mult=lr_mult,
                                name=name+'_conv2d_pointwise')
    return pw_out

def xception_residual_norm(data,
                           in_channels,
                           out_channels,
                           kernel=(3,3),
                           pad=(1,1),
                           stride=(1,1),
                           bias=False,
                           bypass_type='norm', #'bypass_type: norm or separable'
                           name=None,
                           use_global_stats=False,
                           bn_mom=0.9,
                           deformable_conv=False,
                           num_deformable_group=1,
                           lr_mult=1,
                           workspace=512):

    assert stride[0]==stride[1]
    assert stride[0]==1

    sep1 = mx.sym.Activation(data=data,
                             act_type='relu',
                             name=name + '_sep1_relu')
    if mirroring_level >= 1:
        sep1._set_attr(force_mirroring='True')
    if deformable_conv:
        sep1 = Deformable_separable_conv2d(data=sep1,
                                           in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel=kernel,
                                           pad=pad,
                                           stride=stride,
                                           bias=bias,
                                           bn_out=True,
                                           act_out=False,
                                           use_global_stats=use_global_stats,
                                           bn_mom=bn_mom,
                                           num_deformable_group=num_deformable_group,
                                           lr_mult=lr_mult,
                                           name=name + '_sep1_conv')
    else:
        sep1 = Separable_conv2d(data=sep1,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel=kernel,
                                pad=pad,
                                stride=stride,
                                bias=bias,
                                bn_out=True,
                                act_out=False,
                                use_global_stats=use_global_stats,
                                bn_mom=bn_mom,
                                lr_mult=lr_mult,
                                name=name + '_sep1_conv')


    sep1 = mx.sym.BatchNorm(data=sep1,
                            fix_gamma=fix_gamma,
                            eps=eps,
                            momentum=bn_mom,
                            use_global_stats=use_global_stats,
                            sync=True if use_sync_bn else False,
                            attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                            if mirroring_level >= 2 else {},
                            lr_mult=lr_mult,
                            name=name + '_sep1_bn')

    sep2 = mx.sym.Activation(data=sep1,
                             act_type='relu',
                             name=name + '_sep2_relu')
    if mirroring_level >= 1:
        sep2._set_attr(force_mirroring='True')

    if deformable_conv:
        sep2 = Deformable_separable_conv2d(data=sep2,
                                           in_channels=out_channels,
                                           out_channels=out_channels,
                                           kernel=kernel,
                                           pad=pad,
                                           stride=stride,
                                           bias=bias,
                                           bn_out=True,
                                           act_out=False,
                                           use_global_stats=use_global_stats,
                                           bn_mom=bn_mom,
                                           num_deformable_group=num_deformable_group,
                                           lr_mult=lr_mult,
                                           name=name + '_sep2_conv')
    else:
        sep2 = Separable_conv2d(data=sep2,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel=kernel,
                                pad=pad,
                                stride=stride,
                                bias=bias,
                                bn_out=True,
                                act_out=False,
                                use_global_stats=use_global_stats,
                                bn_mom=bn_mom,
                                lr_mult=lr_mult,
                                name=name + '_sep2_conv')


    sep2 = mx.sym.BatchNorm(data=sep2,
                            fix_gamma=fix_gamma,
                            eps=eps,
                            momentum=bn_mom,
                            use_global_stats=use_global_stats,
                            sync=True if use_sync_bn else False,
                            attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                            if mirroring_level >= 2 else {},
                            lr_mult=lr_mult,
                            name=name + '_sep2_bn')

    sep3 = mx.sym.Activation(data=sep2,
                             act_type='relu',
                             name=name + '_sep3_relu')
    if mirroring_level >= 1:
        sep3._set_attr(force_mirroring='True')

    if deformable_conv:
        sep3 = Deformable_separable_conv2d(data=sep3,
                                           in_channels=out_channels,
                                           out_channels=out_channels,
                                           kernel=kernel,
                                           pad=pad,
                                           stride=stride,
                                           bias=bias,
                                           bn_out=True,
                                           act_out=False,
                                           use_global_stats=use_global_stats,
                                           bn_mom=bn_mom,
                                           num_deformable_group=num_deformable_group,
                                           lr_mult=lr_mult,
                                           name=name + '_sep3_conv')
    else:

        sep3 = Separable_conv2d(data=sep3,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel=kernel,
                                pad=pad,
                                stride=stride,
                                bias=bias,
                                bn_out=True,
                                act_out=False,
                                use_global_stats=use_global_stats,
                                bn_mom=bn_mom,
                                lr_mult=lr_mult,
                                name=name + '_sep3_conv')



    sep3 = mx.sym.BatchNorm(data=sep3,
                            fix_gamma=fix_gamma,
                            eps=eps,
                            momentum=bn_mom,
                            use_global_stats=use_global_stats,
                            sync=True if use_sync_bn else False,
                            attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                            if mirroring_level >= 2 else {},
                            lr_mult=lr_mult,
                            name=name + '_sep3_bn')


    if  in_channels==out_channels:
        short_cut = data
    else:
        if bypass_type=='norm':
            short_cut = mx.sym.Convolution(data=data,
                                           num_filter=out_channels,
                                           kernel=(1, 1),
                                           stride=(1, 1),
                                           pad=(0, 0),
                                           num_group=1,
                                           no_bias=False if bias else True,
                                           lr_mult=lr_mult,
                                           workspace=workspace,
                                           name=name + '_conv2d_bypass')


            short_cut = mx.sym.BatchNorm(data=short_cut,
                                         fix_gamma=fix_gamma,
                                         eps=eps,
                                         momentum=bn_mom,
                                         use_global_stats=use_global_stats,
                                         sync=True if use_sync_bn else False,
                                         attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                         if mirroring_level >= 2 else {},
                                         lr_mult=lr_mult,
                                         name=name + '_bypass_bn')

        elif bypass_type=='separable':
            short_cut = Separable_conv2d(data=data,
                                         in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel=kernel,
                                         pad=pad,
                                         stride=stride,
                                         bias=bias,
                                         bn_out=True,
                                         act_out=False,
                                         use_global_stats=use_global_stats,
                                         bn_mom=bn_mom,
                                         lr_mult=lr_mult,
                                         name=name + '_bypass_separable')


            short_cut = mx.sym.BatchNorm(data=short_cut,
                                        fix_gamma=fix_gamma,
                                        eps=eps,
                                        momentum=bn_mom,
                                        use_global_stats=use_global_stats,
                                        sync=True if use_sync_bn else False,
                                        attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                        if mirroring_level >= 2 else {},
                                        lr_mult=lr_mult,
                                        name=name + '_bypass_bn')



        else:
            raise ValueError('no suppport bypass type:{}'
                             'for xception residual norm cell'.format(bypass_type))
    out = sep3+short_cut
    return out


def xception_residual_reduction(data,
                                in_channels,
                                out_channels,
                                kernel=(3,3),
                                pad=(1,1),
                                stride=(2,2),
                                bias=False,
                                bypass_type='norm', #'bypass_type: norm or separable'
                                first_act=True,
                                name=None,
                                use_global_stats=False,
                                bn_mom=0.9,
                                deformable_conv=False,
                                num_deformable_group=1,
                                lr_mult=1,
                                workspace=512):

    assert stride[0]==stride[1]
    assert stride[0]==2

    if first_act:
        sep1 = mx.sym.Activation(data=data,
                                 act_type='relu',
                                 name=name + '_sep1_relu')
        if mirroring_level >= 1:
            sep1._set_attr(force_mirroring='True')
    else:
        sep1 = data
    if deformable_conv:
        sep1 = Deformable_separable_conv2d(data=sep1,
                                           in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel=kernel,
                                           pad=pad,
                                           stride=(1, 1),
                                           bias=bias,
                                           bn_out=True,
                                           act_out=False,
                                           use_global_stats=use_global_stats,
                                           bn_mom=bn_mom,
                                           num_deformable_group=num_deformable_group,
                                           lr_mult=lr_mult,
                                           name=name + '_sep1_conv')
    else:

        sep1 = Separable_conv2d(data=sep1,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel=kernel,
                                pad=pad,
                                stride=(1,1),
                                bias=bias,
                                bn_out=True,
                                act_out=False,
                                use_global_stats=use_global_stats,
                                bn_mom=bn_mom,
                                lr_mult=lr_mult,
                                name=name + '_sep1_conv')



    sep1 = mx.sym.BatchNorm(data=sep1,
                            fix_gamma=fix_gamma,
                            eps=eps,
                            momentum=bn_mom,
                            use_global_stats=use_global_stats,
                            sync=True if use_sync_bn else False,
                            attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                            if mirroring_level >= 2 else {},
                            lr_mult=lr_mult,
                            name=name + '_sep1_bn')

    sep2 = mx.sym.Activation(data=sep1,
                             act_type='relu',
                             name=name + '_sep2_relu')
    if mirroring_level >= 1:
        sep2._set_attr(force_mirroring='True')
    if deformable_conv:
        sep2 = Deformable_separable_conv2d(data=sep2,
                                           in_channels=out_channels,
                                           out_channels=out_channels,
                                           kernel=kernel,
                                           pad=pad,
                                           stride=(1, 1),
                                           bias=bias,
                                           bn_out=True,
                                           act_out=False,
                                           use_global_stats=use_global_stats,
                                           bn_mom=bn_mom,
                                           num_deformable_group=num_deformable_group,
                                           lr_mult=lr_mult,
                                           name=name + '_sep2_conv')
    else:
        sep2 = Separable_conv2d(data=sep2,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel=kernel,
                                pad=pad,
                                stride=(1,1),
                                bias=bias,
                                bn_out=True,
                                act_out=False,
                                use_global_stats=use_global_stats,
                                bn_mom=bn_mom,
                                lr_mult=lr_mult,
                                name=name + '_sep2_conv')


    sep2 = mx.sym.BatchNorm(data=sep2,
                            fix_gamma=fix_gamma,
                            eps=eps,
                            momentum=bn_mom,
                            use_global_stats=use_global_stats,
                            sync=True if use_sync_bn else False,
                            attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                            if mirroring_level >= 2 else {},
                            lr_mult=lr_mult,
                            name=name + '_sep2_bn')


    sep3 = mx.sym.Pooling(data=sep2,
                          kernel=(3, 3),
                          stride=stride,
                          pad=(1, 1),
                          pool_type="max",
                          name=name + '_sep3_max_pooling')
    if bypass_type=='norm':
        short_cut = mx.sym.Convolution(data=data,
                                       num_filter=out_channels,
                                       kernel=(1, 1),
                                       stride=stride,
                                       pad=(0, 0),
                                       num_group=1,
                                       no_bias=False if bias else True,
                                       lr_mult=lr_mult,
                                       workspace=workspace,
                                       name=name + '_conv2d_bypass')


        short_cut = mx.sym.BatchNorm(data=short_cut,
                                    fix_gamma=fix_gamma,
                                    eps=eps,
                                    momentum=bn_mom,
                                    use_global_stats=use_global_stats,
                                    sync=True if use_sync_bn else False,
                                    attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                    if mirroring_level >= 2 else {},
                                    lr_mult=lr_mult,
                                    name=name + '_bypass_bn')


    elif bypass_type=='separable':
        short_cut = Separable_conv2d(data=data,
                                     in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel=kernel,
                                     pad=pad,
                                     stride=stride,
                                     bias=bias,
                                     bn_out=True,
                                     act_out=False,
                                     use_global_stats=use_global_stats,
                                     bn_mom=bn_mom,
                                     lr_mult=lr_mult,
                                     name=name + '_bypass_separable')


        short_cut = mx.sym.BatchNorm(data=short_cut,
                                    fix_gamma=fix_gamma,
                                    eps=eps,
                                    momentum=bn_mom,
                                    use_global_stats=use_global_stats,
                                    sync=True if use_sync_bn else False,
                                    attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                    if mirroring_level >= 2 else {},
                                    lr_mult=lr_mult,
                                    name=name + '_bypass_bn')


    else:
        raise ValueError('no suppport bypass type:{}for xception residual norm cell'.format(bypass_type))
    out = sep3+short_cut
    return out

def xception_residual_reductionbranch(data,
                                      in_channels,
                                      out_channels,
                                      kernel=(3,3),
                                      pad=(1,1),
                                      stride=(2,2),
                                      bias=False,
                                      bypass_type='norm', #'bypass_type: norm or separable'
                                      first_act=True,
                                      name=None,
                                      use_global_stats=False,
                                      bn_mom=0.9,
                                      deformable_conv=False,
                                      num_deformable_group=1,
                                      lr_mult=1,
                                      workspace=512):

    assert stride[0]==stride[1]
    assert stride[0]==2

    if first_act:
        sep1 = mx.sym.Activation(data=data,
                                 act_type='relu',
                                 name=name + '_sep1_relu')
        if mirroring_level >= 1:
            sep1._set_attr(force_mirroring='True')
    else:
        sep1 = data
    if deformable_conv:
        sep1 = Deformable_separable_conv2d(data=sep1,
                                           in_channels=in_channels,
                                           out_channels=in_channels,
                                           kernel=kernel,
                                           pad=pad,
                                           stride=(1, 1),
                                           bias=bias,
                                           bn_out=True,
                                           act_out=False,
                                           use_global_stats=use_global_stats,
                                           bn_mom=bn_mom,
                                           num_deformable_group=num_deformable_group,
                                           lr_mult=lr_mult,
                                           name=name + '_sep1_conv')
    else:
        sep1 = Separable_conv2d(data=sep1,
                                in_channels=in_channels,
                                out_channels=in_channels,
                                kernel=kernel,
                                pad=pad,
                                stride=(1,1),
                                bias=bias,
                                bn_out=True,
                                act_out=False,
                                use_global_stats=use_global_stats,
                                bn_mom=bn_mom,
                                lr_mult=lr_mult,
                                name=name + '_sep1_conv')


    sep1 = mx.sym.BatchNorm(data=sep1,
                            fix_gamma=fix_gamma,
                            eps=eps,
                            momentum=bn_mom,
                            use_global_stats=use_global_stats,
                            sync=True if use_sync_bn else False,
                            attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                            if mirroring_level >= 2 else {},
                            lr_mult=lr_mult,
                             name=name + '_sep1_bn')



    sep2 = mx.sym.Activation(data=sep1,
                             act_type='relu',
                             name=name + '_sep2_relu')
    if mirroring_level >= 1:
        sep2._set_attr(force_mirroring='True')
    if deformable_conv:
        sep2 = Deformable_separable_conv2d(data=sep2,
                                           in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel=kernel,
                                           pad=pad,
                                           stride=(1, 1),
                                           bias=bias,
                                           bn_out=True,
                                           act_out=False,
                                           use_global_stats=use_global_stats,
                                           bn_mom=bn_mom,
                                           num_deformable_group=num_deformable_group,
                                           lr_mult=lr_mult,
                                           name=name + '_sep2_conv')
    else:
        sep2 = Separable_conv2d(data=sep2,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel=kernel,
                                pad=pad,
                                stride=(1,1),
                                bias=bias,
                                bn_out=True,
                                act_out=False,
                                use_global_stats=use_global_stats,
                                bn_mom=bn_mom,
                                lr_mult=lr_mult,
                                name=name + '_sep2_conv')


    sep2 = mx.sym.BatchNorm(data=sep2,
                            fix_gamma=fix_gamma,
                            eps=eps,
                            momentum=bn_mom,
                            use_global_stats=use_global_stats,
                            sync=True if use_sync_bn else False,
                            attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                            if mirroring_level >= 2 else {},
                            lr_mult=lr_mult,
                            name=name + '_sep2_bn')



    sep3 = mx.sym.Pooling(data=sep2,
                          kernel=(3, 3),
                          stride=stride,
                          pad=(1, 1),
                          pool_type="max",
                          name=name + '_sep3_max_pooling')
    if bypass_type=='norm':
        short_cut = mx.sym.Convolution(data=data,
                                       num_filter=out_channels,
                                       kernel=(1, 1),
                                       stride=stride,
                                       pad=(0, 0),
                                       num_group=1,
                                       no_bias=False if bias else True,
                                       lr_mult=lr_mult,
                                       workspace=workspace,
                                       name=name + '_conv2d_bypass')


        short_cut = mx.sym.BatchNorm(data=short_cut,
                                    fix_gamma=fix_gamma,
                                    eps=eps,
                                    momentum=bn_mom,
                                    use_global_stats=use_global_stats,
                                    sync=True if use_sync_bn else False,
                                    attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                    if mirroring_level >= 2 else {},
                                    lr_mult=lr_mult,
                                    name=name + '_bypass_bn')



    elif bypass_type=='separable':
        short_cut = Separable_conv2d(data=data,
                                     in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel=kernel,
                                     pad=pad,
                                     stride=stride,
                                     bias=bias,
                                     bn_out=True,
                                     act_out=False,
                                     use_global_stats=use_global_stats,
                                     bn_mom=bn_mom,
                                     lr_mult=lr_mult,
                                     name=name + '_bypass_separable')


        short_cut = mx.sym.BatchNorm(data=short_cut,
                                    fix_gamma=fix_gamma,
                                    eps=eps,
                                    momentum=bn_mom,
                                    use_global_stats=use_global_stats,
                                    sync=True if use_sync_bn else False,
                                    attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                    if mirroring_level >= 2 else {},
                                    lr_mult=lr_mult,
                                    name=name + '_bypass_bn')

    else:
        raise ValueError('no suppport bypass type:{} for xception residual norm cell'.format(bypass_type))
    out = sep3+short_cut
    return out