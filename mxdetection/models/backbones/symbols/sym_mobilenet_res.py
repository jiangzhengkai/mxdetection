import logging
import mxnet as mx

group_base = 1
dw_act_out = True
kernel_size = (3, 3)
pad_size = (1, 1)
mirroring_level = 0
bn_mom = 0.9
workspace = 512


def add_weight_bias(name, no_bias, **kwargs):
    if 'attr' in kwargs:
        cfg.param_attr = kwargs['attr']
    if 'weight' not in kwargs:
        kwargs['weight'] = get_variable('{}_weight'.format(name))
    if no_bias is False and 'bias' not in kwargs:
        kwargs['bias'] = get_variable('{}_bias'.format(name))
    cfg.param_attr = None
    return kwargs


def _deformable_conv(data, offset, name,
                     num_filter, num_deformable_group,
                     kernel=1, stride=1, pad=-1, dilate=1,
                     num_group=1, no_bias=False, **kwargs):
    if kernel == 1:
        dilate = 1
    if pad < 0:
        assert kernel % 2 == 1, 'Specify pad for an even kernel size'
        pad = ((kernel - 1) * dilate + 1) // 2
    kwargs = add_weight_bias(name, no_bias, **kwargs)
    return mx.contrib.symbol.DeformableConvolution(data=data, offset=offset, name=name,
                                                   num_filter=num_filter,
                                                   num_deformable_group=num_deformable_group,
                                                   kernel=(kernel, kernel),
                                                   stride=(stride, stride),
                                                   pad=(pad, pad),
                                                   dilate=(dilate, dilate),
                                                   num_group=num_group,
                                                   workspace=cfg.workspace,
                                                   no_bias=no_bias, **kwargs)


def Conv(data, num_filter, kernel, stride, pad, depth_mult=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=depth_mult,
                              stride=stride, pad=pad, no_bias=True, workspace=workspace, use_global_stats=True,
                              name='%s%s' % (name, suffix))
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5,
                          momentum=bn_mom, name='%s%s_bn' % (name, suffix),
                          use_global_stats=use_global_stats)
    act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' % (name, suffix))
    return act


def separable_conv_order(data, n_in_ch, n_out_ch, kernel, stride, pad, depth_mult=1,
                         act_out_first=True, bn_out_first=True, act_out_second=True,
                         bn_out_second=True, name=None, suffix='', bn_mom=0.9,
                         workspace=512, num_deformable_group=0, use_global_stats=True):
    if num_deformable_group > 0:
        assert kernel[0] == kernel[1]
        assert stride[0] == stride[1]
        assert pad[0] == pad[1]
        dw_out = deformable_conv(data=data, num_filter=n_in_ch, kernel=kernel[0],
                                 num_group=depth_mult, stride=stride[0], pad=pad[0],
                                 no_bias=True, num_deformable_group=num_deformable_group,
                                 name='%s%s_dw' % (name, suffix))
    else:
        dw_out = mx.sym.Convolution(data=data, num_filter=n_in_ch, kernel=kernel,
                                    num_group=depth_mult, stride=stride, pad=pad,
                                    no_bias=True, workspace=workspace,
                                    name='%s%s_dw' % (name, suffix))
    if bn_out_first:
        dw_out = mx.sym.BatchNorm(data=dw_out, fix_gamma=False, eps=2e-5,
                                  momentum=bn_mom, name='%s%s_dw_bn' % (name, suffix),
                                  use_global_stats=use_global_stats)
    if act_out_first:
        dw_out = mx.sym.Activation(data=dw_out, act_type='relu',
                                   name='%s%s_dw_relu' % (name, suffix))
    # pointwise
    pw_out = mx.sym.Convolution(data=dw_out, num_filter=n_out_ch, kernel=(1, 1),
                                stride=(1, 1), pad=(0, 0), num_group=1, no_bias=True,
                                workspace=workspace, name='%s%s_pw' % (name, suffix))

    if bn_out_second:
        pw_out = mx.sym.BatchNorm(data=pw_out, fix_gamma=False, eps=2e-5,
                                  momentum=bn_mom, name='%s%s_pw_bn' % (name, suffix),
                                  use_global_stats=use_global_stats)
    if act_out_second:
        pw_out = mx.sym.Activation(data=pw_out, act_type='relu', name='%s%s_pw_relu' % (name, suffix))
    return pw_out


def residual_unit_order_new(data, n_in_ch, n_out_ch1, n_out_ch2, dim_match, stride=(1, 1),
                            act_out_first=True, bn_out_first=True, name=None, suffix='',
                            num_deformable_group=0, use_global_stats=True):
    if dim_match:
        sep_out1 = separable_conv_order(data, n_in_ch=n_in_ch, n_out_ch=n_out_ch1, kernel=kernel_size,
                                        stride=stride, pad=pad_size,
                                        depth_mult=n_in_ch // group_base, act_out_first=dw_act_out,
                                        bn_out_first=bn_out_first,
                                        act_out_second=True, bn_out_second=True,
                                        name=name + '_sp1', suffix=suffix, bn_mom=bn_mom, workspace=workspace,
                                        num_deformable_group=num_deformable_group, use_global_stats=True)
    elif n_in_ch == n_out_ch1:
        sep_out1 = separable_conv_order(data, n_in_ch=n_in_ch, n_out_ch=n_out_ch1, kernel=kernel_size,
                                        stride=stride, pad=pad_size,
                                        depth_mult=n_in_ch // group_base, act_out_first=dw_act_out,
                                        bn_out_first=bn_out_first,
                                        act_out_second=True, bn_out_second=True,
                                        name=name + '_sp1', suffix=suffix, bn_mom=bn_mom, workspace=workspace,
                                        num_deformable_group=num_deformable_group, use_global_stats=True)
    else:
        sep_out1_sp1 = separable_conv_order(data, n_in_ch=n_in_ch, n_out_ch=n_out_ch1, kernel=kernel_size,
                                            stride=stride, pad=pad_size,
                                            depth_mult=n_in_ch // group_base, act_out_first=dw_act_out,
                                            bn_out_first=bn_out_first,
                                            act_out_second=True, bn_out_second=True,
                                            name=name + '_sp1_split1', suffix=suffix, bn_mom=bn_mom,
                                            workspace=workspace, num_deformable_group=num_deformable_group,
                                            use_global_stats=True)
        sep_out1_sp2 = separable_conv_order(data, n_in_ch=n_in_ch, n_out_ch=n_out_ch1, kernel=kernel_size,
                                            stride=stride, pad=pad_size,
                                            depth_mult=n_in_ch // group_base, act_out_first=dw_act_out,
                                            bn_out_first=bn_out_first,
                                            act_out_second=True, bn_out_second=True,
                                            name=name + '_sp1_split2', suffix=suffix, bn_mom=bn_mom,
                                            workspace=workspace, num_deformable_group=num_deformable_group,
                                            use_global_stats=True)
        sep_out1 = sep_out1_sp1 + sep_out1_sp2
    sep_out2 = separable_conv_order(sep_out1, n_in_ch=n_out_ch1, n_out_ch=n_out_ch2, kernel=kernel_size,
                                    stride=(1, 1), pad=pad_size, depth_mult=n_out_ch1 // group_base,
                                    act_out_first=dw_act_out,
                                    bn_out_first=bn_out_first, act_out_second=False, bn_out_second=True,
                                    name=name + '_sp2', suffix=suffix,
                                    bn_mom=bn_mom, workspace=workspace, num_deformable_group=num_deformable_group,
                                    use_global_stats=True)

    if dim_match:
        short_cut = data
    elif (stride[0] > 1) and (stride[1] > 1):
        short_cut = separable_conv_order(data=data, n_in_ch=n_in_ch, n_out_ch=n_out_ch2, kernel=kernel_size,
                                         stride=stride, pad=pad_size, depth_mult=n_in_ch // group_base,
                                         act_out_first=dw_act_out,
                                         bn_out_first=bn_out_first, act_out_second=False, bn_out_second=True,
                                         name=name + '_shortcut', suffix=suffix,
                                         bn_mom=bn_mom, workspace=workspace, use_global_stats=True)
    else:
        short_cut = separable_conv_order(data=data, n_in_ch=n_in_ch, n_out_ch=n_out_ch2, kernel=kernel_size,
                                         stride=(1, 1), pad=pad_size, depth_mult=n_in_ch // group_base,
                                         act_out_first=dw_act_out,
                                         bn_out_first=bn_out_first, act_out_second=False, bn_out_second=True,
                                         name=name + '_shortcut', suffix=suffix,
                                         bn_mom=bn_mom, workspace=workspace, use_global_stats=True)

    res_data = sep_out2 + short_cut
    res_act = mx.sym.Activation(data=res_data, act_type='relu', name='%s%s_res_relu' % (name, suffix))
    return res_act


def get_symbol(alpha=0.5, inv_resolution=32, num_deformable_group=0, **kwargs):
    assert inv_resolution == 16 or inv_resolution == 32
    input_dict = kwargs['input_dict'] if 'input_dict' in kwargs else None
    data = kwargs['data'] if 'data' in kwargs else mx.sym.Variable(name='data')
    use_global_stats = kwargs['use_global_stats'] if 'use_global_stats' in kwargs else True
    in_layer_list = []
    act_out_first = True
    bn_out_first = True

    conv1 = Conv(data, num_filter=int(64 * alpha) if alpha == 0.25 else int(32 * alpha), kernel=kernel_size,
                 stride=(2, 2), pad=pad_size, name="conv1", use_global_stats=use_global_stats)
    if input_dict is not None:
        logging.info('conv1: {}'.format(conv1.infer_shape(**input_dict)[1]))

    # res2
    conv2_res = residual_unit_order_new(conv1, n_in_ch=int(64 * alpha) if alpha == 0.25 else int(32 * alpha),
                                        n_out_ch1=int(64 * alpha),
                                        n_out_ch2=int(128 * alpha), dim_match=False,
                                        stride=(2, 2), act_out_first=act_out_first,
                                        bn_out_first=bn_out_first, name='conv2_res', use_global_stats=use_global_stats)
    in_layer_list.append(conv2_res)
    if input_dict is not None:
        logging.info('conv2_res: {}'.format(conv2_res.infer_shape(**input_dict)[1]))

    # res3
    conv3_res = residual_unit_order_new(conv2_res, n_in_ch=int(128 * alpha),
                                        n_out_ch1=int(256 * alpha), n_out_ch2=int(256 * alpha),
                                        dim_match=False, stride=(2, 2),
                                        act_out_first=act_out_first, bn_out_first=bn_out_first,
                                        name='conv3_res', num_deformable_group=num_deformable_group,
                                        use_global_stats=use_global_stats)
    in_layer_list.append(conv3_res)
    if input_dict is not None:
        logging.info('conv3_res: {}'.format(conv3_res.infer_shape(**input_dict)[1]))

    # res4
    conv4_res = residual_unit_order_new(conv3_res, n_in_ch=int(256 * alpha),
                                        n_out_ch1=int(512 * alpha), n_out_ch2=int(512 * alpha),
                                        dim_match=False, stride=(2, 2),
                                        act_out_first=act_out_first, bn_out_first=bn_out_first,
                                        name='conv4_res', use_global_stats=use_global_stats)
    conv5_res_1 = residual_unit_order_new(conv4_res, n_in_ch=int(512 * alpha),
                                          n_out_ch1=int(512 * alpha), n_out_ch2=int(512 * alpha),
                                          dim_match=True, stride=(1, 1),
                                          act_out_first=act_out_first, bn_out_first=bn_out_first,
                                          name='conv5_res_1', use_global_stats=use_global_stats)
    conv5_res_2 = residual_unit_order_new(conv5_res_1, n_in_ch=int(512 * alpha),
                                          n_out_ch1=int(512 * alpha), n_out_ch2=int(512 * alpha),
                                          dim_match=True, stride=(1, 1),
                                          act_out_first=act_out_first, bn_out_first=bn_out_first,
                                          name='conv5_res_2', num_deformable_group=num_deformable_group,
                                          use_global_stats=use_global_stats)
    in_layer_list.append(conv5_res_2)
    if input_dict is not None:
        logging.info('conv5_res_2: {}'.format(conv5_res_2.infer_shape(**input_dict)[1]))

    # res5
    stride = (2, 2) if inv_resolution == 32 else (1, 1)
    conv5_res_3 = residual_unit_order_new(conv5_res_2, n_in_ch=int(512 * alpha),
                                          n_out_ch1=int(1024 * alpha), n_out_ch2=int(1024 * alpha),
                                          dim_match=False, stride=stride,
                                          act_out_first=act_out_first, bn_out_first=bn_out_first,
                                          name='conv5_res_3', num_deformable_group=num_deformable_group,
                                          use_global_stats=use_global_stats)
    conv6_sep_1 = residual_unit_order_new(conv5_res_3, n_in_ch=int(1024 * alpha),
                                          n_out_ch1=int(1024 * alpha),
                                          n_out_ch2=int(1024 * alpha),
                                          dim_match=True, stride=(1, 1),
                                          act_out_first=act_out_first, bn_out_first=bn_out_first,
                                          name='conv6_res_1', num_deformable_group=num_deformable_group,
                                          use_global_stats=use_global_stats)
    in_layer_list.append(conv6_sep_1)
    if input_dict is not None:
        logging.info('conv6_sep_1: {}'.format(conv6_sep_1.infer_shape(**input_dict)[1]))

    return in_layer_list
