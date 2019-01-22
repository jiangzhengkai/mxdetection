import mxnet as mx
import logging

eps = 2e-5
mirroring_level = 2
last_gramma = False





def non_local_block(data, num_filter, mode, prefix=''):
    mid_num_filter = int(num_filter * 0.5)
    if mode == 'gaussian':
        x_reshape = mx.sym.Reshape(data=data, shape=(0, 0, -1))
        f = mx.sym.batch_dot(x_reshape, x_reshape, transpose_a=True, transpose_b=False)
        f = mx.sym.SoftmaxActivation(data=f, mode='channel')
    elif 'embedded_gaussian' in mode:
        x1 = sym.conv(data=data, name=prefix + 'conv_x1', num_filter=mid_num_filter)
        x2 = sym.conv(data=data, name=prefix + 'conv_x2', num_filter=mid_num_filter)
        if 'compress' in mode:
            x1 = sym.pool(data=x1, name=prefix + 'pool_x1', kernel=3, stride=2, pad=1, pool_type='max')
        x1_reshape = mx.sym.Reshape(data=x1, shape=(0, 0, -1))
        x2_reshape = mx.sym.Reshape(data=x2, shape=(0, 0, -1))
        f = mx.sym.batch_dot(x1_reshape, x2_reshape, transpose_a=True, transpose_b=False)
        f = mx.sym.SoftmaxActivation(data=f, mode='channel')
    else:
        raise ValueError("unknown non-local mode {}".format(mode))

    g = sym.conv(data=data, name=prefix + 'conv_g', num_filter=mid_num_filter)
    if mode == 'embedded_gaussian_compress':
        g_pool = sym.pool(data=g, name=prefix + 'pool_g', kernel=3, stride=2, pad=1, pool_type='max')
    else:
        g_pool = g
    g_reshape = mx.sym.Reshape(data=g_pool, shape=(0, 0, -1))
    y = mx.sym.batch_dot(g_reshape, f)
    y = mx.sym.reshape_like(y, g)
    y = sym.conv(data=y, name=prefix + 'conv_y', num_filter=num_filter)
    return data + y

def deformable_conv(data, name, num_filter, num_deformable_group,
                    kernel=3, stride=1, pad=-1, dilate=1,
                    num_group=1, no_bias=False, workspace=512):
    init_zero = mx.init.Zero()

    offset_weight = mx.sym.Variable(name=name+'_offset_weight', attr={'__init__': init_zero.dumps()})
    offset_bias = mx.sym.Variable(name=name+'_offset_bias', attr={'__init__': init_zero.dumps()})
    offset = mx.sym.Convolution(data=data, name=name + '_offset',
                      num_filter=2 * kernel * kernel * num_deformable_group,
                      kernel=(kernel,kernel), stride=(stride,stride), pad=(pad,pad), dilate=(dilate,dilate),
                      num_group=1, no_bias=False, attr={'__init__': init_zero.dumps()}, weight=offset_weight, bias=offset_bias)
    output = mx.contrib.symbol.DeformableConvolution(data=data, offset=offset, name=name,
                                                   num_filter=num_filter,
                                                   num_deformable_group=num_deformable_group,
                                                   kernel=(kernel, kernel),
                                                   stride=(stride, stride),
                                                   pad=(pad, pad),
                                                   dilate=(dilate, dilate),
                                                   num_group=num_group,
                                                   workspace=workspace,
                                                   no_bias=no_bias)
    return output


def residual_unit(data,
                  num_filter,
                  stride,
                  dim_match,
                  bottle_neck=True,
                  kernel_size=(3, 3),
                  dilate=1,
                  num_group=1,
                  bn_mom=0.9,
                  use_se=True,
                  sc_aug=False,
                  name=None,
                  workspace=512,
                  num_deformable=0,
                  use_sync_bn=False,
                  bn_use_global_stats=True):
    pad_size = (((kernel_size[0] - 1) * dilate + 1) // 2,
                ((kernel_size[1] - 1) * dilate + 1) // 2)
    if num_group > 1:
        assert bottle_neck
    if bottle_neck:
        ratio_1 = 0.25
        ratio_2 = 0.25
        if num_group == 32:
            ratio_1 = 0.5
            ratio_2 = 0.5
        if num_group == 64:
            ratio_1 = 1
            ratio_2 = 1

        conv1 = mx.sym.Convolution(data=data,
                                   num_filter=int(num_filter * ratio_1),
                                   kernel=(1, 1),
                                   stride=(1, 1),
                                   pad=(0, 0),
                                   no_bias=True,
                                   workspace=workspace,
                                   name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1,
                               fix_gamma=False,
                               eps=eps,
                               momentum=bn_mom,
                               sync=True if use_sync_bn else False,
                               attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                               if mirroring_level >= 2 else {},
                               use_global_stats=bn_use_global_stats,
                               name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1,
                                 act_type='relu',
                                 name=name + '_relu1')

        if mirroring_level >= 1:
            act1._set_attr(force_mirroring='True')
        if num_deformable > 0:

            conv2 = deformable_conv(data=act1,
                                    num_filter=int(num_filter * ratio_2),
                                    kernel=kernel_size[0],
                                    stride=stride[0],
                                    dilate=dilate,
                                    pad=pad_size[0],
                                    num_group=num_group,
                                    num_deformable_group=num_deformable,
                                    no_bias=True,
                                    workspace=workspace,
                                    name=name + '_conv2'
                                    )
        else:
            conv2 = mx.sym.Convolution(data=act1,
                                   num_filter=int(num_filter * ratio_2),
                                   kernel=kernel_size,
                                   stride=stride,
                                   dilate=(dilate, dilate),
                                   pad=pad_size,
                                   num_group=num_group,
                                   no_bias=True,
                                   workspace=workspace,
                                   name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2,
                               fix_gamma=False,
                               eps=eps,
                               momentum=bn_mom,
                               sync=True if use_sync_bn else False,
                               attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                               if mirroring_level >= 2 else {},
                               use_global_stats=bn_use_global_stats,
                               name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2,
                                 act_type='relu',
                                 name=name + '_relu2')
        if mirroring_level >= 1:
            act2._set_attr(force_mirroring='True')

        conv3 = mx.sym.Convolution(data=act2,
                                   num_filter=num_filter,
                                   kernel=(1, 1),
                                   stride=(1, 1),
                                   pad=(0, 0),
                                   no_bias=True,
                                   workspace=workspace,
                                   name=name + '_conv3')
        if not last_gramma:

            bn3 = mx.sym.BatchNorm(data=conv3,
                                   fix_gamma=False,
                                   eps=eps,
                                   momentum=bn_mom,
                                   sync=True if use_sync_bn else False,
                                   attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                   if mirroring_level >= 2 else {},
                                   use_global_stats=bn_use_global_stats,
                                   name=name + '_bn3')
        else:
            bn3_gamma = mx.sym.var(name=name + "_bn3_gamma", init=mx.init.Constant(0))
            bn3 = mx.sym.BatchNorm(data=conv3,
                                   fix_gamma=False,
                                   eps=eps,
                                   momentum=bn_mom,
                                   gamma=bn3_gamma,
                                   sync=True if use_sync_bn else False,
                                   attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                   if mirroring_level >= 2 else {},
                                   use_global_stats=bn_use_global_stats,
                                   name=name + '_bn3')

        if use_se:
            # se begin
            body_se = mx.sym.Pooling(data=bn3,
                                     global_pool=True,
                                     kernel=(7, 7),
                                     pool_type='avg',
                                     name=name + '_se_pool1')

            body_se = mx.sym.Convolution(data=body_se,
                                         num_filter=num_filter // 16,
                                         kernel=(1, 1),
                                         stride=(1, 1),
                                         pad=(0, 0),
                                         name=name + "_se_conv1",
                                         workspace=workspace)

            body_se = mx.sym.Activation(data=body_se,
                                        act_type='relu',
                                        name=name + '_se_relu')
            if mirroring_level >= 1:
                body_se._set_attr(force_mirroring='True')

            body_se = mx.sym.Convolution(data=body_se,
                                         num_filter=num_filter,
                                         kernel=(1, 1),
                                         stride=(1, 1),
                                         pad=(0, 0),
                                         name=name + "_se_conv2",
                                         workspace=workspace)
            body_se = mx.symbol.Activation(data=body_se,
                                           act_type='sigmoid',
                                           name=name + "_se_sigmoid")
            bn3 = mx.symbol.broadcast_mul(bn3, body_se)

        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=data,
                                          num_filter=num_filter,
                                          kernel=(3, 3) if sc_aug else (1, 1),
                                          pad=(1, 1) if sc_aug else (0, 0),
                                          stride=stride,
                                          no_bias=True,
                                          workspace=workspace,
                                          name=name + '_sc')
            shortcut = mx.sym.BatchNorm(data=shortcut,
                                        fix_gamma=False,
                                        eps=eps,
                                        momentum=bn_mom,
                                        sync=True if use_sync_bn else False,
                                        attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                        if mirroring_level >= 2 else {},
                                        use_global_stats=bn_use_global_stats,
                                        name=name + '_sc_bn')

        data_out = bn3 + shortcut
        data_out = mx.sym.Activation(data=data_out,
                                     act_type='relu',
                                     name=name + '_out_relu')
        if mirroring_level >= 1:
            data_out._set_attr(force_mirroring='True')
        return data_out
    else:
        conv1 = mx.sym.Convolution(data=data,
                                   num_filter=num_filter,
                                   kernel=kernel_size,
                                   stride=stride,
                                   dilate=(dilate, dilate),
                                   pad=pad_size,
                                   no_bias=True,
                                   workspace=workspace,
                                   name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1,
                               fix_gamma=False,
                               momentum=bn_mom,
                               eps=eps,
                               sync=True if use_sync_bn else False,
                               attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                               if mirroring_level >= 2 else {},
                               use_global_stats=bn_use_global_stats,
                               name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1,
                                 act_type='relu',
                                 name=name + '_relu1')
        if mirroring_level >= 1:
            act1._set_attr(force_mirroring='True')
        conv2 = mx.sym.Convolution(data=act1,
                                   num_filter=num_filter,
                                   kernel=kernel_size,
                                   stride=(1, 1),
                                   dilate=(dilate, dilate),
                                   pad=pad_size,
                                   no_bias=True,
                                   workspace=workspace,
                                   name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2,
                               fix_gamma=False,
                               momentum=bn_mom,
                               eps=eps,
                               sync=True if use_sync_bn else False,
                               attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                               if mirroring_level >= 2 else {},
                               use_global_stats=bn_use_global_stats,
                               name=name + '_bn2')

        if use_se:
            # se begin
            body_se = mx.sym.Pooling(data=bn2,
                                     global_pool=True,
                                     kernel=(7, 7),
                                     pool_type='avg',
                                     name=name + '_se_pool1')

            body_se = mx.sym.Convolution(data=body_se,
                                         num_filter=num_filter // 16,
                                         kernel=(1, 1),
                                         stride=(1, 1),
                                         pad=(0, 0),
                                         name=name + "_se_conv1",
                                         workspace=workspace)

            body_se = mx.sym.Activation(data=body_se,
                                        act_type='relu',
                                        name=name + '_se_relu')
            if mirroring_level >= 1:
                body_se._set_attr(force_mirroring='True')

            body_se = mx.sym.Convolution(data=body_se,
                                         num_filter=num_filter,
                                         kernel=(1, 1),
                                         stride=(1, 1),
                                         pad=(0, 0),
                                         name=name + "_se_conv2",
                                         workspace=workspace)
            body_se = mx.symbol.Activation(data=body_se,
                                           act_type='sigmoid',
                                           name=name + "_se_sigmoid")
            bn2 = mx.symbol.broadcast_mul(bn2, body_se)

        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=data,
                                          num_filter=num_filter,
                                          kernel=(1, 1),
                                          stride=stride,
                                          no_bias=True,
                                          workspace=workspace,
                                          name=name + '_sc')
            shortcut = mx.sym.BatchNorm(data=shortcut,
                                        fix_gamma=False,
                                        momentum=bn_mom,
                                        eps=eps,
                                        sync=True if use_sync_bn else False,
                                        attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                        if mirroring_level >= 2 else {},
                                        use_global_stats=bn_use_global_stats,
                                        name=name + '_sc_bn')
        data_out = bn2 + shortcut
        data_out = mx.sym.Activation(data=data_out,
                                     act_type='relu',
                                     name=name + '_out_relu')
        if mirroring_level >= 1:
            data_out._set_attr(force_mirroring='True')
        return data_out


def residual_backbone(data,
                      units,
                      num_stage,
                      filter_list,
                      input_dict,
                      bottle_neck=True,
                      num_group=1,
                      bn_mom=0.9,
                      use_aug=False,
                      workspace=512,
                      deformable_units=None,
                      deformable_group=None,
                      nonlocal_mode=None,
                      bn_use_global_stats=True,
                      use_sync_bn=False):
    num_unit = len(units)
    assert (num_unit == num_stage)
    # data = mx.sym.Variable(name='data')
    body_list = []
    if use_aug:
        body = mx.sym.Convolution(data=data,
                                  num_filter=filter_list[0],
                                  kernel=(3, 3),
                                  stride=(2, 2),
                                  pad=(1, 1),
                                  no_bias=True,
                                  name="conv0",
                                  workspace=workspace)
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=eps,
                                momentum=bn_mom,
                                sync=True if use_sync_bn else False,
                                attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                if mirroring_level >= 2 else {},
                                use_global_stats=bn_use_global_stats,
                                name='bn0')
        body = mx.sym.Activation(data=body,
                                 act_type='relu',
                                 name='relu0')
        if mirroring_level >= 1:
            body._set_attr(force_mirroring='True')

        body = mx.sym.Convolution(data=body,
                                  num_filter=filter_list[0],
                                  kernel=(3, 3),
                                  stride=(1, 1),
                                  pad=(1, 1),
                                  no_bias=True,
                                  name="conv1",
                                  workspace=workspace)
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=eps,
                                momentum=bn_mom,
                                sync=True if use_sync_bn else False,
                                attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                if mirroring_level >= 2 else {},
                                use_global_stats=bn_use_global_stats,
                                name='bn1')
        body = mx.sym.Activation(data=body,
                                 act_type='relu',
                                 name='relu1')
        if mirroring_level >= 1:
            body._set_attr(force_mirroring='True')

        body = mx.sym.Convolution(data=body,
                                  num_filter=2 * filter_list[0],
                                  kernel=(3, 3),
                                  stride=(1, 1),
                                  pad=(1, 1),
                                  no_bias=True,
                                  name="conv2",
                                  workspace=workspace)
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=eps,
                                momentum=bn_mom,
                                sync=True if use_sync_bn else False,
                                attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                if mirroring_level >= 2 else {},
                                use_global_stats=bn_use_global_stats,
                                name='bn2')
        body = mx.sym.Activation(data=body,
                                 act_type='relu',
                                 name='relu2'
                                      '')
        if mirroring_level >= 1:
            body._set_attr(force_mirroring='True')




    else:

        body = mx.sym.Convolution(data=data,
                                  num_filter=filter_list[0],
                                  kernel=(7, 7),
                                  stride=(2, 2),
                                  pad=(3, 3),
                                  no_bias=True,
                                  name="conv0",
                                  workspace=workspace)
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=eps,
                                momentum=bn_mom,
                                sync=True if use_sync_bn else False,
                                attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                if mirroring_level >= 2 else {},
                                use_global_stats=bn_use_global_stats,
                                name='bn0')
        body = mx.sym.Activation(data=body,
                                 act_type='relu',
                                 name='relu0')
        if mirroring_level >= 1:
            body._set_attr(force_mirroring='True')

    body = mx.symbol.Pooling(data=body,
                             kernel=(3, 3),
                             stride=(2, 2),
                             pad=(1, 1),
                             pool_type='max')

    for i in range(num_stage):

        body = residual_unit(data=body,
                             num_filter=filter_list[i + 1], stride=(1 if i == 0 else 2, 1 if i == 0 else 2),
                             dim_match=False,
                             name='stage%d_unit%d' % (i + 1, 1),
                             bottle_neck=bottle_neck,
                             num_group=num_group,
                             sc_aug=False if i == 0 else use_aug,
                             workspace=workspace,
                             num_deformable=deformable_group[i] if units[i] - deformable_units[i] < 1 else 0,
                             bn_use_global_stats=bn_use_global_stats,
                             use_sync_bn=use_sync_bn)

        for j in range(units[i] - 1):

            body = residual_unit(data=body,
                                 num_filter=filter_list[i + 1], stride=(1, 1),
                                 dim_match=True,
                                 name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck,
                                 num_group=num_group,
                                 workspace=workspace,
                                 num_deformable=deformable_group[i] if units[i] - deformable_units[i] < j + 2 else 0,
                                 bn_use_global_stats=bn_use_global_stats,
                                 use_sync_bn=use_sync_bn
                                 )

            if i==2 and j == units[i] - 3 and nonlocal_mode is not None:
                body = non_local_block(data=body,num_filter=filter_list[i+1],mode=nonlocal_mode)
                logging.info('nonlocal_stage{}_{}: {}'.format(i+1,j+2,body.infer_shape(**input_dict)[1]))
        logging.info('stage{}: {}'.format(i+1,body.infer_shape(**input_dict)[1]))
        body_list.append(body)
    return body_list


def get_symbol(data,
               net_depth,
               config,
               num_class=1000,
               num_group=1,
               bn_mom=0.9,
               workspace=512,
               deformable_units=[0,0,0,0],
               num_deformable_group=[0,0,0,0],
               bn_use_global_stats=True,
               bn_use_sync=False,
               input_dict=None
               ):
    if net_depth == 18:
        units = [2, 2, 2, 2]
    elif net_depth == 34:
        units = [3, 4, 6, 3]
    elif net_depth == 50:
        units = [3, 4, 6, 3]
    elif net_depth == 101:
        units = [3, 4, 23, 3]
    elif net_depth == 152:
        units = [3, 8, 36, 3]
    elif net_depth == 154:
        units = [3, 8, 36, 3]
    elif net_depth == 200:
        units = [3, 24, 36, 3]
    elif net_depth == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError("no experiments done on detph {}, you can do it youself".format(net_depth))

    num_stage = 4
    filter_list = [64, 256, 512, 1024, 2048] if net_depth >= 50 else [64, 64, 128, 256, 512]
    bottle_neck = True if net_depth >= 50 else False
    if net_depth == 154:
        use_aug = True
    else:
        use_aug = False


    body_list = residual_backbone(data=data,
                             units=units,
                             num_stage=num_stage,
                             filter_list=filter_list,
                             input_dict=input_dict,
                             bottle_neck=bottle_neck,
                             num_group=num_group,
                             bn_mom=bn_mom,
                             use_aug=use_aug,
                             workspace=workspace,
                             deformable_units=deformable_units,
                             deformable_group=num_deformable_group,
                             bn_use_global_stats=bn_use_global_stats,
                             use_sync_bn=bn_use_sync,
                             nonlocal_mode='embedded_gaussian_compress' if config.network.add_nonlocal else None)


    return body_list





