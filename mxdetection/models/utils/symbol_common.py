import mxnet as mx
from importlib import import_module
from easydict import EasyDict as edict

cfg = edict()
cfg.bn_use_mirroring_level = False
cfg.bn_eps = 2e-5
cfg.workspace = 512
cfg.lr_type = ''
cfg.absorb_bn = False
cfg.bn_use_global_stats = True

def get_symbol_function(name):
    if name is None:
        return None
    parts = name.split('.')
    if len(parts) == 1:
        return globals()[parts[0]]
    module_name = '.'.join(parts[:-1])
    module = import_module(module_name)
    return getattr(module, parts[-1])
    
def _attr_scope_lr(lr_type, lr_owner):
    assert lr_owner in ('weight', 'bias')
    if lr_type == 'alex':
        if lr_owner == 'weight':
            return mx.AttrScope(lr_mult='1.', wd_mult='1.')
        elif lr_owner == 'bias':
            return mx.AttrScope(lr_mult='2.', wd_mult='0.')
    elif lr_type == 'alex10':
        if lr_owner == 'weight':
            return mx.AttrScope(lr_mult='20.', wd_mult='1.')
        elif lr_owner == 'bias':
            return mx.AttrScope(lr_mult='20.', wd_mult='0.')
    elif lr_type == 'torch10':
        if lr_owner == 'weight':
            return mx.AttrScope(lr_mult='10.', wd_mult='1.')
        elif lr_owner == 'bias':
            return mx.AttrScope(lr_mult='10.', wd_mult='0.')
    elif lr_type == 'zeros':
        return mx.AttrScope(lr_mult='0.', wd_mult='0.')
    else:
        return mx.AttrScope()
    

cfg.reuse = False
cfg.save_param_for_reuse = False
cfg.param_dict = {}

def get_variable(name, **kwargs):
    if cfg.reuse:
        assert name in cfg.param_dict
        return cfg.param_dict[name]
    else:
        if name.endswith('weight') or name.endswith('gamma'):
            lr_owner = 'weight'
        elif name.endswith('bias') or name.endswith('beta'):
            lr_owner = 'bias'
        else:
            raise ValueError('unknow variable name: %s' % name)
        with _attr_scope_lr(cfg.lr_type, lr_owner):
            var_kwargs = dict()
            for var_name in ['attr', 'shape', 'init']:
                if var_name in kwargs:
                    var_kwargs[var_name] = kwargs[var_name]
            param = mx.sym.Variable(name, **var_kwargs)
        if cfg.save_param_for_reuse:
            assert name not in cfg.param_dict
            cfg.param_dict[name] = param
        return param
       

def add_weight_bias(name, no_bias, **kwargs):
    if 'attr' in kwargs:
        cfg.param_attr = kwargs['attr']
    if 'weight' not in kwargs:
        kwargs['weight'] = get_variable('{}_weight'.format(name))
    if no_bias is False and 'bias' not in kwargs:
        kwargs['bias'] = get_variable('{}_bias'.format(name))
    cfg.param_attr = None
    return kwargs 
        
def relu(data, name=None):
    return mx.sym.Activation(data, name=name, act_type='relu')

def sigmoid(data, name=None):
    return mx.sym.Activation(data, name=name, act_type='sigmoid')
    
def bn(data, name, fix_gamma=False, **kwargs):
    kwargs['gamma'] = get_variable('{}_gamma'.format(name))
    kwargs['beta'] = get_variable('{}_beta'.format(name))
    if cfg.bn_use_mirroring_level:
        kwargs['attr'] = {'force_mirroring': 'True', 'cudnn_off': 'True'}
    return mx.sym.BatchNorm(data=data, name=name,
                            fix_gamma=fix_gamma, 
                            use_global_stats=cfg.bn_use_global_stats,
                            eps=cfg.bn_eps,
                            **kwargs)
def gn(data, name, channels, gn_group=32, eps=1e-5):
    gn_group = min(channels, gn_group)
    # reshape to N, G, C//G, H, W
    data = mx.sym.Reshape(data=data, shape=(-1, -4, gn_group, -1, -2))
    
    mean = mx.sym.mean(data=data, axis=(2, 3, 4), keepdims=True)
    var = mx.sym.mean(data=(mx.sym.broadcast_sub(data, mean) ** 2), axis=(2, 3, 4), keepdims=True)
    data = mx.sym.broadcast_div(mx.sym.broadcast_sub(data, mean), mx.sym.sqrt(var + eps))
    
    gamma = get_variable(name='%s_gamma' % name, shape=(1, channels, 1, 1), init=mx.initializer.One())
    beta = get_variable(name='%s_beta' % name, shape=(1, channels, 1, 1), init=mx.initializer.Zero())
    data = mx.sym.Reshape(data, shape=(-1, -3, -2))
    data = mx.sym.broadcast_add(mx.sym.broadcast_mul(data, gamma), beta)
    return data

def conv(data, name, num_filter, kernel=1, stride=1, pad=-1, dilate=1,
         num_group=1, no_bias=False, **kwargs):
    if isinstance(kernel, tuple):
        if not isinstance(stride, tuple):
            stride = (stride, stride)
        if not isinstance(dilate, tuple):
            dilate = (dilate, dilate)
        assert isinstance(pad, tuple)
    else:
        assert not isinstance(stride, tuple)
        assert not isinstance(pad, tuple)
        assert not isinstance(dilate, tuple)
        if kernel == 1:
            dilate = 1
        if pad < 0:
            assert kernel % 2 == 1, 'Specify pad for an even kernel size'
            pad = ((kernel - 1) * dilate + 1) // 2
        kernel = (kernel, kernel)
        stride = (stride, stride)
        pad = (pad, pad)
        dilate = (dilate, dilate)
    kwargs = add_weight_bias(name, no_bias, **kwargs)
    return mx.sym.Convolution(data=data, name=name,
                              num_filter=num_filter,
                              kernel=kernel,
                              stride=stride,
                              pad=pad,
                              dilate=dilate,
                              num_group=num_group,
                              workspace=cfg.workspace,
                              no_bias=no_bias, **kwargs)


def _deformable_conv(data, offset, name,
                     num_filter, num_deformable_group,
                     kernel=1, stride=1, pad=-1, dilate=1,
                     num_group=1, no_bias=False, **kwargs):
    if kernel == 1:
        dilate = 1
    if pad < 0:
        assert kernel % 2 == 1, 'Specify pad for an even kernel size'
        pad = ((kernel - 1) * dilate + 1) // 2
    if 'weight' not in kwargs:
        kwargs['weight'] = get_variable('{}_weight'.format(name), **kwargs)
    if no_bias is False and 'bias' not in kwargs:
        kwargs['bias'] = get_variable('{}_bias'.format(name), **kwargs)
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

def deconv(data, name, num_filter, kernel=3, stride=1, pad=0, adj=0,
           target_shape=0, num_group=1, no_bias=False, **kwargs):
    if 'weight' not in kwargs:
        kwargs['weight'] = get_variable('{}_weight'.format(name), **kwargs)
    if no_bias is False and 'bias' not in kwargs:
        kwargs['bias'] = get_variable('{}_bias'.format(name), **kwargs)
    if target_shape > 0:
        return mx.sym.Deconvolution(data=data, name=name,
                                    num_filter=num_filter,
                                    kernel=(kernel, kernel),
                                    stride=(stride, stride),
                                    target_shape=(target_shape, target_shape),
                                    num_group=num_group,
                                    workspace=cfg.workspace,
                                    no_bias=no_bias,
                                    **kwargs)
    else:
        return mx.sym.Deconvolution(data=data, name=name,
                                    num_filter=num_filter,
                                    kernel=(kernel, kernel),
                                    stride=(stride, stride),
                                    pad=(pad, pad),
                                    adj=(adj, adj),
                                    num_group=num_group,
                                    workspace=cfg.workspace,
                                    no_bias=no_bias,
                                    **kwargs)

def fc(data, name, num_hidden, no_bias=False, **kwargs):
    if 'weight' not in kwargs:
        kwargs['weight'] = get_variable('{}_weight'.format(name), **kwargs)
    if no_bias is False and 'bias' not in kwargs:
        kwargs['bias'] = get_variable('{}_bias'.format(name), **kwargs)
    return mx.sym.FullyConnected(data=data, name=name,
                                 num_hidden=num_hidden,
                                 no_bias=no_bias,
                                 **kwargs)

def pool(data, name, kernel=3, stride=2, pad=-1,
         pool_type='max', pooling_convention='valid',
         global_pool=False):
    if global_pool:
        assert pad < 0
        return mx.sym.Pooling(data, name=name,
                              kernel=(1, 1),
                              pool_type=pool_type,
                              global_pool=True)
    else:
        if pad < 0:
            assert kernel % 2 == 1, 'Specify pad for an even kernel size'
            pad = kernel // 2
        return mx.sym.Pooling(data, name=name,
                              kernel=(kernel, kernel),
                              stride=(stride, stride),
                              pad=(pad, pad),
                              pool_type=pool_type,
                              pooling_convention=pooling_convention,
                              global_pool=False)

def upsampling_bilinear(data, name, scale, num_filter, need_train=False):
    if scale == 1:
        return data
    if need_train:
        weight = get_variable('{}_weight'.format(name))
    else:
        old_lr_type = cfg.lr_type
        cfg.lr_type = 'zeros'
        weight = get_variable('{}_weight'.format(name))
        cfg.lr_type = old_lr_type
    return mx.sym.UpSampling(data=data, weight=weight, name=name,
                             scale=int(scale), num_filter=num_filter,
                             sample_type='bilinear', num_args=2,
                             workspace=cfg.workspace)

def upsampling_nearest(data, name, scale):
    return mx.symbol.UpSampling(data, name=name, scale=int(scale), sample_type="nearest")

def softmax_out(data, label, name=None, grad_scale=1.0, ignore_label=None, multi_output=False, normalization='valid'):
    if ignore_label is not None:
        return mx.sym.SoftmaxOutput(data=data, label=label, name=name,
                                    grad_scale=grad_scale,
                                    use_ignore=True,
                                    ignore_label=ignore_label,
                                    multi_output=multi_output,
                                    normalization=normalization)
    else:
        return mx.sym.SoftmaxOutput(data=data, label=label, name=name,
                                    grad_scale=grad_scale,
                                    multi_output=multi_output,
                                    normalization=normalization)


def deformable_conv(data, name, num_filter, num_deformable_group,
                    kernel=3, stride=1, pad=-1, dilate=1,
                    num_group=1, no_bias=False, **kwargs):
    init_zero = mx.init.Zero()
    offset = conv(data=data, name=name + '_offset',
                  num_filter=2 * kernel * kernel * num_deformable_group,
                  kernel=kernel, stride=stride, pad=pad, dilate=dilate,
                  num_group=1, no_bias=False, attr={'__init__': init_zero.dumps()},
                  **kwargs)
    output = _deformable_conv(data=data, offset=offset, name=name,
                              num_filter=num_filter, num_deformable_group=num_deformable_group,
                              kernel=kernel, stride=stride, pad=pad, dilate=dilate,
                              num_group=num_group, no_bias=no_bias, **kwargs)
    return output


# combination
def _dcn_or_conv(data, name, no_bias, num_deformable_group=0, **kwargs):
    if num_deformable_group > 0:
        sym_conv = deformable_conv(data=data, name=name, no_bias=no_bias, num_deformable_group=num_deformable_group, **kwargs)
    else:
        sym_conv = conv(data=data, name=name, no_bias=no_bias, **kwargs)
    return sym_conv

def bnreluconv(data, no_bias=True, absorb_bn=False, prefix='', suffix='', **kwargs):
    sym_bn = data if absorb_bn else bn(data=data, name=prefix + 'bn' + suffix, fix_gamma=False)
    sym_relu = relu(data=sym_bn, name=prefix + 'relu' + suffix)
    sym_conv = _dcn_or_conv(data=sym_relu, name=prefix + 'conv' + suffix, no_bias=no_bias, **kwargs)
    return sym_bn, sym_relu, sym_conv

def convbnrelu(data, no_bias=True, prefix='', suffix='', **kwargs):
    no_bias = False if cfg.absorb_bn else no_bias
    sym_conv = _dcn_or_conv(data=data, name=prefix + 'conv' + suffix, no_bias=no_bias, **kwargs)
    sym_bn = sym_conv if cfg.absorb_bn else bn(data=sym_conv, name=prefix + 'bn' + suffix, fix_gamma=False)
    sym_relu = relu(data=sym_bn, name=prefix + 'relu' + suffix)
    return sym_conv, sym_bn, sym_relu

def convgnrelu(data, no_bias=True, gn_group=32, prefix='', suffix='', **kwargs):
    sym_conv = _dcn_or_conv(data=data, name=prefix + 'conv' + suffix, no_bias=no_bias, **kwargs)
    sym_gn = gn(data=sym_conv, name=prefix + 'gn' + suffix, channels=kwargs['num_filter'], gn_group=gn_group)
    sym_relu = relu(data=sym_gn, name=prefix + 'relu' + suffix)
    return sym_conv, sym_gn, sym_relu

def convbn(data, no_bias=True, prefix='', suffix='', **kwargs):
    no_bias = False if cfg.absorb_bn else no_bias
    sym_conv = _dcn_or_conv(data=data, name=prefix + 'conv' + suffix, no_bias=no_bias, **kwargs)
    sym_bn = sym_conv if cfg.absorb_bn else bn(data=sym_conv, name=prefix + 'bn' + suffix, fix_gamma=False)
    return sym_conv, sym_bn

def convgn(data, no_bias=True, gn_group=32, prefix='', suffix='', **kwargs):
    sym_conv = _dcn_or_conv(data=data, name=prefix + 'conv' + suffix, no_bias=no_bias, **kwargs)
    sym_gn = gn(data=sym_conv, name=prefix + 'gn' + suffix, channels=kwargs['num_filter'], gn_group=gn_group)
    return sym_conv, sym_gn

def reluconv(data, no_bias=False, prefix='', suffix='', **kwargs):
    sym_relu = relu(data=data, name=prefix + 'relu' + suffix)
    sym_conv = _dcn_or_conv(data=sym_relu, name=prefix + 'conv' + suffix, no_bias=no_bias, **kwargs)
    return sym_relu, sym_conv

def convrelu(data, no_bias=False, prefix='', suffix='', **kwargs):
    sym_conv = _dcn_or_conv(data=data, name=prefix + 'conv' + suffix, no_bias=no_bias, **kwargs)
    sym_relu = relu(data=sym_conv, name=prefix + 'relu' + suffix)
    return sym_conv, sym_relu

def fcrelu(data, num_hidden, no_bias=False, prefix='', suffix=''):
    sym_fc = fc(data=data, name=prefix + 'fc' + suffix, num_hidden=num_hidden, no_bias=no_bias)
    sym_relu = relu(data=sym_fc, name=prefix + 'relu' + suffix)
    return sym_fc, sym_relu

def fcreludrop(data, num_hidden, p, no_bias=False, prefix='', suffix=''):
    sym_fc = fc(data=data, name=prefix + 'fc' + suffix, num_hidden=num_hidden, no_bias=no_bias)
    sym_relu = relu(data=sym_fc, name=prefix + 'relu' + suffix)
    sym_drop = dropout(data=sym_relu, name=prefix + 'drop' + suffix, p=p)
    return sym_fc, sym_relu, sym_drop

def remove_batch_norm_paras(w, b, moving_mean, moving_var, gamma, beta, bn_eps=cfg.bn_eps):
    assert w.shape[0] == b.shape[0]
    v = np.zeros(w.shape, dtype=w.dtype)
    c = np.zeros(b.shape, dtype=b.dtype)
    moving_std = np.sqrt(moving_var + bn_eps)
    for i in range(w.shape[0]):
        v[i, :] = w[i, :] * gamma[i] / moving_std[i]
        c[i] = b[i] * gamma[i] / moving_std[i] - moving_mean[i] * gamma[i] / moving_std[i] + beta[i]
    return v, c
