from symbol_collection import *

class Backbone(object):
    def __init__(self, config):
        self.config = config

    def get_symbol(self, data, is_train, **kwargs):
        bn_use_global_stats = self.config.TRAIN.bn_use_global_stats if is_train else True
        bn_use_sync = self.config.TRAIN.bn_use_sync if is_train else False
        inv_resolution = 32 if 'fpn' in self.config.network.task_type else 16

        if self.config.network.net_type == 'resnet':
            in_layer_list = get_resnet_symbol(data=data,
                                              num_layer=self.config.network.num_layer,
                                              net_type=self.config.network.net_type,
                                              inv_resolution=inv_resolution,
                                              deformable_units=self.config.network.deformable_units,
                                              num_deformable_group=self.config.network.num_deformable_group,
                                              bn_use_sync=bn_use_sync,
                                              bn_use_global_stats=bn_use_global_stats,
                                              absorb_bn=self.config.TRAIN.absorb_bn,
                                              input_dict=kwargs['input_dict'])

        elif self.config.network.net_type == 'mobilenet':
            in_layer_list = get_mobilenet_symbol(data=data, inv_resolution=inv_resolution,
                                                 bn_use_global_stats=bn_use_global_stats)

        elif self.config.network.net_type == 'mobilenet_res':
            in_layer_list = get_mobilenet_res_symbol(data=data, inv_resolution=inv_resolution,
                                                     bn_use_global_stats=bn_use_global_stats)

        elif self.config.network.net_type == 'mobilenet_v2_w_bypass':
            in_layer_list = get_molinet_v2_w_bypass_symbol(data=data, inv_resolution=inv_resolution,
                                                           bn_use_global_stats=bn_use_global_stats,
                                                           mirroring_label=self.config.TRAIN.mirroring_label)

        elif self.config.network.net_type == 'mobilenet_v2_wo_bypass':
            in_layer_list = get_molinet_v2_wo_bypass_symbol(data=data, inv_resolution=inv_resolution,
                                                            bn_use_global_stats=bn_use_global_stats)

        elif self.config.network.net_type == 'tiny_xception':
            in_layer_list = get_tiny_xception_symbol(data=data, inv_resolution=inv_resolution,
                                                     bn_use_global_stats=bn_use_global_stats)

        elif self.config.network.net_type == 'senet':
            in_layer_list = get_senet_symbol(data=data, net_depth=self.config.network.num_layer,
                                             config=self.config, num_group=64, input_dict=kwargs['input_dict'],
                                             bn_use_global_stats=bn_use_global_stats, bn_use_sync=bn_use_sync,
                                             deformable_units=self.config.network.deformable_units,
                                             num_deformable_group=self.config.network.num_deformable_group)

        elif self.config.network.net_type == 'resnext_group':
            in_layer_list = get_resnext_symbol(data=data, net_depth=self.config.network.num_layer,
                                               config=self.config, num_group=self.config.network.num_group,
                                               input_dict=kwargs['input_dict'], bn_use_global_stats=bn_use_global_stats,
                                               bn_use_sync=bn_use_sync,
                                               deformable_units=self.config.network.deformable_units,
                                               num_deformable_group=self.config.network.num_deformable_group)

        elif self.config.network.net_type == 'aligned_xception':
            # change config in symbols.unet_config
            use_global_stats_stage1 = True
            use_global_stats_stage2 = bn_use_global_stats
            if len(np.array(self.config.network.deformable_units) > 0) > 0:
                use_deformable_conv = True
            in_layer_list = get_aligned_xception_symbol(data=data, use_deformable_conv=use_deformable_conv,
                                                        use_global_stats_stage1=use_global_stats_stage1,
                                                        use_global_stats_stage2=use_global_stats_stage2,
                                                        use_sync_bn=bn_use_sync)

        else:
            raise ValueError("unknown net_type {}".format(self.config.network.net_type))

        return in_layer_list




