fix_gamma = False
eps = 2e-5
bn_mom_stage1 = 0.9
bn_mom_stage2 = 0.9
workspace=512
mirroring_level=2
use_sync_bn=False
#backbone_

use_global_stats_stage1=True
use_global_stats_stage2=False
if use_sync_bn:
    use_global_stats_stage1=False
bn_stage2=True
lr_mult_stage1=1
lr_mult_stage2=10
use_upscale_head=True
unet_type='unet' # unet or nnet
upscale_type='bilinear' #bilinear or nearest
fusion_type='add'# add or concat
backbone_num_filter=[2048,1024,728,728,256]
out_dim=19
use_rec_list=False
use_deformable_conv=True
lr_mult_offset=1

#dropout

dropout_type='constant' #constant, mutable, poly
base_dropout=0.4
mutable_dropout={'base_p': 0.3,
                'change_p': (0.4, 0.5, 0.6, 0.7),
                'change_step': (30000, 50000, 70000,80000)}

poly_dropout={'start_p': 0.3,
              'end_p': 0.6,
              'max_update': 100000,
              'power': 1.0}