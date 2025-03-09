_base_ = [
    '../base/base_rgb_weather_384.py'
]


lambda_cfg = [128,2048]
training_scheduling = 'normal'



exp_name = 'all_in_one_L_weather' 
dim=48
model = dict(
    type='DMCI',
    generator=dict(type='DMCI_ED_restormer_dual', dim=dim, num_blocks=[2,3,3,4], heads=[4,4,4,4], ffn_expansion_factor=2.66,  
                   bias=False, LayerNorm_type='BiasFree',  N_ch=288, z_ch=4*dim,N_ch_new=True,sigmoid=False,
                   intraEncoder="Restor_Encoderv2_pos3", version_enc="v4", 
                   intraDecoder="Restor_Decoderv2_pos3", version_dec="v4",),
    RD_loss=dict(type='RDLoss', loss_weight=1.0),
        )
resume_from = f'./workdirs/{exp_name}/iter_150000.pth'

 
total_iters = 450000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[total_iters],
    restart_weights=[1],
    min_lr=1e-7)


checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=10000, save_image=False, gpu_collect=True)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./workdirs/{exp_name}'
load_from = None
workflow = [('train', 1)]
