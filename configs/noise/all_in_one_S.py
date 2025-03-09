# model settings ./tools/dist_train_iter.sh   configs/dmci_dev/DMCI_restormer_noise.py  1 ' --test_dataset  ./xxxxxx   '
 
 
lambda_cfg = [128,2048]
training_scheduling = 'normal'

 

exp_name = 'all_in_one_S_noise'
dim=32
model = dict(
    type='DMCI',
    generator=dict(type='DMCI_ED_restormer_dual', dim=32, num_blocks=[2,3,3,4], heads=[4,4,4,4], ffn_expansion_factor=2.66,  
                   bias=False, LayerNorm_type='BiasFree',  N_ch=256, z_ch=4*dim,N_ch_new=True,sigmoid=False,
                   intraEncoder="Restor_Encoderv2_pos3", version_enc="v4", 
                   intraDecoder="Restor_Decoderv2_pos3", version_dec="v4",),
    RD_loss=dict(type='RDLoss', loss_weight=1.0),
        )
train_patch=384
resume_from = f'./workdirs/{exp_name}/iter_150000.pth'

 


train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)
# dataset settings
train_dataset_type = 'AllinOneDataset'
test_dataset_type = 'AllinOneDataset'
val_dataset_type = test_dataset_type

train_pipeline = [
    dict(type='LoadImageFromFile', io_backend='disk', key='lq', channel_order='rgb', ),
    dict(type='LoadImageFromFile', io_backend='disk', key='gt', channel_order='rgb',),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Resize_min', keys=['lq', 'gt'], min_size=train_patch),
    dict(type='PairedRandomCrop', gt_patch_size=train_patch),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='DegradationsRandomChoosev2', data_probs=[0,1], probs=[0, 1], #clean, blur/noise, degraded lq   
         degradations=[
                        dict(type='RandomBlur',
                            params=dict(
                                kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
                                kernel_list=[
                                    'iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                                    'plateau_iso', 'plateau_aniso', 'sinc'
                                ],
                                kernel_prob=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
                                sigma_x=[0.2, 3],
                                sigma_y=[0.2, 3],
                                rotate_angle=[-3.1416, 3.1416],
                                beta_gaussian=[0.5, 4],
                                beta_plateau=[1, 2],
                                sigma_x_step=0.02,
                                sigma_y_step=0.02,
                                rotate_angle_step=0.31416,
                                beta_gaussian_step=0.05,
                                beta_plateau_step=0.1,
                                omega_step=0.0628),
                            keys=['lq']),
                        dict(type='Gaussian_Noise', params=dict( gaussian_sigma=[15, 25, 50], gaussian_gray_noise_prob=0 ), keys=['lq'], ),
                            ]
         ),
 
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path','gt_path']),
    dict(type='ImageToTensor_trans', channel_order='rgb', keys=['lq', 'gt']),
    dict(type='CopyValues', src_keys=['gt'], dst_keys=['lq'], copy_ratio=0.2)
]

 
test_pipeline = [
    dict(type='LoadImageFromFile',io_backend='disk', key='lq',channel_order='rgb', ),
    dict(type='LoadImageFromFile',io_backend='disk',key='gt',channel_order='rgb', ),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor_trans', channel_order='rgb', keys=['lq', 'gt']),
    # dict(type='SyntheticRandomNoise', keys=['lq'], noise_level=4),
    dict(type='Gaussian_Noise_test', keys=['lq'], params=dict(gaussian_sigma=[15, 25, 50]), degrade_type=25),  
    dict(type='Collect',keys=['lq', 'gt'], meta_keys=['lq_path','gt_path']),
    
]

val_pipeline=test_pipeline


data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),  # 2 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=train_dataset_type,
            shuffle=True,
            return_param="all",
            gt_folder='data/all-in-one-train',
            dataset_json='description_v2.json',
            pipeline=train_pipeline,
            scale=1)),
    val=dict(
        type="SRFolderDatasetv3",
        lq_folder="data/all-in-one-test",
        gt_folder="data/all-in-one-test",
        specified_key=["/Kodak_24","/Kodak_24"],
        pipeline=test_pipeline,  
        scale=1),
      test=dict(
        type="SRFolderDatasetv3",
        lq_folder="data/all-in-one-test",
        gt_folder="data/all-in-one-test",
        specified_key=["/Kodak_24","/Kodak_24"],
        pipeline=test_pipeline,  
        scale=1),
)


 

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))
# learning policy
training_scheduling = 'normal'

# total_iters = 300000
total_iters = 400000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[400000],
    restart_weights=[1],
    min_lr=1e-7)



checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=10000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./workdirs/{exp_name}'
load_from = None
workflow = [('train', 1)]
 
 