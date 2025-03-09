 

exp_name = 'base_rgb_weather'
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0, convert_to='y')
 
# dataset settings
train_dataset_type='AllinOneDataset'
test_dataset_type = 'AllinOneDataset'
val_dataset_type = test_dataset_type
# convert_to='yuv'

train_pipeline = [
    dict(type='LoadImageFromFile',io_backend='disk',key='lq', channel_order='rgb'),
    dict(type='LoadImageFromFile',io_backend='disk',key='gt',channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Resize_min', keys=['lq','gt'], min_size=320),
    dict(type='PairedRandomCrop', gt_patch_size=320),
    dict(type='Flip', keys=['lq',  'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['lq',  'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='Collect', keys=['lq','gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor_trans', channel_order='rgb', keys=['lq', 'gt']),
    dict(type='CopyValues', src_keys=['gt'], dst_keys=['lq'], copy_ratio=0.2)
]



test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        channel_order='rgb', 
        ),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        channel_order='rgb', 
        ),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Collect',keys=['lq', 'gt'], meta_keys=['lq_path','gt_path']),
    dict(type='ImageToTensor_trans', channel_order='rgb', keys=['lq', 'gt']),
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
            # gt_folder='/dev/shm/dataset/train',#
            # return_param="gt_path",
            return_param="all",
            gt_folder='data/all-in-one-train',
            dataset_json='all_in_one.json',
            pipeline=train_pipeline,
            scale=1)),
    # val
    val=dict(
        type=test_dataset_type,
        # return_param="gt_path",
        return_param="all",
        gt_folder='data/all-in-one-test-test',
        pipeline=test_pipeline,
        specified_key=["SOTS_outdoor"],
        scale=1),
    # test
    test=dict(
        type=test_dataset_type,
        # return_param="gt_path",
        return_param="all",
        gt_folder='data/all-in-one-test-test',
        pipeline=test_pipeline,
        specified_key=["SOTS_outdoor"],# SOTS_outdoor rain1400_test CSD_test  LOL-v2-test  
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
evaluation = dict(interval=5000, save_image=False, gpu_collect=True)
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
resume_from = None
workflow = [('train', 1)]
 
 