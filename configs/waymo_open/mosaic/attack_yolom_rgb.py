model = dict(
    type='YOLOV4M',
    backbone=dict(
        type='DarknetCSP',
        scale='m5p',
        in_channels=3,
        out_indices=[3, 4, 5]),
    neck=dict(
        type='PACSPFPN',
        in_channels=[192, 384, 768],
        out_channels=[192, 384, 768],
        csp_repetition=1),
    bbox_head=dict(
        type='YOLOV4Head',
        num_classes=3,
        in_channels=[192, 384, 768]
    ),
    use_amp=True
)

adversary = dict(
        sensor='image',
        method='pgd',
        adv_loss='loss_bbox',
        alpha_img=0.5/255,
        alpha_pts=4/255,
        eps_img=2/255,
        eps_pts=0.05,
        norm='l_inf',
        iters=20,
        restrict_region=False,
        patch_size=0.5,
        random_keep=0.0) # At what chance will we keep the original input without perturbing it

train_cfg = dict()

test_cfg = dict(
    min_bbox_size=0,
    nms_pre=-1,
    score_thr=0.001,
    nms=dict(type='nms', iou_threshold=0.65),
    max_per_img=300)

dataset_type = 'WaymoOpenDataset'
data_root = 'data/waymococo_f0/'
img_norm_cfg = dict(
    mean=[114, 114, 114], std=[255, 255, 255], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    # dict(type='GtBBoxesFilter',
    #      min_size=2,
    #      max_aspect_ratio=20),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2020.json',
        img_prefix=data_root + 'train2020/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2020.json',
        img_prefix=data_root + 'val2020/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2020.json',
        img_prefix=data_root + 'val2020/',
        pipeline=test_pipeline))

nominal_batch_size = 64
gpus = 1
accumulate_interval = round(nominal_batch_size / (data['samples_per_gpu'] * gpus))

# optimizer
optimizer = dict(type='SGD', lr=0.01 * 1e-5, momentum=0.937, weight_decay=0.0005,
                 nesterov=True,
                 paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.))

optimizer_config = dict(
    type='AMPGradAccumulateOptimizerHook',
    accumulation=accumulate_interval,
    grad_clip=dict(max_norm=35, norm_type=2),
)

# learning policy
# base learning rate is in optimizer
# start from 1e-5, then to 0.2, then back to 1e-5
# The mmdet implementation uses ratio, so we cannot use 0 as start
lr_config = dict(
    policy='Cyclic',
    target_ratio=(1e5, 1),
    step_ratio_up=0.2  # Peak at 20% training progress
)
# lr_config = dict(
#     policy='CosineAnnealing',
#     min_lr_ratio=0.2,
# )

load_from = '/home/research/joss/universenet/work_dirs/yolov4m_waymo_mosaic/latest.pth'  # noqa
resume_from = None

custom_hooks = [
#     dict(
#         type='YoloV4WarmUpHook',
#         warmup_iters=1000,
#         lr_weight_warmup=0.,
#         lr_bias_warmup=0.1,
#         momentum_warmup=0.9,
#         priority='NORMAL'
#     ),
    dict(
        type='YOLOV4EMAHook',
        momentum=0.9999,
        interval=accumulate_interval,
        warm_up=10000 * accumulate_interval,
        resume_from=resume_from,
        priority='HIGH'
    )
]

total_epochs = 25

evaluation = dict(interval=1, metric='bbox')

checkpoint_config = dict(interval=1)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
# workflow = [('train', 1), ('val', 1)]
workflow = [('train', 1)]

cudnn_benchmark = True
