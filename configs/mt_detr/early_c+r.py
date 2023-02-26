_base_='deformable_detr_twostage_refine_r50_16x2_50e_coco.py'

model = dict(
    backbone=dict(
        _delete_=True,
        type='EarlyFusion',
        bb_type='ConvNeXt',
        in_chans=6,
        args=dict(
            depths=[3, 3, 27, 3], 
            dims=[128, 256, 512, 1024], 
            drop_path_rate=0.7,
            layer_scale_init_value=1.0,
            out_indices=(1, 2, 3),
            pretrained='checkpoint/convnext_base_22k_1k_384.pth',
        ),
    ),
    

    neck=dict(
        _delete_=True,
        type='ChannelMapper',
        in_channels=[256, 512, 1024],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    

    bbox_head=dict(num_classes=2)
        
    )

prefix_list=["data/cam_stereo_left_lut/",
            "data/radar_projection"]


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53]
    ,std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromTwoFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromTwoFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# Modify dataset related settings
classes = ('Vehicle','Pedestrian')
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        img_prefix=prefix_list,
        classes=classes,
        ann_file="data/coco_annotation/train_clear_simple.json",
        pipeline=train_pipeline,
        ),
    val=dict(
        img_prefix=prefix_list,
        classes=classes,
        ann_file="data/coco_annotation/val_clear_simple.json",
        pipeline=test_pipeline,
        ),
    test=dict(
        img_prefix=prefix_list,
        classes=classes,
        ann_file="data/coco_annotation/test_clear_simple.json",
        pipeline=test_pipeline,
        ),
    )


optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW', 
                 lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.7,
                                'decay_type': 'layer_wise',
                                'num_layers': 12})
lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
