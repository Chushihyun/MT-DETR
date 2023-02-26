_base_='deformable_detr_twostage_refine_r50_16x2_50e_coco.py'

model = dict(
    backbone=dict(
        _delete_=True,
        type='CameraOnly',
        net1='ConvNeXt',
        net2='ResNet',
        net3='ResNet',
        args1=dict(
            in_chans=3,
            depths=[3, 3, 27, 3], 
            dims=[128, 256, 512, 1024], 
            drop_path_rate=0.7,
            layer_scale_init_value=1.0,
            out_indices=(1, 2, 3),
            pretrained='checkpoint/convnext_base_22k_1k_384.pth',
            ),
        args2=dict(
            depth=50,
            num_stages=4,
            base_channels=1, # 32
            out_indices=(1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
            ),
        args3=dict(
            depth=50,
            num_stages=4,
            base_channels=1, # 32
            out_indices=(1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
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


# Modify dataset related settings
classes = ('Vehicle','Pedestrian')
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        img_prefix="data/cam_stereo_left_lut/",
        classes=classes,
        ann_file="data/coco_annotation/train_clear_simple.json"),
    val=dict(
        img_prefix="data/cam_stereo_left_lut/",
        classes=classes,
        ann_file="data/coco_annotation/val_clear_simple.json"),
    test=dict(
        img_prefix="data/cam_stereo_left_lut/",
        classes=classes,
        ann_file="data/coco_annotation/test_clear_simple.json"))


optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW', 
                 lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.7,
                                'decay_type': 'layer_wise_multi',
                                'num_layers': 12})
lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)