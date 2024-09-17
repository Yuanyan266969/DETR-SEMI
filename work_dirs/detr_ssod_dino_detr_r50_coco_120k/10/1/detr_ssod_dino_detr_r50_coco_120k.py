dataset_type = 'CocoDataset'
data_root = '/root/paddlejob/workspace/env_run/output/temp/data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Sequential',
        transforms=[
            dict(
                type='RandResize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandFlip', flip_ratio=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Identity'),
                    dict(type='AutoContrast'),
                    dict(type='RandEqualize'),
                    dict(type='RandSolarize'),
                    dict(type='RandColor'),
                    dict(type='RandContrast'),
                    dict(type='RandBrightness'),
                    dict(type='RandSharpness'),
                    dict(type='RandPosterize')
                ])
        ],
        record=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=1),
    dict(type='ExtraAttrs', tag='sup'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor', 'tag'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        type='SemiDataset',
        sup=dict(
            type='CocoDataset',
            ann_file=
            'Split/coco_dataset_10/annotations/instances_train2017_10.json',
            img_prefix='Split/coco_dataset_10/train2017_10/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='Sequential',
                    transforms=[
                        dict(
                            type='RandResize',
                            img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                       (576, 1333), (608, 1333), (640, 1333),
                                       (672, 1333), (704, 1333), (736, 1333),
                                       (768, 1333), (800, 1333)],
                            multiscale_mode='value',
                            keep_ratio=True),
                        dict(type='RandFlip', flip_ratio=0.5),
                        dict(
                            type='OneOf',
                            transforms=[
                                dict(type='Identity'),
                                dict(type='AutoContrast'),
                                dict(type='RandEqualize'),
                                dict(type='RandSolarize'),
                                dict(type='RandColor'),
                                dict(type='RandContrast'),
                                dict(type='RandBrightness'),
                                dict(type='RandSharpness'),
                                dict(type='RandPosterize')
                            ])
                    ],
                    record=True),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=1),
                dict(type='ExtraAttrs', tag='sup'),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_bboxes', 'gt_labels'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'img_norm_cfg', 'pad_shape', 'scale_factor',
                               'tag'))
            ]),
        unsup=dict(
            type='CocoDataset',
            ann_file='coco_data/annotations/annotations_modified.json',
            img_prefix='Semi-DETR/coco_data/images_all/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='MultiBranch',
                    unsup_student=[
                        dict(
                            type='Sequential',
                            transforms=[
                                dict(
                                    type='RandResize',
                                    img_scale=[(480, 1333), (512, 1333),
                                               (544, 1333), (576, 1333),
                                               (608, 1333), (640, 1333),
                                               (672, 1333), (704, 1333),
                                               (736, 1333), (768, 1333),
                                               (800, 1333)],
                                    multiscale_mode='value',
                                    keep_ratio=True),
                                dict(type='RandFlip', flip_ratio=0.5),
                                dict(
                                    type='ShuffledSequential',
                                    transforms=[
                                        dict(
                                            type='OneOf',
                                            transforms=[
                                                dict(type='Identity'),
                                                dict(type='AutoContrast'),
                                                dict(type='RandEqualize'),
                                                dict(type='RandSolarize'),
                                                dict(type='RandColor'),
                                                dict(type='RandContrast'),
                                                dict(type='RandBrightness'),
                                                dict(type='RandSharpness'),
                                                dict(type='RandPosterize')
                                            ]),
                                        dict(
                                            type='OneOf',
                                            transforms=[{
                                                'type': 'RandTranslate',
                                                'x': (-0.1, 0.1)
                                            }, {
                                                'type': 'RandTranslate',
                                                'y': (-0.1, 0.1)
                                            }, {
                                                'type': 'RandRotate',
                                                'angle': (-30, 30)
                                            },
                                                        [{
                                                            'type':
                                                            'RandShear',
                                                            'x': (-30, 30)
                                                        }, {
                                                            'type':
                                                            'RandShear',
                                                            'y': (-30, 30)
                                                        }]])
                                    ]),
                                dict(
                                    type='RandErase',
                                    n_iterations=(1, 5),
                                    size=[0, 0.2],
                                    squared=True)
                            ],
                            record=True),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='Pad', size_divisor=1),
                        dict(type='ExtraAttrs', tag='unsup_student'),
                        dict(type='DefaultFormatBundle'),
                        dict(
                            type='Collect',
                            keys=['img', 'gt_bboxes', 'gt_labels'],
                            meta_keys=('filename', 'ori_shape', 'img_shape',
                                       'img_norm_cfg', 'pad_shape',
                                       'scale_factor', 'tag',
                                       'transform_matrix'))
                    ],
                    unsup_teacher=[
                        dict(
                            type='Sequential',
                            transforms=[
                                dict(
                                    type='RandResize',
                                    img_scale=[(480, 1333), (512, 1333),
                                               (544, 1333), (576, 1333),
                                               (608, 1333), (640, 1333),
                                               (672, 1333), (704, 1333),
                                               (736, 1333), (768, 1333),
                                               (800, 1333)],
                                    multiscale_mode='value',
                                    keep_ratio=True),
                                dict(type='RandFlip', flip_ratio=0.5)
                            ],
                            record=True),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='Pad', size_divisor=1),
                        dict(type='ExtraAttrs', tag='unsup_teacher'),
                        dict(type='DefaultFormatBundle'),
                        dict(
                            type='Collect',
                            keys=['img', 'gt_bboxes', 'gt_labels'],
                            meta_keys=('filename', 'ori_shape', 'img_shape',
                                       'img_norm_cfg', 'pad_shape',
                                       'scale_factor', 'tag',
                                       'transform_matrix'))
                    ])
            ],
            filter_empty_gt=False)),
    val=dict(
        type='CocoDataset',
        ann_file=
        '/root/paddlejob/workspace/env_run/output/temp/data/coco/annotations/instances_val2017.json',
        img_prefix=
        '/root/paddlejob/workspace/env_run/output/temp/data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=1),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file=
        '/root/paddlejob/workspace/env_run/output/temp/data/coco/annotations/instances_val2017.json',
        img_prefix=
        '/root/paddlejob/workspace/env_run/output/temp/data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=1),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    sampler=dict(
        train=dict(
            type='SemiBalanceSampler',
            sample_ratio=[1, 4],
            by_prob=True,
            epoch_length=7330)))
evaluation = dict(interval=4000, metric='bbox', type='SubModulesDistEvalHook')
checkpoint_config = dict(
    interval=4000, create_symlink=False, max_keep_ckpts=5, by_epoch=False)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='MeanTeacher', momentum=0.999, interval=1, warm_up=0),
    dict(type='StepRecord', normalize=False)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
mmdet_base = '../../thirdparty/mmdetection/configs/_base_'
model = dict(
    type='DinoDetrSSOD',
    model=dict(
        type='DINODETR',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet50')),
        bbox_head=dict(
            type='DINODETRSSODHead',
            num_query=900,
            query_dim=4,
            random_refpoints_xy=False,
            bbox_embed_diff_each_layer=False,
            num_classes=6,
            in_channels=2048,
            transformer=dict(type='DINOTransformer'),
            positional_encoding=dict(
                type='SinePositionalEncodingHW',
                temperatureH=20,
                temperatureW=20,
                num_feats=128,
                normalize=True),
            loss_cls1=dict(
                type='TaskAlignedFocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                loss_weight=2.0),
            loss_cls2=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
        train_cfg=dict(
            assigner1=dict(type='O2MAssigner'),
            assigner2=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(
                    type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)),
            warm_up_step=60000),
        test_cfg=dict(max_per_img=300, warm_up_step=60000)),
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.4,
        min_pseduo_box_size=0,
        unsup_weight=4.0,
        aug_query=False),
    test_cfg=dict(inference_on='student'))
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0))))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[120000, 160000])
runner = dict(type='IterBasedRunner', max_iters=120000)
strong_pipeline = [
    dict(
        type='Sequential',
        transforms=[
            dict(
                type='RandResize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandFlip', flip_ratio=0.5),
            dict(
                type='ShuffledSequential',
                transforms=[
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Identity'),
                            dict(type='AutoContrast'),
                            dict(type='RandEqualize'),
                            dict(type='RandSolarize'),
                            dict(type='RandColor'),
                            dict(type='RandContrast'),
                            dict(type='RandBrightness'),
                            dict(type='RandSharpness'),
                            dict(type='RandPosterize')
                        ]),
                    dict(
                        type='OneOf',
                        transforms=[{
                            'type': 'RandTranslate',
                            'x': (-0.1, 0.1)
                        }, {
                            'type': 'RandTranslate',
                            'y': (-0.1, 0.1)
                        }, {
                            'type': 'RandRotate',
                            'angle': (-30, 30)
                        },
                                    [{
                                        'type': 'RandShear',
                                        'x': (-30, 30)
                                    }, {
                                        'type': 'RandShear',
                                        'y': (-30, 30)
                                    }]])
                ]),
            dict(
                type='RandErase',
                n_iterations=(1, 5),
                size=[0, 0.2],
                squared=True)
        ],
        record=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=1),
    dict(type='ExtraAttrs', tag='unsup_student'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor', 'tag', 'transform_matrix'))
]
weak_pipeline = [
    dict(
        type='Sequential',
        transforms=[
            dict(
                type='RandResize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandFlip', flip_ratio=0.5)
        ],
        record=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=1),
    dict(type='ExtraAttrs', tag='unsup_teacher'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor', 'tag', 'transform_matrix'))
]
unsup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiBranch',
        unsup_student=[
            dict(
                type='Sequential',
                transforms=[
                    dict(
                        type='RandResize',
                        img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                   (576, 1333), (608, 1333), (640, 1333),
                                   (672, 1333), (704, 1333), (736, 1333),
                                   (768, 1333), (800, 1333)],
                        multiscale_mode='value',
                        keep_ratio=True),
                    dict(type='RandFlip', flip_ratio=0.5),
                    dict(
                        type='ShuffledSequential',
                        transforms=[
                            dict(
                                type='OneOf',
                                transforms=[
                                    dict(type='Identity'),
                                    dict(type='AutoContrast'),
                                    dict(type='RandEqualize'),
                                    dict(type='RandSolarize'),
                                    dict(type='RandColor'),
                                    dict(type='RandContrast'),
                                    dict(type='RandBrightness'),
                                    dict(type='RandSharpness'),
                                    dict(type='RandPosterize')
                                ]),
                            dict(
                                type='OneOf',
                                transforms=[{
                                    'type': 'RandTranslate',
                                    'x': (-0.1, 0.1)
                                }, {
                                    'type': 'RandTranslate',
                                    'y': (-0.1, 0.1)
                                }, {
                                    'type': 'RandRotate',
                                    'angle': (-30, 30)
                                },
                                            [{
                                                'type': 'RandShear',
                                                'x': (-30, 30)
                                            }, {
                                                'type': 'RandShear',
                                                'y': (-30, 30)
                                            }]])
                        ]),
                    dict(
                        type='RandErase',
                        n_iterations=(1, 5),
                        size=[0, 0.2],
                        squared=True)
                ],
                record=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=1),
            dict(type='ExtraAttrs', tag='unsup_student'),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels'],
                meta_keys=('filename', 'ori_shape', 'img_shape',
                           'img_norm_cfg', 'pad_shape', 'scale_factor', 'tag',
                           'transform_matrix'))
        ],
        unsup_teacher=[
            dict(
                type='Sequential',
                transforms=[
                    dict(
                        type='RandResize',
                        img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                   (576, 1333), (608, 1333), (640, 1333),
                                   (672, 1333), (704, 1333), (736, 1333),
                                   (768, 1333), (800, 1333)],
                        multiscale_mode='value',
                        keep_ratio=True),
                    dict(type='RandFlip', flip_ratio=0.5)
                ],
                record=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=1),
            dict(type='ExtraAttrs', tag='unsup_teacher'),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels'],
                meta_keys=('filename', 'ori_shape', 'img_shape',
                           'img_norm_cfg', 'pad_shape', 'scale_factor', 'tag',
                           'transform_matrix'))
        ])
]
fold = 1
percent = 10
work_dir = 'work_dirs/detr_ssod_dino_detr_r50_coco_120k/10/1/'
cfg_name = 'detr_ssod_dino_detr_r50_coco_120k'
gpu_ids = range(0, 2)
