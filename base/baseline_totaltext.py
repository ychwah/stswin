default_scope = 'mmocr'
custom_imports = dict(imports=['mmocr'], allow_failed_imports=False)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=20, save_best='hmean-iou/hmean', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    visualization=dict(type='VisualizationHook', enable=False, interval=1, show=False, draw_gt=False, draw_pred=False),
)
env_cfg = dict(dist_cfg=dict(backend='nccl'), mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), cudnn_benchmark=False)
log_level = 'INFO'
log_processor = dict(type='LogProcessor', by_epoch=True, window_size=10)
launcher = 'none'
resume = False
work_dir = './work_dirs/v3_totaltext'

pack_meta = dict(
    type='mmocr.PackTextDetInputs',
    meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor')
)

data_root = 'data/totaltext'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmocr.LoadOCRAnnotations', with_bbox=True, with_label=True, with_polygon=True),
    dict(type='mmocr.FixInvalidPolygon'),
    dict(type='RandomResize', keep_ratio=True, ratio_range=(0.5, 2.0), scale=(1600, 960)),
    dict(type='mmocr.TextDetRandomCrop', target_size=(960, 960)),
    dict(type='TorchVisionWrapper', op='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2),
    dict(type='RandomFlip', direction='horizontal', prob=0.5),
    dict(type='Resize', keep_ratio=False, scale=(960, 960)),
    dict(type='mmocr.PackTextDetInputs', meta_keys=('img_path','ori_shape','img_shape','scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', keep_ratio=True, scale=(1920, 1152)),
    dict(type='mmocr.LoadOCRAnnotations', with_bbox=True, with_label=True, with_polygon=True),
    dict(type='mmocr.FixInvalidPolygon'),
    pack_meta,
]

train_dataset = dict(
    type='mmocr.OCRDataset',
    data_root=data_root,
    ann_file='textdet_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
)

test_dataset = dict(
    type='mmocr.OCRDataset',
    data_root=data_root,
    ann_file='textdet_test.json',
    test_mode=True,
    pipeline=test_pipeline,
)

train_dataloader = dict(
    batch_size=2, num_workers=2, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset,
)
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset,
)
test_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset,
)

val_evaluator = dict(type='mmocr.HmeanIOUMetric')
test_evaluator = dict(type='mmocr.HmeanIOUMetric')

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

model = dict(
    type='mmocr.DBNet',
    data_preprocessor=dict(
        type='mmocr.TextDetDataPreprocessor',
        bgr_to_rgb=True, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
        pad_size_divisor=32,
    ),
    backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=7, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.2,
        patch_norm=True, out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'),
    ),
    neck=dict(
        type='mmocr.FPNC',
        in_channels=[96, 192, 384, 768],
        out_channels=128,
        lateral_channels=128,
    ),
    det_head=dict(
        type='mmocr.DBHead',
        in_channels=512,
        module_loss=dict(type='mmocr.DBModuleLoss'),
        postprocessor=dict(
            type='mmocr.DBPostprocessor',
            text_repr_type='poly',  # curved text
            mask_thr=0.3, min_text_score=0.3, min_text_width=5,
            unclip_ratio=1.5, max_candidates=2000,
        ),
    ),
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    loss_scale='dynamic',
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
        )
    ),
)

param_scheduler = [
    dict(type='LinearLR', by_epoch=False, begin=0, end=1500, start_factor=0.001),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=0, end=800, T_max=800, eta_min=1e-6),
]

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='Visualizer', name='visualizer', vis_backends=vis_backends)
