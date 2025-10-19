# DBNet + Swin-T + FPNC on ICDAR2015 (800 epochs, MMOCR 1.x)

_base_ = [
    '../_base_/datasets/icdar2015.py',   # <-- ICDAR2015 base
    '../_base_/default_runtime.py',
    '../dbnet/_base_dbnet_resnet18_fpnc.py',  # model defaults; we'll override below
]

# ----------------------
# Model (Swin-T via MMDetection registry)
# ----------------------
# keep your _base_ list as-is (icdar2015 + default_runtime + dbnet base)
model = dict(
    _delete_=True,  # replace the base model entirely
    type='DBNet',
    data_preprocessor=dict(
        type='TextDetDataPreprocessor',
        pad_size_divisor=32,
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
    ),
    backbone=dict(  # <-- no _delete_ here
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(
            type='Pretrained',
            #Load pretrained swin-t from openmmlab
            checkpoint=('https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth')
        ),
    ),
    neck=dict(  # <-- no _delete_ here
        type='FPNC',
        in_channels=[96, 192, 384, 768],
        lateral_channels=128,
        out_channels=128,
    ),
    det_head=dict(  # <-- no _delete_ here
        type='DBHead',
        in_channels=128,
        module_loss=dict(type='DBModuleLoss'),  # keep only one loss entry
        postprocessor=dict(
            type='DBPostprocessor',
            text_repr_type='quad',   # ICDAR2015
            mask_thr=0.3,
            min_text_score=0.3,
            min_text_width=5,
            max_candidates=2000,
            unclip_ratio=1.5,
        ),
    ),
)



# ----------------------
# Pipelines (preprocessor already normalizes -> no Normalize here)
# ----------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadOCRAnnotations', with_bbox=True, with_polygon=True, with_label=True),
    dict(type='RandomResize', scale=(1600, 960), ratio_range=(0.5, 2.0), keep_ratio=True),
    #dict(type='RandomRotate', max_angle=10, pad_with_fixed_color=True),
    dict(type='TextDetRandomCrop', target_size=(960, 960)),
    #dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Resize', scale=(960, 960), keep_ratio=False),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1920, 1152), keep_ratio=True),
    dict(type='LoadOCRAnnotations', with_bbox=True, with_polygon=True, with_label=True),
    dict(type='PackTextDetInputs'),
]

# ----------------------
# Dataset (ICDAR2015) – use base datasets, override pipelines
# ----------------------
icdar2015_textdet_train = _base_.icdar2015_textdet_train
icdar2015_textdet_train.pipeline = train_pipeline

icdar2015_textdet_test = _base_.icdar2015_textdet_test
icdar2015_textdet_test.pipeline = test_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=icdar2015_textdet_train,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=icdar2015_textdet_test,
)
val_cfg = dict(type='ValLoop')
val_evaluator = dict(type='HmeanIOUMetric')

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=icdar2015_textdet_test,
)
test_cfg = dict(type='TestLoop')
test_evaluator = dict(type='HmeanIOUMetric')

# ----------------------
# Optim & Schedule (800 epochs)
# ----------------------
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        })
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1500),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=0, end=800, T_max=800, eta_min=1e-6),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=800, val_interval=20)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=20, save_best='hmean-iou/hmean'),
    logger=dict(type='LoggerHook', interval=50),
)

randomness = dict(seed=None, deterministic=False)

# Optional: work_dir = './work_dirs/dbnet_swin_t_icdar2015'
