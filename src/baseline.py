# Lightweight DBNet + Swin-T + FPNC for arbitrary-shape text
# Dataset: Total-Text (train/test), 800 epochs

_base_ = [
    'mmocr::_base_/default_runtime.py',
    'mmocr::_base_/det_pipelines.py',
]

# ----------------------
# Model
# ----------------------
model = dict(
    type='DBNet',
    data_preprocessor=dict(
        type='TextDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True
    ),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=7, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3), patch_norm=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin-tiny_16xb64_in1k_20211123-1b40f1ed.pth'
        )
    ),
    neck=dict(
        type='FPNC',
        in_channels=[96, 192, 384, 768],
        lateral_channels=128,
        out_channels=128
    ),
    det_head=dict(
        type='DBHead',
        in_channels=128,
        loss_module=dict(
            type='DBLoss', balance_loss=True, main_loss_type='DiceLoss',
            loss_prob=dict(type='DiceLoss', loss_weight=1.0),
            loss_thresh=dict(type='L1Loss', loss_weight=10.0),
            eps=1e-6
        ),
        postprocessor=dict(
            type='DBPostprocessor',
            text_repr_type='poly',
            mask_thr=0.3,
            min_text_score=0.3,
            min_text_width=5,
            unclip_ratio=1.5,
            max_candidates=2000,
            box_thr=0.6
        )
    )
)

# ----------------------
# Pipelines (high-res + robust but simple augs)
# ----------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadOCRAnnotations', with_bbox=True, with_mask=True, with_label=False),
    dict(type='RandomResize', scale=(1600, 960), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomRotate', max_angle=10, pad_with_fixed_color=True),
    dict(type='TextDetRandomCrop', target_size=(960, 960)),
    dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='PackTextDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1920, 1152), keep_ratio=True),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='LoadOCRAnnotations', with_bbox=True, with_mask=True, with_label=False),
    dict(type='PackTextDetInputs')
]

# ----------------------
# Dataset (Total-Text)
# ----------------------
# Expected layout:
# data/totaltext/
#   images/train/*.jpg
#   images/test/*.jpg
#   annotations/train.json   # MMOCR TextDet JSON (polygons)
#   annotations/test.json
dataset_type = 'TextDetDataset'
data_root = 'data/totaltext/'

train_dataloader = dict(
    batch_size=4, num_workers=4, persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/train/'),
        ann_file='annotations/train.json',
        filter_cfg=dict(filter_empty_gt=True, min_size=8),
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/test/'),
        ann_file='annotations/test.json',
        pipeline=test_pipeline)
)
test_dataloader = val_dataloader

# Polygon-level H-mean (IoU-based) is standard for curved text sets
val_evaluator = dict(type='HmeanIOUMetric', metric='hmean-iou')
test_evaluator = val_evaluator

# ----------------------
# Optim & Schedule (800 epochs)
# ----------------------
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0)
        })
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1500),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=0, end=800, T_max=800, eta_min=1e-6)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=800, val_interval=20)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=20, save_best='hmean-iou/hmean'),
    logger=dict(type='LoggerHook', interval=50)
)

# Repro
randomness = dict(seed=None, deterministic=False)
