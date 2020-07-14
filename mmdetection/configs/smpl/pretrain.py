# model settings
from mmdetection.mmdet.core.utils.smpl_tensorboard import SMPLBoard
import os.path as osp
from mmdetection.mmdet.core.utils.radam import RAdam
from mmdetection.mmdet.core.utils.lr_hooks import SequenceLrUpdaterHook, PowerLrUpdaterHook

model = dict(
    type='SMPLRCNN',
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        # Maybe I should change it to 1 to boost training, but its' better to leave it unchanged now.
        num_classes=2,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    smpl_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    smpl_head=dict(
        type='SMPLHead',
        in_size=14,
        in_channels=256,
        loss_cfg=dict(type='SMPLLoss', normalize_kpts=True),
    ),
    smpl_weight=1,
)
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        mask_size=28,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100,
        mask_thr_binary=0.5))
# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
square_bbox = False
common_train_cfg = dict(
    img_scale=(256, 256),
    img_norm_cfg=img_norm_cfg,
    size_divisor=32,
    flip_ratio=0.5,
    # noise_factor=1e-3,  # To avoid color jitter.
    with_mask=False,
    with_crowd=False,
    with_label=True,
    with_kpts2d=True,
    with_kpts3d=True,
    with_pose=True,
    with_shape=True,
    with_trans=True,
    # max_samples=1024
    square_bbox=square_bbox,
    # rot_factor=30,
)
common_val_cfg = dict(
    img_scale=(256, 256),
    img_norm_cfg=img_norm_cfg,
    size_divisor=32,
    flip_ratio=0,
    noise_factor=1e-3,  # To avoid color jitter.
    with_mask=False,
    with_crowd=False,
    with_label=True,
    with_kpts2d=True,
    with_kpts3d=True,
    with_pose=True,
    with_shape=True,
    with_trans=True,
    max_samples=64,
    square_bbox=square_bbox
)

# h36m_dataset_type = 'H36MDataset'
# h36m_data_root = 'data/h36m/'
# coco_dataset_type = 'COCOKeypoints'
# coco_data_root = 'data/coco/'

dataset_type = 'CommonDataset'
dataset_root = 'data/rcnn-pretrain/'

datasets = [
    dict(
        train=dict(
            type=dataset_type,
            ann_file=dataset_root + 'h36m/' + 'train.pkl',
            img_prefix=dataset_root + 'h36m/' + 'images/',
            sample_weight=0.6,
            **common_train_cfg
        ),
        val=dict(
            type=dataset_type,
            ann_file=dataset_root + 'h36m/' + 'val.pkl',
            img_prefix=dataset_root + 'h36m/' + 'images/',
            sample_weight=0.6,
            **common_val_cfg
        ),
    ),
    dict(
        train=dict(
            type=dataset_type,
            ann_file=dataset_root + 'coco/' + 'train.pkl',
            img_prefix=dataset_root + 'coco/' + 'images/',
            sample_weight=0.3,
            **common_train_cfg
        ),
    ),
    dict(
        train=dict(
            type=dataset_type,
            ann_file=dataset_root + 'lsp/' + 'train.pkl',
            img_prefix=dataset_root + 'lsp/' + 'images/',
            sample_weight=0.3,
            **common_train_cfg
        ),
    ),
    dict(
        train=dict(
            type=dataset_type,
            ann_file=dataset_root + 'mpii/' + 'train.pkl',
            img_prefix=dataset_root + 'mpii/' + 'images/',
            sample_weight=0.3,
            **common_train_cfg
        ),
    ),
    dict(
        train=dict(
            type=dataset_type,
            ann_file=dataset_root + 'mpi_inf_3dhp/' + 'train.pkl',
            img_prefix=dataset_root + 'mpi_inf_3dhp/' + 'images/',
            sample_weight=0.1,
            **common_train_cfg
        ),
    ),
]
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=8,
    train=common_train_cfg,
    val=common_val_cfg,
)
# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type=RAdam, lr=1e-4, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = SequenceLrUpdaterHook(
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    seq=[1e-4]
)
checkpoint_config = dict(interval=1)
# yapf:disable
# runtime settings
total_epochs = 12000
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/pretrain'
load_from = None
resume_from = osp.join(work_dir, 'latest.pth')
workflow = [('train', 1), ('val', 1)]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type=SMPLBoard, log_dir=work_dir, bboxes_only=False, K_SMALLEST=1,
             detail_mode=False)
    ])
# yapf:enable
evaluation = dict(interval=1)
fuse = True
time_limit = 1 * 3000  # In sceonds
