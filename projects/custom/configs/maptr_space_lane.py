"""MapTR (v1, BEVFormer encoder) trained on the in-house SpaceLane dataset.

Why MapTR v1 instead of v2: the v2 LSS encoder needs lidar-derived depth
supervision; we have no lidar. v1's BEVFormer encoder consumes only images
and lidar2img matrices, which fits this rig directly.
"""

_base_ = [
    '../../configs/_base_/default_runtime.py',
]

# Make the registry pick up:
#   - SpaceLaneDataset  (projects.custom.datasets)
#   - ResizeMultiViewImageToFixed (projects.custom.pipelines)
#   - MapTR head / encoder / decoder / transforms (projects.mmdet3d_plugin)
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
custom_imports = dict(
    imports=['projects.custom'],
    allow_failed_imports=False,
)

# -------- problem dimensions --------
point_cloud_range = [-25.0, -25.0, -2.0, 25.0, 25.0, 2.0]
voxel_size = [0.25, 0.25, 4.0]

map_classes = [
    'car-no-go', 'other-no-go', 'column-no-go',
    'wall-no-go', 'outside-no-go',
]
num_map_classes = len(map_classes)

num_vec = 100                       # # of polyline queries
fixed_ptsnum_per_gt_line = 40       # matches converter
fixed_ptsnum_per_pred_line = 40
eval_use_same_gt_sample_num_flag = True

# Camera input target (after ResizeMultiViewImageToFixed)
input_h, input_w = 480, 800

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

# -------- model dims --------
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 200
bev_w_ = 200                         # square because our perception range is 50x50

queue_length = 1                     # single-frame, no temporal queue

# -------- model --------
model = dict(
    type='MapTR',
    use_grid_mask=True,
    video_test_mode=False,
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True,
    ),
    pts_bbox_head=dict(
        type='MapTRHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_vec=num_vec,
        num_pts_per_vec=fixed_ptsnum_per_pred_line,
        num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
        dir_interval=1,
        query_embed_type='instance_pts',
        transform_method='minmax',
        gt_shift_pts_pattern='v2',
        num_classes=num_map_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='MapTRPerceptionTransformer',
            # No temporal queue, no nuScenes can_bus
            rotate_prev_bev=False,
            use_shift=False,
            use_can_bus=False,
            num_cams=10,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=1,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1,
                        ),
                        dict(
                            type='GeometrySptialCrossAttention',
                            pc_range=point_cloud_range,
                            num_cams=10,
                            attention=dict(
                                type='GeometryKernelAttention',
                                embed_dims=_dim_,
                                num_heads=4,
                                dilation=1,
                                kernel_size=(3, 5),
                                num_levels=_num_levels_,
                            ),
                            embed_dims=_dim_,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'),
                ),
            ),
            decoder=dict(
                type='MapTRDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'),
                ),
            ),
        ),
        bbox_coder=dict(
            type='MapTRNMSFreeCoder',
            post_center_range=[-30, -30, -30, -30, 30, 30, 30, 30],
            pc_range=point_cloud_range,
            max_num=num_vec,
            voxel_size=voxel_size,
            num_classes=num_map_classes,
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_pts=dict(type='PtsL1Loss', loss_weight=5.0),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
    ),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='MapTRAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsL1Cost', weight=5),
            pc_range=point_cloud_range,
        ),
    )),
)

# -------- dataset --------
dataset_type = 'SpaceLaneDataset'
data_root = 'data/space_samples/'
ann_root = 'data/space_samples_processed/'

train_pipeline = [
    dict(type='LoadMultiViewImageFromFilesHeterogeneous', to_float32=True),
    dict(type='ResizeMultiViewImageToFixed', size=(input_h, input_w)),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D',
         class_names=map_classes,
         with_gt=False, with_label=False),
    dict(type='CustomCollect3D', keys=['img']),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesHeterogeneous', to_float32=True),
    dict(type='ResizeMultiViewImageToFixed', size=(input_h, input_w)),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(input_w, input_h),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='DefaultFormatBundle3D',
                 class_names=map_classes,
                 with_gt=False, with_label=False),
            dict(type='CustomCollect3D', keys=['img']),
        ],
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=ann_root + 'train.pkl',
        data_root=data_root,
        pipeline=train_pipeline,
        classes=map_classes,
        modality=input_modality,
        test_mode=False,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        box_type_3d='LiDAR',
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_root + 'val.pkl',
        data_root=data_root,
        pipeline=test_pipeline,
        classes=map_classes,
        modality=input_modality,
        test_mode=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        samples_per_gpu=1,
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_root + 'val.pkl',
        data_root=data_root,
        pipeline=test_pipeline,
        classes=map_classes,
        modality=input_modality,
        test_mode=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

# -------- optim --------
optimizer = dict(
    type='AdamW',
    lr=6e-4,
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 24
evaluation = dict(
    interval=2,
    pipeline=test_pipeline,
    metric='chamfer',
    save_best='SpaceLane_chamfer/mAP',
    rule='greater',
)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(max_keep_ckpts=2, interval=2)
fp16 = dict(loss_scale=512.)
find_unused_parameters = True
