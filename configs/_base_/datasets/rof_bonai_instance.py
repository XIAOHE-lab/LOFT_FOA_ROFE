dataset_type = "BONAI"
data_root = "data/BONAI/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="LoadAnnotations",
        with_bbox=True,
        with_mask=True,
        with_offset=True,
        with_footprint_mask=True,
    ),
    dict(type="Resize", img_scale=(1024, 1024), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5, direction=["horizontal", "vertical"]),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="FootprintFormatBundle"),
    dict(
        type="Collect",
        keys=[
            "img",
            "gt_bboxes",
            "gt_labels",
            "gt_masks",
            "gt_offsets",
            "gt_footprint_masks",
        ],
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

cities = ["shanghai", "beijing", "jinan", "haerbin", "chengdu"]
train_ann_file = [
    f"{data_root}coco/bonai_{city}_trainval.json" for city in cities
]
img_prefix = [f"{data_root}trainval/images/" for _ in cities]

data = dict(
    train=dict(
        type=dataset_type,
        ann_file=train_ann_file,
        img_prefix=img_prefix,
        bbox_type="building",
        mask_type="roof",
        pipeline=train_pipeline,
    ),
    train_dataloader=dict(
        samples_per_gpu=2, 
        workers_per_gpu=2
    ),
    val=dict(
        type=dataset_type,
        ann_file=train_ann_file,
        img_prefix=img_prefix,
        bbox_type="building",
        mask_type="roof",
        gt_footprint_csv_file="",
        pipeline=test_pipeline,
    ),
    val_dataloader=dict(
        samples_per_gpu=1, 
        workers_per_gpu=2
    ),
    test=dict(
        type=dataset_type,
        ann_file=train_ann_file,
        img_prefix=img_prefix,
        bbox_type="building",
        mask_type="roof",
        gt_footprint_csv_file="",
        pipeline=test_pipeline,
    ),
    test_dataloader=dict(
        samples_per_gpu=1, 
        workers_per_gpu=2
    ),
)

evaluation = dict(
    interval=1, 
    metric=["segm"], 
    mask_type="direct_footprint"
)
