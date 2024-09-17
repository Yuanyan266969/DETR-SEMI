_base_ = "base_dino_detr_ssod_coco.py"

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        sup=dict(
            type="CocoDataset",
            # ann_file="/root/paddlejob/workspace/env_run/output/temp/data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
            # img_prefix="/root/paddlejob/workspace/env_run/output/temp/data/coco/train2017/",
            ann_file="Split/coco_dataset_10/annotations/instances_train2017_10.json",
            img_prefix="Split/coco_dataset_10/train2017_10/",

        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="coco_data/annotations/annotations_modified.json",
            img_prefix="Semi-DETR/coco_data/images_all/",

        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)

semi_wrapper = dict(
    type="DinoDetrSSOD",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.4,
        min_pseduo_box_size=0,
        unsup_weight=4.0,
        aug_query=False,
        
    ),
    test_cfg=dict(inference_on="student"),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
    dict(type='StepRecord', normalize=False),
]

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=120000)

fold = 1
percent = 1

work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook')
    ],
)
