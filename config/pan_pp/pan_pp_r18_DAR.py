model = dict(
    type='PAN_PP',
    backbone=dict(
        type='resnet18',
        pretrained=True
    ),
    neck=dict(
        type='FPEM_v2',
        in_channels=(64, 128, 256, 512),
        out_channels=128
    ),
    detection_head=dict(
        type='PAN_PP_DetHead',
        in_channels=512,
        hidden_dim=128,
        num_classes=6,
        loss_text=dict(
            type='DiceLoss',
            loss_weight=1.0
        ),
        loss_kernel=dict(
            type='DiceLoss',
            loss_weight=0.5
        ),
        loss_emb=dict(
            type='EmbLoss_v2',
            feature_dim=4,
            loss_weight=0.25
        ),
        use_coordconv=False,
    ),
    recognition_head=dict(
        type='PAN_PP_RecHead',
        input_dim=512,
        hidden_dim=128,
        feature_size=(32, 8)#(8, 32)
    )
)
data = dict(
    batch_size=1,
    train=dict(
        type='PAN_PP_Joint_Train',
        split='train',
        is_transform=True,
        img_size=(736,736),
        short_size=736,
        kernel_scale=0.5,
        read_type='pil',
        with_rec=True
    ),
    test=dict(
        type='PAN_PP_COCO',
        split='train',
        is_transform=True,
        img_size=(736,736),
        short_size=736,
        read_type='pil',
        with_rec=True
    ),
    # dict(
    #     type='PAN_PP_IC15',
    #     split='test',
    #     short_size=720,
    #     read_type='pil',
    #     with_rec=True
    # )
)
train_cfg = dict(
    lr=1e-2,
    schedule='polylr',
    epoch=1000,
    optimizer='Adam'
)
test_cfg = dict(
    min_score=0.2,
    min_area=1,
    min_kernel_area=0.1,
    scale=0.5,
    bbox_type='rect',#rect poly
    result_path='outputs/all_full_180ep',#'outputs/submit_ic15_rec.zip',
    rec_post_process=dict(
        len_thres=1,
        score_thres=0.95,
        unalpha_score_thres=0.9,
        ignore_score_thres=0.93,
        edit_dist_thres=2,
        voc_type=None,
        voc_path=None
    )
)
