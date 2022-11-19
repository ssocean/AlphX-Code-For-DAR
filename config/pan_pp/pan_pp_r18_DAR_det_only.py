# model = dict(
#     type='PAN_PP',
#     backbone=dict(
#         type='resnet18',
#         pretrained=True
#     ),
#     neck=dict(
#         type='FPEM_v2',
#         in_channels=(64, 128, 256, 512),
#         out_channels=128
#     ),
#     detection_head=dict(
#         type='PAN_PP_DetHead',
#         in_channels=512,
#         hidden_dim=128,
#         num_classes=6,
#         loss_text=dict(
#             type='DiceLoss',
#             loss_weight=1.0
#         ),
#         loss_kernel=dict(
#             type='DiceLoss',
#             loss_weight=0.5
#         ),
#         loss_emb=dict(
#             type='EmbLoss_v2',
#             feature_dim=4,
#             loss_weight=0.25
#         ),
#         use_coordconv=False,
#     )
# )
model = dict(
    type='PAN_PP',
    backbone=dict(
        type='resnet50',
        pretrained=True
    ),
    neck=dict(
        type='FPEM_v2',
        in_channels=(256, 512, 1024, 2048),
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
)
# model = dict(
#     type='PAN_PP',
#     backbone=dict(
#         type='resnet101',
#         pretrained=True
#     ),
#     neck=dict(
#         type='FPEM_v2',
#         in_channels=(256, 512, 1024, 2048),
#         out_channels=128
#     ),
#     detection_head=dict(
#         type='PAN_PP_DetHead',
#         in_channels=512,
#         hidden_dim=128,
#         num_classes=6,
#         loss_text=dict(
#             type='DiceLoss',
#             loss_weight=1.0
#         ),
#         loss_kernel=dict(
#             type='DiceLoss',
#             loss_weight=0.5
#         ),
#         loss_emb=dict(
#             type='EmbLoss_v2',
#             feature_dim=4,
#             loss_weight=0.25
#         ),
#         use_coordconv=False,
#     ),
# )
data = dict(
    batch_size=4,
    train=dict(
        type='PAN_PP_Joint_Train',
        split='train',
        is_transform=True,
        img_size=(896,896),
        short_size=896,
        kernel_scale=0.5,
        read_type='pil',
        with_rec=False
    ),
    test=dict(
        type='PAN_PP_COCO',
        split='train',
        is_transform=True,
        img_size=(896,896),
        short_size=896,
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
    # train=dict(
    #     type='PAN_PP_Joint_Train',
    #     split='test',
    #     is_transform=False,
    #     img_size=(896,896),
    #     short_size=896,
    #     kernel_scale=0.5,
    #     read_type='pil',
    #     with_rec=False
    # ),
)
train_cfg = dict(
    lr=1e-2,
    schedule='polylr',
    epoch=300,
    optimizer='Adam'
)
test_cfg = dict(
    min_score=0.85,
    min_area=80,
    min_kernel_area=2.6,
    scale=0.5,
    bbox_type='poly',#rect poly
    result_path='outputs/0915-txt',#'outputs/submit_ic15_rec.zip',
)
