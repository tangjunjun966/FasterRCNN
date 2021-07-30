cfg = dict(

    model=dict(

        backbone=dict(
            pretrained='C:/Users/51102/Desktop/Fasterrcnn_tj/fasterrcnn/resnet50.pth'

        ),

        fpn=dict(
            in_channels=[256, 512, 1024, 2048],  # 输入的各个stage的通道数,用来匹配backbone
            out_channels=256,  # 输出的特征层的通道数
            num_outs=5,  # fpn返回特征金字塔的层数
            add_extra_convs=False,  # 取值为[False on_input on_lateral on_output]
            relu_before_extra_convs=False,

        ),


        rpn=dict(
            # rpn cls与box预测
            in_channels=256,
            num_classes=2,
            use_sigmoid_cls=True,
            feat_channels=256,
            num_anchors=3,# rpn_head中len(anchor_scales)*len(anchor_ratios)
            anchor_scales=None,
            anchor_ratios=None
        ),
        rpn_head=dict(
            # 对anchor进行选择
            rpnhead=dict(
                 num_classes=2,
                 anchor_scales=[4],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),

            ),
            assign=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.006,  # 0.8,  # ≥为正样本anchor
                neg_iou_thr=0.1,  # 可为tuple，表示负样本anchor在之间iou取
                min_pos_iou=0.2,  # 决定gt寻找最好anchor是否需要的阈值（>=）
                ignore_iof_thr=-1,
                gt_max_assign_all=False,  # 默认为True,控制gt满足
            ),
            sampling=True,  # 决定是否使用anchor的采样策略
            sampler=dict(
                type='RandomSampler',
                sampling=True,  # 决定是否使用anchor的采样策略
                num=256,  # 提取正负样本总个数
                pos_fraction=0.5,  # 正样本num中比列
                neg_pos_fraction=-1,  # 负样本个数占正样本比值，-1表示无比列
                add_gt_as_proposals=False
            ),
            rpn_loss=dict(
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)
            )
        ),
        bboxes_head=dict(
            # 对proposal进行选择
            assign=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.8,  # 0.8,  # ≥为正样本anchor
                neg_iou_thr=0.3,  # 可为tuple，表示负样本anchor在之间iou取
                min_pos_iou=0.3,  # 决定gt寻找最好anchor是否需要的阈值（>=）
                ignore_iof_thr=-1,
                gt_max_assign_all=True,  # 默认为True,控制gt满足
            ),
            # sampling=True,  # 决定是否使用anchor的采样策略

            sampler=dict(

                type='CombinedSampler',
                sampling=True,  # 决定是否使用anchor的采样策略
                num=512,  # 提取正负样本总个数
                pos_fraction=0.25,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(type='IoUBalancedNegSampler',
                                 floor_thr=-1,  # 卡控负样本阈值overlap
                                 floor_fraction=0,  # 占负样本比列的floot_thr样本
                                 num_bins=3  # 将0<iou<floor_thr分成num_bins块，平分相同负样本数量
                                 )
            ),

            bbox_roi_extractor=dict(
                # type='SingleRoIExtractor',  # 选择提取roi方法，RoIExtractor类型
                roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),  # ROI具体参数：ROI类型为ROIalign，输出尺寸为7，sample数为2
                out_channels=256,  # 输出通道数
                featmap_strides=[4, 8, 16, 32]  # 特征图的步长
            ),

            bbox_head=dict(
                type='ConvFCBBoxHead',  # 全连接层类型
                num_shared_fcs=2,  # 全连接层数量
                in_channels=256,  # 输入通道数
                fc_out_channels=1024,  # 1024,  # 输出通道数
                roi_feat_size=7,  # ROI特征层尺寸
                num_classes=3,  # 116,  # 分类器的类别数量+1，+1是因为多了一个背景的类别
                target_means=[0., 0., 0., 0.],  # 均值
                target_stds=[0.1, 0.1, 0.2, 0.2],  # 方差
                reg_class_agnostic=False,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                pos_weight=-1
            )
        ),

    ),

    optimizer_info=dict(
        optimizer=dict(lr=0.001, momentum=0.9, weight_decay=0.0001),  # 暂时用SGD
        optimizer_config=dict(type='Optimizerhook', grad_clip=dict(max_norm=35, norm_type=2)),  # 梯度均衡参数
        # learning policy
        lr_strategy=dict(
            type='Step',  # 优化策略
            gamma=0.1,
            warmup='linear',  # 初始的学习率增加的策略，linear为线性增加
            warmup_iters=500,  # 在初始的500次迭代中学习率逐渐增加
            warmup_ratio=1.0 / 3,  # 起始的学习率
            step=[8, 11])  # 在第8和11个epoch时降低学习率

    ),

    train_cfg=dict(

        rpn_proposal=dict(  # 每一层提取最大nms_pre个
            levels_nms=False,  # True 所有层提取proposal需要nms；False所有层按score选择
            nms_level_num=2000,  # 每一层提取最大个数
            nms_post=2000,
            max_total_num=2000,  # 将每层cat后提取数量
            nms_thr=0.8,  # nms中iou大于该值去除
            min_bbox_size=0,  # proposal 高宽必须大于的最小尺寸
        )

    ),

    test_cfg=dict(

        rpn_proposal=dict(  # 每一层提取最大nms_pre个
            levels_nms=False,  # True 所有层提取proposal需要nms；False所有层按score选择
            nms_level_num=1000,  # 每一层提取最大个数
            nms_post=1000,
            max_total_num=1000,  # 将每层cat后提取数量
            nms_thr=0.7,  # nms中iou大于该值去除
            min_bbox_size=0,  # proposal 高宽必须大于的最小尺寸
        ),
        rcnn=dict(score_thr=0.2, nms=dict(type='nms', nms_thr=0.4), max_per_img=50)
    ),

    data_cfg=dict(
        data_root=r'C:\Users\51102\Desktop\VOC2007\data_verity',
        data_loader=dict(batch_size=1,
                         num_workers=1,
                         shuffle=True,
                         drop_last=True, ),

        train_pipeline=[
            dict(type='LoadImageFromFile'),  # 载入图像
            dict(type='LoadAnnotations', with_bbox=True),  # 载入annotations

            dict(type='Resize', img_scale=(256, 128)),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='RandomFlip', direction='vertical', flip_ratio=0.5),

            # dict(type='RandomCropResize', crop_size=(200, 200), crop_ratio=1.1),

            # 加载数据处理模块#

            dict(type='Normalize',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='CategoryWeight', cat_weight={'A1CFB': 4.0}),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
                 img_meta_keys=['gt_bboxes', 'gt_labels', 'cat_weight']),
            # 在results中需要提取的结果
        ],
        test_pipeline=[
            dict(type='Resize', img_scale=(256, 128)),
            dict(type='Normalize'),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'], img_meta_keys=['scale'])

        ],

    ),

    check_point=dict(interval=10),
    total_epochs=300,  # 最大epoch数

    work_dir=r'C:\Users\51102\Desktop\VOC2007\work_dir'  # log文件和模型文件存储路径

)
