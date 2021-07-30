# cfg = dict(
#
#     model=dict(
#
#         rpn_head=dict(
#             # 对anchor进行选择
#             assign=dict(
#                 type='MaxIoUAssigner',
#                 pos_iou_thr=0.006,  # 0.8,  # ≥为正样本anchor
#                 neg_iou_thr=0.1,  # 可为tuple，表示负样本anchor在之间iou取
#                 min_pos_iou=0.2,  # 决定gt寻找最好anchor是否需要的阈值（>=）
#                 ignore_iof_thr=-1,
#                 gt_max_assign_all=False,  # 默认为True,控制gt满足
#             ),
#             sampling=True,  # 决定是否使用anchor的采样策略
#             sampler=dict(
#                 type='RandomSampler',
#                 sampling=True,  # 决定是否使用anchor的采样策略
#                 num=256,  # 提取正负样本总个数
#                 pos_fraction=0.5,  # 正样本num中比列
#                 neg_pos_fraction=-1,  # 负样本个数占正样本比值，-1表示无比列
#                 add_gt_as_proposals=False
#             ),
#             rpn_loss=dict(
#                 loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)
#             )
#         ),
#         bboxes_head=dict(
#             # 对proposal进行选择
#             assign=dict(
#                 type='MaxIoUAssigner',
#                 pos_iou_thr=0.00006,  # 0.8,  # ≥为正样本anchor
#                 neg_iou_thr=0.1,  # 可为tuple，表示负样本anchor在之间iou取
#                 min_pos_iou=0.2,  # 决定gt寻找最好anchor是否需要的阈值（>=）
#                 ignore_iof_thr=-1,
#                 gt_max_assign_all=False,  # 默认为True,控制gt满足
#             ),
#             sampling=True,  # 决定是否使用anchor的采样策略
#
#             sampler=dict(
#
#                 type='CombinedSampler',
#                 sampling=True,  # 决定是否使用anchor的采样策略
#                 num=512,  # 提取正负样本总个数
#                 pos_fraction=0.25,
#                 add_gt_as_proposals=False,
#                 pos_sampler=dict(type='InstanceBalancedPosSampler'),
#                 neg_sampler=dict(type='IoUBalancedNegSampler',
#                                  floor_thr=-1,  # 卡控负样本阈值overlap
#                                  floor_fraction=0,  # 占负样本比列的floot_thr样本
#                                  num_bins=3  # 将0<iou<floor_thr分成num_bins块，平分相同负样本数量
#                                  )
#             ),
#
#             bbox_roi_extractor=dict(
#                 # type='SingleRoIExtractor',  # 选择提取roi方法，RoIExtractor类型
#                 roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),  # ROI具体参数：ROI类型为ROIalign，输出尺寸为7，sample数为2
#                 out_channels=256,  # 输出通道数
#                 featmap_strides=[4, 8, 16, 32]),  # 特征图的步长
#         )
#
#     ),
#
#     train_cfg=dict(
#         rpn_proposal=dict(  # 每一层提取最大nms_pre个
#             levels_nms=False,  # True 所有层提取proposal需要nms；False所有层按score选择
#             nms_level_num=2000,  # 每一层提取最大个数
#             nms_post=2000,
#             max_total_num=2000,  # 将每层cat后提取数量
#             nms_thr=0.8,  # nms中iou大于该值去除
#             min_bbox_size=0,  # proposal 高宽必须大于的最小尺寸
#         ),
#
#
#     )
#
# )

import torch.nn as nn
import torch
from fasterrcnn.resnet import Resnet
from fasterrcnn.fpn import FPN
from fasterrcnn.rpn import RPN
from fasterrcnn.rpn_head import RPNHead
from fasterrcnn.bbox_roi_extractor import SingleRoIExtractor
from fasterrcnn.bboxes_head import ConvFCBBoxHead
from fasterrcnn.utils.anchortarget import AssignSampler
from fasterrcnn.utils.boxroi import bbox2roi


class FasterRCNN(nn.Module):
    def __init__(self, cfg):
        super(FasterRCNN, self).__init__()
        self.cfg = cfg
        backbone_cfg = self.cfg['model']['backbone']
        self.backbone = Resnet(**backbone_cfg)
        fpn_cfg = self.cfg['model']['fpn']
        self.fpn = FPN(**fpn_cfg)
        rpn_cfg = self.cfg['model']['rpn']
        self.rpn = RPN(**rpn_cfg)
        rpnhead_cfg = self.cfg['model']['rpn_head']['rpnhead']
        self.rpn_head = RPNHead(**rpnhead_cfg,rpn_loss=self.cfg['model']['rpn_head']['rpn_loss'])
        roi_extractor_cfg = self.cfg['model']['bboxes_head']['bbox_roi_extractor']
        self.roi_extractor = SingleRoIExtractor(**roi_extractor_cfg)
        bbox_head_cfg = self.cfg['model']['bboxes_head']['bbox_head']
        self.bbox_head = ConvFCBBoxHead(**bbox_head_cfg)

    def forward_train(self, data, img_metas):

        img = data['img']
        x = self.backbone(img)
        x = self.fpn(x)

        rpn_outs = self.rpn(x)  # rpn_outs=rpn_cls_score, rpn_bbox_pred
        rpn_cls_score, rpn_bbox_pred = rpn_outs
        out_rpn_loss = self.rpn_head.loss(rpn_cls_score,  # [batch,anchor*4,h,w]
                                          rpn_bbox_pred,  # [batch,anchor*4,h,w]
                                          img_metas,
                                          assign_cfg=self.cfg['model']['rpn_head']['assign'],
                                          sampler_cfg=self.cfg['model']['rpn_head']['sampler']
                                          )

        cfg_rpn_proposal = self.cfg['train_cfg'].get('rpn_proposal')
        proposal_inputs = rpn_outs + (img_metas, cfg_rpn_proposal)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

        # bboxes

        assignsampler = AssignSampler()
        assign_cfg = self.cfg['model']['bboxes_head']['assign']
        sampler_cfg = self.cfg['model']['bboxes_head']['sampler']
        gt_bboxes = [info.get('gt_bboxes', None) for info in img_metas]
        gt_labels = [info.get('gt_labels', None) for info in img_metas]

        num_imgs = img.size(0)
        sampling_results = []
        for i in range(num_imgs):
            assigned_result = assignsampler.assign(assign_cfg, proposal_list[i], gt_bboxes[i],
                                                   gt_labels[i])  # assigned_gt_inds,max_overlaps
            assigned_gt_inds, max_overlaps, assigned_gt_labels = \
                assigned_result['assigned_gt_inds'], assigned_result['max_overlaps'], assigned_result['assigned_labels']

            if sampler_cfg['sampling']:
                sampler_result = assignsampler.sample(sampler_cfg, assigned_gt_inds, max_overlaps,
                                                      proposal_list[i], gt_bboxes[i], gt_labels[i], assigned_gt_labels)



                pos_bboxes, neg_bboxes = sampler_result['pos_bboxes'], sampler_result['neg_bboxes']


                bboxes = torch.cat([pos_bboxes, neg_bboxes])

                sampling_result = sampler_result
                sampling_result['bboxes_roi'] = bboxes
                sampling_result['pos_bboxes'] = pos_bboxes
                sampling_result['neg_bboxes'] = neg_bboxes

                sampling_results.append(sampling_result)

        # with bboxes

        rois = bbox2roi([res['bboxes_roi'] for res in sampling_results])

        bbox_feats = self.roi_extractor(x[:self.roi_extractor.num_inputs], rois)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        bbox_head_cfg = self.cfg['model']['bboxes_head']['bbox_head']
        bbox_targets = self.bbox_head.get_target(sampling_results,  bbox_head_cfg)
        out_head_loss = self.bbox_head.loss(cls_score, bbox_pred, *bbox_targets)

        losses = dict()
        losses.update(out_rpn_loss)
        losses.update(out_head_loss)

        return losses

    def forward_test(self, data, img_metas):
        x = self.backbone(data)
        x = self.fpn(x)
        rpn_outs = self.rpn(x)
        # rpn_cls_score, rpn_bbox_pred = rpn_outs
        cfg_rpn_proposal = self.cfg['test_cfg'].get('rpn_proposal')
        proposal_inputs = rpn_outs + (img_metas, cfg_rpn_proposal)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        rois = bbox2roi(proposal_list)  # 实际还是proposal_list

        bbox_feats = self.roi_extractor(x[:self.roi_extractor.num_inputs], rois)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        img_shape = img_metas[0]['img_shape']
        scale = img_metas[0]['scale']
        ori_shape = img_metas[0]['ori_shape']
        h_factor, w_factor = ori_shape[0] / scale[0], ori_shape[1] / scale[1]
        scale_factor = [w_factor, h_factor, w_factor, h_factor]

        cfg_rcnn = self.cfg['test_cfg'].get('rcnn')

        pre_bboxes, pre_labels = self.bbox_head.get_pre_bboxes(
            rois,  # rpn 通过预测dx,dy,dw,dh与对应anchor得到rois
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=True,
            cfg=cfg_rcnn
        )
        results = self.bbox_head.convert_pre2result(pre_bboxes, pre_labels, self.bbox_head.num_classes)

        return results






    def forward(self, data, img_metas=None, mode='train'):
        result = None
        if mode == 'train':
            losses = self.forward_train(data, img_metas)
            result = losses
        elif mode == 'test':
            img = data.unsqueeze(0)
            result = self.forward_test(img, img_metas)

        return result
