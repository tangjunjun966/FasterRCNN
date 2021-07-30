import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from fasterrcnn.utils.losses import loss_dict
from fasterrcnn.utils.boxroi import bbox_target
from fasterrcnn.utils.anchortarget import delta2bbox
from fasterrcnn.utils.util import nms
import numpy as np


class ConvFCBBoxHead(nn.Module):

    # noqa: W605
    def __init__(self,
                 num_shared_fcs=2,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 with_cls=True,
                 with_reg=True,
                 with_avg_pool=False,
                 num_classes=81,
                 reg_class_agnostic=False,
                 target_means=None,  # 均值
                 target_stds=None,  # 方差
                 loss_cls=None,
                 loss_bbox=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__()
        assert (num_shared_fcs + num_cls_convs + num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0: assert num_shared_fcs == 0
        if not with_cls: assert num_cls_convs == 0 and num_cls_fcs == 0
        if not with_reg: assert num_reg_convs == 0 and num_reg_fcs == 0

        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.with_avg_pool = with_avg_pool  # 是否平均池化
        self.with_cls, self.with_reg = with_cls, with_reg
        self.target_means, self.target_stds = target_means, target_stds
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic

        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        # 以下是对roi采用共享fc或共享卷积等选择
        # add shared convs and fcs
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        self.shared_fcs, last_layer_dim = self._add_conv_fc_branch(self.num_shared_fcs, self.in_channels, True)
        self.shared_out_channels = last_layer_dim  # 贡献输出通道数

        # # add cls specific branch
        self.cls_fcs, self.cls_last_dim = self._add_conv_fc_branch(self.num_cls_fcs, self.shared_out_channels)
        #
        # add reg specific branch
        self.reg_fcs, self.reg_last_dim = self._add_conv_fc_branch(self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area  # cls输出维度
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)  # 预测class
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 * self.num_classes)  # -->box回归
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)  # 预测box位置

            # loss 进行类的实列化
        self.loss_cls, self.loss_bbox = self.choose_inite_loss(loss_cls, loss_bbox)

    def choose_inite_loss(self, loss_cls, loss_bbox):
        # 选择rpn_loss使用什么方法及对类进行初始化
        loss_cls_new = loss_dict[loss_cls.pop('type')](**loss_cls)  # CrossEntropyLoss
        loss_bbox_new = loss_dict[loss_bbox.pop('type')](**loss_bbox)  # SmoothL1Loss
        return loss_cls_new, loss_bbox_new

    def _add_conv_fc_branch(self, num_branch_fcs, in_channels, is_shared=False):
        """
        Add shared or separable branch
        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers

        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:  # 增加全连接模块
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area  # 得到fc输出维度，即神经元
            for i in range(num_branch_fcs):  # 添加2个线性模块 7*7*256-->1024
                fc_in_channels = (last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHead, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.num_shared_fcs > 0:
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:  # 调用2次共享liner
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

    def get_target(self, sampling_results,  rcnn_train_cfg):
        # 通过sampling_results中选择出已经挑选好的box，label等
        pos_proposals = [res['pos_bboxes'] for res in sampling_results]  # sampling_results['pos_bboxes']
        # [res['neg_bboxes'] for res in sampling_results]
        neg_proposals = [res['neg_bboxes'] for res in sampling_results]
        pos_gt_bboxes = [res['pos_gt_bboxes'] for res in sampling_results]
        pos_gt_labels = [res['pos_gt_labels'] for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets  # labels, label_weights, bbox_targets, bbox_weights

    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc'] = self.accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        return losses

    def accuracy(self, pred, target, topk=1):
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk,)
            return_single = True
        else:
            return_single = False

        maxk = max(topk)
        _, pred_label = pred.topk(maxk, dim=1)
        pred_label = pred_label.t()
        correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / pred.size(0)))
        return res[0] if return_single else res

    def get_pre_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            assert scale_factor is not None, 'lacking scale_factor=[w_factor, h_factor, w_factor, h_factor] '
            scale_factor = torch.tensor(scale_factor).to(bboxes.device)
            bboxes = bboxes.view(bboxes.size(0), -1, 4)
            bboxes = (bboxes * scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = self.multiclass_nms(bboxes, scores,
                                                         cfg['score_thr'], cfg['nms'],
                                                         cfg['max_per_img'])

            return det_bboxes, det_labels

    def multiclass_nms(self, multi_bboxes,
                       multi_scores,
                       score_thr,
                       nms_cfg,
                       max_num=-1,
                       score_factors=None):
        """NMS for multi-class bboxes.

        Args:
            multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
            multi_scores (Tensor): shape (n, #class), where the 0th column
                contains scores of the background class, but this will be ignored.
            score_thr (float): bbox threshold, bboxes with scores lower than it
                will not be considered.
            nms_thr (float): NMS IoU threshold
            max_num (int): if there are more than max_num bboxes after NMS,
                only top max_num will be kept.
            score_factors (Tensor): The factors multiplied to scores before
                applying NMS

        Returns:
            tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
                are 0-based.
        """
        num_classes = multi_scores.shape[1]
        bboxes, labels = [], []
        nms_cfg_ = nms_cfg.copy()
        nms_type = nms_cfg_.pop('type', 'nms')  # 暂时只有nms
        nms_thr = nms_cfg.get('nms_thr', 0.5)
        for i in range(1, num_classes):
            cls_inds = multi_scores[:, i] > score_thr
            if not cls_inds.any():
                continue
            # get bboxes and scores of this class
            if multi_bboxes.shape[1] == 4:  # 表示只有一个类
                _bboxes = multi_bboxes[cls_inds, :]
            else:
                _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
            _scores = multi_scores[cls_inds, i]
            if score_factors is not None:
                _scores *= score_factors[cls_inds]
            cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)

            cls_dets = nms(cls_dets, nms_thr)

            cls_labels = multi_bboxes.new_full((cls_dets.shape[0],),
                                               i - 1,
                                               dtype=torch.long)
            bboxes.append(cls_dets)
            labels.append(cls_labels)
        if bboxes:
            bboxes = torch.cat(bboxes)
            labels = torch.cat(labels)
            if bboxes.shape[0] > max_num:
                _, inds = bboxes[:, -1].sort(descending=True)
                inds = inds[:max_num]
                bboxes = bboxes[inds]
                labels = labels[inds]
        else:
            bboxes = multi_bboxes.new_zeros((0, 5))
            labels = multi_bboxes.new_zeros((0,), dtype=torch.long)

        return bboxes, labels

    def convert_pre2result(self, bboxes, labels, num_classes=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (Tensor): shape (n, 5)
            labels (Tensor): shape (n, )
            num_classes (int): class number, including background class

        Returns:
            list(ndarray): bbox results of each class
        """

        if bboxes.shape[0] == 0:
            return [
                np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
            ]
        else:
            bboxes = bboxes.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            return [bboxes[labels == i, :] for i in range(num_classes - 1)]
