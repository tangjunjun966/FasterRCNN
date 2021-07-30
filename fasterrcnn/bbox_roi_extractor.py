import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

class SingleRoIExtractor(nn.Module):
    """
    Extract RoI features from a every level feature map.
    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.
    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 roi_layer=None,
                 out_channels=None,
                 featmap_strides=None,
                 finest_scale=56
                 ):
        super(SingleRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale


    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        # for s in featmap_strides:


        # layer_roi = getattr(RoIAlign, cfg)

        roi_layers = nn.ModuleList([RoIAlign(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """
        Map rois to corresponding feature levels by scales.
        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt((rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1))  # w*h
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def roi_rescale(self, rois, scale_factor):
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1] + 1
        h = rois[:, 4] - rois[:, 2] + 1
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5 + 0.5
        x2 = cx + new_w * 0.5 - 0.5
        y1 = cy - new_h * 0.5 + 0.5
        y2 = cy + new_h * 0.5 - 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois


    def forward(self, feats, rois, roi_scale_factor=None):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)  # 将roi重新映射level
        roi_feats = feats[0].new_zeros(rois.size(0), self.out_channels, *out_size)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():  # array.any()是或操作，任意一个元素为True，输出为True。
                rois_ = rois[inds, :]  # 提取对应层的roi
                roi_feats_t = self.roi_layers[i](feats[i], rois_)  # roi特征
                roi_feats[inds] = roi_feats_t
        return roi_feats






class RoIAlign(nn.Module):
    def __init__(self,
                 out_size,  # 输出尺寸，一般设置为7
                 spatial_scale,  # feature map的下采样比率（对应感受野大小）
                 sample_num=2,  # 这个是roi align里面选的采样点数目，一般设置为２
                 ):
        super(RoIAlign, self).__init__()

        self.out_size = _pair(out_size)  # _pair 将一个数x重复变成元组(x,x)
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
    def forward(self, features, rois):

        from torchvision.ops import roi_align as tv_roi_align
        return tv_roi_align(features, rois, self.out_size,
                            self.spatial_scale, self.sample_num)
