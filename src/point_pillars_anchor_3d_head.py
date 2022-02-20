import torch
from torch import nn
import numpy as np
from anchors.box_encoder import ResidualCoder
from anchors.anchor_3d_generator import Anchor3DRangeGenerator
from objdet_helper import multiclass_nms, limit_period, bbox_overlaps, box3d_to_bev2d


class PointPillarAnchor3DHead(nn.Module):
    def __init__(self,
                 num_classes=1,
                 feat_channels=384,
                 nms_pre=100,
                 score_thr=0.1,
                 dir_offset=0,
                 encode_angle_by_sin_cos=True,
                 point_cloud_ranges=[[0, -40.0, -3, 70.0, 40.0, 1]],
                 anchor_sizes=[[0.6, 1.0, 1.5]],
                 anchor_rotations=[0, 1.57],
                 iou_thresholds=[[0.35, 0.5]]):
        """
        构造函数

        :param num_classes:
        :param feat_channels:
        :param nms_pre:
        :param score_thr:
        :param dir_offset:
        :param encode_angle_by_sin_cos:
        :param point_cloud_ranges:
        :param anchor_sizes:
        :param anchor_rotations:
        :param iou_thresholds:
        """
        super().__init__()
        self._num_classes = num_classes
        self._feat_channels = feat_channels
        self._nms_pre = nms_pre
        self._score_thr = score_thr
        self._dir_offset = dir_offset
        self._encode_angle_by_sin_cos = encode_angle_by_sin_cos
        self._point_cloud_ranges = point_cloud_ranges
        self._anchor_sizes = anchor_sizes
        self._anchor_rotations = anchor_rotations
        self._iou_thresholds = iou_thresholds

        if len(self._iou_thresholds) != num_classes:
            assert len(self._iou_thresholds) == 1
            self._iou_thresholds = self._iou_thresholds * num_classes
        assert len(self._iou_thresholds) == num_classes

        # 创建anchor生成器
        self._anchor_generator = Anchor3DRangeGenerator(
            point_cloud_ranges=point_cloud_ranges,
            anchor_sizes=anchor_sizes,
            anchor_rotations=anchor_rotations
        )
        # 生成 anchors
        self._anchors = self._anchor_generator.generate_anchors()
        self._num_anchors_per_location = self._anchor_generator.num_anchors_per_location

        # 创建三维矩形框编码器
        self._bbox_coder = ResidualCoder(encode_angle_by_sin_cos=self._encode_angle_by_sin_cos)
        self._box_code_size = self._bbox_coder.box_code_size

        # 构造神经网络预测头
        self._conv_cls = nn.Conv2d(
            in_channels=self._feat_channels,
            out_channels=self._num_anchors_per_location * self._num_classes,
            kernel_size=1)

        self._conv_box = nn.Conv2d(
            in_channels=self._feat_channels,
            out_channels=self._num_anchors_per_location * self._box_code_size,
            kernel_size=1)

        self._conv_dir_cls = nn.Conv2d(
            in_channels=self._feat_channels,
            out_channels=self._num_anchors_per_location * len(self._anchor_rotations),
            kernel_size=1)

        self.init_weights()

    def init_weights(self):
        """
        初始化权重参数

        Returns:
        """
        pi = 0.01
        nn.init.constant_(self._conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self._conv_box.weight, mean=0.0, std=0.001)

    def forward(self, x):
        """
         前向推理函数

        :param x: torch.Tensor, 维度为 (batch_size, 6C, nx/2, ny/2), PillarPoint中 C = 64， nx = 432, ny = 496
        :return:
        """
        # 类别信息前向推理， 输出维度为 (batch_size, self._num_anchors_per_location * self._num_classes, nx/2, ny/2),
        # 默认为 (batch_size, 216, 248, 18)
        cls_preds = self._conv_cls(x)

        # 三维矩形框前向推理， 输出维度为 (batch_size, self._num_anchors_per_location * self._box_code_size, nx/2, ny/2),
        # 默认为 (batch_size, 216, 248, 42)
        box_preds = self._conv_box(x)

        # 三维矩形框方向前向推理， 输出维度为 (batch_size, elf._num_anchors_per_location * len(self._anchor_rotations),
        # nx/2, ny/2), 默认为 (batch_size, 216, 248, 12)
        dir_cls_preds = self._conv_dir_cls(x)

        # 将 channel 维度移动到最后一个维度，cls_preds, box_preds, dir_cls_preds 的维度分别为
        # (batch_size, nx/2, ny/2, self._num_anchors_per_location * self._num_classes),
        # (batch_size, nx/2, ny/2, self._num_anchors_per_location * self._box_code_size),
        # (batch_size, nx/2, ny/2, self._num_anchors_per_location * len(self._anchor_rotations)),
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()

        return cls_preds, box_preds, dir_cls_preds


if __name__ == "__main__":
    # 参考 https://github.com/open-mmlab/OpenPCDet/blob/master/tools/cfgs/kitti_models/pointpillar.yaml
    dense_head = PointPillarAnchor3DHead(
        num_classes=3,
        feat_channels=384,
        nms_pre=100,
        score_thr=0.1,
        dir_offset=0,
        point_cloud_ranges=[
            [0, -39.68, -1.78, 69.12, 39.68, -1.78],        # 车辆的点云范围
            [0, -39.68, -0.6, 69.12, 39.68, 0.6],           # 行人的点云范围
            [0, -39.68, -0.6, 69.12, 39.68, 0.6],           # 自行车的点云范围
        ],
        anchor_sizes=[
            [1.6, 3.9, 1.56],       # 车辆的anchor尺寸, w, l, h
            [0.6, 0.8, 1.73],       # 行人的anchor尺寸, w, l, h
            [0.6, 1.76, 1.73]       # 自行车的anchor尺寸, w, l, h
        ],
        anchor_rotations=[0, 1.57],
        iou_thresholds=[
            [0.45, 0.60],
            [0.35, 0.50],
            [0.35, 0.50],
        ]
    )

    print(dense_head)

    dense_head(torch.rand((4, 384, 216, 248)))

