import torch
from torch import nn
import numpy as np
from anchors.box_encoder import ResidualCoder
from anchors.anchor_generator import AnchorGenerator
from common_utils import limit_period


class PointPillarAnchorHeadSingle(nn.Module):
    def __init__(self,
                 num_classes=1,
                 feat_channels=384,
                 nms_pre=100,
                 score_thr=0.1,
                 dir_offset=0,
                 encode_angle_by_sin_cos=True,
                 point_cloud_range=[0, -39.68, -3.0, 69.12, 39.68, 1.0],
                 anchor_sizes=[[0.6, 1.0, 1.5]],
                 anchor_rotations=[0, 1.57],
                 iou_thresholds=[[0.35, 0.5]]):
        super().__init__()
        self._num_classes = num_classes
        self._feat_channels = feat_channels
        self._nms_pre = nms_pre
        self._score_thr = score_thr
        self._dir_offset = dir_offset
        self._encode_angle_by_sin_cos = encode_angle_by_sin_cos
        self._point_cloud_range = point_cloud_range
        self._anchor_sizes = anchor_sizes
        self._anchor_rotations = anchor_rotations
        self._iou_thresholds = iou_thresholds

        if len(self._iou_thresholds) != num_classes:
            assert len(self._iou_thresholds) == 1
            self._iou_thresholds = self._iou_thresholds * num_classes
        assert len(self._iou_thresholds) == num_classes

        # 创建anchor生成器
        self._anchor_generator = AnchorGenerator(
            point_cloud_ranges=point_cloud_range,
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
            kernel_size=(1, 1))

        self._conv_box = nn.Conv2d(
            in_channels=self._feat_channels,
            out_channels=self._num_anchors_per_location * self._box_code_size,
            kernel_size=(1, 1))

        self._conv_dir_cls = nn.Conv2d(
            in_channels=self._feat_channels,
            out_channels=self._num_anchors_per_location * len(self._anchor_rotations),
            kernel_size=(1, 1))

        self.init_weights()

    @property
    def anchors(self):
        return self._anchors


    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def init_weights(self):
        """
        初始化权重参数

        Returns:
        """
        pi = 0.01
        nn.init.constant_(self._conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self._conv_box.weight, mean=0.0, std=0.001)

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors

        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        dir_offset = self.model_cfg.DIR_OFFSET
        dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
        dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
            else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
        dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

        period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
        dir_rot = limit_period(
            batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
        )
        batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        return batch_cls_preds, batch_box_preds

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
    dense_head = PointPillarAnchorHeadSingle(
        num_classes=3,
        feat_channels=384,
        nms_pre=100,
        score_thr=0.1,
        dir_offset=0,
        point_cloud_ranges=[0, -39.68, -3.0, 69.12, 39.68, 1.0],
        anchor_rotations=[[0, 1.5707963], [0, 1.5707963], [0, 1.5707963]],
        anchor_sizes=[
            [3.9, 1.6, 1.56],  # Car 类别的anchor尺寸, dx, dy, dz
            [0.8, 0.6, 1.73],  # Pedestrian 类别的anchor尺寸, dx, dy, dz
            [1.76, 0.6, 1.73]  # Cyclist 类别的anchor尺寸, dx, dy, dz
        ],
        anchor_bottom_heights=[
            [-1.78],  # Car 类别的 z_offset
            [-0.6],  # Pedestrian 类别的 z_offset
            [-0.6]  # Cyclist 类别的 z_offset
        ],
        align_center=True
    )

    print(dense_head)

    dense_head(torch.rand((4, 384, 216, 248)))

