import torch
from torch import nn
import numpy as np
from anchors.box_encoder import ResidualCoder
from anchors.anchor_3d_generator import Anchor3DRangeGenerator
from common_utils import limit_period


class PointPillarAnchorHeadSingle(nn.Module):
    def __init__(self,
                 input_channels,
                 point_cloud_range,
                 voxel_size,
                 num_dir_bins,
                 dir_offset,
                 dir_limit_offset,
                 anchor_generation_config,
                 use_multi_head=False
                 ):
        """

        :param input_channels:
        :param point_cloud_range:
        :param voxel_size:
        :param num_dir_bins:
        :param dir_offset:
        :param dir_limit_offset:
        :param anchor_generation_config:
        :param use_multi_head:
        """
        super().__init__()
        self._input_channels = input_channels
        self._point_cloud_range = point_cloud_range
        self._voxel_size = voxel_size   # x, y, z
        self._num_dir_bins = num_dir_bins
        self._dir_offset = dir_offset
        self._dir_limit_offset = dir_limit_offset
        self._anchor_generation_config = anchor_generation_config
        self._num_classes = len(self._anchor_generation_config)
        self._use_multi_head = use_multi_head

        # 生成网格（特征图）尺寸信息
        nx = (self._point_cloud_range[3] - self._point_cloud_range[0]) / self._voxel_size[0]
        ny = (self._point_cloud_range[4] - self._point_cloud_range[1]) / self._voxel_size[1]
        assert abs(float(int(nx)) - nx) <= 1e-5
        assert abs(float(int(ny)) - ny) <= 1e-5
        self._grid_size = torch.tensor([nx, ny])

        # 创建三维矩形框编码器
        self._bbox_coder = ResidualCoder(encode_angle_by_sin_cos=True)
        self._box_code_size = self._bbox_coder.box_code_size

        # 创建anchor生成器并生成anchors
        self._anchor_generator = Anchor3DRangeGenerator(
            point_cloud_range=self._point_cloud_range,
            anchor_generation_config=self._anchor_generation_config
        )
        anchors, self._num_anchors_per_location = self.generate_anchors(anchor_ndim=self._box_code_size)
        self._anchors = [x.cuda() for x in anchors]

        # 累加所有类别的所有anchors数量
        self._num_anchors_per_location = sum(self._num_anchors_per_location)

        # 构造神经网络预测头
        self._conv_cls = nn.Conv2d(
            in_channels=self._input_channels,
            out_channels=self._num_anchors_per_location * self._num_classes,
            kernel_size=(1, 1))

        self._conv_box = nn.Conv2d(
            in_channels=self._input_channels,
            out_channels=self._num_anchors_per_location * self._box_code_size,
            kernel_size=(1, 1))

        self._conv_dir_cls = nn.Conv2d(
            in_channels=self._input_channels,
            out_channels=self._num_anchors_per_location * self._num_dir_bins,
            kernel_size=(1, 1))

        self.init_weights()

    def generate_anchors(self, anchor_ndim=7):
        """
        生成anchors

        :param anchor_ndim: 每一个anchor的编码维度
        :return:
        """
        feature_map_size = [self._grid_size[:2] // config['feature_map_stride'] for config in self._anchor_generation_config]
        anchors_list, num_anchors_per_location_list = self._anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                # 在末尾的维度上padding零
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
        根据预测的residuals生成最终的 bounding boxes

        :param batch_size: batch 尺度
        :param cls_preds: 推理得到的 类别预测tensor, 形状为 [4, 216, 248, 18]
        :param box_preds: 推理得到的 box预测tensor, 形状为 [4, 216, 248, 48]
        :param dir_cls_preds: 推理得到的 方向角预测tensor, 形状为 [4, 216, 248, 12]
        :return:
            batch_cls_preds: (batch_size, num_boxes, num_classes)
            batch_box_preds: (batch_size, num_boxes, 7)
        """
        if isinstance(self._anchors, list):
            if self._use_multi_head:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for
                     anchor in self._anchors], dim=0)
            else:
                anchors = torch.cat(self._anchors, dim=-3)
        else:
            anchors = self._anchors

        # 至此，anchors.shape的形状为 [1, 248, 216, 3, 2, 8]
        # num_anchors = 248 * 216 * 3 * 2 = 321408
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]

        # batch_anchors形状为 [4, 321408, 8]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)

        # cls_preds 形状为 [4, 216, 248, 18]
        # batch_cls_preds 形状为 [4, 321408, 3], 321408 = 216 * 248 * 6
        batch_cls_preds = cls_preds.view(
            batch_size, num_anchors, -1).float() if not isinstance(cls_preds, list) else cls_preds

        # box_preds 形状为 [4, 216, 248, 48]
        # batch_box_preds 形状为 [4, 321408, 8], 321408 = 216 * 248 * 6
        batch_box_preds = box_preds.view(
            batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)

        # batch_box_preds解码后的形状为 [4, 321408, 7]
        batch_box_preds = self._bbox_coder.decode(anchors=batch_anchors, deltas=batch_box_preds)

        # dir_offset 为 0.78539，也即45度角
        dir_offset = self._dir_offset
        dir_limit_offset = self._dir_limit_offset

        # dir_cls_preds 形状为 [4, 216, 248, 12]
        # 执行下一行后 dir_cls_preds 形状为 [4, 321408, 2]
        dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
            else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
        dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

        # period = 3.141592653589793
        period = (2 * np.pi / self._num_dir_bins)

        # dir_rot.shape = torch.Size([4, 321408])
        dir_rot = limit_period(batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period)

        #
        batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        # batch_cls_preds.shape = torch.Size([4, 321408, 3])
        # batch_box_preds.shape = torch.Size([4, 321408, 7])
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
        input_channels=384,
        point_cloud_range=[0, -39.68, -3.0, 69.12, 39.68, 1.0],
        voxel_size=[0.16, 0.16, 4.0],
        num_dir_bins=2,
        dir_offset=0.78539,
        dir_limit_offset=0.0,
        anchor_generation_config=[
            {
                'class_name': 'Car',
                'anchor_sizes': [[3.9, 1.6, 1.56]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-1.78],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.6,
                'unmatched_threshold': 0.45
            },
            {
                'class_name': 'Pedestrian',
                'anchor_sizes': [[0.8, 0.6, 1.73]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.6],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            },
            {
                'class_name': 'Cyclist',
                'anchor_sizes': [[1.76, 0.6, 1.73]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.6],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            }
        ]
    )

    dense_head.cuda()
    print(dense_head)
    x = torch.rand((4, 384, 216, 248)).cuda()
    cls_preds, box_preds, dir_cls_preds = dense_head(x)
    print("cls_preds.shape = {}".format(cls_preds.shape))
    print("box_preds.shape = {}".format(box_preds.shape))
    print("dir_cls_preds.shape = {}".format(dir_cls_preds.shape))

    batch_cls_preds, batch_box_preds = dense_head.generate_predicted_boxes(
        batch_size=4,
        cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
    )
