import torch
from torch import nn
from torch.nn import functional as F


def get_paddings_indicator(actual_num, max_num, axis=0):
    """
    计算mask

    :param actual_num: torch.Tensor, 体素中实际有效的点数
    :param max_num: int, 体素化过程中每一个体素的最大点数
    :param axis: 默认为0
    :return: torch.Tensor, 维度为 (batch_size, max_num), 一个典型的输出为：
             tensor([[ True, True, False,  ..., False, False, False],
                    [ True, True, True,  ..., True, True, False],
                    [ True, True, True,  ..., True, False, False],
                    ...,
                    [ True, True, False,  ..., False, False, False],
                    [ True, False, False,  ..., False, False, False],
                    [ True, True, False,  ..., False, False, False]])
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num

    return paddings_indicator


class PFNLayer(nn.Module):
    """
    Pillar特征提取网络，如论文中所介绍的: "we use a simplified version of PointNet where, for each point, a linear
    layer is applied followed by BatchNorm and ReLu to generate a (C,P,N) sized tensor. This is followed
    by a max operation over the channels to create an output tensor of size (C,P)"
    """
    def __init__(self, in_channels=9, out_channels=64):
        """
        构造函数

        :param in_channels: 输入特征维度，默认为9
        :param out_channels: 输出特横维度，默认为64
        """
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._mode = "max"

        self._linear = nn.Linear(self._in_channels, self._out_channels, bias=False)
        self._norm = nn.BatchNorm1d(self._out_channels, eps=1e-3, momentum=0.01)

    def forward(self, inputs):
        """

        :param inputs: 输入体素（或称之为 pillar）特征，即论文中的9维特征，类型为torch.Tensor,
                       形状为(M, max_points_per_voxel, 9)
        :return:
        """
        # 全连接层，输出维度为 (M, max_points_per_voxel, 64)
        x = self._linear(inputs)

        # 执行Batch Normalization
        x = self._norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()

        x = F.relu(x)

        # 对channels（第二维）执行取最大化操作，输出x_max的维度为(M, 1, 64)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        return x_max


class PointPillarFeatureNet(nn.Module):
    """
    Pillar Feature Network的实现，具体架构图参考论文中的图2
    Lang, Alex H., et al. "PointPillars: Fast encoders for object detection from point clouds."
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
    """
    def __init__(self,
                 in_channels=4,
                 feat_channels=64,
                 voxel_size=None,
                 point_cloud_range=None):
        """
        构造函数

        :param in_channels: 输入点云的原始维度，默认为4，指的是 x, y, z, r (激光反射值).
        :param feat_channels: 每一个体素特征提取的输出维度，默认为64，对应论文中的 C 值
        :param voxel_size: pillar的 x 和 y 方向的分辨率尺寸. 默认值为 (0.16, 0.16)，单位为米
        :param point_cloud_range: 考虑的激光雷达坐标系下 x/y/z三个方向的点云范围, (x_min, y_min, z_min, x_max, y_max, z_max).
                                  默认值为(0, -39.68, -3, 69.12, 39.68, 1)，x方向为车辆的正前方，y方向为车辆朝向的左侧正方向，z
                                  方向垂直向上
        """
        super(PointPillarFeatureNet, self).__init__()
        if point_cloud_range is None:
            point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
        if voxel_size is None:
            voxel_size = [0.16, 0.16, 4]

        assert feat_channels > 0
        assert in_channels == 4

        # 正如论文中所阐述: "The points in each pillar are then decorated (augmented) with r, x_c, y_c,
        # z_c, x_p, y_p where r is reflectance, the c subscript denotes distance to the arithmetic mean of all
        # points in the pillar, and the p subscript denotes the offset from the pillar x,y center". 即 原始维度 4
        # 扩充为 9 维
        self._in_channels = in_channels + 5

        self._feat_channels = feat_channels
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range

        # 体素内点云集合的特征提取可以由一系列 PFNLayer 级连而成, PointPillars文章中仅仅使用了一个最简单的单一PFNLayer，即一个全
        # 连接层 + 一个BatchNorm层 + 一个Relu层 + 一个Max层
        self._pfn_layers = nn.ModuleList([
            PFNLayer(
                in_channels=self._in_channels,
                out_channels=self._feat_channels
            )
        ])

        # 每一个体素的物理距离，单位：米
        self._vx = self._voxel_size[0]      # 体素在x方向上的分辨率
        self._vy = self._voxel_size[1]      # 体素在y方向上的分辨率

        # x和y方向上体素的起始索引对应在激光雷达坐标系中的x/y坐标值，单位：米
        self._x_offset = self._vx / 2 + point_cloud_range[0]
        self._y_offset = self._vy / 2 + point_cloud_range[1]

    def forward(self, batch_voxels, batch_zyx_indices, batch_nums_per_voxel, batch_sample_indices):
        """
        前向推理函数

        :param batch_voxels: torch.Tensor, 形状为 (M, max_points_per_voxel, 4). 其中 M 为当前 batch 中点云样本进行体
                             素化后的总体素数量，例如 batch_size = 4时，四个样本的体素化后体素数量分别为 m1, m2, m3, m4.
                             则 M = m1 + m2 + m3 + m4. 4为原始点云的输入维度，分别表示 [x, y, z, intensity]
        :param batch_zyx_indices: 类型为torch.Tensor，形状为 (M, 3), 表示每一个体素（Pillar）对应的坐标索引，顺序为 z, y, x
        :param batch_nums_per_voxel: 类型为torch.Tensor, torch.Tensor, 形状为(M, )，表示每一个体素内的有效点云点数
        :param batch_sample_indices: torch.Tensor, 形状为(M, ), 表示当前体素属于 batch 中的哪一个，即索引
        :return:
        """
        features_ls = [batch_voxels]

        # 对于每一个体素，计算当前体素中各个点距离当前体素中所有点的算数平均值的距离，即论文中的 x_c, y_c, z_c 特征,
        # points_mean的维度大小为(M, 1, 3)
        points_mean = batch_voxels[:, :, :3].sum(dim=1, keepdim=True) / batch_nums_per_voxel.type_as(batch_voxels).view(-1, 1, 1)

        # f_cluster 的维度为 (M, max_points_per_voxel, 3)
        f_cluster = batch_voxels[:, :, :3] - points_mean
        features_ls.append(f_cluster)

        # 对于每一个体素，计算各个点距离体素中心点的距离特征，即论文中的 x_p, y_p 特征,
        # feats_offset的形状为 (M, max_points_per_voxel, 2)
        feats_offset = batch_voxels[:, :, :2].clone().detach()
        feats_offset[:, :, 0] = feats_offset[:, :, 0] - (
                batch_zyx_indices[:, 2].type_as(batch_voxels).unsqueeze(1) * self._vx + self._x_offset)
        feats_offset[:, :, 1] = feats_offset[:, :, 1] - (
                batch_zyx_indices[:, 1].type_as(batch_voxels).unsqueeze(1) * self._vy + self._y_offset)

        features_ls.append(feats_offset)

        # 合并上述的特征，即论文中的feature decorations, 执行完成之后 features 的维度为 (M, max_points_per_voxel, 9)
        features = torch.cat(features_ls, dim=-1)

        # 由于上述计算特征 x_c, y_c, z_c， x_p, y_p 时，没有区分体素化过程中补充的零，实际上这些补充的零在进行特征decoration之后
        # 仍然需要保持为零，所以这里需要将这些补充出来的点云特征重新设置为零
        max_points_per_voxel = features.shape[1]

        # mask 输出维度为 (M, max_points_per_voxel)
        mask = get_paddings_indicator(batch_nums_per_voxel, max_points_per_voxel, axis=0)

        # mask 在最后一维上扩充一维，输出形状为 (M, max_points_per_voxel, 1)
        mask = torch.unsqueeze(mask, -1).type_as(features)

        # 将这些补充出来的点云特征重新设置为零, features 维度为(M, max_points_per_voxel, 9)
        features *= mask

        # 体素特征提取
        for pfn in self._pfn_layers:
            features = pfn(features)

        # 经过 Pillar Feature Net 处理之后的特征维度为 (M, 64)
        features = features.squeeze(dim=1)
        return features
