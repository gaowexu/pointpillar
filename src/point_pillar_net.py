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
             tensor([[ True, False, False,  ..., False, False, False],
                    [ True, False, False,  ..., False, False, False],
                    [ True, False, False,  ..., False, False, False],
                    ...,
                    [ True, False, False,  ..., False, False, False],
                    [ True, False, False,  ..., False, False, False],
                    [ True, False, False,  ..., False, False, False]])
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num

    # paddings_indicator的维度为 (batch_size, max_num)
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
                       形状为(U, max_num_points, 9)
        :return:
        """
        # 全连接层，输出维度为 （U, max_num_points, 64)
        x = self._linear(inputs)

        # 执行Batch Normalization
        x = self._norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()

        x = F.relu(x)

        # 对channels（第二维）执行取最大化操作，输出x_max的维度为（U, 1, 64)
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
                 bin_size=(0.16, 0.16),
                 point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1)):
        """
        构造函数

        :param in_channels: 输入点云的原始维度，默认为4，指的是 x, y, z, r (激光反射值).
        :param feat_channels: 每一个体素特征提取的输出维度，默认为64，对应论文中的 C 值
        :param bin_size: pillar的 x 和 y 方向的分辨率尺寸. 默认值为 (0.16, 0.16)，单位为米
        :param point_cloud_range: 考虑的Velodyne坐标系下 x/y/z三个方向的点云范围, (x_min, y_min, z_min, x_max, y_max, z_max).
                                  默认值为(0, -39.68, -3, 69.12, 39.68, 1)，x方向为车辆的正前方，y方向为车辆朝向的左侧正方向，z
                                  方向垂直向上
        """
        super(PointPillarFeatureNet, self).__init__()
        assert feat_channels > 0
        assert in_channels == 4

        # 正如论文中所阐述: "The points in each pillar are then decorated (augmented) with r, x_c, y_c,
        # z_c, x_p, y_p where r is reflectance, the c subscript denotes distance to the arithmetic mean of all
        # points in the pillar, and the p subscript denotes the offset from the pillar x,y center". 即 原始维度 4
        # 扩充为 9 维
        self._in_channels = in_channels + 5

        self._feat_channels = feat_channels
        self._bin_size = bin_size
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
        self._vx = bin_size[0]
        self._vy = bin_size[1]

        # x和y方向上体素的起始索引对应在Velodyne坐标系中的x/y坐标值，单位：米
        self._x_offset = self._vx / 2 + point_cloud_range[0]
        self._y_offset = self._vy / 2 + point_cloud_range[1]

    def forward(self, features, num_points, coords):
        """
        前向推理函数

        :param features: 类型为 torch.Tensor, 指的是一个batch中样本体素化之后的表征，其形状为 (U, max_num_points, 4),
                         其中 U 为该 batch 中所有样本进行体素化之后的体素数量总和，体素也指Pillar
        :param num_points: 类型为torch.Tensor, 形状为 (U,), 它是一个一维数组，指的是每一个体素中的有效点云数量
        :param coords: 类型为torch.Tensor，形状为 (U, 4), 其中 4 表示 [sample_index, z, y, x]
        :return:
        """
        features_ls = [features]

        # 计算当前体素/pillar中各个点距离所有点的算数平均值的距离，即论文中的 x_c, y_c, z_c 特征
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean            # 形状为 (num_pillars, max_num_points, 3)
        features_ls.append(f_cluster)

        # 计算各个点距离体素/pillar中心点的距离特征，即论文中的 x_p, y_p 特征
        feats_offset = features[:, :, :2].clone().detach()      # 形状为 (num_pillars, max_num_points, 2)
        feats_offset[:, :, 0] = feats_offset[:, :, 0] - (coords[:, 3].type_as(features).unsqueeze(1) * self._vx +
                                                         self._x_offset)
        feats_offset[:, :, 1] = feats_offset[:, :, 1] - (coords[:, 2].type_as(features).unsqueeze(1) * self._vy +
                                                         self._y_offset)
        features_ls.append(feats_offset)

        # 合并上述的特征，即论文中的feature decorations, 执行完成之后 features 的维度为 (num_pillars, max_num_points, 9)
        features = torch.cat(features_ls, dim=-1)

        # 由于上述计算特征 x_c, y_c, z_c， x_p, y_p 时，没有区分体素化过程中补充的零，实际上这些补充的零在进行特征decoration之后
        # 仍然需要保持为零，所以这里需要将这些补充出来的点云重新设置为零
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)      # 形状为 (num_pillars, max_num_points, 1)
        features *= mask

        # 体素特征提取
        for pfn in self._pfn_layers:
            features = pfn(features)

        return features.squeeze(dim=1)
