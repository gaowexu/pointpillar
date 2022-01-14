import torch
from torch import nn
from open3d.ml.torch.ops import voxelize, ragged_to_dense


class PointPillarVoxelization(nn.Module):
    """
    将稀疏的点云进行体素化，在PointPillar这篇文章中voxel实际中等同于pillar, 即voxel是一种更为泛化的表征，其z方向不限制即为pillar
    """
    def __init__(self,
                 voxel_size=(0.16, 0.16, 4.0),
                 point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1),
                 max_num_points=100,
                 max_voxels=9223372036854775807):
        """
        构造函数

        :param voxel_size: 最小体素尺寸，分别为 Velodyne坐标系下 x, y, z 三个方向的尺寸
        :param point_cloud_range: 点云集合中考虑的范围，[x_min, y_min, z_min, x_max, y_max, z_max]，落在该范围之外的点云不作考虑
        :param max_num_points: 每一个体素中最大的点云点数，若落在某一个体素中的点云数量超过该数值，需要进行随机降采样；反之，补零
        :param max_voxels: 考虑的最大体素数量
        """
        super().__init__()
        self._voxel_size = torch.Tensor(voxel_size)
        self._point_cloud_range = torch.Tensor(point_cloud_range)
        self._max_num_points = max_num_points
        self._points_range_min = torch.Tensor(point_cloud_range[:3])    # (x_min, y_min, z_min)
        self._points_range_max = torch.Tensor(point_cloud_range[3:])    # (x_max, y_max, z_max)
        self._max_voxels = max_voxels

    def forward(self, raw_points):
        """
        前向推理函数

        :param raw_points: 原始点云序列，形状为(N, 4), 其中不同帧的点云数量不同，即N不同，4代表的是位置和激光反射强度特征，
                           即[x, y, z, reflectance].
        :return: (out_voxels, out_coords, out_num_points)
                 * out_voxels: 指的是体素的密集表示，它是一个形状为 (num_voxels, max_num_points, 4) 的torch.Tensor, 其中
                               num_voxels指的是输入原始点云 raw_points 经过体素化后的体素数量（即Pillar数量），不同帧的点云
                               序列体素化之后的 num_voxels 一般各不相同，max_num_points 指的是一个体素中设定的最大点云数量，
                               过多则降采样，过少则补零
                 * out_coords: 指的是原始点云体素化后每一个体素对应的坐标，形状为(num_voxels, 3), 注意此处的 3 代表的坐标顺序
                               为 velodyne 坐标系中的 [z, y, x]
                 * out_num_points: 指的是每一个体素中的实际有效的点云数量，它是一个一维数组，类型为torch.Tensor
        """
        points = raw_points[:, :3]      # 形状为(N, 3), 不同帧的点云数量不同，即N不同

        # 计算Velodyne坐标系中 x,y,z三个方向的体素数量，输出为 [432, 496, 1]
        num_voxels = ((self._points_range_max - self._points_range_min) / self._voxel_size).type(torch.int32)

        # 体素化计算
        ans = voxelize(
            points=points,
            row_splits=torch.LongTensor([0, points.shape[0]]).to(points.device),
            voxel_size=self._voxel_size,
            points_range_min=self._points_range_min,
            points_range_max=self._points_range_max,
            max_voxels=self._max_voxels
        )

        # 获取体素化后的体素总数量，论文中将该值限制为最大12000， 即 Max number of Pillars, P
        actual_num_of_voxels = ans.voxel_coords.shape[0]

        # TODO: 当体素化的实际体素数量超过所设定的阈值 self._max_voxels 时，随机降采样？

        # 在点云的第一行前添加一行0，该行索引为0，意味着补0，
        points_with_prepend_zero_row = torch.cat([torch.zeros_like(raw_points[0:1, :]), raw_points])

        # 计算点云dense表达下的索引矩阵，voxels_point_indices_dense的形状为 (actual_num_of_voxels, self._max_num_points)
        voxels_point_indices_dense = ragged_to_dense(
            values=ans.voxel_point_indices,
            row_splits=ans.voxel_point_row_splits,
            out_col_size=self._max_num_points,
            default_value=torch.tensor(-1)
        ) + 1

        # 获取体素矩阵，每一个体素中点云数量为self._max_num_points个（可能包括补充的0），out_voxels形状为
        # (actual_num_of_voxels, self._max_num_points，4)
        out_voxels = points_with_prepend_zero_row[voxels_point_indices_dense]

        # 将点云的索引顺序从 [x,y,z] 改为 [z,y,x]
        out_coords = ans.voxel_coords[:, [2, 1, 0]].contiguous()
        out_num_points = ans.voxel_point_row_splits[1:] - ans.voxel_point_row_splits[:-1]

        # 剔除落在指定x/y 范围内的体素
        in_bounds_y = out_coords[:, 1] < num_voxels[1]
        in_bounds_x = out_coords[:, 2] < num_voxels[0]
        in_bounds = torch.logical_and(in_bounds_x, in_bounds_y)

        out_voxels = out_voxels[in_bounds]
        out_coords = out_coords[in_bounds]
        out_num_points = out_num_points[in_bounds]

        return out_voxels, out_coords, out_num_points
