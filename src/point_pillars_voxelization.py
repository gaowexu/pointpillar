import torch
from spconv.pytorch.utils import PointToVoxel


class VoxelGenerator(object):
    """
    将稀疏的点云进行体素化，在PointPillar这篇文章中voxel实际中等同于pillar, 即voxel是一种更为泛化的表征，其z方向不限制即为pillar
    """
    def __init__(self,
                 voxel_size=[0.16, 0.16, 4.0],
                 point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                 max_num_points_per_voxel=100,
                 max_num_voxels=16000):
        """
        构造函数

        :param voxel_size: 最小体素尺寸，分别为 Velodyne坐标系下 x, y, z 三个方向的尺寸
        :param point_cloud_range: 点云集合中考虑的范围，[x_min, y_min, z_min, x_max, y_max, z_max]，落在该范围之外的点云不作考虑
        :param max_num_points_per_voxel: 每一个体素中最大的点云点数，若落在某一个体素中的点云数量超过该数值，需要进行随机降采样；反之，补零
        :param max_num_voxels: 考虑的最大体素数量
        """
        super().__init__()
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points_per_voxel = max_num_points_per_voxel
        self._points_range_min = point_cloud_range[:3]    # (x_min, y_min, z_min)
        self._points_range_max = point_cloud_range[3:]    # (x_max, y_max, z_max)
        self._max_num_voxels = max_num_voxels

        self._points_to_voxel_converter = PointToVoxel(
            vsize_xyz=self._voxel_size,
            coors_range_xyz=self._point_cloud_range,
            num_point_features=4,
            max_num_points_per_voxel=self._max_num_points_per_voxel,
            max_num_voxels=self._max_num_voxels,
            device=torch.device("cpu:0")
        )

    def generate(self, points):
        """
        对点云数据进行体素化

        :param points: 原始点云序列，形状为(N, 4), 其中不同帧的点云数量不同，即N不同，4代表的是位置和激光反射强度特征，
                       即[x, y, z, intensity].
        :return: (voxels, indices, num_per_voxel)
                 * voxels: 指的是体素的密集表示，它是一个形状为 (num_voxels, max_num_points_per_voxel, 4) 的torch.Tensor,
                 * indices: 指的是原始点云体素化后每一个体素对应的坐标，形状为(num_voxels, 3), 注意此处的 3 代表的体素坐标顺序为
                            为 z, y, x
                 * num_per_voxel: 指的是每一个体素中的实际有效的点云数量，它是一个一维数组，类型为torch.Tensor
        """
        # 体素化计算:
        # 返回 voxels 指的是体素的密集表示，它是一个形状为 (num_voxels, max_num_points_per_voxel, 4) 的
        # torch.Tensor,其中 num_voxels 指的是输入原始点云经过体素化后的体素数量（不超过所设置的最大体素数量
        # max_num_voxels），不同帧的点云序列体素化之后的实际 num_voxels 一般各不相同，4为原始点云的输入维度，
        # 分别表示 [x, y, z, intensity]
        #
        # 返回 indices 形状为 (num_voxels, 3), 表示每一个体素（Pillar）对应的坐标索引，顺序为 z, y, x
        #
        # 返回 num_per_voxel 维度为 (num_voxels, )，表示每一个体素内的有效点云点数
        (voxels, indices, num_per_voxel) = self._points_to_voxel_converter(torch.from_numpy(points).contiguous())

        # 计算点云坐标系中 x,y,z三个方向的体素数量，在 point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1] 和
        # voxel_size=[0.16, 0.16, 4.0] 情况下 输出为 [432, 496, 1]
        voxel_grids = ((torch.Tensor(self._points_range_max) -
                        torch.Tensor(self._points_range_min)) / torch.Tensor(self._voxel_size)).type(torch.int32)

        return voxels, indices, num_per_voxel, voxel_grids


if __name__ == "__main__":
    voxel_generator = VoxelGenerator(
        voxel_size=[0.16, 0.16, 4.0],
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        max_num_points_per_voxel=100,
        max_num_voxels=16000
    )

    import numpy as np
    from tools.visualize import plot_3d_cloud_points
    with open('./temp/points_velodyne_000008.npy', 'rb') as f1:
        points_sample_000008 = np.load(f1)
        points_sample_000008 = points_sample_000008[np.where(points_sample_000008[:, 0] > 0)]

    # plot_3d_cloud_points(points=points_sample_000008)

    voxels, indices, num_per_voxel, voxel_grids = voxel_generator.generate(points=points_sample_000008)
    print("voxels.shape = {}".format(voxels.shape))
    print("indices.shape = {}".format(indices.shape))
    print("num_per_voxel.shape = {}".format(num_per_voxel.shape))
    print("voxel_grids = {}".format(voxel_grids))
