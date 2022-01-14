import torch
from torch import nn
from point_pillar_net import PointPillarFeatureNet
from point_pillars_scatter import PointPillarScatter
from point_pillars_voxelization import PointPillarVoxelization
from torch.nn.functional import pad


class PointPillars(nn.Module):
    def __init__(self,
                 point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1),
                 voxel_size=(0.16, 0.16, 4.0),
                 max_num_points=100,
                 max_pillars=9223372036854775807):
        super().__init__()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._point_cloud_range = point_cloud_range
        self._voxel_size = voxel_size
        self._max_num_points = max_num_points
        self._max_pillars = max_pillars

        # 体素化器
        self._point_pillars_voxelizer = PointPillarVoxelization(
            voxel_size=self._voxel_size,
            point_cloud_range=self._point_cloud_range,
            max_num_points=self._max_num_points,
            max_voxels=self._max_pillars
        )

        # point pillar特征提取网络
        self._point_pillars_feature_net = PointPillarFeatureNet(
            in_channels=4,
            feat_channels=64,
            bin_size=self._voxel_size[0:2],
            point_cloud_range=self._point_cloud_range
        )

        # 伪图像生成器
        self._point_pillars_scatter = PointPillarScatter(
            in_channels=64,
            output_shape=torch.Tensor([69.12/0.16, 39.68 * 2/0.16]).type(torch.int)
        )

    @torch.no_grad()
    def voxelize_batch(self, raw_points_batch):
        """
        对一个Batch的原始点云执行体素化

        :param raw_points_batch: 数组, 其长度为 batch_size, 每一个元素为 torch.Tensor 类型，每一个元素的形状为 (Q, 4).
                                 Q 指的是一帧激光雷达扫描到的点云数量，不同帧之间 Q 通常而言各不相同
        :return: (batch_of_voxels, batch_of_num_points, batch_of_coords)
                 * batch_of_voxels: 类型为 torch.Tensor, 指的是一个batch中样本体素化之后的表征，其形状为 (U, max_num_points, 4),
                                    其中 U 为该 batch 中所有样本进行体素化之后的体素数量总和，体素也指Pillar.
                 * batch_of_num_points: 类型为torch.Tensor, 形状为 (U,), 它是一个一维数组，指的是每一个体素中的有效点云数量
                 * batch_of_coords: 类型为torch.Tensor，形状为 (U, 4), 其中 4 表示 [sample_index, z, y, x]
        """
        batch_of_voxels, batch_of_num_points, batch_of_coords = list(), list(), list()

        for raw_points in raw_points_batch:
            # raw_points 形状为 (Q, 4), Q 指的是一帧激光雷达扫描到的点云数量，不同帧之间 Q 通常而言各不相同
            # 输出 res_voxels 形状为 (N, self._max_num_points, 4)，N 为 该帧点云体素化后的实际体素数量，不同帧之间各不相同
            # 输出 res_coors 形状为 (N, 3)
            # 输出 res_num_points 形状为 (N,)
            res_voxels, res_coors, res_num_points = self._point_pillars_voxelizer(raw_points)

            batch_of_voxels.append(res_voxels)
            batch_of_num_points.append(res_num_points)
            batch_of_coords.append(res_coors)

        coors_batch_with_pad = []
        for i, coord in enumerate(batch_of_coords):
            coord_pad = pad(coord, (1, 0), mode='constant', value=i)
            coors_batch_with_pad.append(coord_pad)

        batch_of_voxels = torch.cat(batch_of_voxels, dim=0)
        batch_of_num_points = torch.cat(batch_of_num_points, dim=0)
        batch_of_coords = torch.cat(coors_batch_with_pad, dim=0)

        return batch_of_voxels, batch_of_num_points, batch_of_coords

    def forward(self, raw_points_batch):
        """
        前向处理函数

        :param raw_points_batch: 数组, 其长度为 batch_size, 每一个元素为 torch.Tensor 类型，每一个元素的形状为 (Q, 4).
                                 Q 指的是一帧激光雷达扫描到的点云数量，不同帧之间 Q 通常而言各不相同
        :return:
        """
        # 步骤一：执行体素化
        # 输出 batch_of_voxels 形状为 (U, max_num_points, 4), U 为该 batch 中所有样本进行体素化之后的体素数量总和
        # 输出 batch_of_num_points 形状为 (U,), 它是一个一维数组，指的是每一个体素中的有效点云数量
        # batch_of_coords 形状为 形状为 (U, 4), 其中 4 表示 [sample_index, z, y, x]
        batch_of_voxels, batch_of_num_points, batch_of_coords = self.voxelize_batch(raw_points_batch=raw_points_batch)

        # 步骤二：提取体素/pillar中的点云特征
        # 输出 pillar_features 的形状为 (U, 64), U 为该 batch 中所有样本进行体素化之后的体素数量总和，
        # 64 为体素中点云进行特征提取后的特征维度
        pillar_features = self._point_pillars_feature_net(batch_of_voxels, batch_of_num_points, batch_of_coords)
        batch_size = batch_of_coords[-1, 0].item() + 1

        # 将学习得到的体素点云特征重新转化为伪图像形式
        # 输出 batch_canvas 的维度信息为 (batch_size, C, ny, nx), 论文中 C=64
        batch_canvas = self._point_pillars_scatter(
            batch_pillar_features=pillar_features,
            batch_coords=batch_of_coords,
            batch_size=batch_size)


        # x = self.backbone(x)
        # x = self.neck(x)
        # outs = self.bbox_head(x)
        # return outs


if __name__ == '__main__':
    import numpy as np
    with open('./temp/points_velodyne_000008.npy', 'rb') as f1:
        points_sample_000008 = np.load(f1)
        points_sample_000008 = points_sample_000008[np.where(points_sample_000008[:, 0] > 0)]

    with open('./temp/points_velodyne_000025.npy', 'rb') as f2:
        points_sample_000025 = np.load(f2)
        points_sample_000025 = points_sample_000025[np.where(points_sample_000025[:, 0] > 0)]

    with open('./temp/points_velodyne_000031.npy', 'rb') as f3:
        points_sample_000031 = np.load(f3)
        points_sample_000031 = points_sample_000031[np.where(points_sample_000031[:, 0] > 0)]

    with open('./temp/points_velodyne_000032.npy', 'rb') as f4:
        points_sample_000032 = np.load(f4)
        points_sample_000032 = points_sample_000032[np.where(points_sample_000032[:, 0] > 0)]

    batch_points = [
        # torch.Tensor(points_sample_000008, device="cpu"),
        # torch.Tensor(points_sample_000025, device="cpu"),
        torch.Tensor(points_sample_000031, device="cpu"),
        torch.Tensor(points_sample_000032, device="cpu"),
    ]

    handler = PointPillars()
    handler(raw_points_batch=batch_points)
