import torch
from torch import nn
from point_pillar_net import PointPillarFeatureNet
from point_pillars_scatter import PointPillarScatter
from point_pillars_backbone import PointPillarBackbone
from point_pillars_anchor_head_single import PointPillarAnchorHeadSingle


class PointPillars(nn.Module):
    def __init__(self,
                 point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1),
                 voxel_size=(0.16, 0.16, 4.0),
                 max_points_per_voxel=100,
                 max_number_of_voxels=(16000, 40000)):
        """
        构造函数

        :param point_cloud_range: 点云范围，[x_min, y_min, z_min, x_max, y_max, z_max]
        :param voxel_size: 体素大小
        :param max_points_per_voxel: 单个体素(pillar)中考虑的最大点数
        :param max_number_of_voxels: 训练和测试阶段考虑的最大体素(pillar)数量
        """
        super().__init__()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._point_cloud_range = point_cloud_range
        self._voxel_size = voxel_size
        self._max_points_per_voxel = max_points_per_voxel
        self._max_number_of_voxels = max_number_of_voxels

        # point pillar特征提取网络
        self._point_pillars_feature_net = PointPillarFeatureNet(
            in_channels=4,
            feat_channels=64,
            voxel_size=self._voxel_size,
            point_cloud_range=self._point_cloud_range
        )

        # 伪图像生成器
        self._point_pillars_scatter = PointPillarScatter(
            in_channels=64,
            nx=(self._point_cloud_range[3] - self._point_cloud_range[0]) / self._voxel_size[0],
            ny=(self._point_cloud_range[4] - self._point_cloud_range[1]) / self._voxel_size[1]
        )

        # Backbone特征提取器
        self._point_pillars_backbone = PointPillarBackbone()

        # 检测头
        self._point_pillars_anchor_3d_head = PointPillarAnchorHeadSingle()

    def forward(self, voxels, indices, nums_per_voxel, sample_indices):
        """
        前向处理函数

        :param voxels: torch.Tensor, 形状为 (M, max_points_per_voxel, 4). 其中 M 为当前 batch 中点云样本进行体素化后的总体素
                       数量，例如 batch_size = 4时，四个样本的体素化后体素数量分别为 m1, m2, m3, m4. 则 M = m1 + m2 + m3 + m4.
                       4为原始点云的输入维度，分别表示 [x, y, z, intensity]
        :param indices: torch.Tensor, 形状为 (M, 3), 表示每一个体素（Pillar）对应的坐标索引，顺序为 z, y, x
        :param nums_per_voxel: torch.Tensor, 形状为(M, )，表示每一个体素内的有效点云点数
        :param sample_indices: torch.Tensor， 形状为(M, ), 表示当前体素属于 batch 中的哪一个，即索引
        :return:
        """
        # 步骤一：提取体素中的点云特征
        # 输出 pillar_features 的形状为 (M, 64), M 为该 batch 中所有样本进行体素化之后的体素数量总和，
        # 64 为体素中点云进行特征提取后的特征维度
        pillar_features = self._point_pillars_feature_net(voxels, indices, nums_per_voxel, sample_indices)

        # 步骤二：将学习得到的体素点云特征重新转化为伪图像形式
        # 输出 batch_canvas 的维度信息为 (batch_size, C, nx, ny), 论文中 C=64, nx=432, ny=496
        batch_size = int(sample_indices[-1].item() + 1)
        batch_canvas = self._point_pillars_scatter(
            batch_pillar_features=pillar_features,
            batch_indices=indices,
            sample_indices=sample_indices,
            batch_size=batch_size)

        # 步骤三：利用Backbone提取伪图像的特征，输出维度为 (batch_size, 6C, nx/2, ny/2)
        backbone_feats = self._point_pillars_backbone(batch_canvas=batch_canvas)
        print("backbone_feats.shape = {}".format(backbone_feats.shape))

        # 步骤五：基于Single Shot Detector (SSD) 对3D物体进行目标检测和回归
        outs = self._point_pillars_anchor_3d_head(backbone_feats)
        return outs


if __name__ == '__main__':
    import numpy as np
    from point_pillars_voxelization import VoxelGenerator

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

    voxel_generator = VoxelGenerator(
        voxel_size=[0.16, 0.16, 4.0],
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        max_num_points_per_voxel=100,
        max_num_voxels=16000
    )

    voxels_000008, indices_000008, num_per_voxel_000008, _ = voxel_generator.generate(points=points_sample_000008)
    voxels_000025, indices_000025, num_per_voxel_000025, _ = voxel_generator.generate(points=points_sample_000025)
    voxels_000031, indices_000031, num_per_voxel_000031, _ = voxel_generator.generate(points=points_sample_000031)
    voxels_000032, indices_000032, num_per_voxel_000032, _ = voxel_generator.generate(points=points_sample_000032)

    print("voxels_000008.shape = {}".format(voxels_000008.shape))
    print("voxels_000025.shape = {}".format(voxels_000025.shape))
    print("voxels_000031.shape = {}".format(voxels_000031.shape))
    print("voxels_000032.shape = {}".format(voxels_000032.shape))

    voxels = torch.cat((voxels_000008, voxels_000025, voxels_000031, voxels_000032), dim=0)
    indices = torch.cat((indices_000008, indices_000025, indices_000031, indices_000032), dim=0)
    nums_per_voxel = torch.cat((num_per_voxel_000008, num_per_voxel_000025, num_per_voxel_000031, num_per_voxel_000032), dim=0)
    sample_indices = torch.cat((
        torch.zeros(voxels_000008.shape[0]),
        torch.ones(voxels_000025.shape[0]),
        2 * torch.ones(voxels_000031.shape[0]),
        3 * torch.ones(voxels_000032.shape[0]),
    ), dim=0)

    print("voxels.shape = {}".format(voxels.shape))
    print("indices.shape = {}".format(indices.shape))
    print("nums_per_voxel.shape = {}".format(nums_per_voxel.shape))
    print("sample_indices.shape = {}".format(sample_indices.shape))

    model = PointPillars()
    model(voxels=voxels, indices=indices, nums_per_voxel=nums_per_voxel, sample_indices=sample_indices)
