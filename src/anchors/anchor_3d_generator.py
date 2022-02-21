import torch


class Anchor3DRangeGenerator(object):
    """
    根据 feature_map_size 生成 anchors
    """
    def __init__(self,
                 point_cloud_range,
                 anchor_rotations,
                 anchor_sizes,
                 anchor_bottom_heights,
                 align_center=True):

        self._point_cloud_range = point_cloud_range
        self._anchor_rotations = anchor_rotations
        self._anchor_sizes = anchor_sizes
        self._anchor_bottom_heights = anchor_bottom_heights
        self._align_center = align_center

    @property
    def num_anchors_per_location(self):
        """
        计算 feature grid map中每一个“像素”位置所有的 anchor_utils 数量
        :return:
        """
        num_rot = len(self._anchor_rotations)
        num_size = torch.tensor(self._anchor_sizes).reshape(-1, 3).size(0)
        return num_rot * num_size

    def grid_anchors(self, feature_map_size, device='cuda'):
        """Generate grid anchors of a single level feature map.

        This function is usually called by method ``self.grid_anchors``.

        Args:
            feature_map_size (tuple[int]): Size of the feature map.
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature map.
        """
        mr_anchors = list()
        for pc_range, anchor_size in zip(self._point_cloud_ranges, self._anchor_sizes):
            mr_anchors.append(
                self.anchors_single_range(
                    feature_map_size=feature_map_size,
                    point_cloud_range=pc_range,
                    anchor_sizes=self._anchor_sizes,
                    rotations=self._anchor_rotations,
                    device=device
                )
            )

            print("\n")

        mr_anchors = torch.cat(mr_anchors, dim=-3)
        return mr_anchors

    @staticmethod
    def anchors_single_range(feature_map_size, point_cloud_range, anchor_sizes, rotations, device='cuda'):
        """

        :param feature_map_size: 特征图大小，为一个数组，默认为 [nx/2, ny/2], 如果是二维，需要转化为三维，即转化为
                                 [1, nx/2, ny/2]，如果传进来的是三维，则代表 [D, H, W] (顺序为激光雷达的 z轴，x 轴， y 轴)
        :param point_cloud_range: 点云范围，[x_min, y_min, z_min, x_max, y_max, z_max]
        :param anchor_sizes: 默认 anchor 的大小，形状为 （N, 3）, 3代表 x, y, z 方向上的尺寸
        :param rotations: anchor 的旋转角度
        :param device: anchors 输出的设备，cpu 或 cuda

        :return: 返回 anchors 集合，其维度为 (D, H, W, len(self._anchor_sizes), len(self._anchor_rotations), 7), 其中
                 D表示在z轴上的分辨率， H = nx/2, W = ny/2, 7 表示anchor参数，即 x, y, z, w, l, h, yaw. 在 PointPillar
                 算法中返回的大小为  (1, 432, 496, len(self._anchor_sizes), len(self._anchor_rotations), 7)
        """
        if len(feature_map_size) == 2:
            feature_map_size = [1, feature_map_size[0], feature_map_size[1]]

        point_cloud_range = torch.tensor(point_cloud_range, device=device)

        x_centers = torch.linspace(
            start=point_cloud_range[0],
            end=point_cloud_range[3],
            steps=feature_map_size[1],
            device=device)

        y_centers = torch.linspace(
            start=point_cloud_range[1],
            end=point_cloud_range[4],
            steps=feature_map_size[2],
            device=device)

        z_centers = torch.linspace(
            start=point_cloud_range[2],
            end=point_cloud_range[5],
            steps=feature_map_size[0],
            device=device)

        anchor_sizes = torch.tensor(anchor_sizes, device=device).reshape(-1, 3)
        rotations = torch.tensor(rotations, device=device)

        # torch.meshgrid default behavior is 'id', np's default is 'xy'
        rets = torch.meshgrid(x_centers, y_centers, z_centers, rotations)
        # torch.meshgrid returns a tuple rather than list
        rets = list(rets)
        tile_shape = [1] * 5
        tile_shape[-2] = int(anchor_sizes.shape[0])

        for i in range(len(rets)):
            rets[i] = rets[i].unsqueeze(-2).repeat(tile_shape).unsqueeze(-1)

        anchor_sizes = anchor_sizes.reshape([1, 1, 1, -1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        anchor_sizes = anchor_sizes.repeat(tile_size_shape)
        rets.insert(3, anchor_sizes)

        # 执行完下一步后输出维度为 (432, 496, 1, len(self._anchor_sizes), len(self._anchor_rotations), 7),
        # 其中432, 496, 1 分别表示 输出的 anchor 在原始激光雷达坐标系上 x,y,z上的box_code, 即为 x, y, z, w, l, h, yaw.
        # 需要注意的是 w, l 分别对应三维物体的宽度和长度，对应着激光雷达坐标系上的 y, x, 在特征图上也对应着 y, x （而不是 x, y）
        # 关于 rets 输出举例说明：
        # rets[0][0][0][0][0] = [0.0000, -39.6800,  -0.6000,   1.6000,   3.9000,   1.5600,   0.0000]
        # rets[0][0][0][1][0] = [0.0000, -39.6800,  -0.6000,   0.6000,   0.8000,   1.7300,   0.0000]
        # rets[0][0][0][2][0] = [0.0000, -39.6800,  -0.6000,   0.6000,   1.7600,   1.7300,   0.0000]
        # rets[0][0][0][0][1] = [0.0000, -39.6800,  -0.6000,   1.6000,   3.9000,   1.5600,   1.5708]
        # rets[1][0][0][0][1] = [0.1604, -39.6800,  -0.6000,   1.6000,   3.9000,   1.5600,   1.5708]
        # rets[-1][0][0][0][1] = [69.1200, -39.6800,  -0.6000,   1.6000,   3.9000,   1.5600,   1.5708]
        # rets[-1][-1][0][0][1] = [69.1200, 39.6800, -0.6000,  1.6000,  3.9000,  1.5600,  1.5708]
        rets = torch.cat(rets, dim=-1)

        return rets


if __name__ == "__main__":
    anchor_3d_range_generator = Anchor3DRangeGenerator(
        point_cloud_range=[0, -39.68, -3.0, 69.12, 39.68, 1.0],
        anchor_rotations=[[0, 1.5707963], [0, 1.5707963], [0, 1.5707963]],
        anchor_sizes=[
            [3.9, 1.6, 1.56],  # Car 类别的anchor尺寸, dx, dy, dz
            [0.8, 0.6, 1.73],  # Pedestrian 类别的anchor尺寸, dx, dy, dz
            [1.76, 0.6, 1.73]  # Cyclist 类别的anchor尺寸, dx, dy, dz
        ],
        anchor_bottom_heights=[
            [-1.78],    # Car 类别的 z_offset
            [-0.6],     # Pedestrian 类别的 z_offset
            [-0.6]      # Cyclist 类别的 z_offset
        ],
        align_center=True
    )

    print(anchor_3d_range_generator.num_anchors_per_location)

    anchor_3d_range_generator.grid_anchors(feature_map_size=(432//2, 496//2))
