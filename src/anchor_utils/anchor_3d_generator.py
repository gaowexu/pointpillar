import torch


class Anchor3DRangeGenerator(object):
    """3D Anchor Generator by range.

    This anchor_utils generator generates anchors by the given range in different
    feature levels.

    Args:
        ranges (list[list[float]]): Ranges of different anchors.
            The ranges are the same across different feature levels. But may
            vary for different anchor_utils sizes if size_per_range is True.
        sizes (list[list[float]]): 3D sizes of anchors.
        rotations (list[float]): Rotations of anchors in a feature grid.
    """

    def __init__(self,
                 point_cloud_range=[[0, -39.68, -3, 69.12, 39.68, 1]],
                 anchor_sizes=[[1.6, 3.9, 1.56]],
                 anchor_rotations=[0, 1.5707963]):
        assert len(point_cloud_range) == len(anchor_sizes)

        self._anchor_sizes = anchor_sizes
        self._point_cloud_range = point_cloud_range
        self._anchor_rotations = anchor_rotations

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
        for pc_range, anchor_size in zip(self._point_cloud_range, self._anchor_sizes):
            print(pc_range, anchor_size)

        #     mr_anchors.append(
        #         self.anchors_single_range(
        #             featmap_size,
        #             pc_range,
        #             anchor_size,
        #             self._anchor_rotations,
        #             device=device
        #         )
        #     )
        #
        # mr_anchors = torch.cat(mr_anchors, dim=-3)
        # return mr_anchors

    @staticmethod
    def anchors_single_range(feature_map_size, point_cloud_range, anchor_size, rotations, device='cuda'):
        """

        :param feature_map_size: 特征图大小，为一个数组，默认为 [nx/2, ny/2]
        :param point_cloud_range: 点云范围，[x_min, y_min, z_min, x_max, y_max, z_max]
        :param anchor_size: 默认 anchor_utils 的大小，
        :param rotations: anchor_utils ua
        :param device:
        :return:
        """
        """Generate anchors in a single range.

        Args:
            feature_size (list[float] | tuple[float]): Feature map size. It is
                either a list of a tuple of [D, H, W](in order of z, y, and x).
            anchor_range (torch.Tensor | list[float]): Range of anchors with
                shape [6]. The order is consistent with that of anchors, i.e.,
                (x_min, y_min, z_min, x_max, y_max, z_max).
            sizes (list[list] | np.ndarray | torch.Tensor): Anchor size with
                shape [N, 3], in order of x, y, z.
            rotations (list[float] | np.ndarray | torch.Tensor): Rotations of
                anchors in a single feature grid.
            device (str): Devices that the anchors will be put on.

        Returns:
            torch.Tensor: Anchors with shape \
                [*feature_size, num_sizes, num_rots, 7].
        """
        if len(feature_map_size) == 2:
            feature_map_size = [1, feature_map_size[0], feature_map_size[1]]

        point_cloud_range = torch.tensor(point_cloud_range, device=device)

        x_centers = torch.linspace(
            start=point_cloud_range[0],
            end=point_cloud_range[3],
            steps=feature_map_size[2],
            device=device)

        y_centers = torch.linspace(
            start=point_cloud_range[1],
            end=point_cloud_range[4],
            steps=feature_map_size[1],
            device=device)

        z_centers = torch.linspace(
            start=point_cloud_range[2],
            end=point_cloud_range[5],
            steps=feature_map_size[0],
            device=device)

        sizes = torch.tensor(sizes, device=device).reshape(-1, 3)
        rotations = torch.tensor(rotations, device=device)

        # torch.meshgrid default behavior is 'id', np's default is 'xy'
        rets = torch.meshgrid(x_centers, y_centers, z_centers, rotations)
        # torch.meshgrid returns a tuple rather than list
        rets = list(rets)
        for i in range(len(rets)):
            rets[i] = rets[i].unsqueeze(-2).unsqueeze(-1)

        sizes = sizes.reshape([1, 1, 1, 1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = sizes.repeat(tile_size_shape)
        rets.insert(3, sizes)

        ret = torch.cat(rets, dim=-1).permute([2, 1, 0, 3, 4, 5])
        # [1, 200, 176, N, 2, 7] for kitti after permute
        return ret


if __name__ == "__main__":
    anchor_3d_range_generator = Anchor3DRangeGenerator(
        point_cloud_range=[
            [0, -40.0, -3.0, 70.4, 40.0, 1.0],
            [0, -20, -2.5, 48.0, 20.0, 0.5],
            [0, -20, -2.5, 48.0, 20.0, 0.5]
        ],
        anchor_sizes=[
            [1.6, 3.9, 1.56],
            [0.6, 0.8, 1.73],
            [0.6, 0.8, 1.73]
        ],
        anchor_rotations=[0, 1.5707963]
    )

    print(anchor_3d_range_generator.num_anchors_per_location)

    anchor_3d_range_generator.grid_anchors(feature_map_size=(432, 496))
