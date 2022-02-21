import torch


class Anchor3DRangeGenerator(object):
    """
    根据 feature_map_size 生成 anchors
    """
    def __init__(self,
                 point_cloud_range,
                 anchor_generation_config):
        """
        构造函数

        :param point_cloud_range: 点云考虑的范围，也是锚框生成时考虑的点云范围，[x_min, y_min, z_min, x_max, y_max, z_max]
        :param anchor_generation_config: 锚框生成配置
        """
        self._point_cloud_range = point_cloud_range
        self._anchor_generation_config = anchor_generation_config

        self._anchor_sizes = [config['anchor_sizes'] for config in anchor_generation_config]
        self._anchor_rotations = [config['anchor_rotations'] for config in anchor_generation_config]
        self._anchor_bottom_heights = [config['anchor_bottom_heights'] for config in anchor_generation_config]
        self._align_center = [config.get('align_center', False) for config in anchor_generation_config]

    def generate_anchors(self, grid_sizes):
        """

        :param grid_sizes:
        :return:
        """
        assert len(grid_sizes) == len(self._anchor_sizes)
        all_anchors = list()
        num_anchors_per_location = list()

        for grid_size, anchor_size, anchor_rotation, anchor_bottom_height, align_center in zip(
                grid_sizes, self._anchor_sizes, self._anchor_rotations,
                self._anchor_bottom_heights, self._align_center):
            num_anchors_per_location.append(len(anchor_size) * len(anchor_rotation) * len(anchor_bottom_height))

            if align_center:
                x_stride = (self._point_cloud_range[3] - self._point_cloud_range[0]) / grid_size[0]
                y_stride = (self._point_cloud_range[4] - self._point_cloud_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (self._point_cloud_range[3] - self._point_cloud_range[0]) / (grid_size[0] - 1)
                y_stride = (self._point_cloud_range[4] - self._point_cloud_range[1]) / (grid_size[1] - 1)
                x_offset, y_offset = 0, 0

            x_shifts = torch.arange(
                start=self._point_cloud_range[0] + x_offset,
                end=self._point_cloud_range[3] + 1e-5,
                step=x_stride,
                dtype=torch.float32,
            ).cuda()

            y_shifts = torch.arange(
                start=self._point_cloud_range[1] + y_offset,
                end=self._point_cloud_range[4] + 1e-5,
                step=y_stride,
                dtype=torch.float32,
            ).cuda()

            z_shifts = torch.tensor(anchor_bottom_height, device="cuda")
            num_anchor_size, num_anchor_rotation = len(anchor_size), len(anchor_rotation)

            anchor_rotation = torch.tensor(anchor_rotation).cuda()
            anchor_size = torch.tensor(anchor_size).cuda()
            x_shifts, y_shifts, z_shifts = torch.meshgrid([x_shifts, y_shifts, z_shifts])

            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)

            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])
            anchors = torch.cat((anchors, anchor_size), dim=-1)

            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat([*anchors.shape[0:3], num_anchor_size, 1, 1])
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)

            # anchors 维度 为 [nx/2, ny/2, nz, len(anchor_size), len(anchor_rotations), 7], 最后一个维度7代表着
            # x, y, z, dx, dy, dz, yaw, 执行完下一步permute(2, 1, 0, 3, 4, 5)后维度变成 [nz, ny/2, nx/2, len(anchor_size),
            # len(anchor_rotations), 7]
            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()

            # 将锚框向上提升至预先设定的中心高度, 即在 anchor_bottom_height 的基础上加上 anchor的dz值的一半
            anchors[..., 2] += anchors[..., 5] / 2

            all_anchors.append(anchors)

        return all_anchors, num_anchors_per_location


if __name__ == "__main__":
    anchor_3d_range_generator = Anchor3DRangeGenerator(
        point_cloud_range=[0, -39.68, -3.0, 69.12, 39.68, 1.0],
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

    all_anchors_out, num_anchors_per_location_out = anchor_3d_range_generator.generate_anchors(
        grid_sizes=[[216, 248], [216, 248], [216, 248]]
    )

    for anchors in all_anchors_out:
        print("anchors.shape = {}".format(anchors.shape))

        print("anchors[0][0][0][0][0] = {}".format(anchors[0][0][0][0][0]))
        print("anchors[0][-1][0][0][0] = {}".format(anchors[0][-1][0][0][0]))
        print("anchors[0][-1][-1][0][0] = {}".format(anchors[0][-1][-1][0][0]))
        print("anchors[0][-1][-1][0][1] = {}".format(anchors[0][-1][-1][0][1]))

        print("\n")

    print(num_anchors_per_location_out)
