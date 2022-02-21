import torch


class AnchorGenerator(object):
    def __init__(self, anchor_range, anchor_sizes, anchor_rotations, anchor_heights, align_center):
        """
        构造函数

        :param anchor_range: 生成anchor的激光点云的范围，[x_min, y_min, z_min, x_max, y_max, z_max]
        :param anchor_sizes: 默认anchor的尺寸，举例为 [
                                 [[3.9, 1.6, 1.56], [5.2, 1.6, 1.56]],
                                 [[0.8, 0.6, 1.73]],
                                 [[1.76, 0.6, 1.73]]
                              ]，类型为一个数组，数组类没一个元素仍为一个尺寸数组，尺寸数组内是三元素数组，三元素的顺序分别为
                              dx, dy, dz.
        :param anchor_rotations:
        :param anchor_heights:
        :param align_center:
        """
        super().__init__()
        self.anchor_range = anchor_range
        self.anchor_sizes = anchor_sizes
        self.anchor_rotations = anchor_rotations
        self.anchor_heights = anchor_heights
        self.align_center = align_center

        print(len(self.anchor_sizes))
        print(len(self.anchor_rotations))
        print(len(self.anchor_heights))

        assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(self.anchor_heights)
        self.num_of_anchor_sets = len(self.anchor_sizes)

    def generate_anchors(self, grid_sizes):
        assert len(grid_sizes) == self.num_of_anchor_sets
        all_anchors = []
        num_anchors_per_location = []
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
                grid_sizes, self.anchor_sizes, self.anchor_rotations, self.anchor_heights, self.align_center):

            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            if align_center:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / grid_size[0]
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (grid_size[0] - 1)
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (grid_size[1] - 1)
                x_offset, y_offset = 0, 0

            x_shifts = torch.arange(
                self.anchor_range[0] + x_offset, self.anchor_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            ).cuda()
            y_shifts = torch.arange(
                self.anchor_range[1] + y_offset, self.anchor_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            ).cuda()
            z_shifts = x_shifts.new_tensor(anchor_height)

            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__()
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)
            anchor_size = x_shifts.new_tensor(anchor_size)
            x_shifts, y_shifts, z_shifts = torch.meshgrid([
                x_shifts, y_shifts, z_shifts
            ])  # [x_grid, y_grid, z_grid]
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)  # [x, y, z, 3]
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])
            anchors = torch.cat((anchors, anchor_size), dim=-1)
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat([*anchors.shape[0:3], num_anchor_size, 1, 1])
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)  # [x, y, z, num_size, num_rot, 7]

            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()
            #anchors = anchors.view(-1, anchors.shape[-1])
            anchors[..., 2] += anchors[..., 5] / 2  # shift to box centers
            all_anchors.append(anchors)

        return all_anchors, num_anchors_per_location


if __name__ == '__main__':
    A = AnchorGenerator(
        anchor_range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
        anchor_sizes=[[[3.9, 1.6, 1.56], [5.2, 1.6, 1.56]], [[0.8, 0.6, 1.73]], [[1.76, 0.6, 1.73]]],
        anchor_rotations=[[0, 0.45, 0.90, 1.57], [0, 1.57], [0, 1.57]],
        anchor_heights=[[-1.78], [-0.6], [-0.6]],
        align_center=[False, False, False]
    )
    # import pdb
    # pdb.set_trace()
    all_anchors, num_anchors_per_location = A.generate_anchors([[216, 248], [216, 248], [216, 248]])

    for item in all_anchors:
        print(item.shape)

    print(num_anchors_per_location)
