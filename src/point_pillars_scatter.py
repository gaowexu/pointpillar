import torch
from torch import nn


class PointPillarScatter(nn.Module):
    """
    将 PFNLayer 中学习得到的点云特征（dense形式）转化为伪图像形式，作为后续的2D卷积神经网络的输入
    """
    def __init__(self, in_channels=64, output_shape=(496, 432)):
        """
        构造函数

        :param in_channels: 点云特征提取的输出维度数，即论文中的 C, 默认为64.
        :param output_shape: 伪图像的输出维度，默认为 (69.12/0.16, 39.68 * 2/0.16)，即为(496, 432)，69.12为x方向考虑的范围，
                             即车辆的正前方；39.68为车辆两侧考虑的范围，即y方向
        """
        super().__init__()
        self._in_channels = in_channels
        self._output_shape = output_shape
        self._ny = output_shape[0]              # 伪图像的行数，即 ny，为 velodyne 坐标系的x方向
        self._nx = output_shape[1]              # 为图像的列数，即 nx, 为 velodyne 坐标系的y方向

    def forward(self, batch_pillar_features_stacked, batch_coords, batch_size):
        """
        Scatter features of single sample

        :param batch_pillar_features_stacked:
        :param batch_coords: Coordinates of each pillar, torch.Tensor with shape (num_pillars, 4), 4 indicates
                       [sample_index, z, y, x]
        :param batch_size: Integer, Number of samples in the current batch.
        :return:
        """
        batch_canvas = list()

        for sample_index in range(batch_size):
            # 针对每一帧点云样本，创建一个值为0的画布，默认大小为(64, 214272), 64为论文中的点云特征输出维度C, 214272=496x432
            canvas = torch.zeros(self._in_channels, self._nx * self._ny,
                                 dtype=batch_pillar_features_stacked.dtype,
                                 device=batch_pillar_features_stacked.device)

            # 将当前帧的点云特征从密集（dense)表达中取出
            batch_mask = batch_coords[:, 0] == sample_index
            this_coords = batch_coords[batch_mask, :]
            pillar_feats = batch_pillar_features_stacked[batch_mask, :]     # 形状为 (Q, 64)，其中Q对于Batch中各个样本各不相同
            transposed_pillar_feats = pillar_feats.t()                      # 形状为 (64, Q）

            #
            indices = this_coords[:, 2] * self._nx + this_coords[:, 3]
            indices = indices.type(torch.long)


            # Now scatter the blob back to the canvas.
            print("indices = {}".format(indices))
            print("indices.shape = {}".format(indices.shape))
            print("canvas[:, indices].shape = {}".format(canvas[:, indices].shape))
            canvas[:, indices] = transposed_pillar_feats

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # 将画布转化为三维，转化为形状为(batch_size, self._in_channels, self._nx * self._ny)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor, batch_canvas shape is (batch_size, 64, 496, 432)
        batch_canvas = batch_canvas.view(batch_size, self._in_channels, self._ny, self._nx)

        return batch_canvas
