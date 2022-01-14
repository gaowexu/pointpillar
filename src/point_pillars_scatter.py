import torch
from torch import nn


class PointPillarScatter(nn.Module):
    """
    将 PFNLayer 中学习得到的点云特征（dense形式）转化为伪图像形式，作为后续的2D卷积神经网络的输入
    """
    def __init__(self, in_channels=64, output_shape=(432, 496)):
        """
        构造函数

        :param in_channels: 点云特征提取的输出维度数，即论文中的 C, 默认为64.
        :param output_shape: 伪图像的输出维度 (nx, ny)，默认为 (69.12/0.16, 39.68 * 2/0.16)，即为(432, 496)，
                             69.12为x方向考虑的范围，即车辆的正前方；39.68为车辆两侧考虑的范围，即y方向
        """
        super().__init__()
        self._in_channels = in_channels
        self._output_shape = output_shape
        self._nx = output_shape[0]
        self._ny = output_shape[1]

    def forward(self, batch_pillar_features, batch_coords, batch_size):
        """
        将点云特征重新转化为伪图像格式

        :param batch_pillar_features: 类型为torch.Tensor，指的是一个batch中所有点云的特征表征，维度为 (U, 64), U 为该 batch 中
                                      所有样本进行体素化之后的体素数量总和
        :param batch_coords: 类型为torch.Tensor，形状为 (U, 4), 其中 4 表示 [sample_index, z, y, x]
        :param batch_size: 整数, batch尺度
        :return:
        """
        batch_canvas = list()

        for sample_index in range(batch_size):
            # 针对每一帧点云样本，创建一个值为0的画布，默认大小为(64, 214272), 64为论文中的点云特征输出维度C, 214272=496x432
            canvas = torch.zeros(self._in_channels, self._nx * self._ny,
                                 dtype=batch_pillar_features.dtype,
                                 device=batch_pillar_features.device)

            # 将当前帧的点云特征从密集（dense)表达中取出
            batch_mask = batch_coords[:, 0] == sample_index
            this_coords = batch_coords[batch_mask, :]               # 形状为 (Q, 4), 4 表示 [sample_index, z, y, x]
            pillar_feats = batch_pillar_features[batch_mask, :]     # 形状为 (Q, 64)，其中Q对于Batch中各个样本各不相同
            transposed_pillar_feats = pillar_feats.t()              # 形状为 (64, Q), Q指的是当前样本的体素数量

            indices = this_coords[:, 2] * self._nx + this_coords[:, 3]
            indices = indices.type(torch.long)

            canvas[:, indices] = transposed_pillar_feats
            batch_canvas.append(canvas)

        # 将画布转化为三维，转化为形状为(batch_size, C, self._nx * self._ny)
        batch_canvas = torch.stack(batch_canvas, 0)

        # 将画布转化为四维，维度信息为 (batch_size, C, 496, 432), 文章中 C=64
        batch_canvas = batch_canvas.view(batch_size, self._in_channels, self._ny, self._nx)

        return batch_canvas
