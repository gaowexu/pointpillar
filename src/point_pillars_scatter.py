import torch
from torch import nn


class PointPillarScatter(nn.Module):
    """
    将 PFNLayer 中学习得到的点云特征（dense形式）转化为伪图像形式，作为后续的2D卷积神经网络的输入
    """
    def __init__(self, in_channels=64, nx=432, ny=496):
        """
        构造函数

        :param in_channels: 点云特征提取的输出维度数，即论文中的 C, 默认为64.
        :param nx: 伪图像的 x 方向 (即为激光雷达 x 方向) 上的维度， 默认为 69.12/0.16 = 432
        :param ny: 伪图像的 y 方向 (即为激光雷达 y 方向) 上的维度， 默认为 39.68 * 2/0.16 = 496
        """
        super().__init__()
        self._in_channels = in_channels

        assert abs(float(int(nx)) - nx) <= 1e-5
        assert abs(float(int(ny)) - ny) <= 1e-5

        self._nx = int(nx)
        self._ny = int(ny)

    def forward(self, batch_pillar_features, batch_indices, sample_indices, batch_size):
        """
        将点云特征重新转化为伪图像格式

        :param batch_pillar_features: 类型为torch.Tensor，指的是一个batch中所有点云的特征表征，维度为 (M, 64), M 为该 batch 中
                                      所有样本进行体素化之后的体素数量总和. 例如 batch_size = 4时，四个样本的体素化后体素数量
                                      分别为 m1, m2, m3, m4. 则 M = m1 + m2 + m3 + m4.
        :param batch_indices: 类型为torch.Tensor，形状为 (M, 3), 表示每一个体素（Pillar）对应的坐标索引，顺序为 z, y, x
        :param sample_indices: torch.Tensor， 形状为(M, ), 表示当前体素属于 batch 中的哪一个，即索引
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
            batch_mask = sample_indices == sample_index
            this_coords = batch_indices[batch_mask, :]              # 形状为 (Q, 3), 3 表示 z, y, x 网格坐标， 其中Q对于Batch中各个样本各不相同
            pillar_feats = batch_pillar_features[batch_mask, :]     # 形状为 (Q, 64)，其中Q对于Batch中各个样本各不相同
            transposed_pillar_feats = pillar_feats.t()              # 形状为 (64, Q), Q指的是当前样本的体素数量

            # 全局Zig-Zag 索引，indices = X * ny + Y, 参考docs/scatter.png说明
            indices = this_coords[:, 2] * self._ny + this_coords[:, 1]
            indices = indices.type(torch.long)

            canvas[:, indices] = transposed_pillar_feats
            batch_canvas.append(canvas)

        # 将画布转化为三维，转化为形状为(batch_size, C, self._nx * self._ny)
        batch_canvas = torch.stack(batch_canvas, 0)

        # 将画布转化为四维，维度信息为 (batch_size, C, nx, ny), 文章中 C=64
        batch_canvas = batch_canvas.view(batch_size, self._in_channels, self._nx, self._ny)

        # 索引反向过来
        batch_canvas = torch.flip(batch_canvas, dims=[2])

        return batch_canvas
