import torch
from torch import nn


class PointPillarBackbone(nn.Module):
    """
    对伪图像点云特征图进行二维卷积进一步提取特征，输出的特征图送入Detection Head进行目标检测，如论文原文所述:
    The car and pedestrian/cyclist backbones are the same except for the stride of the first block (S=2 for car,
    S=1 for pedestrian/cyclist).

    Both network consists of three blocks:
    Block1(S , 4, C), Block2(2S , 6, 2C), and Block3(4S , 6, 4C).

    Each block is upsampled by the following upsampling steps:
    Up1(S, S, 2C), Up2(2S, S, 2C) and Up3(4S, S, 2C).

    Then the features of Up1, Up2 and Up3 are concatenated together to create 6C features for the detection head.
    """
    def __init__(self,
                 in_channels=64,
                 out_channels=[64, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2]):
        super(PointPillarBackbone, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                nn.Conv2d(in_filters[i],
                          out_channels[i],
                          3,
                          bias=False,
                          stride=layer_strides[i],
                          padding=1),
                nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    nn.Conv2d(out_channels[i],
                              out_channels[i],
                              3,
                              bias=False,
                              padding=1))
                block.append(
                    nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """
        前向推理函数

        :param batch_canvas: 一个batch的点云伪图，其形状为 (batch_size, C, ny, nx)
        :return:
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
            print("x.shape = {}".format(x.shape))

        return tuple(outs)


if __name__ == "__main__":
    model = PointPillarBackbone()

    print(model)
    dummy_data = torch.rand(8, 64, 496, 432)

    model(dummy_data)



