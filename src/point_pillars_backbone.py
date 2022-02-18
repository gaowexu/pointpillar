import torch
from torch import nn
import numpy as np


class PointPillarBackbone(nn.Module):
    """
    对伪图像点云特征图进行二维卷积进一步提取特征，输出的特征图送入Detection Head进行目标检测，如论文原文所述:
    The car and pedestrian/cyclist backbones are the same except for the stride of the first block (S=2 for car,
    S=1 for pedestrian/cyclist).

    Both network consists of three blocks:
    Block1(S, 4, C), Block2(2S, 6, 2C), and Block3(4S, 6, 4C).
    A block Block(S , L, F ) has L 3x3 2D conv-layers with F output channels, each followed by BatchNorm and a ReLU.
    The first convolution inside the layer has stride S to ensure the block Sin operates on stride S after receiving
    an input blob of stride Sin. All subsequent convolutions in a block have stride 1.

    Each block is upsampled by the following upsampling steps:
    Up1(S, S, 2C), Up2(2S, S, 2C) and Up3(4S, S, 2C).

    Then the features of Up1, Up2 and Up3 are concatenated together to create 6C features for the detection head.
    """
    def __init__(self,
                 pillar_feats_channels=64,
                 down_sample_out_channels=None,
                 layers_num_in_blocks=None,
                 down_sample_strides=None,
                 up_sample_out_channels=None,
                 up_sample_strides=None,
                 use_conv_for_no_stride=False):
        """
        构造函数

        :param pillar_feats_channels: pillar中点云数据经过特征提取后的特征维度，默认为64
        :param down_sample_out_channels: 卷积block(每一个block中包含数个卷积层）的输出维度
        :param layers_num_in_blocks: 每一个block中包含的卷积层数量，对应者论文中Block(S, L, F)中的L
        :param down_sample_strides: 每一个block中第一个卷积层的stride
        :param up_sample_out_channels: 三个上采样（反卷积）的输出通道数
        :param up_sample_strides: 反卷积的stride
        :param use_conv_for_no_stride:
        """
        super().__init__()
        if up_sample_strides is None:
            up_sample_strides = [1, 2, 4]
        if up_sample_out_channels is None:
            up_sample_out_channels = [128, 128, 128]
        if down_sample_strides is None:
            down_sample_strides = [2, 2, 2]
        if layers_num_in_blocks is None:
            layers_num_in_blocks = [4, 6, 6]
        if down_sample_out_channels is None:
            down_sample_out_channels = [64, 128, 256]

        self._pillar_feats_channels = pillar_feats_channels
        self._down_sample_out_channels = down_sample_out_channels
        self._layers_num_in_blocks = layers_num_in_blocks
        self._down_sample_strides = down_sample_strides
        self._up_sample_strides = up_sample_strides
        self._up_sample_in_channels = down_sample_out_channels
        self._up_sample_out_channels = up_sample_out_channels

        # 卷积部分
        # 卷积层的输入通道数，默认为 [64, 64, 128]
        self._in_filters = [self._pillar_feats_channels, *down_sample_out_channels[:-1]]
        down_sample_blocks = list()
        for i, layer_num in enumerate(self._layers_num_in_blocks):
            block = [
                nn.Conv2d(self._in_filters[i],
                          self._down_sample_out_channels[i],
                          3,
                          bias=False,
                          stride=self._down_sample_strides[i],
                          padding=1),
                nn.BatchNorm2d(self._down_sample_out_channels[i], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num-1):
                block.append(
                    nn.Conv2d(self._down_sample_out_channels[i],
                              self._down_sample_out_channels[i],
                              3,
                              bias=False,
                              padding=1))
                block.append(
                    nn.BatchNorm2d(self._down_sample_out_channels[i], eps=1e-3, momentum=0.01))
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            down_sample_blocks.append(block)

        self._down_sampling_blocks = nn.ModuleList(down_sample_blocks)

        # 反卷积部分
        up_sample_blocks = list()
        for i, out_channel in enumerate(self._up_sample_out_channels):
            stride = self._up_sample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                up_sample_layer = nn.ConvTranspose2d(
                    in_channels=self._up_sample_in_channels[i],
                    out_channels=out_channel,
                    kernel_size=self._up_sample_strides[i],
                    stride=self._up_sample_strides[i],
                    bias=False)
            else:
                stride = np.round(1 / stride).astype(np.int64)
                up_sample_layer = nn.Conv2d(in_channels=self._up_sample_in_channels[i],
                                            out_channels=out_channel,
                                            kernel_size=stride,
                                            stride=stride,
                                            bias=False)

            deblock = nn.Sequential(
                up_sample_layer,
                nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True))
            up_sample_blocks.append(deblock)
        self._up_sample_blocks = nn.ModuleList(up_sample_blocks)
        self.init_weights()

    def init_weights(self):
        """
        初始化卷积层中参数

        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, batch_canvas):
        """
        前向推理函数

        :param batch_canvas: 一个batch的点云伪图，其形状为 (batch_size, C, nx, ny)
        :return: 输出维度为 (batch_size, 6C, nx/2, ny/2)
        """
        block_0_out = self._down_sampling_blocks[0](batch_canvas)
        block_1_out = self._down_sampling_blocks[1](block_0_out)
        block_2_out = self._down_sampling_blocks[2](block_1_out)

        up_0_out = self._up_sample_blocks[0](block_0_out)
        up_1_out = self._up_sample_blocks[1](block_1_out)
        up_2_out = self._up_sample_blocks[2](block_2_out)

        out = torch.cat([up_0_out, up_1_out, up_2_out], dim=1)
        return out

