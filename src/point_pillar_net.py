import torch
from torch import nn
from torchvision import models


class PFNLayer(nn.Module):
    """
    Pillar Feature Net Layer: A feature Encoder Network
    As introduced in paper: "we use a simplified version of PointNet where, for each point, a linear
    layer is applied followed by BatchNorm and ReLu to generate a (C,P,N) sized tensor. This is followed
    by a max operation over the channels to create an output tensor of size (C,P)"
    """
    def __init__(self, in_channels, out_channels, last_layer=False, mode="max"):
        """
        Constructor

        :param in_channels:
        :param out_channels:
        :param last_layer:
        :param mode:
        """
        super().__init__()
        self.name = 'PFNLayer'
        self.mode = mode
        assert self.mode in ['max', 'avg']

    def forward(self, inputs, ):



class PointPillarFeatureNet(nn.Module):
    """
    Implementation of the Pillar Feature Network which is shown in Figure 2 in paper
    Lang, Alex H., et al. "Pointpillars: Fast encoders for object detection from point clouds."
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
    """
    def __init__(self,
                 in_channels=4,
                 feat_channels=64,
                 bin_size=(0.16, 0.16),
                 point_cloud_range=(0, -40.0, -3, 70.0, 40.0, 1)):
        """
        Constructor

        :param in_channels: Number of input features, x, y, z, r (reflectance). Defaults to 4.
        :param feat_channels: Number of features in each of the N PFNLayers. Defaults to 64.
        :param bin_size: Size of bins, unit: meters. Defaults to (0.16, 0.16).
        :param point_cloud_range: Point cloud range, (x_min, y_min, z_min, x_max, y_max, z_max).
                                  Defaults to (0, -40, -3, 70.0, 40, 1).
        """
        super(PointPillarFeatureNet, self).__init__()
        assert feat_channels > 0
        assert in_channels == 4

        # As is introduced in paper: "The points in each pillar are then decorated (augmented) with r, x_c, y_c,
        # z_c, x_p, y_p where r is reflectance, the c subscript denotes distance to the arithmetic mean of all
        # points in the pillar, and the p subscript denotes the offset from the pillar x,y center". i.e., D = 9
        self._in_channels = in_channels + 5

        self._feat_channels = feat_channels
        self._bin_size = bin_size
        self._point_cloud_range = point_cloud_range

        # The feature encoding module could be composed of a series of these layers, but the PointPillars paper
        # only used a single PFNLayer.
        self.pfn_layers = nn.ModuleList([
            PFNLayer(
                in_channels=self._in_channels,
                out_channels=self._feat_channels,
                last_layer=True,
                mode='max'
            )
        ])

    def forward(self, inputs, num_points, coords):
        """
        forward function

        :param inputs: torch.Tensor,
        :param num_points:
        :param coords:
        :return:
        """
        pass


class PointPillarScatter(nn.Module):
    def __init__(self):
        pass

    def forward






#
# class PointPillarNetwork(nn.Module):
#     def __init__(self):
#         super(CustomClassificationModel, self).__init__()
#
#         self._model = models.resnet50(pretrained=True)
#
#         # adapt the last fully connected layer to customized dataset
#         fc_inputs = self._model.fc.in_features
#         self._model.fc = nn.Sequential(
#             nn.Linear(fc_inputs, 256),
#             nn.ReLU(),
#             nn.Linear(256, 6)           # output dimension 6 should be the same with our classification task
#         )
#
#     def forward(self, x):
#         logits = self._model(x)
#         probs = torch.nn.Softmax(dim=1)(logits)
#         return logits, probs
