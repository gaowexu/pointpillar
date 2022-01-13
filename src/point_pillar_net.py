import torch
from torch import nn
from torch.nn import functional as F


def get_paddings_indicator(actual_num, max_num, axis=0):
    """
    Create boolean mask by actually number of a padded tensor.

    :param actual_num: torch.Tensor, actual number of points in each voxel/pillar.
    :param max_num: int, maximum number of points in each voxel/pillar
    :param axis:
    :return: torch.Tensor: Mask indicates which points are valid inside a voxel/pillar.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)

    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)

    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num

    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class PFNLayer(nn.Module):
    """
    Pillar Feature Net Layer: A feature Encoder Network
    As introduced in paper: "we use a simplified version of PointNet where, for each point, a linear
    layer is applied followed by BatchNorm and ReLu to generate a (C,P,N) sized tensor. This is followed
    by a max operation over the channels to create an output tensor of size (C,P)"
    """
    def __init__(self, in_channels, out_channels, mode="max"):
        """
        Constructor

        :param in_channels:
        :param out_channels:
        :param mode:
        """
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._mode = mode
        assert self._mode in ['max', 'avg']

        self._linear = nn.Linear(self._in_channels, self._out_channels, bias=False)
        self._norm = nn.BatchNorm1d(self._out_channels, eps=1e-3, momentum=0.01)

    def forward(self, inputs, num_points):
        """

        :param inputs: torch.Tensor, shape is (num_pillars, max_num_points, 9)
        :param num_points: 1D tensor with the actual number of points in each pillar.
        :return:
        """
        x = self._linear(inputs)
        x = self._norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        return x_max


class PointPillarFeatureNet(nn.Module):
    """
    Implementation of the Pillar Feature Network which is shown in Figure 2 in paper
    Lang, Alex H., et al. "PointPillars: Fast encoders for object detection from point clouds."
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
    """
    def __init__(self,
                 in_channels=4,
                 feat_channels=64,
                 bin_size=(0.16, 0.16),
                 point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1)):
        """
        Constructor

        :param in_channels: Number of input features, x, y, z, r (reflectance). Defaults to 4.
        :param feat_channels: Number of features in each of the N PFNLayers. Defaults to 64.
        :param bin_size: Size of bins, unit: meters. Defaults to (0.16, 0.16).
        :param point_cloud_range: Point cloud range, (x_min, y_min, z_min, x_max, y_max, z_max).
                                  Defaults to (0, -39.68, -3, 69.12, 39.68, 1).
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
        self._pfn_layers = nn.ModuleList([
            PFNLayer(
                in_channels=self._in_channels,
                out_channels=self._feat_channels,
                mode='max'
            )
        ])

        # Need pillar size and x/y offset in order to calculate offset from pillar x/y center
        self._vx = bin_size[0]
        self._vy = bin_size[1]
        self._x_offset = self._vx / 2 + point_cloud_range[0]
        self._y_offset = self._vy / 2 + point_cloud_range[1]

    def forward(self, features, num_points, coords):
        """
        Forward function

        :param features: torch.Tensor, a dense list of point coordinates and features for each pillar.
                         The shape is (num_pillars, max_num_points, 4). Attention: num_pillars here is different with
                         the maximum number of pillars P noted in paper. It is the number of elements in dense list.
        :param coords: coordinates of each pillar, torch.Tensor with shape (num_pillars, 4),
                       4 indicates [sample_index, z, y, x]
        :return:
        """
        features_ls = [features]

        # Find distance to the arithmetic mean of all points in pillars, i.e., the feature x_c, y_c, z_c in paper.
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean            # with shape (num_pillars, max_num_points, 3)
        features_ls.append(f_cluster)

        # Calculate the offset for the pillar x, y center, i.e., the feature x_p, y_p in paper.
        feats_offset = features[:, :, :2].clone().detach()      # with shape (num_pillars, max_num_points, 2)
        feats_offset[:, :, 0] = feats_offset[:, :, 0] - (coords[:, 3].type_as(features).unsqueeze(1) * self._vx +
                                                         self._x_offset)
        feats_offset[:, :, 1] = feats_offset[:, :, 1] - (coords[:, 2].type_as(features).unsqueeze(1) * self._vy +
                                                         self._y_offset)
        features_ls.append(feats_offset)

        # Combine together feature decorations, now features is with shape (num_pillars, max_num_points, 4 + 5)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)      # with shape (num_pillars, max_num_points, 1)
        features *= mask

        for pfn in self._pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze(dim=1)
