import torch
from torch import nn
from torch.nn import functional as F


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

    def forward(self, inputs):
        """

        :param inputs: torch.Tensor, shape = (
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
    Lang, Alex H., et al. "Pointpillars: Fast encoders for object detection from point clouds."
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
        self.pfn_layers = nn.ModuleList([
            PFNLayer(
                in_channels=self._in_channels,
                out_channels=self._feat_channels,
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
        device = features.device

        dtype = features.dtype
        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze()


class PointPillarScatter(nn.Module):
    """
    Converts learned features (output of PFNLayer) from dense tensor to sparse pseudo image.
    """
    def __init__(self, in_channels=64, output_shape=(496, 432)):
        """
        Constructor

        :param in_channels: Channels of input features. Defaults to 64.
        :param output_shape: Required output shape of features. Defaults to (69.12/0.16, 39.68 * 2/0.16),
                             i.e, (y_cols, x_cols)
        """
        super().__init__()
        self._in_channels = in_channels
        self._output_shape = output_shape
        self._ny = output_shape[0]
        self._nx = output_shape[1]

    def forward(self, pillar_features, coords, batch_size):
        """
        Scatter features of single sample

        :param pillar_features: torch.Tensor, pillar features in shape (P, N, D), P is the maximum number of pillars,
                                N is the maximum number of points per pillar, and D is decorated lidar point feature
                                dimension.
        :param coords: Coordinates of each pillar in shape (N, 4). 4 indicates (batch_index, x, y, z)
        :param batch_size: Number of samples in the current batch.
        :return:
        """
        batch_canvas = list()

        for batch_idx in range(batch_size):
            # Create the canvas for this sample, canvas shape = [64, 214272]
            canvas = torch.zeros(
                self._in_channels,
                self._nx * self._ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self._nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            transposed_voxel_features = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = transposed_voxel_features

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor, batch_canvas shape is (batch_size, 64, 496, 432)
        batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.ny, self.nx)

        return batch_canvas
