import torch
from torch import nn
from point_pillar_net import PointPillarFeatureNet, PointPillarScatter
from point_pillars_voxelization import PointPillarVoxelization


class PointPillars(nn.Module):
    def __init__(self,
                 name="PointPillars",
                 device="cuda",
                 point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1),
                 voxel_size=(0.16, 0.16, 4.0),
                 max_num_points=100,
                 max_pillars=12000):
        self._name = name
        self._device = device
        self._point_cloud_range = point_cloud_range
        self._voxel_size = voxel_size
        self._max_num_points = max_num_points
        self._max_pillars = max_pillars

        self._point_pillars_voxelizer = PointPillarVoxelization(
            voxel_size=self._voxel_size,
            point_cloud_range=self._point_cloud_range,
            max_num_points=self._max_num_points,
            max_voxels=self._max_pillars
        )

        self._point_pillars_feature_net = PointPillarFeatureNet(
            in_channels=4,
            feat_channels=64,
            bin_size=self._voxel_size[0:2],
            point_cloud_range=self._point_cloud_range
        )

        self._middle_encoder = PointPillarScatter(

        )

    @torch.no_grad()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_feats(self, points):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        x = self.neck(x)
        return x

    def forward(self, inputs):
        inputs = inputs.point
        x = self.extract_feats(inputs)
        outs = self.bbox_head(x)
        return outs

    def loss(self, ground_truth, predictions):
        pass











