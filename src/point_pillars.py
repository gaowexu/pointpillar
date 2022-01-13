import torch
from torch import nn
from point_pillar_net import PointPillarFeatureNet
from point_pillars_scatter import PointPillarScatter
from point_pillars_voxelization import PointPillarVoxelization
from torch.nn.functional import pad


class PointPillars(nn.Module):
    def __init__(self,
                 name="PointPillars",
                 device="cuda",
                 point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1),
                 voxel_size=(0.16, 0.16, 4.0),
                 max_num_points=100,
                 max_pillars=12000):
        super().__init__()
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

        self._point_pillars_scatter = PointPillarScatter(
            in_channels=64,
            output_shape=torch.Tensor([69.12/0.16, 39.68 * 2/0.16]).type(torch.int)
        )

    @torch.no_grad()
    def voxelize(self, raw_points_batch):
        """
        Apply hard voxelization to points

        :param raw_points_batch: List, a batch of points cloud, each element is a torch.Tensor with shape (Q, 4).
                                 Q is the original amount of points in a single frame, and for different samples,
                                 Q is usually different.
        :return: (batch_of_voxels, batch_of_num_points, batch_of_coords)
                 * batch_of_voxels: torch.Tensor with shape (U, max_num_points, 4), where U is the accumulated value of
                                    dense pillars' amount of all samples in input batch.
                 * batch_of_num_points: torch.Tensor, which is a 1-D tensor indicates how many points in each pillar.
                 * batch_of_coords: torch.Tensor with shape (U, 4), 4 indicates [sample_index, z, y, x]
        """
        batch_of_voxels, batch_of_num_points, batch_of_coords = list(), list(), list()

        for raw_points in raw_points_batch:
            # raw_points is with shape (Q, 4), Q is different for different samples
            # res_voxels is a torch.Tensor with shape (N, self._max_num_points, 4)
            # res_coors is a torch.Tensor with shape (N, 3)
            # res_num_points is a torch.Tensor with shape (N)
            # Note that for each raw_points, N is also different.
            res_voxels, res_coors, res_num_points = self._point_pillars_voxelizer(raw_points)

            batch_of_voxels.append(res_voxels)
            batch_of_num_points.append(res_num_points)
            batch_of_coords.append(res_coors)

        coors_batch_with_pad = []
        for i, coord in enumerate(batch_of_coords):
            coord_pad = pad(coord, (1, 0), mode='constant', value=i)
            coors_batch_with_pad.append(coord_pad)

        batch_of_voxels = torch.cat(batch_of_voxels, dim=0)
        batch_of_num_points = torch.cat(batch_of_num_points, dim=0)
        batch_of_coords = torch.cat(coors_batch_with_pad, dim=0)
        return batch_of_voxels, batch_of_num_points, batch_of_coords

    def extract_feats(self, batch_points):
        """
        Extract feature from samples in a batch

        :param batch_points: points cloud of a batch, with type of list in which each element is a tensor.Tensor
                             with shape (N, max_num_points, 4). Note that N is different for each sample.
        :return:
        """
        # Perform voxelization
        # batch_of_voxels's shape = (U, max_num_points, 4), where U is the accumulated value of dense pillars'
        # amount of all samples in input batch.
        # batch_of_num_points's shape = (U,)
        # batch_of_coords'shape = (U, 4)
        batch_of_voxels, batch_of_num_points, batch_of_coords = self.voxelize(batch_points)

        # Extract pillar features, output pillar_features's shape is (U, 64), 64 is the C in paper.
        pillar_features = self._point_pillars_feature_net(batch_of_voxels, batch_of_num_points, batch_of_coords)
        batch_size = batch_of_coords[-1, 0].item() + 1

        # Converts learned features (output of PFNLayer) from dense tensor to sparse pseudo image.
        # Output batch_canvas's shape is (batch_size, C, x_range/0.16, y_range/0.16), C=64 in paper.
        batch_canvas = self._point_pillars_scatter(
            pillar_features=pillar_features,
            coords=batch_of_coords,
            batch_size=batch_size)
        print("batch_canvas.shape = {}".format(batch_canvas.shape))



        # x = self.backbone(x)
        # x = self.neck(x)
        # return x

    def forward(self, inputs):
        """
        Forward function

        :param inputs: points cloud of a batch, with type of list in which each element is a tensor.Tensor with shape
                       (N, max_num_points, 4). Note that N is different for each sample.
        :return:
        """
        x = self.extract_feats(batch_points=inputs)
        # outs = self.bbox_head(x)
        # return outs

    def loss(self, ground_truth, predictions):
        pass


if __name__ == '__main__':
    import numpy as np
    with open('./temp/points_velodyne_000008.npy', 'rb') as f:
        points_sample_000008 = np.load(f)
        points_sample_000008 = points_sample_000008[np.where(points_sample_000008[:, 0] > 0)]

    with open('./temp/points_velodyne_000025.npy', 'rb') as f:
        points_sample_000025 = np.load(f)
        points_sample_000025 = points_sample_000025[np.where(points_sample_000025[:, 0] > 0)]

    batch_points = [torch.Tensor(points_sample_000008, device="cpu"), torch.Tensor(points_sample_000025, device="cpu")]

    handler = PointPillars()
    handler(inputs=batch_points)







