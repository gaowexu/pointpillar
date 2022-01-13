import torch
from torch import nn
from open3d.ml.torch.ops import voxelize, ragged_to_dense


class PointPillarVoxelization(nn.Module):
    """
    Points cloud voxelization class
    """
    def __init__(self,
                 voxel_size=(0.16, 0.16, 4.0),
                 point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1),
                 max_num_points=100,
                 max_voxels=12000):
        """
        Constructor

        :param voxel_size: voxel edge lengths with format [x, y, z].
        :param point_cloud_range: The valid range of point coordinates as [x_min, y_min, z_min, x_max, y_max, z_max].
        :param max_num_points: The maximum number of points per pillar.
        :param max_voxels: The maximum number of voxels. May be a tuple with values for training and testing.
        """
        super().__init__()
        self._voxel_size = torch.Tensor(voxel_size)
        self._point_cloud_range = torch.Tensor(point_cloud_range)
        self._max_num_points = max_num_points
        self._points_range_min = torch.Tensor(point_cloud_range[:3])
        self._points_range_max = torch.Tensor(point_cloud_range[3:])
        self._max_voxels = max_voxels

    def forward(self, points_feats):
        """
        Forward function

        :param points_feats: Tensor with point coordinates and features. The shape is [N, 4] with N as the number
                             of points. Here 4 indicates [x, y, z, reflectance].
        :return: (out_pillars, out_coords, out_num_points)
                 * out_pillars is a dense list of point coordinates and features for each pillar.
                   The shape is [num_pillars, max_num_points, 4]. Attention: num_pillars here is different with
                   the maximum number of pillars P noted in paper. It is the number of elements in dense list.
                 * out_coords is tensor with the integer pillars coords and shape [num_voxels,3].
                   Note that the order of dims is [z,y,x].
                 * out_num_points is a 1D tensor with the number of points in each pillar.

                A Typical (out_pillars, out_coords, out_num_points) tuple would be like:

                out_voxels = tensor([[[0.1000, 0.1000, 0.1000],
                         [0.1200, 0.1300, 0.4100],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000]],

                        [[0.5000, 0.5000, 0.5000],
                         [0.9000, 0.8000, 0.7500],
                         [0.5000, 0.5000, 0.5000],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000]],

                        [[1.7000, 1.7000, 1.7000],
                         [1.8000, 1.8000, 1.8000],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000]],

                        [[2.3000, 2.1000, 2.4000],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000]]])
                out_coords = tensor([[0, 0, 0],
                        [1, 1, 1],
                        [3, 3, 3],
                        [4, 4, 4]], dtype=torch.int32)
                out_num_points = tensor([2, 3, 2, 1])
        """
        # Points with shape (N, 3)
        points = points_feats[:, :3]

        # (nx, ny, nz)
        num_voxels = ((self._points_range_max - self._points_range_min) / self._voxel_size).type(torch.int32)

        ans = voxelize(
            points=points,
            row_splits=torch.LongTensor([0, points.shape[0]]).to(points.device),
            voxel_size=self._voxel_size,
            points_range_min=self._points_range_min,
            points_range_max=self._points_range_max,
            max_voxels=self._max_voxels
        )

        # Prepend row with zeros which maps to index 0 which maps to void points
        feats = torch.cat([torch.zeros_like(points_feats[0:1, :]), points_feats])

        # Create dense matrix of indices. index 0 maps to the zero vector
        voxels_point_indices_dense = ragged_to_dense(
            values=ans.voxel_point_indices,
            row_splits=ans.voxel_point_row_splits,
            out_col_size=self._max_num_points,
            default_value=torch.tensor(-1)
        ) + 1

        out_voxels = feats[voxels_point_indices_dense]

        # Convert [x,y,z] to [z,y,x] order
        out_coords = ans.voxel_coords[:, [2, 1, 0]].contiguous()
        out_num_points = ans.voxel_point_row_splits[1:] - ans.voxel_point_row_splits[:-1]

        # Filter out pillars generated out of bounds of the pseudo image.
        in_bounds_y = out_coords[:, 1] < num_voxels[1]
        in_bounds_x = out_coords[:, 2] < num_voxels[0]
        in_bounds = torch.logical_and(in_bounds_x, in_bounds_y)

        out_coords = out_coords[in_bounds]
        out_voxels = out_voxels[in_bounds]
        out_num_points = out_num_points[in_bounds]

        return out_voxels, out_coords, out_num_points
