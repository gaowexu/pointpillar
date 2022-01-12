import torch
import open3d.ml.torch as ml3d
from open3d.ml.torch.ops import voxelize, ragged_to_dense


points = torch.Tensor([
  [0.1, 0.1, 0.1],
  [0.5, 0.5, 0.5],
  [1.7, 1.7, 1.7],
  [1.8, 1.8, 1.8],
  [9.3, 9.4, 9.4]])

row_splits = torch.Tensor([0, 2, 5]).to(torch.int64)

ans = ml3d.ops.voxelize(
    points=points,
    row_splits=row_splits,
    voxel_size=torch.Tensor([1.0, 1.0, 1.0]),
    points_range_min=torch.Tensor([0, 0, 0]),
    points_range_max=torch.Tensor([2, 2, 2]))


voxel_coords = ans.voxel_coords
voxel_point_indices = ans.voxel_point_indices
voxel_point_row_splits = ans.voxel_point_row_splits
voxel_batch_splits = ans.voxel_batch_splits

print("voxel_point_indices = {}\n".format(voxel_point_indices))
print("voxel_coords = {}\n".format(voxel_coords))
print("voxel_point_row_splits = {}\n".format(voxel_point_row_splits))
print("voxel_batch_splits = {}".format(voxel_batch_splits))

# Prepend row with zeros which maps to index 0 which maps to void points
feats = torch.cat([torch.zeros_like(points[0:1, :]), points])

# Create dense matrix of indices. index 0 maps to the zero vector
max_num_points = 32
voxels_point_indices_dense = ragged_to_dense(
    values=ans.voxel_point_indices,
    row_splits=ans.voxel_point_row_splits,
    out_col_size=max_num_points,
    default_value=torch.tensor(-1)
) + 1

out_voxels = feats[voxels_point_indices_dense]
out_coords = ans.voxel_coords[:, [2, 1, 0]].contiguous()
out_num_points = ans.voxel_point_row_splits[1:] - ans.voxel_point_row_splits[:-1]

print("out_voxels = {}".format(out_voxels))
print("out_coords = {}".format(out_coords))
print("out_num_points = {}".format(out_num_points))
