import torch
import open3d.ml.torch as ml3d
from open3d.ml.torch.ops import voxelize, ragged_to_dense
import torch.nn.functional as F
from point_pillar_net import get_paddings_indicator

points = torch.Tensor([
  [0.1, 0.1, 0.1],
  [0.12, 0.13, 0.41],
  [0.5, 0.5, 0.5],
  [0.9, 0.8, 0.75],
  [2.3, 2.1, 2.4],
  [0.5, 0.5, 0.5],
  [1.7, 1.7, 1.7],
  [1.8, 1.8, 1.8],
  [9.3, 9.4, 9.4]])

row_splits = torch.LongTensor([0, points.shape[0]])
print("row_splits = {}".format(row_splits))

ans = ml3d.ops.voxelize(
    points=points,
    row_splits=row_splits,
    voxel_size=torch.Tensor([0.5, 0.5, 0.5]),
    points_range_min=torch.Tensor([0, 0, 0]),
    points_range_max=torch.Tensor([2.5, 2.5, 2.5]))


voxel_coords = ans.voxel_coords
voxel_point_indices = ans.voxel_point_indices
voxel_point_row_splits = ans.voxel_point_row_splits
voxel_batch_splits = ans.voxel_batch_splits

# print("voxel_point_indices = {}\n".format(voxel_point_indices))
# print("voxel_coords = {}\n".format(voxel_coords))
# print("voxel_point_row_splits = {}\n".format(voxel_point_row_splits))
# print("voxel_batch_splits = {}".format(voxel_batch_splits))

# Prepend row with zeros which maps to index 0 which maps to void points
feats = torch.cat([torch.zeros_like(points[0:1, :]), points])

# Create dense matrix of indices. index 0 maps to the zero vector
max_num_points = 16
voxels_point_indices_dense = ragged_to_dense(
    values=ans.voxel_point_indices,
    row_splits=ans.voxel_point_row_splits,
    out_col_size=max_num_points,
    default_value=torch.tensor(-1)
) + 1

print("voxels_point_indices_dense = {}".format(voxels_point_indices_dense))

out_voxels = feats[voxels_point_indices_dense]
out_coords = ans.voxel_coords[:, [2, 1, 0]].contiguous()
out_num_points = ans.voxel_point_row_splits[1:] - ans.voxel_point_row_splits[:-1]

print("out_voxels = {}".format(out_voxels))
print("out_coords = {}".format(out_coords))
print("out_num_points = {}".format(out_num_points))

print("\n\n")

num_voxels = torch.Tensor([3, 2])
# Filter out pillars generated out of bounds of the pseudo image.
in_bounds_y = out_coords[:, 1] < num_voxels[1]
in_bounds_x = out_coords[:, 2] < num_voxels[0]
in_bounds = torch.logical_and(in_bounds_x, in_bounds_y)

out_coords = out_coords[in_bounds]
out_voxels = out_voxels[in_bounds]
out_num_points = out_num_points[in_bounds]
print("in_bounds = {}".format(in_bounds))
print("out_voxels = {}".format(out_voxels))
print("out_coords = {}".format(out_coords))
print("out_num_points = {}".format(out_num_points))


print("----------------------------")
print("out_voxels.shape = {}".format(out_voxels.shape))
print("out_num_points.shape = {}".format(out_num_points.shape))
print("out_coords.shape = {}".format(out_coords.shape))
print("\n")
voxels = [out_voxels, out_voxels, out_voxels]
coors = [out_coords, out_coords, out_coords]
num_points = [out_num_points, out_num_points, out_num_points]

voxels = torch.cat(voxels, dim=0)
num_points = torch.cat(num_points, dim=0)
print("voxels.shape = {}".format(voxels.shape))
print("num_points.shape = {}".format(num_points.shape))

coors_batch = []
for i, coor in enumerate(coors):
    coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
    coors_batch.append(coor_pad)
coors_batch = torch.cat(coors_batch, dim=0)
print("coors_batch.shape = {}".format(coors_batch.shape))

print("coors_batch = {}".format(coors_batch))

print("=====================")

features = voxels
features_ls = [features]

vx = 0.5
vy = 0.5
x_offset = 0.5 / 2
y_offset = 0.5 / 2

# Find distance to the arithmetic mean of all points in pillars, i.e., the feature x_c, y_c, z_c in paper.
points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1, 1)
f_cluster = features[:, :, :3] - points_mean  # with shape (num_pillars, max_num_points, 3)
features_ls.append(f_cluster)

# Calculate the offset for the pillar x, y center, i.e., the feature x_p, y_p in paper.
feats_offset = features[:, :, :2].clone().detach()  # with shape (num_pillars, max_num_points, 2)
feats_offset[:, :, 0] = feats_offset[:, :, 0] - (coors_batch[:, 3].type_as(features).unsqueeze(1) * vx +
                                                 x_offset)
feats_offset[:, :, 1] = feats_offset[:, :, 1] - (coors_batch[:, 2].type_as(features).unsqueeze(1) * vy +
                                                 y_offset)
features_ls.append(feats_offset)

# Combine together feature decorations
features = torch.cat(features_ls, dim=-1)

print("features = {}".format(features))
print("features.shape = {}".format(features.shape))

voxel_count = features.shape[1]
mask = get_paddings_indicator(num_points, voxel_count, axis=0)
print("mask = {}".format(mask))
print("mask.shape = {}".format(mask.shape))
mask = torch.unsqueeze(mask, -1).type_as(features)
print("mask = {}".format(mask))
print("mask.shape = {}".format(mask.shape))
features *= mask
print("features = {}".format(features))
print("features.shape = {}".format(features.shape))

print("num_points = {}".format(num_points))




