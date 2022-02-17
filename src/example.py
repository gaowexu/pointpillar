import numpy as np

from cumm import tensorview as tv
from spconv.utils import Point2VoxelCPU3d
from spconv.pytorch.utils import PointToVoxel, gather_features_by_pc_voxel_id
import torch
import numpy as np


np.random.seed(50051)
# voxel gen source code: spconv/csrc/sparse/pointops.py
gen = PointToVoxel(vsize_xyz=[1, 1, 4],
                   coors_range_xyz=[-10, -4, -2, 10, 4, 2],
                   num_point_features=4,
                   max_num_voxels=100,
                   max_num_points_per_voxel=5)

pc = np.array([
    [1.1, 1.9, 1.3, 121.34253],
])
print(pc.shape)
pc_th = torch.from_numpy(pc)
voxels, indices, num_per_voxel = gen(pc_th)

indices = indices.permute(3,2,1,0)

print("voxels = {}".format(voxels))
print("indices = {}".format(indices))
print("num_per_voxel = {}".format(num_per_voxel))




