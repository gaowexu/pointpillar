import torch
from torch import nn


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
        :param coords: Coordinates of each pillar, torch.Tensor with shape (num_pillars, 4), 4 indicates
                       [sample_index, z, y, x]
        :param batch_size: Integer, Number of samples in the current batch.
        :return:
        """
        batch_canvas = list()

        for batch_idx in range(batch_size):
            # Create the canvas for this sample, canvas shape = [64, 214272]
            canvas = torch.zeros(
                self._in_channels,
                self._nx * self._ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self._nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = pillar_features[batch_mask, :]
            transposed_voxel_features = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = transposed_voxel_features

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor, batch_canvas shape is (batch_size, 64, 496, 432)
        batch_canvas = batch_canvas.view(batch_size, self._in_channels, self._ny, self._nx)

        return batch_canvas
