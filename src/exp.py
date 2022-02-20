import torch

anchor_generator_config = [
            {
                'class_name': 'Car',
                'anchor_sizes': [[3.9, 1.6, 1.56]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-1.78],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.6,
                'unmatched_threshold': 0.45
            },
            {
                'class_name': 'Pedestrian',
                'anchor_sizes': [[0.8, 0.6, 1.73]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.6],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            },
            {
                'class_name': 'Cyclist',
                'anchor_sizes': [[1.76, 0.6, 1.73]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.6],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            }
        ]

import numpy as np
grid_size = torch.from_numpy(np.array([432,  496]))
feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_config]

anchor_sizes = [config['anchor_sizes'] for config in anchor_generator_config]
anchor_rotations = [config['anchor_rotations'] for config in anchor_generator_config]
anchor_bottom_heights = [config['anchor_bottom_heights'] for config in anchor_generator_config]
align_center = [config.get('align_center', False) for config in anchor_generator_config]

print('feature_map_size = {}'.format(feature_map_size))
print('anchor_sizes = {}'.format(anchor_sizes))
print('anchor_rotations = {}'.format(anchor_rotations))
print('anchor_bottom_heights = {}'.format(anchor_bottom_heights))
print('align_center = {}'.format(align_center))


