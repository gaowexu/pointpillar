
# 全局配置参数
CONFIG_PARAMS = {
    # 训练过程中考虑的激光点云物理距离，[x_min, y_min, z_min, x_max, y_max, z_max]，落在该范围之外的点云不作考虑，
    # 其中x正方向为车正前方，y正方向为车左侧，z正方向为垂直向上
    "POINT_CLOUD_RANGE": [0, -39.68, -3, 69.12, 39.68, 1.0],

    # 体素大小，尺度为 x,y,z 三个方向的分辨率
    "VOXEL_SIZE": [0.16, 0.16, 4.0],

    # 训练阶段和测试阶段的最大体素数量
    "MAX_NUMBER_OF_VOXELS": [16000, 40000],

    # 锚框生成器配置
    "ANCHOR_GENERATOR_CONFIG": [
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



}
