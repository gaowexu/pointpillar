## PyTorch Implementation of PointPillars
*Solutions Architect: Gaowei Xu (gaowexu1991@gmail.com)*

### Reference:
- https://github.com/tyagi-iiitv/PointPillars
- https://github.com/traveller59/second.pytorch
- https://github.com/hova88/PointPillars_MultiHead_40FPS
- https://github.com/SmallMunich/nutonomy_pointpillars
- KITTI Beginners Tutorial: https://github.com/dtczhl/dtc-KITTI-For-Beginners
- Lang, Alex H., et al. "Pointpillars: Fast encoders for object detection from point clouds." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.


### Environment Setup



### Prepare KITTI Dataset
KITTI 3D object detection benchmark consists of 7481 training images and 7518 test images as well as 
the corresponding point clouds, comprising a total of 80,256 labeled objects. 

One can download the dataset following the [KITTI official website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d), which contains four parts for 3D object detection task:
- Download left color images of object data set (12 GB)
- Download Velodyne point clouds, if you want to use laser information (29 GB)
- Download camera calibration matrices of object data set (16 MB)
- Download training labels of object data set (5 MB)

The corresponding download links are listed below:
```angular2html
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
```

Alternatively, we already downloaded all these datasets and constructed a compressed package,
one can download it with link AAA. After de-compressing it, the directories is:
```angular2html
KITTI_DATASET/
├── testing      <-- 7580 test data
│   ├── calib
│   ├── image_2  <-- for visualization
│   └── velodyne
└── training     <-- 7481 train data
    ├── calib
    ├── image_2  <-- for visualization
    ├── label_2
    └── velodyne
```

A minimum sampled dataset could be downloaded from 

### Model Training





### Model Inference





### TensorRT Acceleration




### Performance Summary 




### Detection Visualization


### License
See the [LICENSE](./LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.

