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
Launch an EC2 with OS Ubuntu 20.04 with GPU supported, and then download PyTorch 1.7.1 with CUDA 11.0 complied: https://github.com/isl-org/open3d_downloads/releases/tag/torch1.7.1

Install open3d and torch with following commands:
```angular2html
pip3 install open3d==0.14.1
pip3 install torch-1.7.1-cp38-cp38-linux_x86_64.whl
```

### Prepare KITTI Dataset
KITTI 3D object detection benchmark consists of 7481 training images and 7518 test images as well as 
the corresponding point clouds, comprising a total of 80,256 labeled objects. 

For the detail about the coordinate system definition, please refer to [Vision meets robotics: The KITTI
dataset](./references/Vision%20meets%20robotics-%20The%20KITTI%20dataset.pdf)

The sensors setup could be referred as the illustration figures below:

![recording_plaform](./setup.png)

- Camera: x = right, y = down, z = forward
- Velodyne: x = forward, y = left, z = up

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
one can download it with link `s3://autonomous-driving-perception/KITTI_3D_OBJECT_DETECTION_DATASET.zip` (`md5: 3033523e4bbf0696443b1d10ab972fe9`). After de-compressing it, the directories are:
```angular2html
KITTI_DATASET/
????????? testing      <-- 7580 test data
???   ????????? calib
???   ????????? image_2  <-- for visualization
???   ????????? velodyne
????????? training     <-- 7481 train data
    ????????? calib
    ????????? image_2  <-- for visualization
    ????????? label_2
    ????????? velodyne
```

A minimum sampled dataset could be downloaded from `s3://autonomous-driving-perception/KITTI_3D_OBJECT_DETECTION_SAMPLED_DATASET.zip`

### Model Training





### Model Inference





### TensorRT Acceleration




### Performance Summary 




### Detection Visualization



### TODO List
- [x] KITTI data inspection
- [x] voxelization
- [x] point pillar net (PFNLayer)
- [x] Scatter
- [x] 2D backbone
- [ ] anchor head single (i.e., anchor generation)
- [ ] axis aligned target assigner (for training)
- [ ] post-processing (NMS)
- [ ] loss function
- [ ] train code
- [ ] augmentation

### License
See the [LICENSE](./LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.

