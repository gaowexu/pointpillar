from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class Projection(object):
    def __init__(self, image_full_path, calib_full_path, label_full_path, cloud_points_full_path):
        """
        Constructor

        :param image_full_path: full path of image
        :param calib_full_path: full path of calibration
        :param label_full_path: 3D object label file's full path
        :param cloud_points_full_path: cloud point full path
        """
        self._image_full_path = image_full_path
        self._calib_full_path = calib_full_path
        self._label_full_path = label_full_path
        self._cloud_points_full_path = cloud_points_full_path

    def run(self):
        left_camera_image = Image.open(self._image_full_path).convert('RGB')
        velo_cloud_points = np.fromfile(self._cloud_points_full_path, dtype=np.float32).reshape(-1, 4)
        print(velo_cloud_points[0:100, :])

        # plt.figure()
        # plt.imshow(left_camera_image)
        # plt.axis("off")
        # plt.show()

    def velodyne3d_to_cam2d(self):
        """
        P0: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 0.000000000000e+00 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
        P1: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.875744000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
        P2: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 4.485728000000e+01 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.163791000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.745884000000e-03
        P3: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.395242000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.199936000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.729905000000e-03
        R0_rect: 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03 7.402527000000e-03 4.351614000000e-03 9.999631000000e-01
        Tr_velo_to_cam: 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01
        Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01

        :return:
        """


    def cam2d_to_velodyne3d(self):
        pass


if __name__ == "__main__":
    handler = Projection(
        image_full_path="../dataset/KITTI_3D_OBJECT_DETECTION_SAMPLED_DATASET/training/image_2/000011.png",
        calib_full_path="../dataset/KITTI_3D_OBJECT_DETECTION_SAMPLED_DATASET/training/calib/000011.txt",
        label_full_path="../dataset/KITTI_3D_OBJECT_DETECTION_SAMPLED_DATASET/training/label_2/000011.txt",
        cloud_points_full_path="../dataset/KITTI_3D_OBJECT_DETECTION_SAMPLED_DATASET/training/velodyne/000011.bin",
    )

    handler.run()



