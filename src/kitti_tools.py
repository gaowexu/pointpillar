import cv2
import numpy as np


class KITTITools(object):
    """
    Reference: https://github.com/dtczhl/dtc-KITTI-For-Beginners/blob/master/python/object_viewer.py
    """
    def __init__(self, image_full_path, calib_full_path, label_full_path, cloud_points_full_path):
        """
        Constructor

        :param image_full_path: full path of image
        :param calib_full_path: full path of calibration
        :param label_full_path: 3D object label file's full path
        :param cloud_points_full_path: cloud point full path, raw data is with form of [x0 y0 z0 r0 x1 y1 z1 r1 ...].
        """
        self._image_full_path = image_full_path
        self._calib_full_path = calib_full_path
        self._label_full_path = label_full_path
        self._cloud_points_full_path = cloud_points_full_path

    def load_calibration_matrix(self):
        """
        load calibration matrix

        P0 = gray_L (left gray camera)
        P1 = gray_R (right gray camera)
        P2 = rgb_L (left color camera)
        P3 = rgb_R (right color camera)

        :return: P0, P1, P2, P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo

        Tr_velo_to_cam is the RT (rotation/translation) matrix from cloud point to image

        For projecting a velodyne co-ordinate point into the camera-2 image:
        [u v 1]^T = P2 * R0_rect * Tr_velo_to_cam * [x y z 1]^T
        where [u, v]^T is the image pixel index (column/row), and [x, y, z]^T is the cloud point coordinate
        """
        with open(self._calib_full_path) as rf:
            all_lines = rf.readlines()

        P0 = np.matrix([float(x) for x in all_lines[0].strip('\n').split(' ')[1:]]).reshape(3, 4)
        P1 = np.matrix([float(x) for x in all_lines[1].strip('\n').split(' ')[1:]]).reshape(3, 4)
        P2 = np.matrix([float(x) for x in all_lines[2].strip('\n').split(' ')[1:]]).reshape(3, 4)
        P3 = np.matrix([float(x) for x in all_lines[3].strip('\n').split(' ')[1:]]).reshape(3, 4)

        R0_rect = np.matrix([float(x) for x in all_lines[4].strip('\n').split(' ')[1:]]).reshape(3, 3)
        Tr_velo_to_cam = np.matrix([float(x) for x in all_lines[5].strip('\n').split(' ')[1:]]).reshape(3, 4)
        Tr_imu_to_velo = np.matrix([float(x) for x in all_lines[6].strip('\n').split(' ')[1:]]).reshape(3, 4)

        # add a 1 in bottom-right, reshape R0_rect to 4 x 4
        R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0], axis=0)
        R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0, 1], axis=1)

        # add a row in bottom of Tr_velo_to_cam, reshape Tr_velo_to_cam to 4 x 4
        Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)

        return P0, P1, P2, P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo

    def load_annotation(self):
        """
        Load annotations of KITTI 3D object detection

        :return:
        """
        pass

    def run(self):
        # load calibration matrix
        P0, P1, P2, P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo = self.load_calibration_matrix()

        left_camera_image = cv2.imread(self._image_full_path, cv2.IMREAD_COLOR)         # BGR
        pc_data = np.fromfile(self._cloud_points_full_path, dtype='<f4').reshape(-1, 4)
        print(pc_data.shape)


if __name__ == "__main__":
    tools = KITTITools(
        image_full_path="../dataset/KITTI_3D_OBJECT_DETECTION_SAMPLED_DATASET/training/image_2/000011.png",
        calib_full_path="../dataset/KITTI_3D_OBJECT_DETECTION_SAMPLED_DATASET/training/calib/000011.txt",
        label_full_path="../dataset/KITTI_3D_OBJECT_DETECTION_SAMPLED_DATASET/training/label_2/000011.txt",
        cloud_points_full_path="../dataset/KITTI_3D_OBJECT_DETECTION_SAMPLED_DATASET/training/velodyne/000011.bin",
    )

    tools.run()
