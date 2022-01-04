import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class KITTITools(object):
    """
    Tools class for KITTI dataset

    Reference:
    1. https://github.com/dtczhl/dtc-KITTI-For-Beginners/blob/master/python/object_viewer.py
    2. https://github.com/azureology/kitti-velo2cam
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

        self._categories_color = {
            'Car': [1, 0, 0],               # red
            'DontCare': [0, 0, 0],          # black
            'Pedestrian': [0, 0, 1],        # blue
            'Van': [1, 1, 0],               # yellow
            'Cyclist': [1, 0, 1],           # magenta
            'Truck': [0, 1, 1],             # cyan
            'Misc': [0.5, 0, 0],            # maroon
            'Tram': [0, 0.5, 0],            # green
            'Person_sitting': [0, 0, 0.5]   # navy
        }

    def load_calibration_matrix(self):
        """
        load calibration matrix

        P0 = gray_L (left gray camera)
        P1 = gray_R (right gray camera)
        P2 = rgb_L (left color camera)
        P3 = rgb_R (right color camera)

        :return: p0, p1, p2, p3, r0_rect, tr_velo_to_cam, tr_imu_to_velo

        tr_velo_to_cam is the RT (rotation/translation) matrix from cloud point to image

        For projecting a velodyne co-ordinate point into the camera-2 image:
        [u v 1]^T = p2 * r0_rect * tr_velo_to_cam * [x y z 1]^T
        where [u, v]^T is the image pixel index (column/row), and [x, y, z]^T is the cloud point coordinate
        """
        with open(self._calib_full_path) as rf:
            all_lines = rf.readlines()

        p0 = np.matrix([float(x) for x in all_lines[0].strip('\n').split()[1:]]).reshape(3, 4)
        p1 = np.matrix([float(x) for x in all_lines[1].strip('\n').split()[1:]]).reshape(3, 4)
        p2 = np.matrix([float(x) for x in all_lines[2].strip('\n').split()[1:]]).reshape(3, 4)
        p3 = np.matrix([float(x) for x in all_lines[3].strip('\n').split()[1:]]).reshape(3, 4)

        r0_rect = np.matrix([float(x) for x in all_lines[4].strip('\n').split()[1:]]).reshape(3, 3)
        tr_velo_to_cam = np.matrix([float(x) for x in all_lines[5].strip('\n').split()[1:]]).reshape(3, 4)
        tr_imu_to_velo = np.matrix([float(x) for x in all_lines[6].strip('\n').split()[1:]]).reshape(3, 4)

        # add a 1 in bottom-right, reshape r0_rect to 4 x 4
        r0_rect = np.insert(r0_rect, 3, values=[0, 0, 0], axis=0)
        r0_rect = np.insert(r0_rect, 3, values=[0, 0, 0, 1], axis=1)

        # add a row in bottom of tr_velo_to_cam, reshape tr_velo_to_cam to 4 x 4
        tr_velo_to_cam = np.insert(tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)

        return p0, p1, p2, p3, r0_rect, tr_velo_to_cam, tr_imu_to_velo

    def load_annotation(self):
        """
        Load annotations of KITTI 3D object detection

        :return: a list of all ground truth labels in current annotation file
        """
        with open(self._label_full_path, 'r') as rf:
            all_lines = rf.readlines()

        ground_truth = list()
        for line in all_lines:
            line = line.strip()
            labels = line.split()

            # 3D object's category information
            category = labels[0]

            # truncated Float from 0 (non-truncated) to 1 (truncated)
            truncated_degree = float(labels[1])

            # occluded Integer (0,1,2,3) indicating occlusion state: 0 = fully visible,
            # 1 = partly occluded 2 = largely occluded, 3 = unknown
            occluded_degree = float(labels[2])

            # alpha Observation angle of object, ranging [-pi..pi]
            observation_angle = float(labels[3])

            # bbox 2D bounding box of object in the image (0-based index):
            # contains left, top, right, bottom pixel coordinates
            left_pixel = float(labels[4])
            top_pixel = float(labels[5])
            right_pixel = float(labels[6])
            bottom_pixel = float(labels[7])

            # 3 dimensions 3D object dimensions: height, width, length (in meters)
            height = float(labels[8])
            width = float(labels[9])
            length = float(labels[9])

            # 3 location 3D object location x,y,z in camera coordinates (in meters)
            x = float(labels[10])
            y = float(labels[11])
            z = float(labels[12])

            # rotation ry around Y-axis in camera coordinates system, ranging [-pi..pi]
            rotation_y = float(labels[13])

            ground_truth.append(
                {
                    "category": category,
                    "truncated_degree": truncated_degree,
                    "occluded_degree": occluded_degree,
                    "observation_angle": observation_angle,
                    "image_bbox": [left_pixel, top_pixel, right_pixel, bottom_pixel],   # [x_min, y_min, x_max, y_max]
                    "height": height,
                    "width": width,
                    "length": length,
                    "center_point": [x, y, z],
                    "rotation": rotation_y
                }
            )

        return ground_truth

    @staticmethod
    def project_velo_points_in_image(pts3d, p2, r0_rect, tr_velo_to_cam):
        """
        project cloud points into plannar image

        :param pts3d: Nx4 numpy array, cloud points data
        :param p2: projection matrix (after rectification) from a 3D coordinate in camera coordinate (x,y,z,1) to image
                   plannar coordinate (u, v, 1)
        :param r0_rect: rectifying rotation matrix of the reference camera, 4 x 4 matrix
        :param tr_velo_to_cam: RT (rotation/translation) matrix from cloud point to image
        :return:
        """
        cloud_points = copy.deepcopy(pts3d)

        # replaces the reflectance value by 1, and transpose the array, so points can be directly multiplied
        # by the camera projection matrix
        indices = cloud_points[:, 3] > 0
        cloud_points = cloud_points[indices, :]
        cloud_points[:, 3] = 1.0

        # now cloud points is with shape (4, M), reflectances is with shape (M, )
        cloud_points = cloud_points.transpose()
        reflectances = pts3d[indices, 3]

        # [u v z]^T = p2 * r0_rect * tr_velo_to_cam * [x y z 1]^T
        pts3d_cam = np.matmul(r0_rect, np.matmul(tr_velo_to_cam, cloud_points))     # x = right, y = down, z = forward
        pts3d_cam = np.array(pts3d_cam)

        # filter out points in front of camera
        pts3d_cam_indices = pts3d_cam[2, :] > 0
        pts3d_cam_in_front_of_camera = pts3d_cam[:, pts3d_cam_indices]
        pts3d_cam_in_front_of_camera = np.matrix(pts3d_cam_in_front_of_camera)

        pts2d_cam = np.matmul(p2, pts3d_cam_in_front_of_camera)
        pts2d_cam_normed = pts2d_cam
        pts2d_cam_normed[:2] = pts2d_cam[:2] / pts2d_cam[2, :]
        reflectances = reflectances[pts3d_cam_indices]

        assert pts2d_cam_normed.shape[1] == pts3d_cam_in_front_of_camera.shape[1] == reflectances.shape[0]
        return pts2d_cam_normed, pts3d_cam_in_front_of_camera, reflectances

    def visualize_3d_object_in_points_cloud(self, ):
        pass

    def visualize_points_cloud_in_image(self, image_data, pc_data, p2, r0_rect, tr_velo_to_cam):
        """
        visualize points cloud in plannar image

        :param image_data: image data, RGB order
        :param pc_data: Nx4 numpy array, cloud points data
        :param p2: projection matrix (after rectification) from a 3D coordinate in camera coordinate
                   (x,y,z,1) to image plannar coordinate (u, v, 1), 3 x 4 matrix
        :param r0_rect: rectifying rotation matrix of the reference camera, 4 x 4 matrix
        :param tr_velo_to_cam: RT (rotation/translation) matrix from cloud point to image, 4 x 4 matrix
        :return:
        """
        height, width, channels = image_data.shape

        pts2d_cam_normed, pts3d_cam_in_front_of_camera, reflectances = self.project_velo_points_in_image(
            pts3d=pc_data,
            p2=p2,
            r0_rect=r0_rect,
            tr_velo_to_cam=tr_velo_to_cam
        )

        plt.figure(figsize=(12, 4), dpi=96, tight_layout=True)
        plt.imshow(image_data)

        # plot cloud points
        u, v, z = pts2d_cam_normed
        u_in = np.logical_and(u >= 0, u < width)
        v_in = np.logical_and(v >= 0, v < height)
        fov_indices = np.array(np.logical_and(u_in, v_in))
        fov_indices = np.squeeze(fov_indices)
        pts2d_cam_normed_fov = pts2d_cam_normed[:, fov_indices]
        references_fov = reflectances[fov_indices]

        # generate color map from depth, z is the depth
        u, v, z = pts2d_cam_normed_fov
        plt.scatter([u], [v], c=[z], cmap='rainbow_r', alpha=0.5, s=2)

        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def visualize_2d_object_in_image(self, image_data, gts):
        """
        visualize 2D object in image (frame) which is captured with left color camera

        :param image_data: RGB data of camera frame
        :param gts: ground truth, which is a list
        :return: None
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(image_data)

        for gt in gts:
            [left_pixel, top_pixel, right_pixel, bottom_pixel] = gt["image_bbox"]
            rect = patches.Rectangle(
                (left_pixel, top_pixel), right_pixel-left_pixel, bottom_pixel-top_pixel,
                linewidth=2, edgecolor=self._categories_color[gt["category"]], facecolor='none')
            ax.add_patch(rect)

        plt.axis("off")
        fig.tight_layout()
        plt.show()

    def run(self):
        # load calibration matrix
        _, _, p2, _, r0_rect, tr_velo_to_cam, _ = self.load_calibration_matrix()

        # load annotations
        gts = self.load_annotation()

        # read camera data, shape = (375, 1242, 3)
        left_camera_image = cv2.imread(self._image_full_path, cv2.IMREAD_COLOR)         # BGR
        left_camera_image = cv2.cvtColor(left_camera_image, cv2.COLOR_BGR2RGB)          # RGB

        # read cloud point data
        pc_data = np.fromfile(self._cloud_points_full_path, dtype='<f4').reshape(-1, 4)

        # self.visualize_2d_object_in_image(
        #     image_data=left_camera_image,
        #     gts=gts
        # )

        self.visualize_points_cloud_in_image(
            image_data=left_camera_image,
            pc_data=pc_data,
            p2=p2,
            r0_rect=r0_rect,
            tr_velo_to_cam=tr_velo_to_cam
        )

        # self.visualize_3d_object_in_points_cloud(
        #     point_cloud_data=pc_data,
        #     gts=gts,
        #     p2=p2,
        #     r0_rect=r0_rect,
        #     tr_velo_to_cam=tr_velo_to_cam
        # )


if __name__ == "__main__":
    tools = KITTITools(
        image_full_path="../dataset/KITTI_3D_OBJECT_DETECTION_SAMPLED_DATASET/training/image_2/000011.png",
        calib_full_path="../dataset/KITTI_3D_OBJECT_DETECTION_SAMPLED_DATASET/training/calib/000011.txt",
        label_full_path="../dataset/KITTI_3D_OBJECT_DETECTION_SAMPLED_DATASET/training/label_2/000011.txt",
        cloud_points_full_path="../dataset/KITTI_3D_OBJECT_DETECTION_SAMPLED_DATASET/training/velodyne/000011.bin",
    )

    tools.run()
