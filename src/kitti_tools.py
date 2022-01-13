import cv2
import numpy as np
import copy
import open3d as o3d   # pip3 install open3d==0.14.1
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class KITTITools(object):
    """
    Tools class for KITTI dataset

    Reference:
    1. https://github.com/dtczhl/dtc-KITTI-For-Beginners/blob/master/python/object_viewer.py
    2. https://github.com/azureology/kitti-velo2cam
    3. https://github.com/kuixu/kitti_object_vis
    """
    def __init__(self, image_full_path, calibration_full_path, label_full_path, cloud_points_full_path):
        """
        Constructor

        :param image_full_path: full path of image
        :param calibration_full_path: full path of calibration file
        :param label_full_path: 3D object label file's full path
        :param cloud_points_full_path: cloud point full path, raw data is with form of [x0 y0 z0 r0 x1 y1 z1 r1 ...].
        """
        self._image_full_path = image_full_path
        self._calibration_full_path = calibration_full_path
        self._label_full_path = label_full_path
        self._cloud_points_full_path = cloud_points_full_path

        self._categories_color = {
            'Car': [255, 0, 0],             # red
            'DontCare': [0, 0, 0],          # black
            'Pedestrian': [0, 0, 255],      # blue
            'Van': [255, 255, 0],           # yellow
            'Cyclist': [255, 0, 255],       # magenta
            'Truck': [0, 255, 255],         # cyan
            'Misc': [127, 0, 0],            # maroon
            'Tram': [0, 127, 0],            # green
            'Person_sitting': [0, 0, 127]   # navy
        }

        self._p2, self._r0_rect, self._tr_velodyne_to_camera = self.load_calibration_matrix()
        self._gts_in_camera_coordinate_system = self.load_annotation()

        # convert the ground truth from camera coordinate system to velodyne coordinate system
        self._gts_in_velodyne_coordinate_system = self.convert_gt_from_camera_to_velodyne_coordinate_system()

    def load_calibration_matrix(self):
        """
        load calibration matrix from calibration file

        P0 = gray_L (left gray camera)
        P1 = gray_R (right gray camera)
        P2 = rgb_L (left color camera)
        P3 = rgb_R (right color camera)

        :return: p2, r0_rect, tr_velodyne_to_camera

        p2: projection matrix (after rectification) from a 3D coordinate in camera coordinate (x,y,z,1) to image
            plane coordinate (u, v, 1)
        r0_rect: rectifying rotation matrix of the reference camera, 4 x 4 matrix
        tr_velodyne_to_camera: RT (rotation/translation) matrix from cloud point to image
        """
        with open(self._calibration_full_path) as rf:
            all_lines = rf.readlines()

        p0 = np.matrix([float(x) for x in all_lines[0].strip('\n').split()[1:]]).reshape(3, 4)
        p1 = np.matrix([float(x) for x in all_lines[1].strip('\n').split()[1:]]).reshape(3, 4)
        p2 = np.matrix([float(x) for x in all_lines[2].strip('\n').split()[1:]]).reshape(3, 4)
        p3 = np.matrix([float(x) for x in all_lines[3].strip('\n').split()[1:]]).reshape(3, 4)

        r0_rect = np.matrix([float(x) for x in all_lines[4].strip('\n').split()[1:]]).reshape(3, 3)
        tr_velodyne_to_camera = np.matrix([float(x) for x in all_lines[5].strip('\n').split()[1:]]).reshape(3, 4)
        tr_imu_to_velodyne = np.matrix([float(x) for x in all_lines[6].strip('\n').split()[1:]]).reshape(3, 4)

        # add a 1 in bottom-right, reshape r0_rect to 4 x 4
        r0_rect = np.insert(r0_rect, 3, values=[0, 0, 0], axis=0)
        r0_rect = np.insert(r0_rect, 3, values=[0, 0, 0, 1], axis=1)

        # add a row in bottom of tr_velodyne_to_camera, reshape tr_velodyne_to_camera to 4 x 4
        tr_velodyne_to_camera = np.insert(tr_velodyne_to_camera, 3, values=[0, 0, 0, 1], axis=0)

        return p2, r0_rect, tr_velodyne_to_camera

    def load_annotation(self):
        """
        Load annotations of KITTI 3D object detection

        :return: a list of all ground truth for various 3D objects in current annotation file
        """
        with open(self._label_full_path, 'r') as rf:
            all_lines = rf.readlines()

        ground_truth = list()
        for line in all_lines:
            line = line.strip()
            labels = line.split()

            # 3D object's category information, 'DontCare' labels denote regions in which objects have not been labeled,
            # for example because they have been too far away from the laser scanner.
            type = labels[0]

            # truncated Float from 0 (non-truncated) to 1 (truncated)
            truncation = float(labels[1])

            # occluded Integer (0,1,2,3) indicating occlusion state: 0 = fully visible,
            # 1 = partly occluded 2 = largely occluded, 3 = unknown
            occlusion = float(labels[2])

            # alpha observation angle of object, ranging [-pi..pi]
            alpha = float(labels[3])

            # 2D bounding box of object in the image: contains left, top, right, bottom pixel coordinates
            x_min = float(labels[4])
            y_min = float(labels[5])
            x_max = float(labels[6])
            y_max = float(labels[7])

            # 3D object dimensions: height, width, length (in meters)
            height = float(labels[8])
            width = float(labels[9])
            length = float(labels[10])

            # 3D object location x,y,z in camera coordinates (in meters)
            x = float(labels[11])
            y = float(labels[12])
            z = float(labels[13])

            # Rotation ry around Y-axis in camera coordinates [-pi..pi]
            ry = float(labels[14])

            ground_truth.append(
                {
                    "type": type,
                    "truncation": truncation,
                    "occlusion": occlusion,
                    "alpha": alpha,
                    "box2d": [x_min, y_min, x_max, y_max],
                    "height": height,
                    "width": width,
                    "length": length,
                    "location": [x, y, z],
                    "ry": ry
                }
            )

        return ground_truth

    def project_velodyne_points_to_image_plane(self, pts3d):
        """
        project cloud points into image plane

        :param pts3d: Nx4 numpy array, cloud points data
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

        # [u v z]^T = p2 * r0_rect * tr_velodyne_to_cam * [x y z 1]^T
        pts3d_cam = np.matmul(self._r0_rect, np.matmul(self._tr_velodyne_to_camera, cloud_points))
        pts3d_cam = np.array(pts3d_cam)

        # filter out points in front of camera
        pts3d_cam_indices = pts3d_cam[2, :] > 0
        pts3d_cam_in_front_of_camera = pts3d_cam[:, pts3d_cam_indices]
        pts3d_cam_in_front_of_camera = np.matrix(pts3d_cam_in_front_of_camera)

        pts2d_cam = np.matmul(self._p2, pts3d_cam_in_front_of_camera)
        pts2d_cam_normed = pts2d_cam
        pts2d_cam_normed[:2] = pts2d_cam[:2] / pts2d_cam[2, :]
        reflectances = reflectances[pts3d_cam_indices]

        assert pts2d_cam_normed.shape[1] == pts3d_cam_in_front_of_camera.shape[1] == reflectances.shape[0]
        return pts2d_cam_normed, pts3d_cam_in_front_of_camera, reflectances

    def convert_gt_from_camera_to_velodyne_coordinate_system(self):
        """
        convert the ground truth frame camera coordinate system to velodyne coordinate system
        - Camera: x = right, y = down, z = forward
        - Velodyne: x = forward, y = left, z = up

        :return:
        """
        ret_gt_list = list()

        for index, gt in enumerate(self._gts_in_camera_coordinate_system):
            height, width, length = gt["height"], gt["width"], gt["length"]
            [x_c, y_c, z_c] = gt["location"]
            ry = gt["ry"]

            # The reference point for the 3D bounding box for each object is centered on the bottom face of the box,
            # as is shown in cs_overview.pdf. The corners of bounding box are computed as follows with
            # respect to the reference point and in the object coordinate system.
            x_corners = np.array([length/2, length/2, -length/2, -length/2, length/2, length/2, -length/2, -length/2])
            y_corners = np.array([0, 0, 0, 0, -height, -height, -height, -height])
            z_corners = np.array([width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2, width/2])
            base_corners3d = np.array([x_corners, y_corners, z_corners])

            # compute rotational matrix around yaw axis
            rotation_matrix = np.array([
                [np.cos(ry), 0.0, np.sin(ry)],
                [0.0, 1.0, 0.0],
                [-np.sin(ry), 0.0, np.cos(ry)]
            ])

            # corner3d in camera coordinate system
            corners3d = np.dot(rotation_matrix, base_corners3d) + np.array([[x_c], [y_c], [z_c]])

            corners3d_in_camera_coordinate_system = np.insert(corners3d, 3, np.ones(8, ), axis=0)
            inverse_tr_velodyne_to_camera = np.linalg.inv(self._tr_velodyne_to_camera)
            corners3d_in_velodyne_coordinate_system = np.dot(
                inverse_tr_velodyne_to_camera,
                corners3d_in_camera_coordinate_system)

            corners3d_in_velodyne_coordinate_system = corners3d_in_velodyne_coordinate_system[0:3, :]

            ret_gt_list.append(
                {
                    "type": gt["type"],
                    "truncation": gt["truncation"],
                    "occlusion": gt["occlusion"],
                    "alpha": gt["alpha"],
                    "box3d": corners3d_in_velodyne_coordinate_system.tolist()
                }
            )

        return ret_gt_list

    def plot_velodyne_points_in_image_plane(self, image_data, pc_data):
        """
        visualize points cloud in plannar image

        :param image_data: image data, RGB order
        :param pc_data: Nx4 numpy array, cloud points data
        :return:
        """
        height, width, channels = image_data.shape
        pts2d_cam_normed, pts3d_cam_in_front_of_camera, reflectances = self.project_velodyne_points_to_image_plane(
            pts3d=pc_data
        )

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(image_data)

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

    def plot_2d_box_in_image_plane(self, image_data, gts_in_camera_coordinate_system, thickness=2):
        """
        plot 2D object in image (frame) which is captured with left color camera

        :param image_data: RGB data of camera frame
        :param gts_in_camera_coordinate_system: ground truth in camera coordinate system
        :param thickness: thickness of 3D box plotted in image plane
        :return: None
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(image_data)

        for gt in gts_in_camera_coordinate_system:
            [left_pixel, top_pixel, right_pixel, bottom_pixel] = gt["box2d"]
            rect = patches.Rectangle(
                (left_pixel, top_pixel),
                right_pixel-left_pixel,
                bottom_pixel-top_pixel,
                linewidth=thickness,
                edgecolor=np.array(self._categories_color[gt["type"]]) / 255.0,
                facecolor='none')
            ax.add_patch(rect)

        plt.axis("off")
        fig.tight_layout()
        plt.show()

    def project_corners_from_velodyne_to_image_plane(self, box3d):
        """
        project 8 corners from velodyne coordinate system to image plane system

        :param box3d: a numpy array with shape 8 x 3
        :return: a numpy array with shape 8 x 3
        """
        box3d = np.insert(box3d, 3, np.ones(shape=(8, )), axis=1)
        box3d = box3d.transpose()   # shape is 4 x 8

        # [u v z]^T = p2 * r0_rect * tr_velo_to_cam * [x y z 1]^T
        box3d_cam = np.matmul(self._r0_rect, np.matmul(self._tr_velodyne_to_camera, box3d))
        box3d_cam = np.matmul(self._p2, box3d_cam)

        box3d_cam_normed = box3d_cam
        box3d_cam_normed[:2] = box3d_cam_normed[:2] / box3d_cam_normed[2, :]
        box3d_cam_normed = box3d_cam_normed.transpose()

        return box3d_cam_normed

    def plot_3d_box_in_image_plane(self, image_data, gts_in_velodyne_coordinate_system, thickness=2):
        """
        plot 3D object in image (frame) which is captured with left color camera

        :param image_data: RGB data of camera frame
        :param gts_in_velodyne_coordinate_system: ground truth in velodyne coordinate system
        :param thickness: thickness of 3D box plotted in image plane
        :return: None
        """

        for index, gt in enumerate(gts_in_velodyne_coordinate_system):
            type = gt["type"]

            # box3d is a (8,3) array of vertices for the 3d box in following order:
            #     1 -------- 0
            #    /|         /|
            #   2 -------- 3 .
            #   | |        | |
            #   . 5 -------- 4
            #   |/         |/
            #   6 -------- 7

            box3d = np.array(gt["box3d"])
            box3d = np.transpose(box3d)

            # project 3D box corners from velodyne coordinate system to image plane coordinate system
            box3d_in_image_plane = self.project_corners_from_velodyne_to_image_plane(box3d)

            color = self._categories_color[type]
            qs = box3d_in_image_plane.astype(np.int32)

            # Ref: https://github.com/kuixu/kitti_object_vis/blob/master/kitti_util.py
            for k in range(0, 4):
                i, j = k, (k + 1) % 4
                cv2.line(image_data, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
                i, j = k + 4, (k + 1) % 4 + 4
                cv2.line(image_data, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
                i, j = k, k + 4
                cv2.line(image_data, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(image_data)
        plt.axis("off")
        fig.tight_layout()
        plt.show()

    def plot_3d_box_in_velodyne_coordinate_system(self, pc_data, gts_in_velodyne_coordinate_system):
        """
        plot 3D bounding boxes in velodyne coordinate system (cloud points' coordinate system)

        :param pc_data: a numpy array with shape M x 4
        :param gts_in_velodyne_coordinate_system: ground truth of 3D bounding boxes in velodyne coordinate system
        :return: None
        """
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pc_data[:, 0:3])

        # dump cloud points for network debug
        with open('./temp/points_velodyne_000032.npy', 'wb') as f:
            np.save(f, pc_data)

        geometries = [point_cloud]

        for index, gt in enumerate(gts_in_velodyne_coordinate_system):
            type = gt["type"]
            if type == 'DontCare':
                continue

            points_box = np.array(gt["box3d"])
            points_box = points_box.transpose()
            lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3],
                                  [4, 5], [4, 7], [5, 6], [6, 7],
                                  [0, 4], [1, 5], [2, 6], [3, 7]])
            colors = np.array([self._categories_color[type] for _ in range(len(lines_box))])

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points_box)
            line_set.lines = o3d.utility.Vector2iVector(lines_box)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(line_set)

        o3d.visualization.draw_geometries(geometries)

    def run(self):
        # read camera data, shape = (375, 1242, 3)
        left_camera_image = cv2.imread(self._image_full_path, cv2.IMREAD_COLOR)         # BGR
        left_camera_image = cv2.cvtColor(left_camera_image, cv2.COLOR_BGR2RGB)          # RGB

        # read cloud point data
        pc_data = np.fromfile(self._cloud_points_full_path, dtype='<f4').reshape(-1, 4)

        # visualization
        self.plot_velodyne_points_in_image_plane(
            image_data=left_camera_image,
            pc_data=pc_data)

        self.plot_2d_box_in_image_plane(
            image_data=left_camera_image,
            gts_in_camera_coordinate_system=self._gts_in_camera_coordinate_system)

        self.plot_3d_box_in_image_plane(
            image_data=left_camera_image,
            gts_in_velodyne_coordinate_system=self._gts_in_velodyne_coordinate_system)

        self.plot_3d_box_in_velodyne_coordinate_system(
            pc_data=pc_data,
            gts_in_velodyne_coordinate_system=self._gts_in_velodyne_coordinate_system)


if __name__ == "__main__":
    tools = KITTITools(
        image_full_path="../dataset/KITTI_3D_OBJECT_DETECTION_SAMPLED_DATASET/training/image_2/000032.png",
        calibration_full_path="../dataset/KITTI_3D_OBJECT_DETECTION_SAMPLED_DATASET/training/calib/000032.txt",
        label_full_path="../dataset/KITTI_3D_OBJECT_DETECTION_SAMPLED_DATASET/training/label_2/000032.txt",
        cloud_points_full_path="../dataset/KITTI_3D_OBJECT_DETECTION_SAMPLED_DATASET/training/velodyne/000032.bin",
    )

    tools.run()
