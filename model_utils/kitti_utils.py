import numpy as np
import cv2
cv2.setNumThreads(0)
import os, math
import matplotlib.pyplot as plt

from scipy.optimize import leastsq
from PIL import Image

TOP_Y_MIN = -30
TOP_Y_MAX = +30
TOP_X_MIN = 0
TOP_X_MAX = 100
TOP_Z_MIN = -3.5
TOP_Z_MAX = 0.6

TOP_X_DIVISION = 0.05
TOP_Y_DIVISION = 0.05
TOP_Z_DIVISION = 0.02

cbox = np.array([[0, 70.4], [-40, 40], [-3, 2]])


def convertRot2Alpha(ry3d, z3d, x3d):
    alpha = ry3d - math.atan2(x3d, z3d)

    # equivalent
    equ_alpha = ry3d - math.atan2(x3d, z3d)

    while alpha > math.pi: alpha -= math.pi * 2
    while alpha < (-math.pi): alpha += math.pi * 2

    return alpha


'''calib utils'''
def cart2hom(pts_3d):
    """ Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))   # pts_3d shape: (n, 3). pts_3d_hom shape: (n, 4)
    return pts_3d_hom


def project_rect_to_image(P,pts_3d_rect):
    """ Input: nx3 points in rect camera coord.
        Output: nx2 points in image2 coord.
    """
    pts_3d_rect = cart2hom(pts_3d_rect)    # Left pts_3d_rect shape: (n, 4)
    pts_2d = np.dot(pts_3d_rect, np.transpose(P))  # pts_2d shape: (n, 3). self.P shape: (3, 4)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2], pts_2d[:, 2] # return (c_u, c_v), z


class Object3d(object):
    """ 3d object label """

    def __init__(self, label_file_line):
        data = label_file_line.split(" ")
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(
            data[2]
        )  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax], dtype=np.float32)

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = np.array((float(data[11]), float(data[12]), float(data[13])), dtype=np.float32)  # location (x,y,z) in camera coord.

        self.dis_to_cam = np.linalg.norm(self.t)

        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.real_alpha = data[3]
        self.ray = math.atan2(self.t[0], self.t[2])
        self.alpha = convertRot2Alpha(self.ry, self.t[2], self.t[0])

        # difficulty level
        self.level_str = None
        self.level = self.get_kitti_obj_level()

    def get_kitti_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])

        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.t

        return corners3d

    def print_object(self):
        print(
            "Type, truncation, occlusion, alpha: %s, %d, %d, %f"
            % (self.type, self.truncation, self.occlusion, self.alpha)
        )
        print(
            "2d bbox (x0,y0,x1,y1): %f, %f, %f, %f"
            % (self.xmin, self.ymin, self.xmax, self.ymax)
        )
        print("3d bbox h,w,l: %f, %f, %f" % (self.h, self.w, self.l))
        print(
            "3d bbox location, ry: (%f, %f, %f), %f"
            % (self.t[0], self.t[1], self.t[2], self.ry)
        )
        print("Difficulty of estimation: {}".format(self.estimate_diffculty()))


def read_calib_file(filepath):
    """ Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, "r") as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            key, value = line.split(":", 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def read_calib_from_video(calib_root_dir):
    """ Read calibration for camera 2 from video calib files.
        there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
    """
    data = {}
    cam2cam = read_calib_file(
        os.path.join(calib_root_dir, "calib_cam_to_cam.txt")
    )
    velo2cam = read_calib_file(
        os.path.join(calib_root_dir, "calib_velo_to_cam.txt")
    )
    Tr_velo_to_cam = np.zeros((3, 4))
    Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam["R"], [3, 3])
    Tr_velo_to_cam[:, 3] = velo2cam["T"]
    data["Tr_velo_to_cam"] = np.reshape(Tr_velo_to_cam, [12])
    data["R0_rect"] = cam2cam["R_rect_00"]
    data["P2"] = cam2cam["P_rect_02"]
    return data


def inverse_rigid_trans(Tr):
    """ Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr

def project_image_to_rect(uv_depth,calib_dict):
    """ Input: nx3 first two channels are uv, 3rd channel
               is depth in rect camera coord.
        Output: nx3 points in rect camera coord.
    """
    n = uv_depth.shape[0]
    x = ((uv_depth[:, 0] - calib_dict['c_u']) * uv_depth[:, 2]) / calib_dict['f_u'] + calib_dict['b_x']
    y = ((uv_depth[:, 1] - calib_dict['c_u']) * uv_depth[:, 2]) / calib_dict['f_v'] + calib_dict['b_y']

    if isinstance(uv_depth, np.ndarray):
        pts_3d_rect = np.zeros((n, 3))
    else:
        # torch.Tensor or torch.cuda.Tensor
        pts_3d_rect = uv_depth.new(uv_depth.shape).zero_()

    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = uv_depth[:, 2]

    return pts_3d_rect


def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects


def getcalib(calib_filepath, from_video=False, use_right_cam=False):
    if from_video:
        calibs = read_calib_from_video(calib_filepath)
    else:
        calibs = read_calib_file(calib_filepath)

    # Projection matrix from rect camera coord to image coord
    P = calibs["P3"] if use_right_cam else calibs["P2"]
    P = np.reshape(P, [3, 4])

    # Rigid transform from Velodyne coord to reference camera coord
    V2C = calibs["Tr_velo_to_cam"]
    V2C = np.reshape(V2C, [3, 4])
    C2V = inverse_rigid_trans(V2C)

    # Rotation from reference camera coord to rect camera coord
    R0 = calibs["R0_rect"]
    R0 = np.reshape(R0, [3, 3])

    # Camera intrinsics and extrinsics
    c_u = P[0, 2]
    c_v = P[1, 2]
    f_u = P[0, 0]
    f_v = P[1, 1]
    b_x = P[0, 3] / (-f_u)  # relative
    b_y = P[1, 3] / (-f_v)
    return {'P':P,'R0':R0,'C2V':C2V,'c_u':c_u,'c_v':c_v,'f_u':f_u,'f_v':f_v,'b_x':b_x,'b_y':b_y}


def approx_proj_center(proj_center, surface_centers, img_size):
    # proj_center: 3D center projected on 2D plane. surface_centers: 2D center.
    # surface_inside
    img_w, img_h = img_size
    surface_center_inside_img = (surface_centers[:, 0] >= 0) & (surface_centers[:, 1] >= 0) & \
                                (surface_centers[:, 0] <= img_w - 1) & (surface_centers[:, 1] <= img_h - 1)

    if surface_center_inside_img.sum() > 0:
        target_surface_center = surface_centers[surface_center_inside_img.argmax()]
        # y = ax + b
        a, b = np.polyfit([proj_center[0], target_surface_center[0]], [proj_center[1], target_surface_center[1]], 1)
        valid_intersects = []
        valid_edge = []

        left_y = b
        if (0 <= left_y <= img_h - 1):
            valid_intersects.append(np.array([0, left_y]))
            valid_edge.append(0)

        right_y = (img_w - 1) * a + b
        if (0 <= right_y <= img_h - 1):
            valid_intersects.append(np.array([img_w - 1, right_y]))
            valid_edge.append(1)

        top_x = -b / a
        if (0 <= top_x <= img_w - 1):
            valid_intersects.append(np.array([top_x, 0]))
            valid_edge.append(2)

        bottom_x = (img_h - 1 - b) / a
        if (0 <= bottom_x <= img_w - 1):
            valid_intersects.append(np.array([bottom_x, img_h - 1]))
            valid_edge.append(3)

        valid_intersects = np.stack(valid_intersects)
        min_idx = np.argmin(np.linalg.norm(valid_intersects - proj_center.reshape(1, 2), axis=1))

        return valid_intersects[min_idx], valid_edge[min_idx]
    else:
        return None

'''
heatmap util sets
'''
def ellip_gaussian2D(shape, sigma_x, sigma_y):
	m, n = [(ss - 1.) / 2. for ss in shape]
	y, x = np.ogrid[-m:m + 1, -n:n + 1]

	# generate meshgrid
	h = np.exp(-(x * x) / (2 * sigma_x * sigma_x) - (y * y) / (2 * sigma_y * sigma_y))
	h[h < np.finfo(h.dtype).eps * h.max()] = 0

	return h


def draw_umich_gaussian_2D(heatmap, center, radius_x, radius_y, k=1):
    diameter_x, diameter_y = 2 * radius_x + 1, 2 * radius_y + 1
    gaussian = ellip_gaussian2D((diameter_y, diameter_x), sigma_x=diameter_x / 6, sigma_y=diameter_y / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius_x), min(width - x, radius_x + 1)
    top, bottom = min(y, radius_y), min(height - y, radius_y + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius_y - top:radius_y + bottom, radius_x - left:radius_x + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def gaussian_radius(height, width, min_overlap=0.7):
	a1 = 1
	b1 = (height + width)
	c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
	sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
	r1 = (b1 + sq1) / 2

	a2 = 4
	b2 = 2 * (height + width)
	c2 = (1 - min_overlap) * width * height
	sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
	r2 = (b2 + sq2) / 2

	a3 = 4 * min_overlap
	b3 = -2 * min_overlap * (height + width)
	c3 = (min_overlap - 1) * width * height
	sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
	r3 = (b3 + sq3) / 2

	return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
	m, n = [(ss - 1.) / 2. for ss in shape]
	y, x = np.ogrid[-m:m + 1, -n:n + 1]

	# generate meshgrid
	h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
	h[h < np.finfo(h.dtype).eps * h.max()] = 0

	return h


def draw_umich_gaussian(heatmap, center, radius, k=1, ignore=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    '''
    Assign all pixels within the area as -1. They will be further suppressed by heatmap from other 
    objects with the maximum. Therefore, these don't care objects won't influence other positive samples.
    '''

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        if ignore:
            masked_heatmap[masked_heatmap == 0] = -1
        else:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap