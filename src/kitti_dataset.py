import logging
import numpy as np

from config import TYPE_ID_CONVERSION
import mindspore.dataset as ds

from Monodde.model_utils.kitti_utils import *
from .distributed_sampler import DistributedSampler
from .transformers import *



class KittiDataset:
    '''kitti Dataset '''
    def __init__(self,root,cfg):
        self.cfg = cfg  # setup
        self.root = root
        self.image_dir = os.path.join(root, "image_2")
        self.image_right_dir = os.path.join(root, "image_3")
        self.label_dir = os.path.join(root, "label_2")
        self.calib_dir = os.path.join(root, "calib")

        self.split = cfg.DATASETS.TRAIN_SPLIT if cfg.is_training else cfg.DATASETS.TEST_SPLIT  # whether split datasets or not
        self.is_train = np.array(cfg.is_training)  # train or test
        self.imageset_txt = os.path.join(root, "ImageSets", "{}.txt".format(self.split))
        assert os.path.exists(self.imageset_txt), "ImageSets file not exist, dir = {}".format(self.imageset_txt)

        image_files = []
        for line in open(self.imageset_txt, "r"):
            base_name = line.replace("\n", "")
            image_name = base_name + ".png"
            image_files.append(image_name)

        self.image_files = image_files  # load images
        self.label_files = [i.replace(".png", ".txt") for i in self.image_files]  # load labels
        self.classes = cfg.DATASETS.DETECT_CLASSES  # ("Car", "Pedestrian", "Cyclist")
        self.num_classes = len(self.classes)  # 3
        self.num_samples = len(self.image_files)  # images files sum  3712

        # whether to use right-view image
        self.use_right_img = cfg.DATASETS.USE_RIGHT_IMAGE & cfg.is_training  # True

        # self.augmentation = get_composed_augmentations() if (
        #             self.is_train and augment) else None  # Only RandomHorizontallyFlip.

        # input and output shapes
        self.input_width = cfg.INPUT.WIDTH_TRAIN  # 1280
        self.input_height = cfg.INPUT.HEIGHT_TRAIN  # 384
        self.down_ratio = cfg.MODEL.BACKBONE.DOWN_RATIO  # 4
        self.output_width = self.input_width // cfg.MODEL.BACKBONE.DOWN_RATIO  # 320
        self.output_height = self.input_height // cfg.MODEL.BACKBONE.DOWN_RATIO  # 96
        self.output_size = [self.output_width, self.output_height]  # [320,96]

        # maximal length of extracted feature map when appling edge fusion
        self.max_edge_length = (self.output_width + self.output_height) * 2  # 832
        self.max_objs = cfg.DATASETS.MAX_OBJECTS  # 40

        # filter invalid annotations
        self.filter_annos = cfg.DATASETS.FILTER_ANNO_ENABLE  # True
        self.filter_params = cfg.DATASETS.FILTER_ANNOS  # [0.9,20]

        # handling truncation
        self.consider_outside_objs = cfg.DATASETS.CONSIDER_OUTSIDE_OBJS  # True
        self.use_approx_center = cfg.INPUT.USE_APPROX_CENTER  # whether to use approximate representations for outside objects  (True)
        self.proj_center_mode = cfg.INPUT.APPROX_3D_CENTER  # the type of approximate representations for outside objects  #'intersect'

        # for edge feature fusion
        self.enable_edge_fusion = cfg.MODEL.HEAD.ENABLE_EDGE_FUSION  # True

        # True
        self.use_modify_keypoint_visible = cfg.INPUT.KEYPOINT_VISIBLE_MODIFY

        PI = np.pi
        self.orientation_method = cfg.INPUT.ORIENTATION  # 'muti-bin'
        self.multibin_size = cfg.INPUT.ORIENTATION_BIN_SIZE
        self.alpha_centers = np.array([0, PI / 2, PI, - PI / 2])  # centers for multi-bin orientation

        # use '2D' or '3D' center for heatmap prediction
        self.heatmap_center = cfg.INPUT.HEATMAP_CENTER  # 3D
        self.adjust_edge_heatmap = cfg.INPUT.ADJUST_BOUNDARY_HEATMAP  # True
        self.edge_heatmap_ratio = cfg.INPUT.HEATMAP_RATIO  # radius / 2d box, 0.5

        self.logger = logging.getLogger("monoflex.dataset")
        self.logger.info("Initializing KITTI {} set with {} files loaded.".format(self.split, self.num_samples))

    def __len__(self):
        if self.use_right_img:
            return self.num_samples * 2
        else:
            return self.num_samples


    def get_right_image(self, idx):
        img_filename = os.path.join(self.image_right_dir, self.image_files[idx])
        img = Image.open(img_filename).convert('RGB')
        return img

    def get_calibration(self, idx, use_right_cam=False):
        calib_filename = os.path.join(self.calib_dir, self.label_files[idx])
        return getcalib(calib_filename, use_right_cam=use_right_cam)

    def get_image(self, idx):
        img_filename = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_filename).convert('RGB')
        return img

    def get_label_objects(self, idx):
        if self.split != 'test':
            label_filename = os.path.join(self.label_dir, self.label_files[idx])

        return read_label(label_filename)

    def filtrate_objects(self, obj_list):
        """
        Discard objects which are not in self.classes (or its similar classes)
        :param obj_list: list
        :return: list
        """
        type_whitelist = self.classes
        valid_obj_list = []

        if obj_list is None:
            return None

        for obj in obj_list:
            if obj.type not in type_whitelist:
                continue

            valid_obj_list.append(obj)

        return valid_obj_list

    def pad_image(self, image):
        img = np.array(image)
        h, w, c = img.shape
        ret_img = np.zeros((self.input_height, self.input_width, c))
        pad_y = (self.input_height - h) // 2
        pad_x = (self.input_width - w) // 2

        ret_img[pad_y: pad_y + h, pad_x: pad_x + w] = img
        pad_size = np.array([pad_x, pad_y])

        return Image.fromarray(ret_img.astype(np.uint8)), pad_size

    def get_edge_utils(self, image_size, pad_size, down_ratio=4):
        img_w, img_h = image_size

        x_min, y_min = np.ceil(pad_size[0] / down_ratio), np.ceil(pad_size[1] / down_ratio)
        x_max, y_max = (pad_size[0] + img_w - 1) // down_ratio, (pad_size[1] + img_h - 1) // down_ratio
        # print('x_min',x_min,'y_min',y_min,'x_max',x_max,'y_max',y_max)

        step = 1
        # boundary idxs
        edge_indices = []

        # left
        y = np.arange(y_min, y_max, step)
        x = np.ones(len(y)) * x_min

        edge_indices_edge = np.vstack((x, y)).T
        edge_indices_edge[:, 0] = np.clip(edge_indices_edge[:, 0], x_min, float("inf"))
        edge_indices_edge[:, 1] = np.clip(edge_indices_edge[:, 1], y_min, float("inf"))
        edge_indices_edge = np.unique(edge_indices_edge, axis=0)
        edge_indices.append(edge_indices_edge)

        # bottom
        x = np.arange(x_min, x_max, step)
        y = np.ones(len(x)) * y_max

        edge_indices_edge = np.vstack((x, y)).T
        edge_indices_edge[:, 0] = np.clip(edge_indices_edge[:, 0], x_min, float("inf"))
        edge_indices_edge[:, 1] = np.clip(edge_indices_edge[:, 1], y_min, float("inf"))
        edge_indices_edge = np.unique(edge_indices_edge, axis=0)
        edge_indices.append(edge_indices_edge)

        # right
        y = np.arange(y_max, y_min, -step)
        x = np.ones(len(y)) * x_max

        # edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge = np.vstack((x, y)).T
        edge_indices_edge[:, 0] = np.clip(edge_indices_edge[:, 0], x_min, float("inf"))
        edge_indices_edge[:, 1] = np.clip(edge_indices_edge[:, 1], y_min, float("inf"))
        edge_indices_edge = np.flipud(np.unique(edge_indices_edge, axis=0))
        edge_indices.append(edge_indices_edge)

        # top
        x = np.arange(x_max, x_min - 1, -step)
        y = np.ones(len(x)) * y_min

        edge_indices_edge = np.vstack((x, y)).T
        edge_indices_edge[:, 0] = np.clip(edge_indices_edge[:, 0], x_min, float("inf"))
        edge_indices_edge[:, 1] = np.clip(edge_indices_edge[:, 1], y_min, float("inf"))

        edge_indices_edge = np.flipud(np.unique(edge_indices_edge, axis=0))
        edge_indices.append(edge_indices_edge)

        edge_indices = np.concatenate([index for index in edge_indices], axis=0)
        return edge_indices

    def encode_alpha_multibin(self, alpha, num_bin=2, margin=1 / 6):
        # encode alpha (-PI ~ PI) to 2 classes and 1 regression
        encode_alpha = np.zeros(num_bin * 2)
        bin_size = 2 * np.pi / num_bin  # pi
        margin_size = bin_size * margin  # pi / 6

        bin_centers = self.alpha_centers
        range_size = bin_size / 2 + margin_size

        offsets = alpha - bin_centers
        offsets[offsets > np.pi] = offsets[offsets > np.pi] - 2 * np.pi
        offsets[offsets < -np.pi] = offsets[offsets < -np.pi] + 2 * np.pi

        for i in range(num_bin):
            offset = offsets[i]
            if abs(offset) < range_size:
                encode_alpha[i] = 1
                encode_alpha[i + num_bin] = offset

        return encode_alpha


    def __getitem__(self, idx):
        if idx >= self.num_samples:
            # utilize right color image
            idx = idx % self.num_samples
            img = self.get_right_image(idx)
            calib = self.get_calibration(idx, use_right_cam=True)
            objs = None if self.split == 'test' else self.get_label_objects(idx)
            use_right_img = True
            # generate the bboxes for right color image
            right_objs = []
            img_w, img_h = img.size
            for obj in objs:
                corners_3d = obj.generate_corners3d()
                corners_2d, _ = project_rect_to_image(calib['P'],corners_3d)
                obj.box2d = np.array([max(corners_2d[:, 0].min(), 0), max(corners_2d[:, 1].min(), 0),
                                      min(corners_2d[:, 0].max(), img_w - 1), min(corners_2d[:, 1].max(), img_h - 1)],
                                     dtype=np.float32)

                obj.xmin, obj.ymin, obj.xmax, obj.ymax = obj.box2d
                right_objs.append(obj)

            objs = right_objs
        else:
            # utilize left color image
            img = self.get_image(idx)
            calib = self.get_calibration(idx)
            objs = None if self.split == 'test' else self.get_label_objects(idx)

            use_right_img = False

        original_idx = self.image_files[idx][:6]
        objs = self.filtrate_objects(objs)  # remove objects of irrelevant classes
        # random horizontal flip
        # if self.augmentation is not None:
        #     img, objs, calib = self.augmentation(img, objs, calib)

        # for i, obj in enumerate(objs):
        #     print('________', i, ':', obj.type)
        # pad image
        img_before_aug_pad = np.array(img).copy()
        img_w, img_h = img.size
        img, pad_size = self.pad_image(img)
        # for training visualize, use the padded images
        ori_img = np.array(img).copy() if self.is_train else img_before_aug_pad

        # the boundaries of the image after padding
        x_min, y_min = int(np.ceil(pad_size[0] / self.down_ratio)), int(np.ceil(pad_size[1] / self.down_ratio))
        x_max, y_max = (pad_size[0] + img_w - 1) // self.down_ratio, (pad_size[1] + img_h - 1) // self.down_ratio

        input_edge_count=None
        input_edge_indices=None
        if self.enable_edge_fusion:
            # generate edge_indices for the edge fusion module
            input_edge_indices = np.zeros([self.max_edge_length, 2], dtype=np.int32)
            edge_indices = self.get_edge_utils((img_w, img_h),
                                               pad_size)  # edge_indices: The coordinates of edges in anticlockwise order.
            input_edge_count = edge_indices.shape[0]
            input_edge_indices[:edge_indices.shape[0]] = edge_indices
            input_edge_count = input_edge_count - 1  # explain ?

        if self.split == 'test':
            # for inference we parametrize with original size
            target={'image_size':img.size,'is_train':self.is_train}
            # target = ParamsList(image_size=img.size, is_train=self.is_train)
            target["pad_size"]=pad_size
            target["calib"]=calib
            target["ori_img"]=ori_img
            if self.enable_edge_fusion:
                target['edge_len']=input_edge_count
                target['edge_indices']=input_edge_indices

            # if self.transforms is not None: img, target = self.transforms(img, target)

            return img, target, original_idx

        # heatmap
        heat_map = np.zeros([self.num_classes, self.output_height, self.output_width], dtype=np.float32)
        ellip_heat_map = np.zeros([self.num_classes, self.output_height, self.output_width], dtype=np.float32)
        # classification
        cls_ids = np.zeros([self.max_objs], dtype=np.int32)
        target_centers = np.zeros([self.max_objs, 2], dtype=np.int32)
        # 2d bounding boxes
        gt_bboxes = np.zeros([self.max_objs, 4], dtype=np.float32)
        bboxes = np.zeros([self.max_objs, 4], dtype=np.float32)
        # keypoints: 2d coordinates and visible(0/1)
        keypoints = np.zeros([self.max_objs, 10, 3], dtype=np.float32)
        keypoints_depth_mask = np.zeros([self.max_objs, 3],
                                        dtype=np.float32)  # whether the depths computed from three groups of keypoints are valid
        # 3d dimension
        dimensions = np.zeros([self.max_objs, 3], dtype=np.float32)  # (h, w, l)
        # 3d location
        locations = np.zeros([self.max_objs, 3], dtype=np.float32)  # (x, y, z)
        # rotation y
        rotys = np.zeros([self.max_objs], dtype=np.float32)
        # alpha (local orientation)
        alphas = np.zeros([self.max_objs], dtype=np.float32)
        # offsets from center to expected_center
        offset_3D = np.zeros([self.max_objs, 2], dtype=np.float32)

        # occlusion and truncation
        occlusions = np.zeros(self.max_objs)
        truncations = np.zeros(self.max_objs)

        if self.orientation_method == 'head-axis':
            orientations = np.zeros([self.max_objs, 3], dtype=np.float32)
        else:
            orientations = np.zeros([self.max_objs, self.multibin_size * 2],
                                    dtype=np.float32)  # multi-bin loss: 2 cls + 2 offset

        reg_mask = np.zeros([self.max_objs], dtype=np.uint8)  # regression mask
        trunc_mask = np.zeros([self.max_objs], dtype=np.uint8)  # outside object mask
        reg_weight = np.zeros([self.max_objs], dtype=np.float32)  # regression weight

        GRM_keypoints_visible = np.zeros([self.max_objs, 11], dtype=np.bool_)  # The visibility of 11 keypoints.

        for i, obj in enumerate(objs):
            cls = obj.type
            cls_id = TYPE_ID_CONVERSION[cls]
            if cls_id < 0: continue

            # TYPE_ID_CONVERSION = {
            #     'Car': 0,
            #     'Pedestrian': 1,
            #     'Cyclist': 2,
            #     'Van': -4,
            #     'Truck': -4,
            #     'Person_sitting': -2,
            #     'Tram': -99,
            #     'Misc': -99,
            #     'DontCare': -1,
            # }

            float_occlusion = float(
                obj.occlusion)  # 0 for normal, 0.33 for partially, 0.66 for largely, 1 for unknown (mostly very far and small objs)
            float_truncation = obj.truncation  # 0 ~ 1 and stands for truncation level

            # bottom centers ==> 3D centers
            locs = obj.t.copy()
            locs[1] = locs[1] - obj.h / 2
            if locs[-1] <= 0: continue  # objects which are behind the image

            # generate 8 corners of 3d bbox
            corners_3d = obj.generate_corners3d()
            corners_2d, _ = project_rect_to_image(calib['P'], corners_3d)
            projected_box2d = np.array([corners_2d[:, 0].min(), corners_2d[:, 1].min(),
                                        corners_2d[:, 0].max(), corners_2d[:, 1].max()])

            if projected_box2d[0] >= 0 and projected_box2d[1] >= 0 and \
                    projected_box2d[2] <= img_w - 1 and projected_box2d[3] <= img_h - 1:
                box2d = projected_box2d.copy()
            else:
                box2d = obj.box2d.copy()

                # filter some unreasonable annotations
            if self.filter_annos:
                if float_truncation >= self.filter_params[0] and (box2d[2:] - box2d[:2]).min() <= \
                        self.filter_params[1]: continue

            # project 3d location to the image plane
            proj_center, depth = project_rect_to_image(calib['P'], locs.reshape(-1, 3))
            proj_center = proj_center[0]

            # generate approximate projected center when it is outside the image
            proj_inside_img = (0 <= proj_center[0] <= img_w - 1) & (0 <= proj_center[1] <= img_h - 1)

            approx_center = False
            if not proj_inside_img:
                if self.consider_outside_objs:
                    approx_center = True

                    center_2d = (box2d[:2] + box2d[2:]) / 2
                    if self.proj_center_mode == 'intersect':
                        target_proj_center, edge_index = approx_proj_center(proj_center, center_2d.reshape(1, 2),
                                                                            (img_w, img_h))
                    else:
                        raise NotImplementedError
                else:
                    continue
            else:
                target_proj_center = proj_center.copy()

            # 10 keypoints
            bot_top_centers = np.stack((corners_3d[:4].mean(axis=0), corners_3d[4:].mean(axis=0)), axis=0)
            keypoints_3D = np.concatenate((corners_3d, bot_top_centers), axis=0)
            keypoints_2D, _ = project_rect_to_image(calib['P'], keypoints_3D)

            # keypoints mask: keypoint must be inside the image and in front of the camera
            keypoints_x_visible = (keypoints_2D[:, 0] >= 0) & (keypoints_2D[:, 0] <= img_w - 1)
            keypoints_y_visible = (keypoints_2D[:, 1] >= 0) & (keypoints_2D[:, 1] <= img_h - 1)
            keypoints_z_visible = (keypoints_3D[:, -1] > 0)

            # xyz visible
            keypoints_visible = keypoints_x_visible & keypoints_y_visible & keypoints_z_visible
            GRM_keypoint_visible = np.concatenate((keypoints_visible, np.array([proj_inside_img], dtype=np.bool_)),
                                                  axis=0)  # GRM_keypoint_visible shape: (11,)

            # center, diag-02, diag-13
            keypoints_depth_valid = np.stack((
                                             keypoints_visible[[8, 9]].all(), keypoints_visible[[0, 2, 4, 6]].all(),
                                             keypoints_visible[[1, 3, 5, 7]].all()))

            if self.use_modify_keypoint_visible:
                keypoints_visible = np.append(np.tile(keypoints_visible[:4] | keypoints_visible[4:8], 2),
                                              np.tile(keypoints_visible[8] | keypoints_visible[9], 2))
                keypoints_depth_valid = np.stack((keypoints_visible[[8, 9]].all(),
                                                  keypoints_visible[[0, 2, 4, 6]].all(),
                                                  keypoints_visible[[1, 3, 5, 7]].all()))

                keypoints_visible = keypoints_visible.astype(np.float32)
                keypoints_depth_valid = keypoints_depth_valid.astype(np.float32)

            # downsample bboxes, points to the scale of the extracted feature map (stride = 4)
            keypoints_2D = (keypoints_2D + pad_size.reshape(1, 2)) / self.down_ratio
            target_proj_center = (target_proj_center + pad_size) / self.down_ratio
            proj_center = (proj_center + pad_size) / self.down_ratio

            box2d[0::2] += pad_size[0]
            box2d[1::2] += pad_size[1]
            box2d /= self.down_ratio
            # 2d bbox center and size
            bbox_center = (box2d[:2] + box2d[2:]) / 2
            bbox_dim = box2d[2:] - box2d[:2]

            # target_center: the point to represent the object in the downsampled feature map
            if self.heatmap_center == '2D':
                target_center = bbox_center.round().astype(np.int32)
            else:
                target_center = target_proj_center.round().astype(np.int32)

            # clip to the boundary
            target_center[0] = np.clip(target_center[0], x_min, x_max)
            target_center[1] = np.clip(target_center[1], y_min, y_max)

            pred_2D = True  # In fact, there are some wrong annotations where the target center is outside the box2d
            if not (target_center[0] >= box2d[0] and target_center[1] >= box2d[1] and target_center[0] <= box2d[
                2] and target_center[1] <= box2d[3]):
                pred_2D = False

            if (bbox_dim > 0).all() and (0 <= target_center[0] <= self.output_width - 1) and (
                    0 <= target_center[1] <= self.output_height - 1):
                rot_y = obj.ry
                alpha = obj.alpha

                # generating heatmap
                if self.adjust_edge_heatmap and approx_center:
                    # for outside objects, generate 1-dimensional heatmap
                    bbox_width = min(target_center[0] - box2d[0], box2d[2] - target_center[0])
                    bbox_height = min(target_center[1] - box2d[1], box2d[3] - target_center[1])
                    radius_x, radius_y = bbox_width * self.edge_heatmap_ratio, bbox_height * self.edge_heatmap_ratio
                    radius_x, radius_y = max(0, int(radius_x)), max(0, int(radius_y))
                    assert min(radius_x, radius_y) == 0
                    heat_map[cls_id] = draw_umich_gaussian_2D(heat_map[cls_id], target_center, radius_x, radius_y)
                else:
                    # for inside objects, generate circular heatmap
                    radius = gaussian_radius(bbox_dim[1], bbox_dim[0])
                    radius = max(0, int(radius))
                    heat_map[cls_id] = draw_umich_gaussian(heat_map[cls_id], target_center, radius)

                cls_ids[i] = cls_id
                target_centers[i] = target_center
                # offset due to quantization for inside objects or offset from the interesection to the projected 3D center for outside objects
                offset_3D[i] = proj_center - target_center

                # 2D bboxes
                gt_bboxes[i] = obj.box2d.copy()  # for visualization
                if pred_2D: bboxes[i] = box2d

                # local coordinates for keypoints
                keypoints[i] = np.concatenate(
                    (keypoints_2D - target_center.reshape(1, -1), keypoints_visible[:, np.newaxis]), axis=1)
                keypoints_depth_mask[i] = keypoints_depth_valid

                dimensions[i] = np.array([obj.l, obj.h, obj.w])
                locations[i] = locs
                rotys[i] = rot_y
                alphas[i] = alpha

                orientations[i] = self.encode_alpha_multibin(alpha, num_bin=self.multibin_size)

                reg_mask[i] = 1
                reg_weight[i] = 1  # all objects are of the same weights (for now)
                trunc_mask[i] = int(approx_center)  # whether the center is truncated and therefore approximate
                occlusions[i] = float_occlusion
                truncations[i] = float_truncation
                GRM_keypoints_visible[i] = GRM_keypoint_visible


        # size_istrain=np.array([img.size[0],img.size[1],self.is_train.astype(np.int32)])
        # cls_ids=cls_ids[:,np.newaxis]
        # reg_mask=reg_mask[:, np.newaxis]
        # reg_weight=reg_weight[:, np.newaxis]
        # rotys=rotys[:, np.newaxis]
        # trunc_mask=trunc_mask[:, np.newaxis]
        # alphas=alphas[:, np.newaxis]
        # occlusions=occlusions[:, np.newaxis]
        # truncations=truncations[:, np.newaxis]
        # cat_array=np.concatenate((cls_ids,target_centers,keypoints_depth_mask,dimensions,locations,reg_mask,reg_weight,offset_3D
        #                           ,bboxes,rotys,trunc_mask,alphas,orientations,gt_bboxes,occlusions,truncations,GRM_keypoints_visible), axis=1)
        # target=[keypoints,pad_size, ori_img, heat_map]
        # target={'image_size':img.size,'is_train':self.is_train, 'cls_ids':cls_ids,
        #         'target_centers':target_centers, "keypoints": keypoints, "keypoints_depth_mask": keypoints_depth_mask,
        #         "dimensions": dimensions, "locations": locations, "calib": calib, "reg_mask": reg_mask, "reg_weight": reg_weight,
        #         "offset_3D": offset_3D, "2d_bboxes": bboxes, "pad_size": pad_size, "ori_img": ori_img, "rotys": rotys, "trunc_mask": trunc_mask,
        #         "alphas": alphas, "orientations": orientations, "hm": heat_map, "gt_bboxes": gt_bboxes, "occlusions": occlusions,
        # "truncations": truncations, "GRM_keypoints_visible": GRM_keypoints_visible}
        # if self.enable_edge_fusion:
        #     pass
        #     # size_istrain=np.append(size_istrain,input_edge_count)
        #     # target['edge_len']= input_edge_count
        #     # target['edge_indices']= input_edge_indices
        # else:
        #     input_edge_count=0
        #     # size_istrain = np.append(size_istrain, 0)
        #     input_edge_indices=None
        return img,img.size,self.is_train, cls_ids,target_centers, keypoints, keypoints_depth_mask,\
               dimensions, locations, reg_mask, reg_weight,\
               offset_3D, bboxes, pad_size, ori_img, rotys, trunc_mask,\
               alphas, orientations, heat_map, gt_bboxes,  \
               GRM_keypoints_visible, calib['P'],calib['R0'],calib['C2V'],calib['c_u'], calib['c_v'],\
               calib['f_u'],calib['f_v'],calib['b_x'],calib['b_y'],input_edge_count,input_edge_indices
        # return img, size_istrain,cat_array,keypoints,pad_size, ori_img, heat_map,original_idx,input_edge_indices


class DatasetCatalog:
    '''load dataset root by different setup'''
    def __init__(self, data_root = "./kitti"):
        self.DATA_DIR = data_root
        self.DATASETS = {
            "kitti_train": {
                "root": "training",
            },
            "kitti_test": {
                "root": "testing",
            },
            "kitti_demo":{
                "root": "kitti_demo",
            },
            "nuscenes_train":{
                "root": "train",
            },
            "nuscenes_val":{
                "root": "val",
            }
        }

    def get(self, name):
        if "kitti" in name:
            data_dir = self.DATA_DIR
            attrs = self.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["root"]),
            )
            return dict(
                factory="KITTIDataset",
                args=args,
            )
        elif "nuscenes" in name:
            data_dir = self.DATA_DIR
            attrs = self.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["root"]),
            )
            return dict(
                factory="NuscenesDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


def create_kitti_dataset(cfg):
    '''create kitti'''
    dataset_list = cfg.DATASETS.TRAIN if cfg.is_training else cfg.DATASETS.TEST
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasetcatalog =DatasetCatalog(data_root=cfg.DATASETS.DATA_ROOT)
    datasets = []
    device_num=cfg.group_size
    # cores = multiprocessing.cpu_count()
    # num_parallel_workers = int(cores / device_num)
    if cfg.is_training:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert images_per_batch % device_num == 0, \
            "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used." \
                .format(images_per_batch, device_num)

        images_per_gpu = images_per_batch // device_num
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert images_per_batch % device_num == 0, \
            "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used." \
                .format(images_per_batch, device_num)

        images_per_gpu = images_per_batch // device_num
    for dataset_name in dataset_list:  #create list of dataset
        data=datasetcatalog.get(dataset_name)
        kitti_dataset = KittiDataset(data["args"]['root'], cfg)
        dataset_column_names=['img','size','is_train', 'cls_ids','target_centers', 'keypoints', 'keypoints_depth_mask',\
               'dimensions', 'locations', 'reg_mask', 'reg_weight',\
               'offset_3D', 'bboxes', 'pad_size', 'ori_img', 'rotys', 'trunc_mask',\
               'alphas', 'orientations', 'heat_map', 'gt_bboxes',  'GRM_keypoints_visible', 'P','R0','C2V','c_u', 'c_v','f_u','f_v','b_x','b_y','input_edge_count','input_edge_indices']
        distributed_sampler = DistributedSampler(len(kitti_dataset), device_num, cfg.rank, shuffle=True)
        transformers_fn=Normalization(cfg)
        kitti_dataset.transforms = transformers_fn
        dataset = ds.GeneratorDataset(kitti_dataset, column_names=dataset_column_names, sampler=distributed_sampler,
                                      python_multiprocessing=True, num_parallel_workers=images_per_gpu)
        dataset=dataset.map(operations=transformers_fn,input_columns=['img'],num_parallel_workers=images_per_gpu, python_multiprocessing=True)
        dataset = dataset.batch(images_per_gpu, num_parallel_workers=images_per_gpu, drop_remainder=True)
        datasets.append(dataset)
    if cfg.is_training:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(datasets) == 1
        return datasets[0]
    return datasets
