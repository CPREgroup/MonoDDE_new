# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import csv
import os
from collections import defaultdict
from collections import deque

import numpy
import numpy as np
from mindspore import Tensor, ops, nn
import mindspore as ms
from shapely import Polygon


class SmoothedValue():
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=10):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def value(self):
        d = Tensor(list(self.deque))
        return d[-1].item()

    @property
    def median(self):
        d = Tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = Tensor(list(self.deque))
        return d.mean().asnumpy()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger():
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, Tensor) or isinstance(v, numpy.ndarray):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("{} object has no attribute {}".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            # loss_str.append(
            #     "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            # )
            loss_str.append(
                "{}: {:.4f}".format(name, meter.avg)
            )
        return self.delimiter.join(loss_str)


def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


class AllReduce(nn.Cell):
    def __init__(self):
        super(AllReduce, self).__init__()
        self.all_reduce = ops.AllReduce()

    def construct(self, x):
        return self.all_reduce(x)


def nms_hm(heat_map, kernel=3, reso=1):
    kernel = int(kernel / reso)
    if kernel % 2 == 0:
        kernel += 1

    pad = (kernel - 1) // 2
    maxpool2d = nn.MaxPool2d(kernel_size=kernel, stride=1, pad_mode='pad', padding=pad)
    hmax = maxpool2d(heat_map)

    eq_index = (hmax == heat_map)

    return heat_map * eq_index


def get_iou_3d(pred_corners, target_corners):
    """
    :param corners3d: (N, 8, 3) in rect coords
    :param query_corners3d: (N, 8, 3)
    :return: IoU
    """
    min = ops.Minimum()
    max = ops.Maximum()
    zeros=ops.Zeros()

    A, B = pred_corners, target_corners
    N = A.shape[0]

    # init output
    iou3d = zeros((N,),ms.float32)

    # for height overlap, since y face down, use the negative y
    min_h_a = - A[:, 0:4, 1].sum(axis=1) / 4.0
    max_h_a = - A[:, 4:8, 1].sum(axis=1) / 4.0
    min_h_b = - B[:, 0:4, 1].sum(axis=1) / 4.0
    max_h_b = - B[:, 4:8, 1].sum(axis=1) / 4.0

    # overlap in height
    h_max_of_min = max(min_h_a, min_h_b)
    h_min_of_max = min(max_h_a, max_h_b)
    h_overlap = max(zeros(h_min_of_max.shape,ms.float32),h_min_of_max - h_max_of_min)

    # x-z plane overlap
    for i in range(N):
        bottom_a, bottom_b =  Polygon(ops.transpose(A[i, 0:4, [0, 2]],(1,0))), Polygon(ops.transpose(B[i, 0:4, [0, 2]],(1,0)))

        if bottom_a.is_valid and bottom_b.is_valid:
            # check is valid,  A valid Polygon may not possess any overlapping exterior or interior rings.
            bottom_overlap = bottom_a.intersection(bottom_b).area
        else:
            bottom_overlap =0

        overlap3d = bottom_overlap * h_overlap[i]
        union3d = bottom_a.area * (max_h_a[i] - min_h_a[i]) + bottom_b.area  * (max_h_b[i] - min_h_b[i]) - overlap3d

        iou3d[i] = overlap3d / union3d

    return iou3d


def select_topk(heat_map, K=100):
    '''
    Args:
        heat_map: heat_map in [N, C, H, W]
        K: top k samples to be selected
        score: detection threshold

    Returns:

    '''
    batch, cls, height, width = heat_map.shape

    # First select topk scores in all classes and batchs
    # [N, C, H, W] -----> [N, C, H*W]
    heat_map = heat_map.view(batch, cls, -1)
    # Both in [N, C, K]
    topk_scores_all, topk_inds_all = ops.topk(heat_map, K)

    # topk_inds_all = topk_inds_all % (height * width) # todo: this seems redudant
    topk_ys = ops.cast(topk_inds_all / width,ms.float32)
    topk_xs = ops.cast(topk_inds_all % width,ms.float32)

    # assert isinstance(topk_xs, ops.cuda.FloatTensor)
    # assert isinstance(topk_ys, ops.cuda.FloatTensor)

    # Select topK examples across channel (classes)
    # [N, C, K] -----> [N, C*K]
    topk_scores_all = topk_scores_all.view(batch, -1)
    # Both in [N, K]
    topk_scores, topk_inds = ops.topk(topk_scores_all, K)
    topk_clses = ops.cast(topk_inds / K,ms.float32)

    # assert isinstance(topk_clses, ops.cuda.FloatTensor)

    # First expand it as 3 dimension
    topk_inds_all = _gather_feat(topk_inds_all.view(batch, -1, 1), topk_inds).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_inds).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_inds).view(batch, K)

    return topk_scores, topk_inds_all, topk_clses, topk_ys, topk_xs


def _gather_feat(feat, ind):
    '''
    Select specific indexs on feature map
    Args:
        feat: all results in 3 dimensions
        ind: positive index

    Returns:

    '''
    channel = feat.shape[-1]
    size=ms.Tensor(np.array([ind.shape[0], ind.shape[1], channel]),ms.int32)
    ind=ops.expand_dims(ind,-1).expand(size)
    feat = feat.gather_elements(1, ind)

    return feat


def get_iou3d(pred_bboxes, target_bboxes):
    num_query = target_bboxes.shape[0]

    # compute overlap along y axis
    min_h_a = - (pred_bboxes[:, 1] + pred_bboxes[:, 4] / 2)
    max_h_a = - (pred_bboxes[:, 1] - pred_bboxes[:, 4] / 2)
    min_h_b = - (target_bboxes[:, 1] + target_bboxes[:, 4] / 2)
    max_h_b = - (target_bboxes[:, 1] - target_bboxes[:, 4] / 2)

    # overlap in height
    h_max_of_min = ops.max(min_h_a, min_h_b)
    h_min_of_max = ops.min(max_h_a, max_h_b)
    h_overlap = (h_min_of_max - h_max_of_min).clamp_(min=0)

    # volumes of bboxes
    pred_volumes = pred_bboxes[:, 3] * pred_bboxes[:, 4] * pred_bboxes[:, 5]
    target_volumes = target_bboxes[:, 3] * target_bboxes[:, 4] * target_bboxes[:, 5]

    # derive x y l w alpha
    pred_bboxes = pred_bboxes[:, [0, 2, 3, 5, 6]]
    target_bboxes = target_bboxes[:, [0, 2, 3, 5, 6]]

    # convert bboxes to corners
    pred_corners = get_corners(pred_bboxes)
    target_corners = get_corners(target_bboxes)
    iou_3d = pred_bboxes.new_zeros(num_query)

    for i in range(num_query):
        ref_polygon = Polygon(pred_corners[i])
        target_polygon = Polygon(target_corners[i])
        overlap = ref_polygon.intersection(target_polygon).area
        # multiply bottom overlap and height overlap
        # for 3D IoU
        overlap3d = overlap * h_overlap[i]
        union3d = ref_polygon.area * (max_h_a[0] - min_h_a[0]) + target_polygon.area * (max_h_b[i] - min_h_b[i]) - overlap3d
        iou_3d[i] = overlap3d / union3d

    return iou_3d


def get_corners(bboxes):
    # bboxes: x, y, w, l, alpha; N x 5
    corners = ops.zeros((bboxes.shape[0], 4, 2), dtype=ms.float32)
    x, y, w, l = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    # compute cos and sin
    cos_alpha = ops.cos(bboxes[:, -1])
    sin_alpha = ops.sin(bboxes[:, -1])
    # front left
    corners[:, 0, 0] = x - w / 2 * cos_alpha - l / 2 * sin_alpha
    corners[:, 0, 1] = y - w / 2 * sin_alpha + l / 2 * cos_alpha

    # rear left
    corners[:, 1, 0] = x - w / 2 * cos_alpha + l / 2 * sin_alpha
    corners[:, 1, 1] = y - w / 2 * sin_alpha - l / 2 * cos_alpha

    # rear right
    corners[:, 2, 0] = x + w / 2 * cos_alpha + l / 2 * sin_alpha
    corners[:, 2, 1] = y + w / 2 * sin_alpha - l / 2 * cos_alpha

    # front right
    corners[:, 3, 0] = x + w / 2 * cos_alpha - l / 2 * sin_alpha
    corners[:, 3, 1] = y + w / 2 * sin_alpha + l / 2 * cos_alpha

    return corners


def uncertainty_guided_prune(separate_depths, GRM_uncern, cfg, depth_range=None, initial_use_uncern=True):
    '''
    Description:
        Prune the unresonable depth prediction results calculated by SoftGRM based on uncertainty.
    Input:
        separate_depths: The depths solved by SoftGRM. shape: (val_objs, 20).
        GRM_uncern: The estimated variance of SoftGRM equations. shape: (val_objs, 20).
        cfg: The config object.
        depth_range: The range of reasonable depth. Format: (depth_min, depth_max)
    Output:
        pred_depths: The solved depths. shape: (val_objs,)
        pred_uncerns: The solved uncertainty. shape: (val_objs,)
    '''
    objs_depth_list = []
    objs_uncern_list = []
    sigma_param = cfg.TEST.UNCERTAINTY_GUIDED_PARAM
    for obj_id in range(separate_depths.shape[0]):
        obj_depths = separate_depths[obj_id]
        obj_uncerns = GRM_uncern[obj_id]
        # Filter the depth estimations out of possible range.
        if depth_range != None:
            valid_depth_mask = (obj_depths > depth_range[0]) & (obj_depths < depth_range[1])
            obj_depths = obj_depths[valid_depth_mask]
            obj_uncerns = obj_uncerns[valid_depth_mask]
        # If all objects are filtered.
        if obj_depths.shape[0] == 0:
            objs_depth_list.append(ops.expand_dims(separate_depths[obj_id].mean(),0))
            objs_uncern_list.append(ops.expand_dims(GRM_uncern[obj_id].mean, 0))
            continue

        if initial_use_uncern:
            considered_index = obj_uncerns.argmin()
        else:
            considered_index = find_crowd_index(obj_depths)

        considered_mask = ops.zeros(obj_depths.shape, dtype=ms.bool_)
        considered_mask[considered_index] = True
        obj_depth_mean = obj_depths[considered_index]
        obj_depth_sigma = ops.sqrt(obj_uncerns[considered_index])
        flag = True
        search_cnt = 0
        while flag == True:
            search_cnt += 1
            flag = False
            new_considered_mask = (obj_depths > obj_depth_mean - sigma_param * obj_depth_sigma) & (
                        obj_depths < obj_depth_mean + sigma_param * obj_depth_sigma)
            if considered_mask.equal(new_considered_mask).sum()<considered_mask.shape[0] or search_cnt > 20:  # No new elements are considered.
                objs_depth_list.append(ops.expand_dims(obj_depth_mean,0))
                objs_uncern_list.append(ops.expand_dims((obj_depth_sigma * obj_depth_sigma),0))
                break
            else:
                considered_mask = new_considered_mask
                considered_depth = obj_depths[considered_mask]
                considered_uncern = obj_uncerns[considered_mask]
                considered_w = 1 / considered_uncern
                considered_w = considered_w / considered_w.sum()
                obj_depth_mean = (considered_w * considered_depth).sum()
                obj_depth_sigma = ops.sqrt((considered_w * considered_uncern).sum())
                flag = True

    pred_depths = ops.cat(objs_depth_list, axis=0)
    pred_uncerns = ops.cat(objs_uncern_list, axis=0)
    return pred_depths, pred_uncerns


def find_crowd_index(obj_depths):
    '''
    Description:
        Find the depth at the most crowded index for each objects.
    Input:
        obj_depths: The estimated depths of an object. shape: (num_depth,).
    Output:
        crowd_index: Int.
    '''
    num_depth = obj_depths.shape[0]

    depth_matrix = ops.expand_dims(obj_depths,0).expand(num_depth, num_depth)
    cost_matrix = (ops.expand_dims(obj_depths,1) - depth_matrix).abs()    # cost_matrix shape: (num_depth, num_depth)
    crowd_index = cost_matrix.sum(axis = 1).argmin()
    return crowd_index


def error_from_uncertainty(uncern):
    '''
    Description:
        Get the error derived from uncertainty.
    Input:
        uncern: uncertainty tensor. shape: (val_objs, 20)
    Output:
        error: The produced error. shape: (val_objs,)
    '''
    if uncern.ndim != 2:
        raise Exception("uncern must be a 2-dim tensor.")
    weights = 1 / uncern	# weights shape: (total_num_objs, 20)
    weights = weights / ops.sum(weights, dim = 1, keepdim = True)
    error = ops.sum(weights * uncern, dim = 1)	# error shape: (valid_objs,)
    return error


def nms_3d(results, bboxes, scores, iou_threshold = 0.2):
    '''
    Description:
        Given the 3D bounding boxes of objects and confidence scores, remove the overlapped ones with low confidence.
    Input:
        results: The result tensor for KITTI. shape: (N, 14)
        bboxes: Vertex coordinates of 3D bounding boxes. shape: (N, 8, 3)
        scores: Confidence scores. shape: (N).
        iou_threshold: The IOU threshold for filering overlapped objects. Type: float.
    Output:
        preserved_results: results after NMS.
    '''
    descend_index = ops.flip(ops.argsort(scores, axis = 0), dims = (0,))
    results = results[descend_index, :]
    sorted_bboxes = bboxes[descend_index, :, :]

    box_indices = np.arange(0, sorted_bboxes.shape[0])
    suppressed_box_indices = []
    tmp_suppress = []

    while len(box_indices) > 0:

        if box_indices[0] not in suppressed_box_indices:
            selected_box = box_indices[0]
            tmp_suppress = []

            for i in range(len(box_indices)):
                if box_indices[i] != selected_box:
                    selected_iou = get_iou_3d(sorted_bboxes[selected_box].squeeze(0), sorted_bboxes[box_indices[i]].squeeze(0))[0]
                    if selected_iou > iou_threshold:
                        suppressed_box_indices.append(box_indices[i])
                        tmp_suppress.append(i)

        box_indices = np.delete(box_indices, tmp_suppress, axis=0)
        box_indices = box_indices[1:]

    preserved_index = np.setdiff1d(np.arange(0, sorted_bboxes.shape[0]), np.array(suppressed_box_indices), assume_unique=True)
    preserved_results = results[preserved_index.tolist(), :]

    return preserved_results

