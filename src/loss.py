import mindspore.ops as ops
import mindspore.nn as nn
import mindspore as ms
import mindspore.numpy as mnp
import numpy as np
import pdb
from shapely.geometry import Polygon
from .net_utils import Converter_key2channel,project_image_to_rect
PI = np.pi


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def warpper(iteration):
        if iteration < rampup_length:
            iteration = np.clip(iteration, 0.0, rampup_length)
            phase = 1.0 - iteration / rampup_length
            # weight increase from 0.007~1
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return warpper


def reweight_loss(loss, weight):
    '''
    Description:
        Reweight loss by weight.
    Input:
        loss: Loss vector. shape: (val_objs,).
        weight: Weight vector. shape: (val_objs,).
    Output:
        w_loss: Float.
    '''
    w_loss = loss * weight
    # factor = loss.sum() / torch.clamp(w_loss.sum(), min = 1e-9)
    # w_loss = w_loss * factor.detach()
    return w_loss


'''depth loss'''
class Berhu_Loss(nn.Cell):
    def __init__(self):
        super(Berhu_Loss, self).__init__()
        # according to ECCV18 Joint taskrecursive learning for semantic segmentation and depth estimation
        self.c = 0.2

    def construct(self, prediction, target):
        pdb.set_trace()
        differ=(prediction - target).abs()
        c=ops.clip_by_value(self.c,differ.max() * self.c,1e-4,)   #有疑问
        # larger than c: l2 loss
        # smaller than c: l1 loss
        large_idx = ops.nonzero(differ > c)
        small_idx = ops.nonzero(differ <= c)

        loss = differ[small_idx].sum() + ((differ[large_idx] ** 2) / c + c).sum() / 2

        return loss


class Inverse_Sigmoid_Loss(nn.Cell):
    def __init__(self):
        super(Inverse_Sigmoid_Loss, self).__init__()
        self.l1loss = nn.L1Loss(reduction='none')
        self.sigmoid=ops.Sigmoid()

    def construct(self, prediction, target, weight=None):
        trans_prediction = 1 / self.sigmoid(target) - 1
        loss=self.l1loss(trans_prediction, target)
        if weight is not None:
            loss = loss * weight

        return loss


class Log_L1_Loss(nn.Cell):
    def __init__(self):
        super(Log_L1_Loss, self).__init__()
        self.log=ops.Log()
        self.l1loss=nn.L1Loss(reduction='none')

    def construct(self, prediction, target, weight=None):
        loss=self.l1loss(self.log(prediction), self.log(target))

        if weight is not None:
            loss = loss * weight
        return loss


''''''
class FocalLoss(nn.Cell):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.pow=ops.Pow()

    def construct(self, prediction, target):
        positive_index = target.equal(1)
        negative_index = (target.lt(1) & target.ge(0))
        ignore_index = ops.equal(target,-1)  # ignored pixels

        negative_weights = self.pow(1 - target, self.beta)
        loss = 0.

        positive_loss = ops.log(prediction) \
                        * self.pow(1 - prediction, self.alpha) * positive_index

        negative_loss = ops.log(1 - prediction) \
                        * self.pow(prediction, self.alpha) * negative_weights * negative_index

        num_positive = positive_index.sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        loss = - negative_loss - positive_loss

        return loss, num_positive


'''iou loss'''
class IOULoss(nn.Cell):
    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type
        self.min = ops.Minimum()
        self.max = ops.Maximum()
        self.log=ops.Log()

    def construct(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = self.min(pred_left, target_left) + self.min(pred_right, target_right)
        g_w_intersect = self.max(pred_left, target_left) + self.max(pred_right, target_right)
        h_intersect = self.min(pred_bottom, target_bottom) + self.min(pred_top, target_top)
        g_h_intersect = self.max(pred_bottom, target_bottom) + self.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect

        area_union = target_area + pred_area - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loss_type == 'iou':
            losses = -self.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        return losses, ious


'''multibin loss'''
def Real_MultiBin_loss(vector_ori, gt_ori, num_bin=4):
    gt_ori = gt_ori.view((-1, gt_ori.shape[-1]))  # bin1 cls, bin1 offset, bin2 cls, bin2 offst
    cls_losses = ms.Tensor(0,ms.float32)
    reg_losses = ms.Tensor(0,ms.float32)

    reg_cnt = ms.Tensor(0,ms.float32)
    l1loss=nn.L1Loss(reduction='none')
    sin=ops.Sin()
    cos=ops.Cos()
    for i in range(num_bin):
        # bin cls loss
        cls_ce_loss = ops.cross_entropy(vector_ori[:, (i * 2): (i * 2 + 2)], ops.expand_dims(gt_ori[:, i],-1), reduction='none')  #gt_ori  p
        # regression loss
        valid_mask_i = (gt_ori[:, i] == 1)
        cls_losses += cls_ce_loss.mean()
        if valid_mask_i.sum() > 0:
            s = num_bin * 2 + i * 2
            e = s + 2
            pred_offset = ops.L2Normalize()(vector_ori[valid_mask_i, s: e])
            reg_loss = l1loss(pred_offset[:, 0], sin((gt_ori[valid_mask_i, num_bin + i]))) + \
                       l1loss(pred_offset[:, 1], cos((gt_ori[valid_mask_i, num_bin + i])))

            reg_losses += reg_loss.sum()
            reg_cnt += valid_mask_i.sum()

    return ops.div(cls_losses, ms.Tensor(num_bin,ms.float32)) + ops.div(reg_losses, reg_cnt)


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


class Mono_loss(nn.Cell):
    def __init__(self,cfg):
        super(Mono_loss, self).__init__()
        self.cfg = cfg
        self.key2channel = Converter_key2channel(keys=cfg.MODEL.HEAD.REGRESSION_HEADS,
                                                 channels=cfg.MODEL.HEAD.REGRESSION_CHANNELS)
        self.loss_weight_ramper = exp_rampup(cfg.SOLVER.RAMPUP_ITERATIONS)

        self.max_objs = cfg.DATASETS.MAX_OBJECTS
        self.center_sample = cfg.MODEL.HEAD.CENTER_SAMPLE
        self.regress_area = cfg.MODEL.HEAD.REGRESSION_AREA
        self.heatmap_type = cfg.MODEL.HEAD.HEATMAP_TYPE
        self.corner_depth_sp = cfg.MODEL.HEAD.SUPERVISE_CORNER_DEPTH
        self.loss_keys = cfg.MODEL.HEAD.LOSS_NAMES

        self.dim_weight = ops.Tensor(cfg.MODEL.HEAD.DIMENSION_WEIGHT).view(1, 3)
        self.uncertainty_range = cfg.MODEL.HEAD.UNCERTAINTY_RANGE

        # loss functions
        loss_types = cfg.MODEL.HEAD.LOSS_TYPE
        self.cls_loss_fnc = FocalLoss(cfg.MODEL.HEAD.LOSS_PENALTY_ALPHA,
                                      cfg.MODEL.HEAD.LOSS_BETA)  # penalty-reduced focal loss
        self.iou_loss = IOULoss(loss_type=loss_types[2])  # iou loss for 2D detection

        # depth loss
        if loss_types[3] == 'berhu':
            self.depth_loss = Berhu_Loss()
        elif loss_types[3] == 'inv_sig':
            self.depth_loss = Inverse_Sigmoid_Loss()
        elif loss_types[3] == 'log':
            self.depth_loss = Log_L1_Loss()
        elif loss_types[3] == 'L1':
            self.depth_loss = nn.L1Loss(reduction='none')
        else:
            raise ValueError

        # regular regression loss
        self.reg_loss = loss_types[1]
        self.reg_loss_fnc = nn.L1Loss(reduction='none') if loss_types[1] == 'L1' else nn.SmoothL1Loss
        self.keypoint_loss_fnc = nn.L1Loss(reduction='none')

        # multi-bin loss setting for orientation estimation
        self.multibin = (cfg.INPUT.ORIENTATION == 'multi-bin')
        self.orien_bin_size = cfg.INPUT.ORIENTATION_BIN_SIZE
        self.trunc_offset_loss_type = cfg.MODEL.HEAD.TRUNCATION_OFFSET_LOSS

        self.loss_weights = {}
        for key, weight in zip(cfg.MODEL.HEAD.LOSS_NAMES, cfg.MODEL.HEAD.INIT_LOSS_WEIGHT): self.loss_weights[
            key] = weight

        # whether to compute corner loss
        self.compute_direct_depth_loss = 'depth_loss' in self.loss_keys
        self.compute_keypoint_depth_loss = 'keypoint_depth_loss' in self.loss_keys
        self.compute_weighted_depth_loss = 'weighted_avg_depth_loss' in self.loss_keys
        self.compute_corner_loss = 'corner_loss' in self.loss_keys
        self.separate_trunc_offset = 'trunc_offset_loss' in self.loss_keys
        self.compute_combined_depth_loss = 'combined_depth_loss' in self.loss_keys
        self.compute_GRM_loss = 'GRM_loss' in self.loss_keys
        self.compute_SoftGRM_loss = 'SoftGRM_loss' in self.loss_keys
        self.compute_IOU3D_predict_loss = 'IOU3D_predict_loss' in self.loss_keys

        # corner_with_uncertainty is whether to use corners with uncertainty to solve the depth, rather than directly applying uncertainty to corner estimation itself.
        self.pred_direct_depth = 'depth' in self.key2channel.keys
        self.depth_with_uncertainty = 'depth_uncertainty' in self.key2channel.keys
        self.compute_keypoint_corner = 'corner_offset' in self.key2channel.keys
        self.corner_with_uncertainty = 'corner_uncertainty' in self.key2channel.keys

        self.corner_offset_uncern = 'corner_offset_uncern' in self.key2channel.keys
        self.dim_uncern = '3d_dim_uncern' in self.key2channel.keys
        self.combined_depth_uncern = 'combined_depth_uncern' in self.key2channel.keys
        self.corner_loss_uncern = 'corner_loss_uncern' in self.key2channel.keys

        self.perdict_IOU3D = 'IOU3D_predict' in self.key2channel.keys

        self.uncertainty_weight = cfg.MODEL.HEAD.UNCERTAINTY_WEIGHT  # 1.0
        self.keypoint_xy_weights = cfg.MODEL.HEAD.KEYPOINT_XY_WEIGHT  # [1, 1]
        self.keypoint_norm_factor = cfg.MODEL.HEAD.KEYPOINT_NORM_FACTOR  # 1.0
        self.modify_invalid_keypoint_depths = cfg.MODEL.HEAD.MODIFY_INVALID_KEYPOINT_DEPTH

        # depth used to compute 8 corners
        self.corner_loss_depth = cfg.MODEL.HEAD.CORNER_LOSS_DEPTH
        self.eps = 1e-5
        self.SoftGRM_loss_weight = ms.Tensor(self.cfg.MODEL.HEAD.SOFTGRM_LOSS_WEIGHT)
        self.dynamic_thre = cfg.SOLVER.DYNAMIC_THRESHOLD

        self.exp=ops.Exp()
        self.log=ops.Log()
        self.div=ops.Div()
        self.reducmean=ops.ReduceMean()
        self.reducesum=ops.ReduceSum()
        self.expanddims=ops.ExpandDims()
        self.concat=ops.Concat(axis=1)
        self.argminwithvalue=ops.ArgMinWithValue(axis=1)


    def construct(self, targets_heatmap,predictions,pred_targets, preds, reg_nums, weights,iteration):
        pred_heatmap = predictions['cls']
        # heatmap loss
        if self.heatmap_type == 'centernet':
            hm_loss, num_hm_pos = self.cls_loss_fnc(pred_heatmap, targets_heatmap)
            hm_loss = ops.Tensor(self.loss_weights['hm_loss']) * hm_loss  # Heatmap loss.
            hm_loss = hm_loss / ops.clamp(num_hm_pos, 1)
        else:
            raise ValueError
        num_reg_2D = reg_nums['reg_2D']
        num_reg_3D = reg_nums['reg_3D']
        num_reg_obj = reg_nums['reg_obj']

        trunc_mask = ops.cast(pred_targets['trunc_mask_3D'],ms.bool_)
        num_trunc = trunc_mask.sum()
        num_nontrunc = num_reg_obj - num_trunc

        #loss define
        depth_MAE = ms.Tensor(0)
        reg_2D_loss = ms.Tensor(0)
        iou_2D=ms.Tensor(0)
        offset_3D_loss=ms.Tensor(0)
        trunc_offset_loss=ms.Tensor(0)
        corner_3D_loss=ms.Tensor(0)
        depth_3D_loss=ms.Tensor(0)
        real_depth_3D_loss=ms.Tensor(0)
        keypoint_loss=ms.Tensor(0)
        center_MAE=ms.Tensor(0)
        keypoint_02_MAE=ms.Tensor(0)
        keypoint_13_MAE=ms.Tensor(0)
        lower_MAE=ms.Tensor(0)
        hard_MAE=ms.Tensor(0)
        soft_MAE=ms.Tensor(0)
        mean_MAE=ms.Tensor(0)
        keypoint_depth_loss=ms.Tensor(0)
        IOU3D_predict_loss=ms.Tensor(0)
        soft_depth_loss=ms.Tensor(0)
        combined_depth_loss=ms.Tensor(0)
        GRM_loss=ms.Tensor(0)
        SoftGRM_loss=ms.Tensor(0)
        orien_3D_loss=ms.Tensor(0)
        log_valid_keypoint_depth_loss=ms.Tensor(0)
        trunc_offset_loss = ms.Tensor(0)

        # IoU loss for 2D detection
        if num_reg_2D > 0:
            reg_2D_loss, iou_2D = self.iou_loss(preds['reg_2D'], pred_targets['reg_2D'])
            reg_2D_loss = ops.Tensor(self.loss_weights['bbox_loss'], ms.float32) * reg_2D_loss.mean()
            iou_2D = iou_2D.mean()
            depth_MAE = (preds['depth_3D'] - pred_targets['depth_3D']).abs() / pred_targets[
                'depth_3D']  # MAE for direct depth regression.

        objs_IOU = ops.Tensor(get_iou_3d(preds['corners_3D'], pred_targets['corners_3D']).asnumpy()) # objects_IOU shape: (val_objs,)
        pred_IoU_3D = objs_IOU.mean()

        if self.cfg.SOLVER.DYNAMIC_WEIGHT:
            objs_weight = objs_IOU
            # objs_weight.requires_grad = False

            thre_mask = objs_IOU < self.dynamic_thre
            objs_weight[thre_mask] = 1
            thre_mask = objs_IOU >= self.dynamic_thre
            objs_weight[thre_mask]=self.exp(-(objs_IOU[thre_mask] - self.dynamic_thre))
            # thre_mask1=thre_mask.asnumpy().tolist()
            # objs_IOU_exp=self.exp(-(objs_IOU[thre_mask1] - self.dynamic_thre))
            # for i in range(len(objs_IOU_exp)):
            #     objs_weight[thre_mask] = objs_IOU_exp[i]
            objs_weight = ops.stop_gradient(objs_weight)

        if self.compute_IOU3D_predict_loss:
            IOU_label = ops.Tensor(objs_IOU.asnumpy())
            IOU_label = 2 * IOU_label - 0.5
            IOU_label[IOU_label > 1] = 1
            IOU_label[IOU_label < 0] = 0
            IOU3D_predict_loss = ops.binary_cross_entropy_with_logits(preds['IOU3D_predict'].squeeze(1), IOU_label,
                                                ops.Tensor(np.array([1.0, 1.0, 1.0]), ms.float32),reduction='none')
            IOU3D_predict_loss = self.loss_weights['IOU3D_predict_loss'] * IOU3D_predict_loss.mean()

        dims_3D_loss=ms.Tensor(0)
        if num_reg_3D > 0:
            # direct depth loss
            if self.compute_direct_depth_loss:
                depth_3D_loss = ms.Tensor(self.loss_weights['depth_loss'],ms.float32) * self.depth_loss(preds['depth_3D'],pred_targets['depth_3D'])
                real_depth_3D_loss = depth_3D_loss.asnumpy().mean()

                if self.depth_with_uncertainty:
                    depth_3D_loss = depth_3D_loss * self.exp(- preds['depth_uncertainty']) + \
                                    preds['depth_uncertainty'] * ops.Tensor(self.loss_weights['depth_loss'],ms.float32)

                depth_3D_loss = depth_3D_loss.mean()

            # offset_3D loss
            offset_3D_loss = self.reg_loss_fnc(preds['offset_3D'], pred_targets['offset_3D']).sum(axis=1) # offset_3D_loss shape: (val_objs,)
            # offset_3D_loss=ops.cast(offset_3D_loss

            # use different loss functions for inside and outside objects
            if self.separate_trunc_offset:
                if self.trunc_offset_loss_type == 'L1':
                    trunc_offset_loss = offset_3D_loss[trunc_mask]

                elif self.trunc_offset_loss_type == 'log':
                    trunc_offset_loss = self.log(1 + offset_3D_loss[trunc_mask])   #trunc_mask have a problem.
                # trunc_offset_loss=ops.cast(trunc_offset_loss,ms.float64)

                trunc_offset_loss = self.loss_weights['trunc_offset_loss'] * trunc_offset_loss.sum() / ops.maximum(trunc_mask.sum(),1)
                offset_3D_loss = self.loss_weights['offset_loss'] * offset_3D_loss[~trunc_mask].mean()
            else:
                offset_3D_loss = self.loss_weights['offset_loss'] * offset_3D_loss.mean()

            # orientation loss
            if self.multibin:
                orien_3D_loss = ops.Tensor(self.loss_weights['orien_loss'],ms.float32) * \
                                Real_MultiBin_loss(preds['orien_3D'], pred_targets['orien_3D'],
                                                   num_bin=self.orien_bin_size)
            # dimension loss
            dims_3D_loss = self.reg_loss_fnc(preds['dims_3D'], pred_targets['dims_3D']) * ops.cast(self.dim_weight,preds['dims_3D'].dtype)
            if self.dim_uncern:
                dims_3D_loss = dims_3D_loss / preds['dim_uncern'] + self.log(preds['dim_uncern'])
            dims_3D_loss = dims_3D_loss.sum(axis=1)
            if self.cfg.SOLVER.DYNAMIC_WEIGHT:
                dims_3D_loss = reweight_loss(dims_3D_loss, objs_weight)
            dims_3D_loss = self.loss_weights['dims_loss'] * dims_3D_loss.mean()

            # corner loss
            if self.compute_corner_loss:
                # N x 8 x 3
                corner_3D_loss = self.reg_loss_fnc(preds['corners_3D'], pred_targets['corners_3D']).sum(axis=2).mean(axis=1)
                if self.corner_loss_uncern:
                    corner_loss_uncern = preds['corner_loss_uncern'].squeeze(1)
                    corner_3D_loss = corner_3D_loss / corner_loss_uncern + self.log(corner_loss_uncern)
                if self.cfg.SOLVER.DYNAMIC_WEIGHT:
                    corner_3D_loss = reweight_loss(corner_3D_loss, objs_weight)
                corner_3D_loss = self.loss_weight_ramper(iteration) * self.loss_weights[
                    'corner_loss'] * corner_3D_loss.mean()

            if self.compute_keypoint_corner:
                if self.corner_offset_uncern:
                    keypoint_loss = self.keypoint_loss_fnc(preds['keypoints'], pred_targets['keypoints'])  # keypoint_loss shape: (val_objs, 10, 2)
                    keypoint_loss_mask = ops.cast(ops.expand_dims(pred_targets['keypoints_mask'],2).expand_as(
                        keypoint_loss),ms.bool_)  # keypoint_loss_mask shape: (val_objs, 10, 2)
                    keypoint_loss_uncern = preds['corner_offset_uncern'].view(-1, 10, 2)  # keypoint_loss_uncern shape: (val_objs, 10, 2)

                    valid_keypoint_loss = keypoint_loss[keypoint_loss_mask]  # valid_keypoint_loss shape: (valid_equas,)
                    invalid_keypoint_loss = ops.Tensor(keypoint_loss[~keypoint_loss_mask].asnumpy())  # invalid_keypoint_loss shape: (invalid_equas,)
                    valid_keypoint_uncern = keypoint_loss_uncern[keypoint_loss_mask]  # valid_keypoint_uncern shape: (valid_equas,)
                    invalid_keypoint_uncern = keypoint_loss_uncern[~keypoint_loss_mask]  # invalid_keypoint_uncern: (invalid_equas,)

                    valid_keypoint_loss = valid_keypoint_loss / valid_keypoint_uncern + self.log(valid_keypoint_uncern)
                    if self.cfg.SOLVER.DYNAMIC_WEIGHT:
                        keypoint_objs_weight = ops.expand_dims(ops.expand_dims(objs_weight,1),2).expand_as(
                            keypoint_loss)  # keypoint_objs_weight shape: (val_objs, 10, 2)
                        valid_keypoint_objs_weight = keypoint_objs_weight[
                            keypoint_loss_mask]  # valid_keypoint_objs_weight shape: (valid_equas,)
                        valid_keypoint_loss = reweight_loss(valid_keypoint_loss, valid_keypoint_objs_weight)
                    valid_keypoint_loss = valid_keypoint_loss.sum() / ops.maximum(keypoint_loss_mask.sum(), 1)

                    invalid_keypoint_loss = invalid_keypoint_loss / invalid_keypoint_uncern
                    invalid_keypoint_loss = invalid_keypoint_loss.sum() / ops.clamp(invalid_keypoint_loss.sum(), 1)

                    if self.modify_invalid_keypoint_depths:
                        keypoint_loss = self.loss_weights['keypoint_loss'] * (
                                    valid_keypoint_loss + invalid_keypoint_loss)
                    else:
                        keypoint_loss = self.loss_weights['keypoint_loss'] * valid_keypoint_loss
                else:
                    # N x K x 3
                    keypoint_loss = self.keypoint_loss_fnc(preds['keypoints'], pred_targets['keypoints']).sum(2) * pred_targets['keypoints_mask']
                    keypoint_loss = keypoint_loss.sum(axis=1)  # Left keypoints_loss shape: (val_objs,)
                    if self.cfg.SOLVER.DYNAMIC_WEIGHT:
                        keypoint_loss = reweight_loss(keypoint_loss, objs_weight)
                    keypoint_loss = ops.Tensor(self.loss_weights['keypoint_loss'],ms.float32) * keypoint_loss.sum() / ops.clamp(pred_targets['keypoints_mask'].sum(), 1)

                if self.compute_keypoint_depth_loss:
                    pred_keypoints_depth, keypoints_depth_mask = preds['keypoints_depths'], ops.cast(pred_targets['keypoints_depth_mask'],ms.bool_)
                    target_keypoints_depth = self.expanddims(pred_targets['depth_3D'],-1).tile((1,3))

                    valid_pred_keypoints_depth = pred_keypoints_depth[keypoints_depth_mask]
                    invalid_pred_keypoints_depth = ops.Tensor(pred_keypoints_depth[~keypoints_depth_mask].asnumpy()) # The depths decoded from invalid keypoints are not used for updating networks.

                    # valid and non-valid
                    valid_keypoint_depth_loss = self.loss_weights['keypoint_depth_loss'] * self.reg_loss_fnc(
                        valid_pred_keypoints_depth,
                        target_keypoints_depth[keypoints_depth_mask])

                    invalid_keypoint_depth_loss = self.loss_weights['keypoint_depth_loss'] * self.reg_loss_fnc(
                        invalid_pred_keypoints_depth,
                        target_keypoints_depth[~keypoints_depth_mask])

                    # for logging
                    log_valid_keypoint_depth_loss = ops.Tensor(valid_keypoint_depth_loss.asnumpy()).mean()

                    if self.corner_with_uncertainty:
                        # center depth, corner 0246 depth, corner 1357 depth
                        pred_keypoint_depth_uncertainty = preds['corner_offset_uncertainty']

                        valid_uncertainty = pred_keypoint_depth_uncertainty[keypoints_depth_mask]
                        invalid_uncertainty = pred_keypoint_depth_uncertainty[~keypoints_depth_mask]

                        valid_keypoint_depth_loss = valid_keypoint_depth_loss * self.exp(- valid_uncertainty) + \
                                                    self.loss_weights['keypoint_depth_loss'] * valid_uncertainty

                        invalid_keypoint_depth_loss = invalid_keypoint_depth_loss * self.exp(
                            - invalid_uncertainty)  # Lead to infinite uncertainty for invisible keypoints.

                    # average
                    valid_keypoint_depth_loss = valid_keypoint_depth_loss.sum() / ops.clamp(
                        keypoints_depth_mask.sum(), 1)
                    invalid_keypoint_depth_loss = invalid_keypoint_depth_loss.sum() / ops.clamp(
                        (~keypoints_depth_mask).sum(), 1)  #all ~.problem

                    # the gradients of invalid depths are not back-propagated
                    if self.modify_invalid_keypoint_depths:
                        keypoint_depth_loss = (valid_keypoint_depth_loss + invalid_keypoint_depth_loss)
                    else:
                        keypoint_depth_loss = valid_keypoint_depth_loss

                # compute the average error for each method of depth estimation
                keypoint_MAE = (preds['keypoints_depths'] - ops.expand_dims(pred_targets['depth_3D'],-1)).abs() \
                               / ops.expand_dims(pred_targets['depth_3D'],-1)

                center_MAE = keypoint_MAE[:, 0].mean()
                keypoint_02_MAE = keypoint_MAE[:, 1].mean()
                keypoint_13_MAE = keypoint_MAE[:, 2].mean()

                if self.corner_with_uncertainty:
                    if self.pred_direct_depth and self.depth_with_uncertainty:
                        combined_depth = self.concat((ops.expand_dims(preds['depth_3D'],1), (preds['keypoints_depths'])))
                        combined_uncertainty = self.concat(
                            (ops.expand_dims(preds['depth_uncertainty'],1), preds['corner_offset_uncertainty'])).exp()
                        combined_MAE = self.concat((ops.expand_dims(depth_MAE,1), keypoint_MAE))
                    else:
                        combined_depth = preds['keypoints_depths']
                        combined_uncertainty = preds['corner_offset_uncertainty'].exp()
                        combined_MAE = keypoint_MAE

                    # the oracle MAE
                    lower_MAE = ops.min(combined_MAE,1)[0]
                    # the hard ensemble
                    hard_MAE = combined_MAE[ops.arange(combined_MAE.shape[0]), combined_uncertainty.argmin(axis=1)]
                    # the soft ensemble
                    combined_weights = 1 / combined_uncertainty
                    combined_weights = combined_weights / combined_weights.sum(axis=1,keepdims=True)
                    soft_depths = ops.sum(combined_depth * combined_weights, 1)
                    soft_MAE = (soft_depths - pred_targets['depth_3D']).abs() / pred_targets['depth_3D']
                    # the average ensemble
                    mean_depths = combined_depth.mean(axis=1)
                    mean_MAE = (mean_depths - pred_targets['depth_3D']).abs() / pred_targets['depth_3D']

                    # average
                    lower_MAE, hard_MAE, soft_MAE, mean_MAE = lower_MAE.mean(), hard_MAE.mean(), soft_MAE.mean(), mean_MAE.mean()

                    if self.compute_weighted_depth_loss:
                        soft_depth_loss = self.loss_weights['weighted_avg_depth_loss'] * \
                                          self.reg_loss_fnc(soft_depths, pred_targets['depth_3D'])

            depth_MAE = depth_MAE.mean()

            if self.compute_combined_depth_loss:  # The loss for final estimated depth.
                combined_depth_loss = self.reg_loss_fnc(preds['combined_depth'], pred_targets['depth_3D'])
                if self.combined_depth_uncern:
                    combined_depth_uncern = preds['combined_depth_uncern'].squeeze(1)
                    combined_depth_loss = combined_depth_loss / combined_depth_uncern + self.log(combined_depth_uncern)
                if self.cfg.SOLVER.DYNAMIC_WEIGHT:
                    combined_depth_loss = reweight_loss(combined_depth_loss, objs_weight)
                combined_depth_loss = self.loss_weight_ramper(iteration) * self.loss_weights[
                    'combined_depth_loss'] * combined_depth_loss.mean()

            if self.compute_GRM_loss:
                GRM_valid_items = pred_targets['GRM_valid_items']  # GRM_valid_items shape: (val_objs, 25)
                # GRM_valid_items_inverse=(~GRM_valid_items).tolist()
                # GRM_valid_items=GRM_valid_items.tolist()
                valid_GRM_A = preds['GRM_A'][GRM_valid_items, :]  # valid_GRM_A shape: (valid_equas, 3)
                valid_GRM_B = preds['GRM_B'][GRM_valid_items, :]  # valid_GRM_B shape: (valid_equas, 1)
                invalid_GRM_A = ops.Tensor(preds['GRM_A'][~GRM_valid_items, :].asnumpy())  # invalid_GRM_A shape: (invalid_equas, 3)
                invalid_GRM_B = ops.Tensor(preds['GRM_B'][~GRM_valid_items, :].asnumpy()) # invalid_GRM_B shape: (invalid_equas, 1)
                valid_target_location = ops.expand_dims(pred_targets['locations'],1).expand_as(preds['GRM_A'])[
                    GRM_valid_items]  # # valid_target_location shape: (valid_equas, 3)
                invalid_target_location = ops.expand_dims(pred_targets['locations'],1).expand_as(preds['GRM_A'])[
                    ~GRM_valid_items]  # # valid_target_location shape: (invalid_equas, 3)
                valid_uncern = preds['GRM_uncern'][GRM_valid_items]  # shape: (valid_equas,)
                invalid_uncern = preds['GRM_uncern'][~GRM_valid_items]  # shape: (invalid_equas,)~
                # gvis = ops.Tensor(sum(np.array(GRM_valid_items).astype(int)), ms.float32)
                # gvis_inverse = ops.Tensor(sum(np.array(GRM_valid_items).astype(int)), ms.float32)

                valid_GRM_loss = self.reg_loss_fnc(ops.reduce_sum((valid_GRM_A * valid_target_location),1),
                                                   valid_GRM_B.squeeze(1))  # valid_GRM_loss shape: (valid_equas, 1)
                valid_GRM_loss = valid_GRM_loss / valid_uncern + self.log(valid_uncern)
                valid_GRM_loss = valid_GRM_loss.sum() / ops.clamp(GRM_valid_items.sum(), 1)

                invalid_GRM_loss = self.reg_loss_fnc(ops.reduce_sum((invalid_GRM_A * invalid_target_location),1),
                                                     invalid_GRM_B.squeeze(1))  # invalid_GRM_loss shape: (invalid_equas, 1)
                invalid_GRM_loss = invalid_GRM_loss / invalid_uncern
                invalid_GRM_loss = invalid_GRM_loss.sum() / ops.clamp((~GRM_valid_items).sum(), 1)

                if self.modify_invalid_keypoint_depths:
                    GRM_loss = self.loss_weights['GRM_loss'] * (valid_GRM_loss + invalid_GRM_loss)
                else:
                    GRM_loss = self.loss_weights['GRM_loss'] * valid_GRM_loss

            if self.compute_SoftGRM_loss:
                GRM_valid_items = pred_targets['GRM_valid_items']  # GRM_valid_items shape: (val_objs, 20)
                # GRM_valid_items_inverse=(~GRM_valid_items).tolist()[0]
                # GRM_valid_items=GRM_valid_items.tolist()
                separate_depths = preds['separate_depths']  # separate_depths shape: (val_objs, 20)
                valid_target_depth = ops.expand_dims(pred_targets['depth_3D'],1).expand_as(separate_depths)[GRM_valid_items]  # shape: (valid_equas,)
                invalid_target_depth = ops.expand_dims(pred_targets['depth_3D'],1).expand_as(separate_depths)[~GRM_valid_items]  # shape: (invalid_equas,) problem~
                valid_separate_depths = separate_depths[GRM_valid_items]  # shape: (valid_equas,)
                invalid_separate_depths = ms.Tensor(separate_depths[~GRM_valid_items].asnumpy())  # shape: (invalid_equas,) ~
                valid_uncern = preds['GRM_uncern'][GRM_valid_items]  # shape: (valid_equas,)
                invalid_uncern = preds['GRM_uncern'][~GRM_valid_items]  # shape: (invalid_equas,)~
                SoftGRM_weight = self.SoftGRM_loss_weight
                SoftGRM_weight = ops.function.broadcast_to(ops.expand_dims(SoftGRM_weight,0),separate_depths.shape)
                valid_SoftGRM_weight = SoftGRM_weight[GRM_valid_items]
                invalid_SoftGRM_weight = SoftGRM_weight[~GRM_valid_items]

                valid_SoftGRM_loss = self.reg_loss_fnc(valid_separate_depths, valid_target_depth) / valid_uncern + self.log(valid_uncern)
                if self.cfg.SOLVER.DYNAMIC_WEIGHT:
                    equas_weight = ops.broadcast_to(ops.expand_dims(objs_weight,1),separate_depths.shape)
                    valid_equas_weight = equas_weight[GRM_valid_items]
                    valid_SoftGRM_loss = reweight_loss(valid_SoftGRM_loss, valid_equas_weight)
                # gvis=ops.Tensor(np.array([sum(np.array(GRM_valid_items))]).astype(int),ms.float32)
                # gvis_inverse = ops.Tensor(np.array([sum(np.array(GRM_valid_items_inverse))]).astype(int), ms.float32)+ms.Tensor(20.0,ms.float32)
                valid_SoftGRM_loss = self.div(ops.reduce_sum(valid_SoftGRM_loss * valid_SoftGRM_weight) , ops.clamp(GRM_valid_items.sum(), 1))
                invalid_SoftGRM_loss = self.div(self.reg_loss_fnc(ops.Tensor(invalid_separate_depths), invalid_target_depth) , invalid_uncern)
                invalid_SoftGRM_loss = self.div((invalid_SoftGRM_loss * invalid_SoftGRM_weight).sum() , ops.clamp(
                    (~GRM_valid_items).sum(),1))  #problem,all ~. Avoid the occasion that no invalid equations and the returned value is NaN.

                if self.modify_invalid_keypoint_depths:
                    SoftGRM_loss = (self.loss_weight_ramper(iteration) * self.loss_weights['SoftGRM_loss'] * (
                                valid_SoftGRM_loss + invalid_SoftGRM_loss))
                else:
                    SoftGRM_loss = self.loss_weight_ramper(iteration) * self.loss_weights[
                        'SoftGRM_loss'] * valid_SoftGRM_loss

        loss_dict = {
            'hm_loss': hm_loss,
            'bbox_loss': reg_2D_loss,
            'dims_loss': dims_3D_loss,
            'orien_loss': orien_3D_loss,
        }

        log_loss_dict = {
            '2D_IoU': iou_2D,
            '3D_IoU': pred_IoU_3D,
        }

        MAE_dict = {}

        if self.separate_trunc_offset:
            loss_dict['offset_loss'] = offset_3D_loss
            loss_dict['trunc_offset_loss'] = trunc_offset_loss
        else:
            loss_dict['offset_loss'] = offset_3D_loss

        if self.compute_corner_loss:
            loss_dict['corner_loss'] = corner_3D_loss

        if self.compute_direct_depth_loss:
            loss_dict['depth_loss'] = depth_3D_loss
            log_loss_dict['depth_loss'] = real_depth_3D_loss
            MAE_dict['depth_MAE'] = depth_MAE

        if self.compute_keypoint_corner:
            loss_dict['keypoint_loss'] = keypoint_loss

            MAE_dict.update({
                'center_MAE': center_MAE,
                '02_MAE': keypoint_02_MAE,
                '13_MAE': keypoint_13_MAE,
            })

            if self.corner_with_uncertainty:
                MAE_dict.update({
                    'lower_MAE': lower_MAE,
                    'hard_MAE': hard_MAE,
                    'soft_MAE': soft_MAE,
                    'mean_MAE': mean_MAE,
                })

        if self.compute_keypoint_depth_loss:
            loss_dict['keypoint_depth_loss'] = keypoint_depth_loss
            log_loss_dict['keypoint_depth_loss'] = log_valid_keypoint_depth_loss

        if self.compute_IOU3D_predict_loss:
            loss_dict['IOU3D_predict_loss'] = IOU3D_predict_loss
            log_loss_dict['IOU3D_predict_loss'] = IOU3D_predict_loss

        if self.compute_weighted_depth_loss:
            loss_dict['weighted_avg_depth_loss'] = soft_depth_loss

        if self.compute_combined_depth_loss:
            loss_dict['combined_depth_loss'] = combined_depth_loss
            log_loss_dict['combined_depth_loss'] = combined_depth_loss

        if self.compute_GRM_loss:
            loss_dict['GRM_loss'] = GRM_loss
            log_loss_dict['GRM_loss'] = GRM_loss

        if self.compute_SoftGRM_loss:
            loss_dict['SoftGRM_loss'] = SoftGRM_loss
            log_loss_dict['SoftGRM_loss'] = SoftGRM_loss

        # loss_dict ===> log_loss_dict
        for key, value in loss_dict.items():
            if key not in log_loss_dict:
                log_loss_dict[key] = value

        # stop when the loss has NaN or Inf
        # for v in loss_dict.values():
        #     if ops.isnan(v).sum() > 0:
        #         pdb.set_trace()
        #     if ops.isinf(v).sum() > 0:
        #         pdb.set_trace()

        log_loss_dict.update(MAE_dict)
        losses = sum(loss for loss in loss_dict.values()).mean()

        return losses


class Anno_Encoder(nn.Cell):
    def __init__(self, cfg):
        super(Anno_Encoder, self).__init__()
        device = cfg.MODEL.DEVICE
        self.INF = 100000000
        self.EPS = 1e-9

        # center related
        self.num_cls = len(cfg.DATASETS.DETECT_CLASSES)
        self.min_radius = cfg.DATASETS.MIN_RADIUS
        self.max_radius = cfg.DATASETS.MAX_RADIUS
        self.center_ratio = cfg.DATASETS.CENTER_RADIUS_RATIO
        self.target_center_mode = cfg.INPUT.HEATMAP_CENTER
        # if mode == 'max', centerness is the larger value, if mode == 'area', assigned to the smaller bbox
        self.center_mode = cfg.MODEL.HEAD.CENTER_MODE

        # depth related
        self.depth_mode = cfg.MODEL.HEAD.DEPTH_MODE
        self.depth_range = cfg.MODEL.HEAD.DEPTH_RANGE
        self.depth_ref = ops.Tensor(cfg.MODEL.HEAD.DEPTH_REFERENCE)

        # dimension related
        self.dim_mean = ops.Tensor(cfg.MODEL.HEAD.DIMENSION_MEAN)
        self.dim_std = ops.Tensor(cfg.MODEL.HEAD.DIMENSION_STD)
        self.dim_modes = cfg.MODEL.HEAD.DIMENSION_REG

        # orientation related
        self.alpha_centers = ops.Tensor([0, PI / 2, PI, - PI / 2])
        self.multibin = (cfg.INPUT.ORIENTATION == 'multi-bin')
        self.orien_bin_size = cfg.INPUT.ORIENTATION_BIN_SIZE

        # offset related
        self.offset_mean = cfg.MODEL.HEAD.REGRESSION_OFFSET_STAT[0]
        self.offset_std = cfg.MODEL.HEAD.REGRESSION_OFFSET_STAT[1]

        # output info
        self.down_ratio = cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_height = cfg.INPUT.HEIGHT_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_width = cfg.INPUT.WIDTH_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.K = self.output_width * self.output_height

        self.zeros=ops.Zeros()
        self.exp=ops.Exp()
        self.sigmoid=ops.Sigmoid()
        self.zeros=ops.Zeros()
        self.l2_norm=ops.L2Normalize()


    def encode_box3d(self, rotys, dims, locs):
        '''
        construct 3d bounding box for each object.
        Args:
                rotys: rotation in shape N
                dims: dimensions of objects
                locs: locations of objects

        Returns:

        '''
        if len(rotys.shape) == 2:
            rotys = rotys.flatten()
        if len(dims.shape) == 3:
            dims = dims.view(-1, 3)
        if len(locs.shape) == 3:
            locs = locs.view(-1, 3)

        # device = rotys.device
        N = rotys.shape[0]
        ry = self.rad_to_matrix(rotys, N)

        # l, h, w
        dims_corners = dims.view((-1, 1)).tile((1, 8))
        dims_corners = dims_corners * 0.5
        dims_corners[:, 4:] = -dims_corners[:, 4:]
        index = ops.Tensor([[4, 5, 0, 1, 6, 7, 2, 3],
                              [0, 1, 2, 3, 4, 5, 6, 7],
                              [4, 0, 1, 5, 6, 2, 3, 7]],ms.int32).tile((N, 1))

        box_3d_object = ops.gather_elements(dims_corners, 1, index)
        b=box_3d_object.view((N, 3, -1))
        box_3d = ops.matmul(ry, b)  #ry:[11,3,3]   box_3d_object:[11,3,8]
        box_3d += ops.expand_dims(locs,-1).tile((1, 1, 8))

        return ops.transpose(box_3d,(0, 2, 1))


    @staticmethod
    def rad_to_matrix(rotys, N):
        # device = rotys.device

        cos, sin = ops.cos(rotys), ops.sin(rotys)

        i_temp = ops.Tensor([[1, 0, 1],
                             [0, 1, 0],
                             [-1, 0, 1]], dtype=ms.float32)

        ry = i_temp.tile((N, 1)).view(N, -1, 3)

        ry[:, 0, 0] *= cos
        ry[:, 0, 2] *= sin
        ry[:, 2, 0] *= sin
        ry[:, 2, 2] *= cos

        return ry

    def decode_depth(self, depths_offset):
        if self.depth_mode == 'exp':
            depth = ops.exp(depths_offset)
        elif self.depth_mode == 'linear':
            depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]
        elif self.depth_mode == 'inv_sigmoid':
            depth = 1 / self.sigmoid(depths_offset) - 1
        else:
            raise ValueError

        if self.depth_range is not None:
            depth = ops.clamp(depth, self.depth_range[0], self.depth_range[1])

        return depth

    def decode_location_flatten(self, points, offsets, depths, calibs, pad_size, batch_idxs):
        batch_size = len(calibs)
        gts = ops.unique(batch_idxs)[0].asnumpy().tolist()
        locations = self.zeros((points.shape[0], 3), ms.float32)
        points = (points + offsets) * self.down_ratio - pad_size[batch_idxs]  # Left points: The 3D centers in original images.

        for idx, gt in enumerate(gts):
            corr_pts_idx = ops.nonzero(batch_idxs == gt).squeeze(-1)
            calib = calibs[gt]
            # concatenate uv with depth
            corr_pts_depth = ops.cat((points[corr_pts_idx], depths[corr_pts_idx, None]),axis=1)
            locations[corr_pts_idx] = project_image_to_rect(corr_pts_depth,calib)
        return locations

    def decode_depth_from_keypoints(self, pred_offsets, pred_keypoints, pred_dimensions, calibs, avg_center=False):
        # pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center
        assert len(calibs) == 1  # for inference, batch size is always 1

        calib = calibs[0]
        # we only need the values of y
        pred_height_3D = pred_dimensions[:, 1]
        pred_keypoints = pred_keypoints.view(-1, 10, 2)
        # center height -> depth
        if avg_center:
            updated_pred_keypoints = pred_keypoints - pred_offsets.view(-1, 1, 2)
            center_height = updated_pred_keypoints[:, -2:, 1]
            center_depth = calib.f_v * ops.expand_dims(pred_height_3D,-1) / (center_height.abs() * self.down_ratio * 2)
            center_depth = ops.ReduceMean(keep_dims=False)(center_depth, 1)
        else:
            center_height = pred_keypoints[:, -2, 1] - pred_keypoints[:, -1, 1]
            center_depth = calib.f_v * pred_height_3D / (center_height.abs() * self.down_ratio)

        # corner height -> depth
        corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1]
        corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1]
        corner_02_depth = calib.f_v * pred_height_3D.unsqueeze(-1) / (corner_02_height * self.down_ratio)
        corner_13_depth = calib.f_v * pred_height_3D.unsqueeze(-1) / (corner_13_height * self.down_ratio)
        corner_02_depth = ops.ReduceMean(keep_dims=False)(corner_02_depth, 1)
        corner_13_depth = ops.ReduceMean(keep_dims=False)(corner_13_depth, 1)
        # K x 3
        pred_depths = ops.Stack(axis=1)((center_depth, corner_02_depth, corner_13_depth))

        return pred_depths

    def decode_depth_from_keypoints_batch(self, pred_keypoints, pred_dimensions, calibs, batch_idxs=None):
        # pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center (bottom first)
        # pred_keypoints[k,10,2]
        # pred_dimensions[k,3]
        pred_height_3D = pred_dimensions[:, 1]  #[k,]
        batch_size = len(calibs)
        if batch_size == 1:
            batch_idxs = self.zeros(pred_dimensions.shape[0],ms.float32)

        center_height = pred_keypoints[:, -2, 1] - pred_keypoints[:, -1, 1]  #[2]

        corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1] #[2,2]
        corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1] #[2,2]

        pred_keypoint_depths = {'center': [], 'corner_02': [], 'corner_13': []}

        for idx, gt_idx in enumerate(ops.unique(ops.cast(batch_idxs,ms.int32))[0].asnumpy().tolist()):
            calib = calibs[idx]
            corr_pts_idx = ops.nonzero(batch_idxs == gt_idx).squeeze(-1)
            center_depth = calib['f_v'] * pred_height_3D[corr_pts_idx] / (
                        ops.relu(center_height[corr_pts_idx]) * self.down_ratio + self.EPS)
            corner_02_depth = calib['f_v'] * ops.expand_dims(pred_height_3D[corr_pts_idx], -1) / (
                    ops.relu(corner_02_height[corr_pts_idx]) * self.down_ratio + self.EPS)
            corner_13_depth = calib['f_v'] * ops.expand_dims(pred_height_3D[corr_pts_idx], -1) / (
                    ops.relu(corner_13_height[corr_pts_idx]) * self.down_ratio + self.EPS)

            corner_02_depth = corner_02_depth.mean(axis=1)
            corner_13_depth = corner_13_depth.mean(axis=1)

            pred_keypoint_depths['center'].append(center_depth)
            pred_keypoint_depths['corner_02'].append(corner_02_depth)
            pred_keypoint_depths['corner_13'].append(corner_13_depth)

        for key, depths in pred_keypoint_depths.items():
            pred_keypoint_depths[key] = ops.clamp(ops.cat(depths), self.depth_range[0], self.depth_range[1])
        pred_depths = ops.stack(([depth for depth in pred_keypoint_depths.values()]),axis=1)

        return pred_depths

    def decode_dimension(self, cls_id, dims_offset):
        '''
        retrieve object dimensions
        Args:
            cls_id: each object id
            dims_offset: dimension offsets, shape = (N, 3)

        Returns:

        '''
        cls_dimension_mean = self.dim_mean[cls_id, :]

        if self.dim_modes[0] == 'exp':
            dims_offset = self.exp(dims_offset)

        if self.dim_modes[2]:
            cls_dimension_std = self.dim_std[cls_id, :]
            dimensions = dims_offset * cls_dimension_std + cls_dimension_mean
        else:
            dimensions = dims_offset * cls_dimension_mean

        return dimensions

    def decode_axes_orientation(self, vector_ori, locations=None, dict_for_3d_center=None):
        '''
        Description:
            Compute global orientation (rotys) and relative angle (alphas). Relative angle is calculated based on $vector_ori.
            When $locations is provided, we use locations_x and locations_z to compute $rays ($rotys-$alphas). If $dict_for_3d_center
            is provided, rays is derived from $center_3D_x, $f_x and $c_u.
        Args:
            vector_ori: local orientation in [axis_cls, head_cls, sin, cos] format. shape: (valid_objs, 16)
            locations: object location. shape: None or (valid_objs, 3)
            dict_for_3d_center: A dictionary that contains information relative to $rays. If not None, its components are as follows:
                dict_for_3d_center['target_centers']: Target centers. shape: (valid_objs, 2)
                dict_for_3d_center['offset_3D']: The offset from target centers to 3D centers. shape: (valid_objs, 2)
                dict_for_3d_center['pad_size']: The pad size for the original image. shape: (B, 2)
                dict_for_3d_center['calib']: A list contains calibration objects. Its length is B.
                dict_for_3d_center['batch_idxs']: The bacth index of input batch. shape: None or (valid_objs,)

        Returns: for training we only need roty
                         for testing we need both alpha and roty

        '''
        if self.multibin:
            pred_bin_cls = vector_ori[:, : self.orien_bin_size * 2].view(-1, self.orien_bin_size, 2)
            pred_bin_cls = nn.Softmax(axis=1)(pred_bin_cls)[..., 1]
            orientations = ops.zeros((vector_ori.shape[0],), vector_ori.dtype)
            for i in range(self.orien_bin_size):
                mask_i = (pred_bin_cls.argmax(axis=1) == i)
                s = self.orien_bin_size * 2 + i * 2
                e = s + 2
                pred_bin_offset = vector_ori[mask_i, s: e]
                orientations[mask_i] = ops.atan2(pred_bin_offset[:, 0], pred_bin_offset[:, 1]) + self.alpha_centers[i]
        else:
            axis_cls = nn.Softmax(axis=1)(vector_ori[:, :2])
            axis_cls = axis_cls[:, 0] < axis_cls[:, 1]
            head_cls = nn.Softmax(axis=1)(vector_ori[:, 2:4])
            head_cls = head_cls[:, 0] < head_cls[:, 1]
            # cls axis
            orientations = self.alpha_centers[axis_cls + head_cls * 2]
            sin_cos_offset = self.l2_norm(vector_ori[:, 4:])
            orientations += ops.atan(sin_cos_offset[:, 0] / sin_cos_offset[:, 1])

        if locations is not None:  # Compute rays based on 3D locations.
            locations = locations.view(-1, 3)
            rays = ops.atan2(locations[:, 0], locations[:, 2])
        elif dict_for_3d_center is not None:  # Compute rays based on 3D centers projected on 2D plane.
            if len(dict_for_3d_center['calib']) == 1:  # Batch size is 1.
                batch_idxs = ops.zeros((vector_ori.shape[0],), ms.uint8)
            else:
                batch_idxs = dict_for_3d_center['batch_idxs']
            centers_3D = self.decode_3D_centers(dict_for_3d_center['target_centers'], dict_for_3d_center['offset_3D'],
                                                dict_for_3d_center['pad_size'], batch_idxs)
            centers_3D_x = centers_3D[:, 0]  # centers_3D_x shape: (total_num_objs,)

            c_u = [calib['c_u'] for calib in dict_for_3d_center['calib']]
            f_u = [calib['f_u'] for calib in dict_for_3d_center['calib']]
            b_x = [calib['b_x'] for calib in dict_for_3d_center['calib']]

            rays = ops.zeros_like(orientations)
            for idx, gt_idx in enumerate(ops.unique(batch_idxs)[0].asnumpy().tolist()):
                corr_idx = ops.cast(ops.nonzero(batch_idxs == gt_idx).squeeze(-1),ms.int32)
                rays[corr_idx] = ops.atan2(centers_3D_x[corr_idx] - c_u[idx],f_u[idx])  # This is exactly an approximation.
        else:
            raise Exception("locations and dict_for_3d_center should not be None simultaneously.")
        alphas = orientations
        rotys = alphas + rays

        larger_idx = (rotys > -PI).nonzero()
        small_idx = (rotys < -PI).nonzero()
        if len(larger_idx) != 0:
            rotys[larger_idx] -= 2 * PI
        if len(small_idx) != 0:
            rotys[[small_idx]] += 2 * PI

        larger_idx = (alphas > PI).nonzero()
        small_idx = (alphas < -PI).nonzero()
        if len(larger_idx) != 0:
            alphas[larger_idx] -= 2 * PI
        if len(small_idx) != 0:
            alphas[small_idx] += 2 * PI
        return rotys, alphas

    def decode_3D_centers(self, target_centers, offset_3D, pad_size, batch_idxs):
        '''
        Description:
            Decode the 2D points that 3D centers projected on the original image rather than the heatmap.
        Input:
            target_centers: The points that represent targets. shape: (total_num_objs, 2)
            offset_3D: The offset from target_centers to 3D centers. shape: (total_num_objs, 2)
            pad_size: The size of padding. shape: (B, 2)
            batch_idxs: The batch index of various objects. shape: (total_num_objs,)
        Output:
            target_centers: The 2D points that 3D centers projected on the 2D plane. shape: (total_num_objs, 2)
        '''
        centers_3D = ops.zeros_like(target_centers,dtype=ms.float32)
        for idx, gt_idx in enumerate(ops.unique(batch_idxs)[0].asnumpy().tolist()):
            corr_idx = ops.nonzero(batch_idxs == gt_idx).squeeze(-1)
            centers_3D[corr_idx, :] = (target_centers[corr_idx, :] + offset_3D[corr_idx, :]) * self.down_ratio - pad_size[idx]  # The points of 3D centers projected on 2D plane.
        return centers_3D

    def decode_2D_keypoints(self, target_centers, pred_keypoint_offset, pad_size, batch_idxs):
        '''
        Description:
            Calculate the positions of keypoints on original image.
        Args:
            target_centers: The position of target centers on heatmap. shape: (total_num_objs, 2)
            pred_keypoint_offset: The offset of keypoints related to target centers on the feature maps. shape: (total_num_objs, 2 * keypoint_num)
            pad_size: The padding size of the original image. shape: (B, 2)
            batch_idxs: The batch index of various objects. shape: (total_num_objs,)
        Returns:
            pred_keypoints: The decoded 2D keypoints. shape: (total_num_objs, 10, 2)
        '''
        total_num_objs, _ = pred_keypoint_offset.shape

        pred_keypoint_offset = pred_keypoint_offset.reshape((total_num_objs, -1, 2))  # It could be 8 or 10 keypoints.
        pred_keypoints = ops.expand_dims(target_centers,1) + pred_keypoint_offset  # pred_keypoints shape: (total_num_objs, keypoint_num, 2)

        for idx, gt_idx in enumerate(ops.unique(ops.cast(batch_idxs,ms.int32))[0].asnumpy().tolist()):
            corr_idx = ops.cast(ops.nonzero(batch_idxs == gt_idx).squeeze(-1),ms.int32)
            pred_keypoints[corr_idx, :] = pred_keypoints[corr_idx, :] * self.down_ratio - pad_size[idx]

        return pred_keypoints


    def decode_from_GRM(self, pred_rotys, pred_dimensions, pred_keypoint_offset, pred_direct_depths, targets_dict,
                        GRM_uncern=None, GRM_valid_items=None,
                        batch_idxs=None, cfg=None):
        '''
        Description:
            Compute the 3D locations based on geometric constraints.
        Input:
            pred_rotys: The predicted global orientation. shape: (total_num_objs, 1)
            pred_dimensions: The predicted dimensions. shape: (num_objs, 3)
            pred_keypoint_offset: The offset of keypoints related to target centers on the feature maps. shape: (total_num_obs, 20)
            pred_direct_depths: The directly estimated depth of targets. shape: (total_num_objs, 1)
            targets_dict: The dictionary that contains somre required information. It must contains the following 4 items:
                targets_dict['target_centers']: Target centers. shape: (valid_objs, 2)
                targets_dict['offset_3D']: The offset from target centers to 3D centers. shape: (valid_objs, 2)
                targets_dict['pad_size']: The pad size for the original image. shape: (B, 2)
                targets_dict['calib']: A list contains calibration objects. Its length is B.
            GRM_uncern: The estimated uncertainty of 25 equations in GRM. shape: None or (total_num_objs, 25).
            GRM_valid_items: The effectiveness of 25 equations. shape: None or (total_num_objs, 25)
            batch_idxs: The batch index of various objects. shape: None or (total_num_objs,)
            cfg: The config object. It could be None.
        Output:
            pinv: The decoded positions of targets. shape: (total_num_objs, 3)
            A: Matrix A of geometric constraints. shape: (total_num_objs, 25, 3)
            B: Matrix B of geometric constraints. shape: (total_num_objs, 25, 1)
        '''
        target_centers = targets_dict[
            'target_centers']  # The position of target centers on heatmap. shape: (total_num_objs, 2)
        offset_3D = targets_dict['offset_3D']  # shape: (total_num_objs, 2)
        calibs = [targets_dict['calib']]  # The list contains calibration objects. Its length is B.
        pad_size = targets_dict['pad_size']  # shape: (B, 2)

        if GRM_uncern is None:
            GRM_uncern = ops.ones((pred_rotys.shape[0], 25), ms.float32)

        if len(calibs) == 1:  # Batch size is 1.
            batch_idxs = ops.ones((pred_rotys.shape[0],), ms.uint8)

        if GRM_valid_items is None:  # All equations of GRM is valid.
            GRM_valid_items = ops.ones((pred_rotys.shape[0], 25), ms.bool_)

        # For debug
        '''pred_keypoint_offset = targets_dict['keypoint_offset'][:, :, 0:2].contiguous().view(-1, 20)
        pred_rotys = targets_dict['rotys'].view(-1, 1)
        pred_dimensions = targets_dict['dimensions']
        locations = targets_dict['locations']
        pred_direct_depths = locations[:, 2].view(-1, 1)'''

        pred_keypoints = self.decode_2D_keypoints(target_centers, pred_keypoint_offset,
                                                  targets_dict['pad_size'],
                                                  batch_idxs=batch_idxs)  # pred_keypoints: The 10 keypoint position original image. shape: (total_num_objs, 10, 2)

        c_u = ops.Tensor([calib['c_u'] for calib in calibs])  # c_u shape: (B,)
        c_v = ops.Tensor([calib['c_v'] for calib in calibs])
        f_u = ops.Tensor([calib['f_u'] for calib in calibs])
        f_v = ops.Tensor([calib['f_v'] for calib in calibs])
        b_x = ops.Tensor([calib['b_x'] for calib in calibs])
        b_y = ops.Tensor([calib['b_y'] for calib in calibs])

        n_pred_keypoints = pred_keypoints  # n_pred_keypoints shape: (total_num_objs, 10, 2)

        for idx, gt_idx in enumerate(ops.unique(ops.cast(batch_idxs,ms.int32))[0].asnumpy().tolist()):
            corr_idx = ops.nonzero(batch_idxs == gt_idx).squeeze(-1).astype(ms.float64)
            n_pred_keypoints[corr_idx, :, 0] = (n_pred_keypoints[corr_idx, :, 0] - c_u[idx]) / f_u[idx]
            n_pred_keypoints[corr_idx, :, 1] = (n_pred_keypoints[corr_idx, :, 1] - c_v[idx]) / f_v[idx]

        total_num_objs = n_pred_keypoints.shape[0]

        centers_3D = self.decode_3D_centers(target_centers, offset_3D, pad_size,
                                            batch_idxs)  # centers_3D: The positions of 3D centers of objects projected on original image.
        n_centers_3D = centers_3D
        for idx, gt_idx in enumerate(ops.Unique()(ops.cast(batch_idxs,ms.int32))[0].asnumpy().tolist()):
            corr_idx = ops.nonzero(batch_idxs == gt_idx).squeeze(-1).astype(ms.float64)
            n_centers_3D[corr_idx, 0] = (n_centers_3D[corr_idx, 0] - c_u[idx]) / f_u[
                idx]  # n_centers_3D shape: (total_num_objs, 2)
            n_centers_3D[corr_idx, 1] = (n_centers_3D[corr_idx, 1] - c_v[idx]) / f_v[idx]

        kp_group = ops.cat([(ops.Reshape(n_pred_keypoints,(total_num_objs, 20)), n_centers_3D,
                              ops.Zeros((total_num_objs, 2), ms.float32))],axis=1)  # kp_group shape: (total_num_objs, 24)
        coe = ops.zeros((total_num_objs, 24, 2), ms.float32)
        coe[:, 0:: 2, 0] = -1
        coe[:, 1:: 2, 1] = -1
        A = ops.cat((coe, ops.expand_dims(kp_group,2)),axis=2)
        coz = ops.zeros((total_num_objs, 1, 3), ms.float32)
        coz[:, :, 2] = 1
        A = ops.cat((A, coz),axis=1)  # A shape: (total_num_objs, 25, 3)

        pred_rotys = pred_rotys.view(total_num_objs, 1)
        cos = ops.cos(pred_rotys)  # cos shape: (total_num_objs, 1)
        sin = ops.sin(pred_rotys)  # sin shape: (total_num_objs, 1)

        pred_dimensions = pred_dimensions.view(total_num_objs, 3)
        l = pred_dimensions[:, 0: 1]  # h shape: (total_num_objs, 1). The same as w and l.
        h = pred_dimensions[:, 1: 2]
        w = pred_dimensions[:, 2: 3]

        B = ops.zeros((total_num_objs, 25, 1), ms.float32)
        B[:, 0, :] = l / 2 * cos + w / 2 * sin
        B[:, 2, :] = l / 2 * cos - w / 2 * sin
        B[:, 4, :] = -l / 2 * cos - w / 2 * sin
        B[:, 6, :] = -l / 2 * cos + w / 2 * sin
        B[:, 8, :] = l / 2 * cos + w / 2 * sin
        B[:, 10, :] = l / 2 * cos - w / 2 * sin
        B[:, 12, :] = -l / 2 * cos - w / 2 * sin
        B[:, 14, :] = -l / 2 * cos + w / 2 * sin
        B[:, 1: 8: 2, :] = ops.expand_dims((h / 2),1)
        B[:, 9: 16: 2, :] = -ops.expand_dims((h / 2),1)
        B[:, 17, :] = h / 2
        B[:, 19, :] = -h / 2

        total_num_objs = n_pred_keypoints.shape[0]
        pred_direct_depths = pred_direct_depths.view(total_num_objs, )
        for idx, gt_idx in enumerate(ops.unique(batch_idxs.astype(ms.int32))[0].asnumpy().tolist()):
            corr_idx = ops.nonzero(batch_idxs == gt_idx).squeeze(-1).astype(ms.float64)
            B[corr_idx, 22, 0] = -(centers_3D[corr_idx, 0] - c_u[idx]) * pred_direct_depths[corr_idx] / f_u[idx] - b_x[
                idx]
            B[corr_idx, 23, 0] = -(centers_3D[corr_idx, 1] - c_v[idx]) * pred_direct_depths[corr_idx] / f_v[idx] - b_y[
                idx]
            B[corr_idx, 24, 0] = pred_direct_depths[corr_idx]

        C = ops.zeros((total_num_objs, 25, 1), ms.float32)
        kps = n_pred_keypoints.view(total_num_objs, 20)  # kps_x shape: (total_num_objs, 20)
        C[:, 0, :] = kps[:, 0: 1] * (-l / 2 * sin + w / 2 * cos)
        C[:, 1, :] = kps[:, 1: 2] * (-l / 2 * sin + w / 2 * cos)
        C[:, 2, :] = kps[:, 2: 3] * (-l / 2 * sin - w / 2 * cos)
        C[:, 3, :] = kps[:, 3: 4] * (-l / 2 * sin - w / 2 * cos)
        C[:, 4, :] = kps[:, 4: 5] * (l / 2 * sin - w / 2 * cos)
        C[:, 5, :] = kps[:, 5: 6] * (l / 2 * sin - w / 2 * cos)
        C[:, 6, :] = kps[:, 6: 7] * (l / 2 * sin + w / 2 * cos)
        C[:, 7, :] = kps[:, 7: 8] * (l / 2 * sin + w / 2 * cos)
        C[:, 8, :] = kps[:, 8: 9] * (-l / 2 * sin + w / 2 * cos)
        C[:, 9, :] = kps[:, 9: 10] * (-l / 2 * sin + w / 2 * cos)
        C[:, 10, :] = kps[:, 10: 11] * (-l / 2 * sin - w / 2 * cos)
        C[:, 11, :] = kps[:, 11: 12] * (-l / 2 * sin - w / 2 * cos)
        C[:, 12, :] = kps[:, 12: 13] * (l / 2 * sin - w / 2 * cos)
        C[:, 13, :] = kps[:, 13: 14] * (l / 2 * sin - w / 2 * cos)
        C[:, 14, :] = kps[:, 14: 15] * (l / 2 * sin + w / 2 * cos)
        C[:, 15, :] = kps[:, 15: 16] * (l / 2 * sin + w / 2 * cos)

        B = B - C  # B shape: (total_num_objs, 25, 1)

        # A = A[:, 22:25, :]
        # B = B[:, 22:25, :]

        weights = 1 / GRM_uncern  # weights shape: (total_num_objs, 25)

        ##############  Block the invalid equations ##############
        # A = A * GRM_valid_items.unsqueeze(2)
        # B = B * GRM_valid_items.unsqueeze(2)

        ##############  Solve pinv for Coordinate loss ##############
        A_coor = A
        B_coor = B
        if cfg is not None and not cfg.MODEL.COOR_ATTRIBUTE:  # Do not use Coordinate loss to train attributes.
            A_coor = A_coor.asnumpy()
            B_coor = B_coor.asnumpy()

        weights_coor = weights
        if cfg is not None and not cfg.MODEL.COOR_UNCERN:  # Do not use Coordinate loss to train uncertainty.
            weights_coor = weights_coor.asnumpy()

        A_coor = A_coor * ops.expand_dims(weights_coor,2)
        B_coor = B_coor * ops.expand_dims(weights_coor,2)

        AT_coor = ops.transpose(A_coor,(0, 2, 1))  # A shape: (total_num_objs, 25, 3)
        pinv = ops.bmm(AT_coor, A_coor)
        pinv = ops.inverse(pinv)
        pinv = ops.bmm(pinv, AT_coor)
        pinv = ops.bmm(pinv, B_coor)

        ##############  Solve A_uncern and B_uncern for GRM loss ##############
        A_uncern = A
        B_uncern = B
        if cfg is not None and not cfg.MODEL.GRM_ATTRIBUTE:  # Do not use GRM loss to train attributes.
            A_uncern = A_uncern.asnumpy()   #.detach()
            B_uncern = B_uncern.asnumpy()   #.detach()

        weights_uncern = weights
        if cfg is not None and not cfg.MODEL.GRM_UNCERN:  # Do not use GRM loss to train uncertainty.
            weights_uncern = weights_uncern.asnumpy()   #.detach()

        A_uncern = A_uncern * ops.expand_dims(weights_uncern,2)
        B_uncern = B_uncern * ops.expand_dims(weights_uncern,2)

        return pinv.view(-1, 3), A_uncern, B_uncern


    def decode_from_SoftGRM(self, pred_rotys, pred_dimensions, pred_keypoint_offset, pred_combined_depth,
                            targets_dict, GRM_uncern=None, GRM_valid_items=None,
                            batch_idxs=None):
        '''
        Description:
            Decode depth from geometric constraints directly.
        Input:
            pred_rotys: The predicted global orientation. shape: (val_objs, 1)
            pred_dimensions: The predicted dimensions. shape: (val_objs, 3)
            pred_keypoint_offset: The offset of keypoints related to target centers on the feature maps. shape: (val_objs, 16)
            pred_combined_depth: The depth decoded from direct regression and keypoints. (val_objs, 4)
            targets_dict: The dictionary that contains somre required information. It must contains the following 4 items:
                targets_dict['target_centers']: Target centers. shape: (valid_objs, 2)
                targets_dict['offset_3D']: The offset from target centers to 3D centers. shape: (valid_objs, 2)
                targets_dict['pad_size']: The pad size for the original image. shape: (B, 2)
                targets_dict['calib']: A list contains calibration objects. Its length is B.
            GRM_uncern: The estimated uncertainty of 20 equations in GRM. shape: None or (val_objs, 20).
            GRM_valid_items: The effectiveness of 20 equations. shape: None or (val_objs, 20)
            batch_idxs: The batch index of various objects. shape: None or (val_objs,)
            weighted_sum: Whether to directly weighted add the depths decoded by 20 equations separately.
        Output:
            depth: The depth solved considering all geometric constraints. shape: (val_objs)
            separate_depths: The depths produced by 24 equations, respectively. shape: (val_objs, 24).
        '''
        val_objs_num, _ = pred_combined_depth.shape

        target_centers = targets_dict['target_centers']  # The position of target centers on heatmap. shape: (val_objs, 2)
        offset_3D = targets_dict['offset_3D']  # shape: (val_objs, 2)
        calibs = targets_dict['calib']  # The list contains calibration objects. Its length is B.
        pad_size = targets_dict['pad_size']  # shape: (B, 2)

        if len(calibs) == 1:  # Batch size is 1.
            batch_idxs = ops.Zeros()((val_objs_num,), ms.uint8)

        if GRM_uncern is not None:
            assert GRM_uncern.shape[1] == 20
        else:
            GRM_uncern = ops.ones((val_objs_num, 20), ms.float32)

        if GRM_valid_items is not None:
            assert GRM_valid_items.shape[1] == 20
        else:
            GRM_valid_items = ops.ones((val_objs_num, 20), ms.float32)

        assert pred_keypoint_offset.shape[1] == 16  # Do not use the bottom center and top center. Only 8 vertexes.

        pred_keypoints = self.decode_2D_keypoints(target_centers, pred_keypoint_offset, targets_dict['pad_size'],
                                                  batch_idxs=batch_idxs)  # pred_keypoints: The 10 keypoint position original image. shape: (total_num_objs, 8, 2)

        c_u = [calib['c_u'] for calib in calibs]  # c_u shape: (B,)
        c_v = [calib['c_v'] for calib in calibs]
        f_u = [calib['f_u'] for calib in calibs]
        f_v = [calib['f_v'] for calib in calibs]
        b_x = [calib['b_x'] for calib in calibs]
        b_y = [calib['b_y'] for calib in calibs]

        n_pred_keypoints = pred_keypoints  # n_pred_keypoints shape: (total_num_objs, 8, 2)

        for idx, gt_idx in enumerate(ops.unique(batch_idxs)[0].asnumpy().tolist()):
            corr_idx = ops.nonzero(batch_idxs == gt_idx).squeeze(-1)
            n_pred_keypoints[corr_idx, :, 0] = (n_pred_keypoints[corr_idx, :, 0] - c_u[idx]) / f_u[idx]  # The normalized keypoint coordinates on 2D plane. shape: (total_num_objs, 8, 2)
            n_pred_keypoints[corr_idx, :, 1] = (n_pred_keypoints[corr_idx, :, 1] - c_v[idx]) / f_v[idx]

        centers_3D = self.decode_3D_centers(target_centers, offset_3D, pad_size, batch_idxs)  # centers_3D: The positions of 3D centers of objects projected on original image.
        n_centers_3D = centers_3D
        for idx, gt_idx in enumerate(ops.unique(batch_idxs)[0].asnumpy().tolist()):
            corr_idx = ops.nonzero(batch_idxs == gt_idx).squeeze(-1).astype(ms.int32)
            n_centers_3D[corr_idx, 0] = (n_centers_3D[corr_idx, 0] - c_u[idx]) / f_u[idx]  # n_centers_3D: The normalized 3D centers on 2D plane. shape: (total_num_objs, 2)
            n_centers_3D[corr_idx, 1] = (n_centers_3D[corr_idx, 1] - c_v[idx]) / f_v[idx]

        A = ops.zeros((val_objs_num, 20, 1), ms.float32)
        B = ops.zeros((val_objs_num, 20, 1), ms.float32)
        C = ops.zeros((val_objs_num, 20, 1), ms.float32)

        A[:, 0:16, 0] = (n_pred_keypoints - ops.expand_dims(n_centers_3D, 1)).view(-1, 16)
        A[:, 16:20, 0] = 1

        cos = ops.cos(pred_rotys)  # cos shape: (val_objs, 1)
        sin = ops.sin(pred_rotys)  # sin shape: (val_objs, 1)

        l = pred_dimensions[:, 0: 1]  # h shape: (total_num_objs, 1). The same as w and l.
        h = pred_dimensions[:, 1: 2]
        w = pred_dimensions[:, 2: 3]

        B[:, 0, :] = l / 2 * cos + w / 2 * sin
        B[:, 2, :] = l / 2 * cos - w / 2 * sin
        B[:, 4, :] = -l / 2 * cos - w / 2 * sin
        B[:, 6, :] = -l / 2 * cos + w / 2 * sin
        B[:, 8, :] = l / 2 * cos + w / 2 * sin
        B[:, 10, :] = l / 2 * cos - w / 2 * sin
        B[:, 12, :] = -l / 2 * cos - w / 2 * sin
        B[:, 14, :] = -l / 2 * cos + w / 2 * sin
        B[:, 1: 8: 2, :] = ops.expand_dims((h / 2),1)
        B[:, 9: 16: 2, :] = ops.expand_dims(-(h / 2),1)
        B[:, 16:20, 0] = pred_combined_depth[:,:4]  # Direct first keypoint next

        kps = n_pred_keypoints.view(val_objs_num, 16)  # kps_x shape: (total_num_objs, 16)
        C[:, 0, :] = kps[:, 0: 1] * (-l / 2 * sin + w / 2 * cos)
        C[:, 1, :] = kps[:, 1: 2] * (-l / 2 * sin + w / 2 * cos)
        C[:, 2, :] = kps[:, 2: 3] * (-l / 2 * sin - w / 2 * cos)
        C[:, 3, :] = kps[:, 3: 4] * (-l / 2 * sin - w / 2 * cos)
        C[:, 4, :] = kps[:, 4: 5] * (l / 2 * sin - w / 2 * cos)
        C[:, 5, :] = kps[:, 5: 6] * (l / 2 * sin - w / 2 * cos)
        C[:, 6, :] = kps[:, 6: 7] * (l / 2 * sin + w / 2 * cos)
        C[:, 7, :] = kps[:, 7: 8] * (l / 2 * sin + w / 2 * cos)
        C[:, 8, :] = kps[:, 8: 9] * (-l / 2 * sin + w / 2 * cos)
        C[:, 9, :] = kps[:, 9: 10] * (-l / 2 * sin + w / 2 * cos)
        C[:, 10, :] = kps[:, 10: 11] * (-l / 2 * sin - w / 2 * cos)
        C[:, 11, :] = kps[:, 11: 12] * (-l / 2 * sin - w / 2 * cos)
        C[:, 12, :] = kps[:, 12: 13] * (l / 2 * sin - w / 2 * cos)
        C[:, 13, :] = kps[:, 13: 14] * (l / 2 * sin - w / 2 * cos)
        C[:, 14, :] = kps[:, 14: 15] * (l / 2 * sin + w / 2 * cos)
        C[:, 15, :] = kps[:, 15: 16] * (l / 2 * sin + w / 2 * cos)

        B = B - C  # B shape: (total_num_objs, 24, 1)

        weights = 1 / GRM_uncern  # weights shape: (val_objs, 20)

        separate_depths = (B / (A + self.EPS)).squeeze(2)  # separate_depths: The depths produced by 24 equations, respectively. shape: (val_objs, 20).
        separate_depths = ops.clamp(separate_depths, self.depth_range[0], self.depth_range[1])

        weights = (weights / ops.sum(weights, dim=1,keepdim=True))
        depth = ops.sum(weights * separate_depths, 1)

        return depth, separate_depths


def select_point_of_interest(batch, index, feature_maps):
    '''
    Select POI(point of interest) on feature map
    Args:
        batch: batch size
        index: in point format or index format
        feature_maps: regression feature map in [N, C, H, W]

    Returns:

    '''
    w = feature_maps.shape[3]
    if len(index.shape) == 3:
        index = index[:, :, 1] * w + index[:, :, 0]
    index = index.view(batch, -1)
    # [N, C, H, W] -----> [N, H, W, C]
    feature_maps = ops.transpose(feature_maps,(0, 2, 3, 1))
    channel = feature_maps.shape[-1]
    # [N, H, W, C] -----> [N, H*W, C]
    feature_maps = feature_maps.view(batch, -1, channel)
    # expand index in channels
    index = ops.tile(ops.expand_dims(index,-1),(1, 1, channel))   #[1,80,72]
    # select specific features bases on POIs
    feature_maps = feature_maps.gather_elements(1,index)  # Left feature_maps shape: (B, num_objs, C)

    return feature_maps
