import logging
import os
import shutil

import mindspore
import mindspore as ms
from mindspore import nn
from mindspore import ops
import mindspore.numpy as mnp
from .backbone import *
from .predictor import *
from .loss import *
from Monodde.model_utils.utils import *
from Monodde.model_utils.timer import Timer, get_time_str
from .evaluation import evaluate_python
from .evaluation import generate_kitti_3d_detection


class Mono_net(nn.Cell):
    '''
    Generalized structure for keypoint based object detector.
    main parts:
    - backbone:include dla network and dcn-v2network
        -dla:deep layer aggregation module
        -dcn-v2:deformable convolutional networks
    - heads:predictor module
    '''

    def __init__(self, cfg):
        super(Mono_net, self).__init__()

        # dla_dcn.NORM_TYPE = cfg.MODEL.BACKBONE.NORM_TYPE

        if cfg.MODEL.BACKBONE.CONV_BODY == 'dla34':
            self.backbone = build_backbone(cfg)
        elif cfg.MODEL.BACKBONE.CONV_BODY == 'dla34_noDCN':
            # self.backbone = dla_noDCN.DLA(cfg)
            self.backbone.out_channels = 64

        self.heads = bulid_head(cfg, self.backbone.out_channels)

        self.test = cfg.DATASETS.TEST_SPLIT == 'test'
        self.training=cfg.is_training

    def construct(self, images,targets,edge_infor,iteration):
        images=ops.transpose(images,(0,3,1,2))
        features = self.backbone(images)
        edge_count=edge_infor[0]
        edge_indices=edge_infor[-1]
        output = self.heads(features, edge_count,edge_indices, iteration=iteration)
        return output



class MonoddeWithLossCell(nn.Cell):
    '''MonoDDE loss'''
    def __init__(self,network,cfg):
        super(MonoddeWithLossCell, self).__init__()
        self.cfg=cfg
        self.per_batch=cfg.SOLVER.IMS_PER_BATCH
        self.mono_network=Mono_net(cfg)
        self.loss_block=Mono_loss(cfg)
        self.anno_encoder = Anno_Encoder(cfg)
        self.corner_loss_depth=cfg.MODEL.HEAD.CORNER_LOSS_DEPTH

        # flatten keys and channels
        self.keys = [key for key_group in cfg.MODEL.HEAD.REGRESSION_HEADS for key in key_group]
        self.channels = [channel for channel_groups in cfg.MODEL.HEAD.REGRESSION_CHANNELS for channel in channel_groups]

        # self.key2channel = Converter_key2channel(keys=cfg.MODEL.HEAD.REGRESSION_HEADS,
        #                                          channels=cfg.MODEL.HEAD.REGRESSION_CHANNELS)
        self.pred_direct_depth = 'depth' in self.keys
        self.pred_direct_depth = 'depth' in self.keys
        self.depth_with_uncertainty = 'depth_uncertainty' in self.keys
        self.compute_keypoint_corner = 'corner_offset' in self.keys
        self.corner_with_uncertainty = 'corner_uncertainty' in self.keys

        self.corner_offset_uncern = 'corner_offset_uncern' in self.keys
        self.dim_uncern = '3d_dim_uncern' in self.keys
        self.combined_depth_uncern = 'combined_depth_uncern' in self.keys
        self.corner_loss_uncern = 'corner_loss_uncern' in self.keys
        self.uncertainty_range = cfg.MODEL.HEAD.UNCERTAINTY_RANGE
        self.perdict_IOU3D = 'IOU3D_predict' in self.keys

        self.concat=ops.Concat(axis=1)
        self.ones=ops.Ones()
        self.relu=ops.ReLU()

    def key2channel(self, key):
        # find the corresponding index
        index = self.keys.index(key)

        s = sum(self.channels[:index])
        e = s + self.channels[index]

        return slice(s, e, 1)


    def prepare_targets(self,data):
        iamges=data[0]
        edge_infor=[data[-3],data[-2]]
        heatmaps=data[19]
        cls_ids=data[3]
        offset_3D=data[11]
        target_centers=data[4]
        bboxes=data[12]
        keypoints=data[5]
        keypoints_depth_mask=data[6]
        dimensions=data[7]
        locations=data[8]
        rotys=data[15]
        alphas=data[17]
        orientations=data[18]
        pad_size=data[13]
        calibs=[]
        for i in range(self.per_batch):
            calibs.append(dict(P=ops.cast(data[22][i,:,:],ms.float32),R0=ops.cast(data[23][i,:,:],ms.float32),C2V=ops.cast(data[24][i,:,:],ms.float32),c_u=ops.cast(data[25][i],ms.float32),c_v=ops.cast(data[26][i],ms.float32),f_u=ops.cast(data[27][i],ms.float32),
                        f_v=ops.cast(data[28][i],ms.float32),b_x=ops.cast(data[29][i],ms.float32),b_y=ops.cast(data[30][i],ms.float32)))
        reg_mask=ops.cast(data[9],ms.int32)
        reg_weight=data[10]
        ori_imgs=ops.cast(data[14],ms.int32)
        trunc_mask=ops.cast(data[16],ms.int32)
        GRM_keypoints_visible=data[21]
        return_dict = dict(cls_ids=cls_ids, target_centers=target_centers, bboxes=bboxes, keypoints=keypoints,
                           dimensions=dimensions,
                           locations=locations, rotys=rotys, alphas=alphas, calib=calibs, pad_size=pad_size,
                           reg_mask=reg_mask, reg_weight=reg_weight,
                           offset_3D=offset_3D, ori_imgs=ori_imgs, trunc_mask=trunc_mask, orientations=orientations,
                           keypoints_depth_mask=keypoints_depth_mask,
                           GRM_keypoints_visible=GRM_keypoints_visible
                           )


        return iamges,edge_infor, heatmaps, return_dict


    def prepare_predictions(self, targets_variables, predictions):
        target_corner_keypoints=ms.Tensor(0)
        pred_keypoints_3D=ms.Tensor(0)
        pred_direct_depths_3D=ms.Tensor(0)
        GRM_uncern=ms.Tensor(0)
        pred_keypoints_depths_3D=ms.Tensor(0)
        pred_corner_depth_3D=ms.Tensor(0)
        pred_regression = predictions['reg']
        batch, channel, feat_h, feat_w = pred_regression.shape

        # 1. get the representative points
        targets_bbox_points = targets_variables["target_centers"]  # representative points

        reg_mask_gt = targets_variables["reg_mask"] # reg_mask_gt shape: (B, num_objs)
        flatten_reg_mask_gt = ops.cast(reg_mask_gt.view(-1),ms.bool_)# flatten_reg_mask_gt shape: (B * num_objs)
        # the corresponding image_index for each object, used for finding pad_size, calib and so on

        batch_idxs = ops.arange(batch).view(-1,1).expand_as(reg_mask_gt).reshape(-1) # batch_idxs shape: (B * num_objs)
        batch_idxs=batch_idxs[flatten_reg_mask_gt]
        valid_targets_bbox_points = targets_bbox_points.view(-1, 2)[flatten_reg_mask_gt]# valid_targets_bbox_points shape: (valid_objs, 2)

        # fcos-style targets for 2D
        target_bboxes_2D = targets_variables['bboxes'].view(-1, 4)[flatten_reg_mask_gt]# target_bboxes_2D shape: (valid_objs, 4). 4 -> (x1, y1, x2, y2)
        target_bboxes_height = target_bboxes_2D[:, 3] - target_bboxes_2D[:,1]  # target_bboxes_height shape: (valid_objs,)
        target_bboxes_width = target_bboxes_2D[:, 2] - target_bboxes_2D[:,0]  # target_bboxes_width shape: (valid_objs,)

        target_regression_2D = self.concat(
            (valid_targets_bbox_points - target_bboxes_2D[:, :2], target_bboxes_2D[:, 2:] - valid_targets_bbox_points))  # offset to 2D bbox boundaries.
        mask_regression_2D = (target_bboxes_height > 0) & (target_bboxes_width > 0)
        target_regression_2D=target_regression_2D[mask_regression_2D] # target_regression_2D shape: (valid_objs, 4)

        # targets for 3D
        target_clses = targets_variables["cls_ids"].view(-1)[flatten_reg_mask_gt] # target_clses shape: (val_objs,)
        target_depths_3D = targets_variables['locations'][..., -1].view(-1)[flatten_reg_mask_gt]  # target_depths_3D shape: (val_objs,)
        target_rotys_3D = targets_variables['rotys'].view(-1)[flatten_reg_mask_gt] # target_rotys_3D shape: (val_objs,)
        target_alphas_3D = targets_variables['alphas'].view(-1)[flatten_reg_mask_gt] # target_alphas_3D shape: (val_objs,)
        target_offset_3D = targets_variables["offset_3D"].view(-1, 2)[flatten_reg_mask_gt] # The offset from target centers to projected 3D centers. target_offset_3D shape: (val_objs, 2)
        target_dimensions_3D = targets_variables['dimensions'].view(-1, 3)[flatten_reg_mask_gt] # target_dimensions_3D shape: (val_objs, 3)

        target_orientation_3D = targets_variables['orientations'].view(-1, targets_variables['orientations'].shape[-1])[flatten_reg_mask_gt]# target_orientation_3D shape: (num_objs, 8)
        target_locations_3D = self.anno_encoder.decode_location_flatten(valid_targets_bbox_points, target_offset_3D,
                                                                        target_depths_3D,
                                                                        targets_variables['calib'],
                                                                        targets_variables['pad_size'],
                                                                        batch_idxs)  # target_locations_3D shape: (valid_objs, 3)

        target_corners_3D = self.anno_encoder.encode_box3d(target_rotys_3D, target_dimensions_3D,target_locations_3D)  # target_corners_3D shape: (valid_objs, 8, 3)
        target_bboxes_3D = self.concat((target_locations_3D, target_dimensions_3D, target_rotys_3D[:, None]))  # target_bboxes_3D shape: (valid_objs, 7)

        target_trunc_mask = targets_variables['trunc_mask'].view(-1)[flatten_reg_mask_gt]  # target_trunc_mask shape(valid_objs,)
        obj_weights = targets_variables["reg_weight"].view(-1)[flatten_reg_mask_gt]  # obj_weights shape: (valid_objs,)

        keypoints_visible = targets_variables["GRM_keypoints_visible"].view(-1, targets_variables["GRM_keypoints_visible"].shape[-1])[flatten_reg_mask_gt]  # keypoints_visible shape: (valid_objs, 11)
        if self.corner_loss_depth == 'GRM':
            keypoints_visible = ops.tile(ops.expand_dims(keypoints_visible,2),(1, 1, 2)).reshape((keypoints_visible.shape[0],
                                                                                       -1))  # The effectness of first 22 GRM equations.
            GRM_valid_items = self.concat((keypoints_visible,
                                         self.ones((keypoints_visible.shape[0], 3), ms.bool_)))  # GRM_valid_items shape: (valid_objs, 25)
        elif self.corner_loss_depth == 'soft_GRM':
            keypoints_visible = ops.tile(ops.expand_dims(keypoints_visible[:, 0:8],2),(1, 1, 2)).reshape((keypoints_visible.shape[0], -1))  # The effectiveness of the first 16 equations. shape: (valid_objs, 16)
            direct_depth_visible = self.ones((keypoints_visible.shape[0], 1),ms.bool_)
            veritical_group_visible = targets_variables["keypoints_depth_mask"].view(-1, 3)[flatten_reg_mask_gt].astype(ms.bool_)  # veritical_group_visible shape: (valid_objs, 3)
            GRM_valid_items = self.concat((keypoints_visible, direct_depth_visible, veritical_group_visible))  # GRM_valid_items shape: (val_objs, 20)
        else:
            GRM_valid_items = None
        # 2. extract corresponding predictions
        pred_regression_pois_3D = select_point_of_interest(batch, targets_bbox_points, pred_regression).view(-1, channel)[flatten_reg_mask_gt] # pred_regression_pois_3D shape: (valid_objs, C)

        pred_regression_2D = self.relu(pred_regression_pois_3D[mask_regression_2D,self.key2channel('2d_dim')])  # pred_regression_2D shape: (valid_objs, 4)
        pred_offset_3D = pred_regression_pois_3D[:,self.key2channel('3d_offset')]  # pred_offset_3D shape: (valid_objs, 2)
        pred_dimensions_offsets_3D = pred_regression_pois_3D[:, self.key2channel('3d_dim')]  # pred_dimensions_offsets_3D shape: (valid_objs, 3)
        pred_orientation_3D = self.concat((pred_regression_pois_3D[:, self.key2channel('ori_cls')],
                                         pred_regression_pois_3D[:, self.key2channel('ori_offset')]))  # pred_orientation_3D shape: (valid_objs, 16)

        # decode the pred residual dimensions to real dimensions
        pred_dimensions_3D = self.anno_encoder.decode_dimension(target_clses, pred_dimensions_offsets_3D)

        # preparing outputs
        targets = {'reg_2D': target_regression_2D, 'offset_3D': target_offset_3D, 'depth_3D': target_depths_3D,
                   'orien_3D': target_orientation_3D,
                   'dims_3D': target_dimensions_3D, 'corners_3D': target_corners_3D, 'width_2D': target_bboxes_width,
                   'rotys_3D': target_rotys_3D,
                   'cat_3D': target_bboxes_3D, 'trunc_mask_3D': target_trunc_mask, 'height_2D': target_bboxes_height,
                   'GRM_valid_items': GRM_valid_items,
                   'locations': target_locations_3D
                   }

        preds = {'reg_2D': pred_regression_2D, 'offset_3D': pred_offset_3D, 'orien_3D': pred_orientation_3D,
                 'dims_3D': pred_dimensions_3D}

        reg_nums = {'reg_2D': mask_regression_2D.sum(), 'reg_3D': flatten_reg_mask_gt.sum(),
                    'reg_obj': flatten_reg_mask_gt.sum()}
        weights = {'object_weights': obj_weights}

        # predict the depth with direct regression
        if self.pred_direct_depth:
            pred_depths_offset_3D = pred_regression_pois_3D[:, self.key2channel('depth')].squeeze(-1)
            pred_direct_depths_3D = self.anno_encoder.decode_depth(pred_depths_offset_3D)
            preds['depth_3D'] = pred_direct_depths_3D  # pred_direct_depths_3D shape: (valid_objs,)

        # predict the uncertainty of depth regression
        if self.depth_with_uncertainty:
            preds['depth_uncertainty'] = pred_regression_pois_3D[:, self.key2channel('depth_uncertainty')].squeeze(-1)  # preds['depth_uncertainty'] shape: (val_objs,)

            if self.uncertainty_range is not None:
                preds['depth_uncertainty'] = ops.clamp(preds['depth_uncertainty'],self.uncertainty_range[0],
                                                         self.uncertainty_range[1])

        # else:
        # 	print('depth_uncertainty: {:.2f} +/- {:.2f}'.format(
        # 		preds['depth_uncertainty'].mean().item(), preds['depth_uncertainty'].std().item()))

        # predict the keypoints
        if self.compute_keypoint_corner:
            # targets for keypoints
            target_corner_keypoints = targets_variables["keypoints"].view(flatten_reg_mask_gt.shape[0], -1, 3)[flatten_reg_mask_gt]  # target_corner_keypoints shape: (val_objs, 10, 3)
            targets['keypoints'] = target_corner_keypoints[..., :2]  # targets['keypoints'] shape: (val_objs, 10, 2)
            targets['keypoints_mask'] = target_corner_keypoints[..., -1]  # targets['keypoints_mask'] shape: (val_objs, 10)
            reg_nums['keypoints'] = targets['keypoints_mask'].sum()
            # mask for whether depth should be computed from certain group of keypoints
            target_corner_depth_mask = targets_variables["keypoints_depth_mask"].view(-1, 3)[flatten_reg_mask_gt]
            targets['keypoints_depth_mask'] = ops.cast(target_corner_depth_mask,ms.int32)  # target_corner_depth_mask shape: (val_objs, 3)

            # predictions for keypoints
            pred_keypoints_3D = pred_regression_pois_3D[:, self.key2channel('corner_offset')]
            pred_keypoints_3D = pred_keypoints_3D.view((int(flatten_reg_mask_gt.sum()), -1, 2))
            preds['keypoints'] = pred_keypoints_3D  # pred_keypoints_3D shape: (val_objs, 10, 2)

            pred_keypoints_depths_3D = self.anno_encoder.decode_depth_from_keypoints_batch(pred_keypoints_3D,
                                                                                           pred_dimensions_3D,
                                                                                           targets_variables['calib'],
                                                                                           batch_idxs)
            preds['keypoints_depths'] = pred_keypoints_depths_3D  # pred_keypoints_depths_3D shape: (val_objs, 3)

        # Optimize keypoint offset with uncertainty.
        if self.corner_offset_uncern:
            corner_offset_uncern = pred_regression_pois_3D[:, self.key2channel('corner_offset_uncern')]
            preds['corner_offset_uncern'] = ops.exp(ops.clamp(corner_offset_uncern, self.uncertainty_range[0],
                                                        self.uncertainty_range[1]))

        # Optimize dimension with uncertainty.
        if self.dim_uncern:
            dim_uncern = pred_regression_pois_3D[:, self.key2channel('3d_dim_uncern')]
            preds['dim_uncern'] = ops.clip_by_value(dim_uncern, self.uncertainty_range[0],
                                              self.uncertainty_range[1]).exp()

        # Optimize combined_depth with uncertainty
        if self.combined_depth_uncern:
            combined_depth_uncern = pred_regression_pois_3D[:, self.key2channel('combined_depth_uncern')]
            preds['combined_depth_uncern'] = ops.clamp(combined_depth_uncern, self.uncertainty_range[0],
                                                         self.uncertainty_range[1]).exp()

        # Optimize corner coordinate loss with uncertainty
        if self.corner_loss_uncern:
            corner_loss_uncern = pred_regression_pois_3D[:, self.key2channel('corner_loss_uncern')]
            preds['corner_loss_uncern'] = ops.clamp(corner_loss_uncern, self.uncertainty_range[0],
                                                      self.uncertainty_range[1]).exp()

        # predict the uncertainties of the solved depths from groups of keypoints
        if self.corner_with_uncertainty:
            preds['corner_offset_uncertainty'] = pred_regression_pois_3D[:, self.key2channel(
                'corner_uncertainty')]  # preds['corner_offset_uncertainty'] shape: (val_objs, 3)

            if self.uncertainty_range is not None:
                preds['corner_offset_uncertainty'] = ops.clamp(preds['corner_offset_uncertainty'],
                                                                 self.uncertainty_range[0],
                                                                 self.uncertainty_range[1])

        if self.corner_loss_depth == 'GRM':
            GRM_uncern = pred_regression_pois_3D[:, self.key2channel('GRM_uncern')]  # GRM_uncern shape: (num_objs, 25)
            GRM_uncern = ops.clamp(GRM_uncern, self.uncertainty_range[0], self.uncertainty_range[1]).exp()
            # Decode rot_y
            # Verify the correctness of orientation decoding.
            '''gt_ori_code = ops.zeros_like(pred_orientation_3D).to(pred_orientation_3D.device)	# gt_ori_code shape: (num_objs, 16)
            gt_ori_code[:, 0:8:2] = 0.1
            gt_ori_code[:, 1:8:2] = target_orientation_3D[:, 0:4]
            gt_ori_code[:, 8::2] = ops.sin(target_orientation_3D[:, 4:8])
            gt_ori_code[:, 9::2] = ops.cos(target_orientation_3D[:, 4:8])
            pred_orientation_3D = gt_ori_code'''
            info_dict = {'target_centers': valid_targets_bbox_points.float(), 'offset_3D': target_offset_3D,
                         'pad_size': targets_variables['pad_size'],
                         'calib': targets_variables['calib'], 'batch_idxs': batch_idxs}
            GRM_rotys, _ = self.anno_encoder.decode_axes_orientation(pred_orientation_3D, dict_for_3d_center=info_dict)

            info_dict.update({'ori_imgs': targets_variables['ori_imgs'], 'keypoint_offset': target_corner_keypoints,
                              'locations': target_locations_3D,
                              'dimensions': target_dimensions_3D, 'rotys': target_rotys_3D})
            GRM_locations, GRM_A, GRM_B = self.anno_encoder.decode_from_GRM(ops.expand_dims(GRM_rotys,1), pred_dimensions_3D,
                                                                            pred_keypoints_3D.view(-1, 20),
                                                                            pred_direct_depths_3D.view(-1, 1),
                                                                            GRM_uncern=GRM_uncern,
                                                                            GRM_valid_items=GRM_valid_items,
                                                                            batch_idxs=batch_idxs, cfg=self.cfg,
                                                                            targets_dict=info_dict)
            pred_corner_depth_3D = GRM_locations[:, 2]

            preds.update(
                {'combined_depth': pred_corner_depth_3D, 'GRM_A': GRM_A, 'GRM_B': GRM_B, 'GRM_uncern': GRM_uncern})

        elif self.corner_loss_depth == 'soft_GRM':
            if 'GRM_uncern' in self.keys:
                GRM_uncern = pred_regression_pois_3D[:,self.key2channel('GRM_uncern')]  # GRM_uncern shape: (num_objs, 20)
            elif 'GRM1_uncern' in self.keys:
                uncern_GRM1 = pred_regression_pois_3D[:,self.key2channel('GRM1_uncern')]  # uncern_GRM1 shape: (num_objs, 8)
                uncern_GRM2 = pred_regression_pois_3D[:,self.key2channel('GRM2_uncern')]  # uncern_GRM1 shape: (num_objs, 8)
                uncern_Mono_Direct = pred_regression_pois_3D[:,self.key2channel('Mono_Direct_uncern')]  # uncern_Mono_Direct shape: (num_objs, 1)
                uncern_Mono_Keypoint = pred_regression_pois_3D[:, self.key2channel('Mono_Keypoint_uncern')]  # uncern_Mono_Keypoint shape: (num_objs, 3)
                GRM_uncern = ops.cat((ops.expand_dims(uncern_GRM1,2), ops.expand_dims(uncern_GRM2,2)),axis=2).view(-1, 16)
                GRM_uncern = ops.cat((GRM_uncern, uncern_Mono_Direct, uncern_Mono_Keypoint),axis=1)  # GRM_uncern shape: (num_objs, 20)
            GRM_uncern = ops.clamp(GRM_uncern, self.uncertainty_range[0], self.uncertainty_range[1]).exp()
            assert GRM_uncern.shape[1] == 20

            pred_combined_depths = ops.cat((ops.expand_dims(pred_direct_depths_3D,1), pred_keypoints_depths_3D),axis=1)  # pred_combined_depths shape: (valid_objs, 4)
            info_dict = {'target_centers': valid_targets_bbox_points, 'offset_3D': target_offset_3D,
                         'pad_size': targets_variables['pad_size'],
                         'calib': targets_variables['calib'], 'batch_idxs': batch_idxs}
            GRM_rotys, _ = self.anno_encoder.decode_axes_orientation(pred_orientation_3D, dict_for_3d_center=info_dict)

            pred_vertex_offset = pred_keypoints_3D[:, 0:8, :]  # Do not use the top center and bottom center.
            pred_corner_depth_3D, separate_depths = self.anno_encoder.decode_from_SoftGRM(ops.expand_dims(GRM_rotys,1),
                                                                                          pred_dimensions_3D,
                                                                                          pred_vertex_offset.reshape(-1,16),
                                                                                          pred_combined_depths,
                                                                                          targets_dict=info_dict,
                                                                                          GRM_uncern=GRM_uncern,
                                                                                          batch_idxs=batch_idxs)  # pred_corner_depth_3D shape: (val_objs,), separate_depths shape: (val_objs, 20)
            preds.update(
                {'combined_depth': pred_corner_depth_3D, 'separate_depths': separate_depths, 'GRM_uncern': GRM_uncern})

        elif self.corner_loss_depth == 'direct':
            pred_corner_depth_3D = pred_direct_depths_3D  # Only use estimated depth.

        elif self.corner_loss_depth == 'keypoint_mean':
            pred_corner_depth_3D = preds['keypoints_depths'].mean(axis=1)  # Only use depth solved by keypoints.

        else:
            assert self.corner_loss_depth in ['soft_combine', 'hard_combine']
            # make sure all depths and their uncertainties are predicted
            pred_combined_uncertainty = ops.Concat(axis=1)(
                ops.exp((ops.expand_dims(preds['depth_uncertainty'],-1), preds['corner_offset_uncertainty'])))  # pred_combined_uncertainty shape: (val_objs, 4)
            pred_combined_depths = ops.Concat(axis=1)((ops.expand_dims(pred_direct_depths_3D,-1), preds['keypoints_depths']))  # pred_combined_depths shape: (val_objs, 4)

            if self.corner_loss_depth == 'soft_combine':  # Weighted sum.
                pred_uncertainty_weights = 1 / pred_combined_uncertainty
                pred_uncertainty_weights = pred_uncertainty_weights / ops.ReduceSum(keep_dims=True)(pred_uncertainty_weights,axis=1)
                pred_corner_depth_3D = ops.ReduceSum()(pred_combined_depths * pred_uncertainty_weights, 1)
                preds['weighted_depths'] = pred_corner_depth_3D  # pred_corner_depth_3D shape: (val_objs,)

            elif self.corner_loss_depth == 'hard_combine':  # Directly use the depth with the smallest uncertainty.
                pred_corner_depth_3D = pred_combined_depths[
                    mnp.arange(pred_combined_depths.shape[0]), pred_combined_uncertainty.argmin(axis=1)]

        if self.perdict_IOU3D:
            preds['IOU3D_predict'] = pred_regression_pois_3D[:, self.key2channel('IOU3D_predict')]

        pred_locations_3D = self.anno_encoder.decode_location_flatten(valid_targets_bbox_points, pred_offset_3D,
                                                                      pred_corner_depth_3D,
                                                                      targets_variables['calib'],
                                                                      targets_variables['pad_size'],
                                                                      batch_idxs)  # pred_locations_3D shape: (val_objs, 3)
        # decode rotys and alphas
        pred_rotys_3D, _ = self.anno_encoder.decode_axes_orientation(pred_orientation_3D,
                                                                     pred_locations_3D)  # pred_rotys_3D shape: (val_objs,)
        # encode corners
        pred_corners_3D = self.anno_encoder.encode_box3d(pred_rotys_3D, pred_dimensions_3D,
                                                         pred_locations_3D)  # pred_corners_3D shape: (val_objs, 8, 3)
        pred_corners_3D=pred_corners_3D
        pred_dimensions_3D=pred_dimensions_3D
        pred_locations_3D=pred_locations_3D
        # concatenate all predictions
        pred_bboxes_3D = ops.cat((pred_locations_3D, pred_dimensions_3D, pred_rotys_3D[:, None]),axis=1)  # pred_bboxes_3D shape: (val_objs, 7)

        preds.update({'corners_3D': pred_corners_3D, 'rotys_3D': pred_rotys_3D, 'cat_3D': pred_bboxes_3D})

        return targets, preds, reg_nums, weights


    def construct(self,data,iteration):
        images,edge_infor,targets_heatmap, targets_variables=self.prepare_targets(data)
        predictions=self.mono_network(images, targets_variables, edge_infor, iteration)
        pred_targets, preds, reg_nums, weights = self.prepare_predictions(targets_variables, predictions)
        loss=self.loss_block(targets_heatmap, predictions, pred_targets, preds, reg_nums, weights, iteration)
        return loss


class EvalWrapper:
    def __init__(self, cfg, network, dataset):
        super(EvalWrapper, self).__init__()
        self.cfg=cfg
        self.network=network
        self.dataset=dataset[0]
        self.device_num=cfg.group_size
        self.anno_encoder = Anno_Encoder(cfg)
        self.key2channel = Converter_key2channel(keys=cfg.MODEL.HEAD.REGRESSION_HEADS,
                                            channels=cfg.MODEL.HEAD.REGRESSION_CHANNELS)
        self.det_threshold = cfg.TEST.DETECTIONS_THRESHOLD
        self.max_detection = cfg.TEST.DETECTIONS_PER_IMG
        self.eval_dis_iou = cfg.TEST.EVAL_DIS_IOUS
        self.eval_depth = cfg.TEST.EVAL_DEPTH

        self.survey_depth = cfg.TEST.SURVEY_DEPTH
        self.depth_statistics_path = os.path.join(self.cfg.OUTPUT_DIR, 'depth_statistics')
        if self.survey_depth:
            if os.path.exists(self.depth_statistics_path):
                shutil.rmtree(self.depth_statistics_path)
            os.makedirs(self.depth_statistics_path)

        self.output_width = cfg.INPUT.WIDTH_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_height = cfg.INPUT.HEIGHT_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_depth = cfg.MODEL.HEAD.OUTPUT_DEPTH
        self.pred_2d = cfg.TEST.PRED_2D

        self.pred_direct_depth = 'depth' in self.key2channel.keys
        self.depth_with_uncertainty = 'depth_uncertainty' in self.key2channel.keys
        self.regress_keypoints = 'corner_offset' in self.key2channel.keys
        self.keypoint_depth_with_uncertainty = 'corner_uncertainty' in self.key2channel.keys

        self.GRM_with_uncertainty = 'GRM_uncern' in self.key2channel.keys or 'GRM1_uncern' in self.key2channel.keys
        self.predict_IOU3D_as_conf = 'IOU3D_predict' in self.key2channel.keys

        # use uncertainty to guide the confidence
        self.uncertainty_as_conf = cfg.TEST.UNCERTAINTY_AS_CONFIDENCE
        self.uncertainty_range = cfg.MODEL.HEAD.UNCERTAINTY_RANGE

        self.variance_list = []
        # if cfg.eval_parallel:
        #     self.reduce = AllReduce()


    def prepare_targets(self, data):
        iamges = ms.Tensor(data['img'],ms.float32)
        size=ms.Tensor(data['size'][0])
        edge_infor = [ops.Tensor(data['input_edge_count'],ms.int32), ops.Tensor(data['input_edge_indices'],ms.int32)]
        cls_ids = ms.Tensor(data['cls_ids'],ms.float32)
        offset_3D = ms.Tensor(data['offset_3D'],ms.float32)
        target_centers = ms.Tensor(data['target_centers'],ms.float32)
        bboxes = ms.Tensor(data['bboxes'],ms.float32)
        keypoints = ms.Tensor(data['keypoints'],ms.float32)
        keypoints_depth_mask = ms.Tensor(data['keypoints_depth_mask'],ms.float32)
        dimensions = ms.Tensor(data['dimensions'],ms.float32)
        locations = ms.Tensor(data['locations'],ms.float32)
        rotys = ms.Tensor(data['rotys'],ms.float32)
        alphas = ms.Tensor(data['alphas'],ms.float32)
        orientations = ms.Tensor(data['orientations'],ms.float32)
        pad_size = ms.Tensor(data['pad_size'],ms.float32)
        calibs = []
        calibs.append(dict(P=ops.Tensor(data['P'][0, :, :], ms.float32), R0=ops.Tensor(data['R0'][0, :, 0], ms.float32),
                           C2V=ops.Tensor(data['C2V'][0, :, :], ms.float32), c_u=ops.Tensor(data['c_u'][0], ms.float32),
                           c_v=ops.Tensor(data['c_v'][0], ms.float32), f_u=ops.Tensor(data['f_u'][0], ms.float32),
                           f_v=ops.Tensor(data['f_v'][0], ms.float32), b_x=ops.Tensor(data['b_x'][0], ms.float32),
                           b_y=ops.Tensor(data['b_y'][0], ms.float32)))
        reg_mask = ops.Tensor(data['reg_mask'], ms.int32)
        reg_weight = ops.Tensor(data['reg_weight'],ms.float32)
        ori_imgs = ops.Tensor(data['ori_img'], ms.int32)
        trunc_mask = ops.Tensor(data['trunc_mask'], ms.int32)
        GRM_keypoints_visible = data['GRM_keypoints_visible']
        return_dict = dict(cls_ids=cls_ids, size=size, target_centers=target_centers, bboxes=bboxes, keypoints=keypoints,
                           dimensions=dimensions,
                           locations=locations, rotys=rotys, alphas=alphas, calib=calibs, pad_size=pad_size,
                           reg_mask=reg_mask, reg_weight=reg_weight,
                           offset_3D=offset_3D, ori_imgs=ori_imgs, trunc_mask=trunc_mask, orientations=orientations,
                           keypoints_depth_mask=keypoints_depth_mask,
                           GRM_keypoints_visible=GRM_keypoints_visible,img_ids=ops.Tensor(data['original_idx'],ms.int32)
                           )

        return iamges, edge_infor, return_dict



    def synchronize(self):
        sync = Tensor(np.array([1]).astype(np.int32))
        # sync = self.reduce(sync)    # For synchronization
        sync = sync.asnumpy()[0]
        if sync != self.device_num:
            raise ValueError(
                f"Sync value {sync} is not equal to number of device {self.device_num}. "
                f"There might be wrong with devices."
            )


    def inference(self, iteration, save_all_results = True, metrics=['R40'],dataset_name = 'kitti'):
        if cfg.OUTPUT_DIR:
            output_folder = os.path.join(cfg.OUTPUT_DIR, dataset_name, "inference_{}".format(iteration))
            os.makedirs(output_folder, exist_ok=True)
        num_devices = self.cfg.group_size
        logger = logging.getLogger("monoflex.inference")

        logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(self.dataset)))
        predict_folder = os.path.join(output_folder, 'data')

        if os.path.exists(predict_folder):
            shutil.rmtree(predict_folder)
        os.makedirs(predict_folder, exist_ok=True)

        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()

        dis_ious = defaultdict(list)
        depth_errors = defaultdict(list)
        for index, data in enumerate(self.dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
            output, eval_utils, visualize_preds,image_ids = self.inference_once(data, iteration)
            dis_iou = eval_utils['dis_ious']
            if dis_iou is not None:
                for key in dis_iou: dis_ious[key] += dis_iou[key].tolist()

            depth_error = eval_utils['depth_errors']
            if depth_error is not None:
                for key in depth_error: depth_errors[key] += depth_error[key].tolist()

            # if vis:
            #     show_image_with_boxes(vis_target.get_field('ori_img'), output, vis_target,
            #                           visualize_preds, vis_scores=eval_utils['vis_scores'],
            #                           vis_save_path=os.path.join(vis_folder, image_ids[0] + '.png'))

            # For the validation of the training phase, $save_all_results is True and all files should be saved.
            # During the validation of evaluation phase, $save_all_results is False and only the files containining detected tartgets are saved.
            if save_all_results or output.shape[0] != 0:
                predict_txt = image_ids.asnumpy()[0] + '.txt'
                predict_txt = os.path.join(predict_folder, predict_txt)
                generate_kitti_3d_detection(output, predict_txt, dataset_name)
            # disentangling IoU
        for key, value in dis_ious.items():
            mean_iou = sum(value) / len(value)
            dis_ious[key] = mean_iou

        for key, value in depth_errors.items():
            value = np.array(value)
            value[value > 1] = 1  # Limit the uncertainty below 1. Some estimated value could be really large.
            value = value.tolist()
            mean_error = sum(value) / len(value)
            depth_errors[key] = mean_error

        for key, value in dis_ious.items():
            logger.info("{}, MEAN IOU = {:.4f}".format(key, value))

        for key, value in depth_errors.items():
            logger.info("{}, MEAN ERROR/UNCERTAINTY = {:.4f}".format(key, value))

        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        logger.info(
            "Total run time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(self.dataset), num_devices
            )
        )

        if save_all_results is False:
            return None, None, None

        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info(
            "Model inference time: {} ({} s / img per device, on {} devices)".format(
                total_infer_time,
                inference_timer.total_time * num_devices / len(self.dataset),
                num_devices,
            )
        )
        # if not comm.is_main_process():
        #     return None, None, None

        logger.info('Finishing generating predictions, start evaluating ...')
        ret_dicts = []

        # ! We observe bugs in the evaluation code of MonoFlex. Thus, the results of this inference process should only be for inference.
        # for metric in metrics:
        #     result, ret_dict = evaluate_python(label_path=dataset.label_dir,
        #                                        result_path=predict_folder,
        #                                        label_split_file=dataset.imageset_txt,
        #                                        current_class=dataset.classes,
        #                                        metric=metric)
        #
        #     logger.info('metric = {}'.format(metric))
        #     logger.info('\n' + result)
        #
        #     ret_dicts.append(ret_dict)
        return ret_dicts, dis_ious
        # return ret_dicts, result, dis_ious

    def inference_once(self, data, iteration):
        images,edge_infor, targets_variables=self.prepare_targets(data)
        image_ids=targets_variables['img_ids']
        predictions = self.network(images, targets_variables, edge_infor, iteration)
        pred_heatmap, pred_regression = predictions['cls'], predictions['reg']
        batch = pred_heatmap.shape[0]

        calib, pad_size = targets_variables['calib'], targets_variables['pad_size']
        img_size = targets_variables['size']

        # evaluate the disentangling IoU for each components in (3D center offset, depth, dimension, orientation)
        dis_ious = self.evaluate_3D_detection(targets_variables, pred_regression) if self.eval_dis_iou else None

        # evaluate the accuracy of predicted depths
        depth_errors = self.evaluate_3D_depths(targets_variables, pred_regression) if self.eval_depth else None

        if self.survey_depth: self.survey_depth_statistics(targets_variables, pred_regression, image_ids)

        # max-pooling as nms for heat-map
        heatmap = nms_hm(pred_heatmap)
        visualize_preds = {'heat_map': pred_heatmap}

        # select top-k of the predicted heatmap
        scores, indexs, clses, ys, xs = select_topk(heatmap, K=self.max_detection)

        pred_bbox_points = ops.cat([xs.view(-1, 1), ys.view(-1, 1)], axis=1)
        pred_regression_pois = select_point_of_interest(batch, indexs, pred_regression).view(-1, pred_regression.shape[1])

        # For debug
        self.det_threshold = 0

        # thresholding with score
        scores = scores.view(-1)
        if self.cfg.TEST.DEBUG:
            valid_mask = scores >= 0
        else:
            valid_mask = scores >= self.det_threshold

        # No valid predictions and not the debug mode.
        if valid_mask.sum() == 0:
            result = scores.new_zeros((1, 14))
            visualize_preds['keypoints'] = scores.new_zeros((1, 20))
            visualize_preds['proj_center'] = scores.new_zeros((1, 2))
            eval_utils = {'dis_ious': dis_ious, 'depth_errors': depth_errors, 'vis_scores': scores.new_zeros((1)),
                          'uncertainty_conf': scores.new_zeros((1)), 'estimated_depth_error': scores.new_zeros((1))}

            return result, eval_utils, visualize_preds,image_ids

        scores = scores[valid_mask]

        clses = clses.view(-1)[valid_mask]
        pred_bbox_points = pred_bbox_points[valid_mask]
        pred_regression_pois = pred_regression_pois[valid_mask]

        pred_2d_reg = ops.relu(pred_regression_pois[:, self.key2channel('2d_dim')])
        pred_offset_3D = pred_regression_pois[:, self.key2channel('3d_offset')]
        pred_dimensions_offsets = pred_regression_pois[:, self.key2channel('3d_dim')]
        pred_orientation = ops.cat((pred_regression_pois[:, self.key2channel('ori_cls')],
                                      pred_regression_pois[:, self.key2channel('ori_offset')]), axis=1)
        visualize_preds['proj_center'] = pred_bbox_points + pred_offset_3D

        pred_box2d = self.anno_encoder.decode_box2d_fcos(pred_bbox_points, pred_2d_reg, pad_size, img_size)
        pred_dimensions = self.anno_encoder.decode_dimension(ops.cast(clses,ms.int32), pred_dimensions_offsets)

        if self.pred_direct_depth:
            pred_depths_offset = pred_regression_pois[:, self.key2channel('depth')].squeeze(-1)
            pred_direct_depths = self.anno_encoder.decode_depth(pred_depths_offset)

        if self.depth_with_uncertainty:
            pred_direct_uncertainty = pred_regression_pois[:, self.key2channel('depth_uncertainty')].exp()
            visualize_preds['depth_uncertainty'] = pred_regression[:, self.key2channel('depth_uncertainty'),
                                                   ...].squeeze(1)

        if self.regress_keypoints:
            pred_keypoint_offset = pred_regression_pois[:, self.key2channel('corner_offset')]
            pred_keypoint_offset = pred_keypoint_offset.view(-1, 10, 2)
            # solve depth from estimated key-points
            pred_keypoints_depths = self.anno_encoder.decode_depth_from_keypoints_batch(pred_keypoint_offset,
                                                                                        pred_dimensions, calib)
            visualize_preds['keypoints'] = pred_keypoint_offset

        if self.keypoint_depth_with_uncertainty:
            pred_keypoint_uncertainty = pred_regression_pois[:, self.key2channel('corner_uncertainty')].exp()

        estimated_depth_error = None
        # For debug
        # reg_mask_gt = targets_variables["reg_mask"]	# reg_mask_gt shape: (B, num_objs)
        # flatten_reg_mask_gt = ops.cast(reg_mask_gt.view(-1),ms.bool_)	# flatten_reg_mask_gt shape: (B * num_objs)
        # pred_bbox_points = targets_variables['target_centers'].view(-1, 2)[flatten_reg_mask_gt]	# target centers
        # pred_offset_3D = targets_variables['offset_3D'].view(-1, 2)[flatten_reg_mask_gt]	# Offset from target centers to 3D ceneters
        # pred_dimensions = targets_variables['dimensions'].view(-1, 3)[flatten_reg_mask_gt]	# dimensions
        # target_orientation_3D = targets_variables['orientations'].view(-1, 8)[flatten_reg_mask_gt]	# The label of orientation
        # pred_orientation = ops.zeros((target_orientation_3D.shape[0], 16), dtype = ms.float32)	# gt_ori_code shape: (num_objs, 16)
        # pred_orientation[:, 0:8:2] = 0.1
        # pred_orientation[:, 1:8:2] = target_orientation_3D[:, 0:4]
        # pred_orientation[:, 8::2] = ops.sin(target_orientation_3D[:, 4:8])
        # pred_orientation[:, 9::2] = ops.cos(target_orientation_3D[:, 4:8])	# Orientation
        # pred_keypoint_offset = targets_variables['keypoints'][0, :, :, 0:2]
        # pred_keypoint_offset = pred_keypoint_offset[flatten_reg_mask_gt]	# Offset from target centers to keypoints.
        # pred_direct_depths = targets_variables['locations'][0, :, -1][flatten_reg_mask_gt]	# Direct depth estimation.
        # pred_keypoints_depths = ops.expand_dims(pred_direct_depths,1).tile((1, 3))	# pred_keypoints_depths shape: (num_objs, 3)
        # clses = targets_variables['cls_ids'][0][flatten_reg_mask_gt]	# Category information
        # pred_box2d = targets_variables['bboxes'].view(-1, 4)[flatten_reg_mask_gt] # 2D bboxes
        # pred_box2d = pred_box2d * self.cfg.MODEL.BACKBONE.DOWN_RATIO - pad_size.tile((1, 2))
        # scores = ops.ones((pred_bbox_points.shape[0],), dtype = ms.float32)	# 2D confidence

        '''reg_mask_gt = targets_variables["reg_mask"]	# reg_mask_gt shape: (B, num_objs)
        flatten_reg_mask_gt = ops.cast(reg_mask_gt.view(-1),ms.bool_)	# flatten_reg_mask_gt shape: (B * num_objs)
        target_depths = targets_variables['locations'][0, :, -1][flatten_reg_mask_gt]'''

        if self.output_depth == 'GRM':
            if not self.GRM_with_uncertainty:
                raise Exception("When compute depth with GRM, the GRM head of network must be loaded.")
            GRM_uncern = pred_regression_pois[:, self.key2channel('GRM_uncern')]  # (valid_objs, 25)
            GRM_uncern = ops.clamp(GRM_uncern, min=self.uncertainty_range[0], max=self.uncertainty_range[1]).exp()

            # For debug !!!
            # GRM_uncern = 0.01 * ops.ones((pred_bbox_points.shape[0], 25), dtype = ms.float32)

            info_dict = {'target_centers': pred_bbox_points, 'offset_3D': pred_offset_3D, 'pad_size': pad_size,
                         'calib': calib, 'batch_idxs': None}
            GRM_rotys, GRM_alphas = self.anno_encoder.decode_axes_orientation(pred_orientation, dict_for_3d_center=info_dict)

            GRM_locations, _, _ = self.anno_encoder.decode_from_GRM(ops.expand_dims(GRM_rotys,1), pred_dimensions,
                                                                    pred_keypoint_offset.reshape(-1, 20),
                                                                    pred_direct_depths.unsqueeze(1), targets_dict=info_dict,
                                                                    GRM_uncern=GRM_uncern)  # pred_locations_3D shape: (valid_objs, 3)
            pred_depths = GRM_locations[:, 2]

            weights = 1 / GRM_uncern  # weights shape: (total_num_objs, 25)
            weights = weights / ops.sum(weights, dim=1, keepdim=True)
            estimated_depth_error = ops.sum(weights * GRM_uncern, dim=1)  # estimated_depth_error shape: (valid_objs,)

        elif self.output_depth == 'soft_GRM':
            if not self.GRM_with_uncertainty:
                raise Exception("When compute depth with GRM, the GRM head of network must be loaded.")
            if 'GRM_uncern' in self.key2channel.keys:
                GRM_uncern = pred_regression_pois[:, self.key2channel('GRM_uncern')]  # GRM_uncern shape: (valid_objs, 20)
            elif 'GRM1_uncern' in self.key2channel.keys:
                uncern_GRM1 = pred_regression_pois[:, self.key2channel('GRM1_uncern')]  # uncern_GRM1 shape: (valid_objs, 8)
                uncern_GRM2 = pred_regression_pois[:, self.key2channel('GRM2_uncern')]  # uncern_GRM1 shape: (valid_objs, 8)
                uncern_Mono_Direct = pred_regression_pois[:, self.key2channel( 'Mono_Direct_uncern')]  # uncern_Mono_Direct shape: (valid_objs, 1)
                uncern_Mono_Keypoint = pred_regression_pois[:, self.key2channel( 'Mono_Keypoint_uncern')]  # uncern_Mono_Keypoint shape: (valid_objs, 3)
                GRM_uncern = ops.cat((ops.expand_dims(uncern_GRM1,2), ops.expand_dims(uncern_GRM2,2)), axis=2).view(-1, 16)
                GRM_uncern = ops.cat((GRM_uncern, uncern_Mono_Direct, uncern_Mono_Keypoint), axis=1)  # GRM_uncern shape: (valid_objs, 20)
            GRM_uncern = ops.clamp(GRM_uncern, min=self.uncertainty_range[0], max=self.uncertainty_range[1]).exp()
            assert GRM_uncern.shape[1] == 20

            # For debug !!!
            # GRM_uncern = 0.01 * ops.ones((pred_bbox_points.shape[0], 20), dtype = ms.float32)

            pred_combined_depths = ops.cat((ops.expand_dims(pred_direct_depths,1), pred_keypoints_depths), axis=1)  # pred_combined_depths shape: (valid_objs, 4)

            info_dict = {'target_centers': pred_bbox_points, 'offset_3D': pred_offset_3D, 'pad_size': pad_size,
                         'calib': calib, 'batch_idxs': None}
            GRM_rotys, GRM_alphas = self.anno_encoder.decode_axes_orientation(pred_orientation, dict_for_3d_center=info_dict)

            # For debug !!!
            # GRM_rotys = targets_variables['rotys'].view(-1)[flatten_reg_mask_gt]

            pred_vertex_offset = pred_keypoint_offset[:, 0:8, :]  # Do not use the top center and bottom center.
            pred_depths, separate_depths = self.anno_encoder.decode_from_SoftGRM(ops.expand_dims(GRM_rotys,1),
                                                                                 pred_dimensions,
                                                                                 pred_vertex_offset.reshape(-1, 16),
                                                                                 pred_combined_depths,
                                                                                 targets_dict=info_dict,
                                                                                 GRM_uncern=GRM_uncern)  # pred_depths shape: (total_num_objs,)

            ### For the experiments of ablation study on depth estimation ###
            '''separate_depths = ops.cat((separate_depths[:, 0:16], separate_depths[:, 19:20]), axis = 1)
            GRM_uncern = ops.cat((GRM_uncern[:, 0:16], GRM_uncern[:, 19:20]), axis = 1)
            self.variance_list.append(ops.var(separate_depths).item())
            print('Mean variance:', sum(self.variance_list) / len(self.variance_list))'''

            if self.cfg.TEST.UNCERTAINTY_3D == "GRM_uncern":
                estimated_depth_error = error_from_uncertainty(GRM_uncern)  # estimated_depth_error shape: (valid_objs,)
            elif self.cfg.TEST.UNCERTAINTY_3D == "combined_depth_uncern":
                combined_depth_uncern = pred_regression_pois[:, self.key2channel('combined_depth_uncern')]
                combined_depth_uncern = ops.clamp(combined_depth_uncern, min=self.uncertainty_range[0], max=self.uncertainty_range[1]).exp()
                estimated_depth_error = error_from_uncertainty(combined_depth_uncern)
            elif self.cfg.TEST.UNCERTAINTY_3D == 'corner_loss_uncern':
                corner_loss_uncern = pred_regression_pois[:, self.key2channel('corner_loss_uncern')]
                corner_loss_uncern = ops.clamp(corner_loss_uncern, min=self.uncertainty_range[0], max=self.uncertainty_range[1]).exp()
                estimated_depth_error = error_from_uncertainty(corner_loss_uncern)
            elif self.cfg.TEST.UNCERTAINTY_3D == 'uncern_soft_avg':
                GRM_error = error_from_uncertainty(GRM_uncern)

                combined_depth_uncern = pred_regression_pois[:, self.key2channel('combined_depth_uncern')]
                combined_depth_uncern = ops.clamp(combined_depth_uncern, min=self.uncertainty_range[0], max=self.uncertainty_range[1]).exp()
                combined_depth_error = error_from_uncertainty(combined_depth_uncern)

                corner_loss_uncern = pred_regression_pois[:, self.key2channel('corner_loss_uncern')]
                corner_loss_uncern = ops.clamp(corner_loss_uncern, min=self.uncertainty_range[0], max=self.uncertainty_range[1]).exp()
                corner_loss_error = error_from_uncertainty(corner_loss_uncern)

                estimated_depth_error = ops.cat(
                    (GRM_error.unsqueeze(1), ops.expand_dims(combined_depth_error[:GRM_error.shape[0]],1), ops.expand_dims(corner_loss_error[:GRM_error.shape[0]],1)), axis=1)
                estimated_depth_error = error_from_uncertainty(estimated_depth_error)

            # Uncertainty guided pruning to filter the unreasonable estimation results.
            if self.cfg.TEST.UNCERTAINTY_GUIDED_PRUNING:
                pred_depths, _ = uncertainty_guided_prune(separate_depths, GRM_uncern, cfg=self.cfg,
                                                          depth_range=self.anno_encoder.depth_range,
                                                          initial_use_uncern=True)

        ### For obtain the Oracle results ###
        # pred_depths, estimated_depth_error = self.get_oracle_depths(pred_box2d, clses, separate_depths, GRM_uncern, targets[0])

        elif self.output_depth == 'direct':
            pred_depths = pred_direct_depths

            if self.depth_with_uncertainty: estimated_depth_error = pred_direct_uncertainty.squeeze(axis=1)

        elif self.output_depth.find('keypoints') >= 0:
            if self.output_depth == 'keypoints_avg':
                pred_depths = pred_keypoints_depths.mean(axis=1)
                if self.keypoint_depth_with_uncertainty: estimated_depth_error = pred_keypoint_uncertainty.mean(axis=1)

            elif self.output_depth == 'keypoints_center':
                pred_depths = pred_keypoints_depths[:, 0]
                if self.keypoint_depth_with_uncertainty: estimated_depth_error = pred_keypoint_uncertainty[:, 0]

            elif self.output_depth == 'keypoints_02':
                pred_depths = pred_keypoints_depths[:, 1]
                if self.keypoint_depth_with_uncertainty: estimated_depth_error = pred_keypoint_uncertainty[:, 1]

            elif self.output_depth == 'keypoints_13':
                pred_depths = pred_keypoints_depths[:, 2]
                if self.keypoint_depth_with_uncertainty: estimated_depth_error = pred_keypoint_uncertainty[:, 2]

            else:
                raise ValueError

        # hard ensemble, soft ensemble and simple average
        elif self.output_depth in ['hard', 'soft', 'mean', 'oracle']:
            if self.pred_direct_depth and self.depth_with_uncertainty:
                pred_combined_depths = ops.cat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths), axis=1)
                pred_combined_uncertainty = ops.cat((pred_direct_uncertainty, pred_keypoint_uncertainty), axis=1)
            else:
                pred_combined_depths = pred_keypoints_depths
                pred_combined_uncertainty = pred_keypoint_uncertainty

            depth_weights = 1 / pred_combined_uncertainty
            visualize_preds['min_uncertainty'] = depth_weights.argmax(axis=1)

            if self.output_depth == 'hard':
                pred_depths = pred_combined_depths[ops.arange(pred_combined_depths.shape[0]), depth_weights.argmax(axis=1)]

                # the uncertainty after combination
                estimated_depth_error = pred_combined_uncertainty.min(axis=1).value

            elif self.output_depth == 'soft':
                depth_weights = depth_weights / depth_weights.sum(axis=1, keepdims=True)
                pred_depths = ops.sum(pred_combined_depths * depth_weights, axis=1)

                # the uncertainty after combination
                estimated_depth_error = ops.sum(depth_weights * pred_combined_uncertainty, axis=1)

            elif self.output_depth == 'mean':
                pred_depths = pred_combined_depths.mean(axis=1)

                # the uncertainty after combination
                estimated_depth_error = pred_combined_uncertainty.mean(axis=1)

            # the best estimator is always selected
            elif self.output_depth == 'oracle':
                pred_depths, estimated_depth_error = self.get_oracle_depths(pred_box2d, clses, pred_combined_depths,
                                                                            pred_combined_uncertainty, data[0])

        batch_idxs = pred_depths.new_zeros(pred_depths.shape[0],dtype=ms.int32)
        pred_locations = self.anno_encoder.decode_location_flatten(pred_bbox_points, pred_offset_3D, pred_depths, calib,
                                                                   pad_size, batch_idxs)
        pred_center_locations = pred_locations
        pred_rotys, pred_alphas = self.anno_encoder.decode_axes_orientation(pred_orientation, locations=pred_locations)

        pred_locations[:, 1] += pred_dimensions[:, 1] / 2
        clses = clses.view(-1, 1)
        pred_alphas = pred_alphas.view(-1, 1)
        pred_rotys = pred_rotys.view(-1, 1)
        scores = scores.view(-1, 1)
        # change dimension back to h,w,l
        if cfg.MODEL.DEVICE=='CPU':
            pred_dimensions = ms.numpy.roll(pred_dimensions,shift=[-1], axis=[1])
        else:
            pred_dimensions=pred_dimensions.roll(shifts=-1, dims=1)

        # the uncertainty of depth estimation can reflect the confidence for 3D object detection
        vis_scores = scores

        if self.predict_IOU3D_as_conf:
            IOU3D_predict = pred_regression_pois[:, self.key2channel('IOU3D_predict')]
            scores = scores * ops.sigmoid(IOU3D_predict)
            uncertainty_conf, estimated_depth_error = None, None

        elif self.uncertainty_as_conf and estimated_depth_error is not None:
            '''[bias_thre = (pred_dimensions[:, 0] + pred_dimensions[:, 2]) / 2 * 0.3
            conf_list = []
            for i in range(bias_thre.shape[0]):
                conf = 2 * stats.norm.cdf(bias_thre[i].item(), 0, estimated_depth_error[i].item()) - 1
                conf_list.append(conf)
            uncertainty_conf = ms.Tensor(conf_list)
            uncertainty_conf = ops.clamp(uncertainty_conf, min=0.01, max=1)'''

            uncertainty_conf = 1 - ops.clamp(estimated_depth_error, min=0.01, max=1)
            scores = scores * uncertainty_conf.view(-1, 1)
        else:
            uncertainty_conf, estimated_depth_error = None, None

        # kitti output format
        result = ops.cat([clses, pred_alphas, pred_box2d, pred_dimensions, pred_locations, pred_rotys, scores], axis=1)

        if self.cfg.TEST.USE_NMS:
            bboxes = self.anno_encoder.encode_box3d(pred_rotys, pred_dimensions, pred_center_locations)
            result = nms_3d(result, bboxes, scores.squeeze(1), iou_threshold=self.cfg.TEST.NMS_THRESHOLD)

        eval_utils = {'dis_ious': dis_ious, 'depth_errors': depth_errors, 'uncertainty_conf': uncertainty_conf,
                      'estimated_depth_error': estimated_depth_error, 'vis_scores': vis_scores}

        # Filter 2D confidence * 3D confidence
        result_mask = result[:, -1] > self.cfg.TEST.DETECTIONS_3D_THRESHOLD
        result = result[result_mask, :]

        return result,eval_utils,visualize_preds, image_ids


    def evaluate_3D_detection(self, targets, pred_regression):
        # computing disentangling 3D IoU for offset, depth, dimension, orientation
        batch, channel = pred_regression.shape[:2]

        # 1. extract prediction in points of interest
        target_points = targets['target_centers'].float()
        pred_regression_pois = select_point_of_interest(  # pred_regression_pois shape: (B, num_objs, C)
            batch, target_points, pred_regression
        )

        # 2. get needed predictions
        pred_regression_pois = pred_regression_pois.view(-1, channel)
        reg_mask = targets['reg_mask'].view(-1).bool()
        pred_regression_pois = pred_regression_pois[reg_mask]
        target_points = target_points[0][reg_mask]

        pred_offset_3D = pred_regression_pois[:, self.key2channel('3d_offset')]
        pred_dimensions_offsets = pred_regression_pois[:, self.key2channel('3d_dim')]
        pred_orientation = ops.cat((pred_regression_pois[:, self.key2channel('ori_cls')],
                                      pred_regression_pois[:, self.key2channel('ori_offset')]), dim=1)
        pred_keypoint_offset = pred_regression_pois[:, self.key2channel('corner_offset')].view(-1, 10, 2)

        # 3. get ground-truth
        target_clses = targets['cls_ids'].view(-1)[reg_mask]
        target_offset_3D = targets['offset_3D'].view(-1, 2)[reg_mask]
        target_locations = targets['locations'].view(-1, 3)[reg_mask]
        target_dimensions = targets['dimensions'].view(-1, 3)[reg_mask]
        target_rotys = targets['rotys'].view(-1)[reg_mask]

        target_depths = target_locations[:, -1]

        # 4. decode prediction
        pred_dimensions = self.anno_encoder.decode_dimension(
            target_clses,
            pred_dimensions_offsets,
        )

        pred_depths_offset = pred_regression_pois[:, self.key2channel('depth')].squeeze(-1)

        if self.output_depth == 'GRM':
            if not self.GRM_with_uncertainty:
                raise Exception("When compute depth with GRM, the GRM head of network must be loaded.")
            GRM_uncern = pred_regression_pois[:, self.key2channel('GRM_uncern')].exp()  # (valid_objs, 25)
            info_dict = {'target_centers': target_points, 'offset_3D': pred_offset_3D, 'pad_size': targets["pad_size"],
                         'calib': targets['calib'], 'batch_idxs': None}
            pred_rotys, _ = self.anno_encoder.decode_axes_orientation(pred_orientation, dict_for_3d_center=info_dict)
            pred_direct_depths = self.anno_encoder.decode_depth(pred_depths_offset)
            pred_locations, _, _ = self.anno_encoder.decode_from_GRM(pred_rotys.unsqueeze(1), pred_dimensions,
                                                                     pred_keypoint_offset.reshape(-1, 20),
                                                                     pred_direct_depths.unsqueeze(1),
                                                                     targets_dict=info_dict,
                                                                     GRM_uncern=GRM_uncern)  # pred_locations_3D shape: (valid_objs, 3)
            pred_depths = pred_locations[:, 2]

        elif self.output_depth == 'direct':
            pred_depths = self.anno_encoder.decode_depth(pred_depths_offset)

        elif self.output_depth == 'keypoints':
            pred_depths = self.anno_encoder.decode_depth_from_keypoints(pred_offset_3D, pred_keypoint_offset,
                                                                        pred_dimensions, targets['calib'])
            pred_uncertainty = pred_regression_pois[:, self.key2channel('corner_uncertainty')].exp()
            pred_depths = pred_depths[ops.arange(pred_depths.shape[0]), pred_uncertainty.argmin(
                dim=1)]  # Use the depth estimation with the smallest uncertainty.

        elif self.output_depth == 'combine':
            pred_direct_depths = self.anno_encoder.decode_depth(pred_depths_offset)
            pred_keypoints_depths = self.anno_encoder.decode_depth_from_keypoints(pred_offset_3D, pred_keypoint_offset,
                                                                                  pred_dimensions, targets['calib'])
            pred_combined_depths = ops.cat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths), dim=1)

            pred_direct_uncertainty = pred_regression_pois[:, self.key2channel('depth_uncertainty')].exp()
            pred_keypoint_uncertainty = pred_regression_pois[:, self.key2channel('corner_uncertainty')].exp()
            pred_combined_uncertainty = ops.cat((pred_direct_uncertainty, pred_keypoint_uncertainty), dim=1)
            pred_depths = pred_combined_depths[
                ops.arange(pred_combined_depths.shape[0]), pred_combined_uncertainty.argmin(
                    dim=1)]  # # Use the depth estimation with the smallest uncertainty.

        elif self.output_depth == 'soft':
            pred_direct_depths = self.anno_encoder.decode_depth(pred_depths_offset)
            pred_keypoints_depths = self.anno_encoder.decode_depth_from_keypoints(pred_offset_3D, pred_keypoint_offset,
                                                                                  pred_dimensions, targets['calib'])
            pred_combined_depths = ops.cat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths), dim=1)

            pred_direct_uncertainty = pred_regression_pois[:, self.key2channel('depth_uncertainty')].exp()
            pred_keypoint_uncertainty = pred_regression_pois[:, self.key2channel('corner_uncertainty')].exp()
            pred_combined_uncertainty = ops.cat((pred_direct_uncertainty, pred_keypoint_uncertainty), dim=1)

            depth_weights = 1 / pred_combined_uncertainty
            depth_weights = depth_weights / depth_weights.sum(dim=1, keepdim=True)
            pred_depths = ops.sum(pred_combined_depths * depth_weights, dim=1)

        batch_idxs = pred_depths.new_zeros(pred_depths.shape[0]).long()
        # Decode 3D location from using ground truth target points, target center offset (estimation or label) and depth (estimation or label).
        pred_locations_offset = self.anno_encoder.decode_location_flatten(target_points, pred_offset_3D, target_depths,
                                                                          # pred_offset_3D is the target center offset.
                                                                          targets['calib'], targets["pad_size"],
                                                                          batch_idxs)

        pred_locations_depth = self.anno_encoder.decode_location_flatten(target_points, target_offset_3D, pred_depths,
                                                                         targets['calib'], targets["pad_size"],
                                                                         batch_idxs)

        pred_locations = self.anno_encoder.decode_location_flatten(target_points, pred_offset_3D, pred_depths,
                                                                   targets['calib'], targets["pad_size"], batch_idxs)

        pred_rotys, _ = self.anno_encoder.decode_axes_orientation(
            pred_orientation,
            target_locations,
        )

        fully_pred_rotys, _ = self.anno_encoder.decode_axes_orientation(pred_orientation, pred_locations)

        # fully predicted
        pred_bboxes_3d = ops.cat((pred_locations, pred_dimensions, fully_pred_rotys[:, None]), axis=1)
        # ground-truth
        target_bboxes_3d = ops.cat((target_locations, target_dimensions, target_rotys[:, None]), axis=1)
        # disentangling
        offset_bboxes_3d = ops.cat((pred_locations_offset, target_dimensions, target_rotys[:, None]),
                                     axis=1)  # The offset is target center offset.
        depth_bboxes_3d = ops.cat((pred_locations_depth, target_dimensions, target_rotys[:, None]), axis=1)
        dims_bboxes_3d = ops.cat((target_locations, pred_dimensions, target_rotys[:, None]), axis=1)
        orien_bboxes_3d = ops.cat((target_locations, target_dimensions, pred_rotys[:, None]), axis=1)

        # 6. compute 3D IoU
        pred_IoU = get_iou3d(pred_bboxes_3d, target_bboxes_3d)
        offset_IoU = get_iou3d(offset_bboxes_3d, target_bboxes_3d)
        depth_IoU = get_iou3d(depth_bboxes_3d, target_bboxes_3d)
        dims_IoU = get_iou3d(dims_bboxes_3d, target_bboxes_3d)
        orien_IoU = get_iou3d(orien_bboxes_3d, target_bboxes_3d)
        output = dict(pred_IoU=pred_IoU, offset_IoU=offset_IoU, depth_IoU=depth_IoU, dims_IoU=dims_IoU,
                      orien_IoU=orien_IoU)

        return output

    def evaluate_3D_depths(self, targets, pred_regression):
        # computing disentangling 3D IoU for offset, depth, dimension, orientation
        batch, channel = pred_regression.shape[:2]

        # 1. extract prediction in points of interest
        target_points = targets['target_centers']  # target_points shape: (num_objs, 2). (x, y)
        pred_regression_pois = select_point_of_interest(batch, target_points, pred_regression)

        pred_regression_pois = pred_regression_pois.view(-1, channel)
        reg_mask = ops.cast(targets['reg_mask'].view(-1),ms.bool_)
        pred_regression_pois = pred_regression_pois[reg_mask]
        target_points = target_points[0][reg_mask]
        target_offset_3D = targets['offset_3D'].view(-1, 2)[reg_mask, :]

        # depth predictions
        pred_depths_offset = pred_regression_pois[:, self.key2channel('depth')]  # pred_direct_depths shape: (num_objs,)
        pred_keypoint_offset = pred_regression_pois[:, self.key2channel('corner_offset')]  # pred_keypoint_offset: (num_objs, 20)

        # Orientatiion predictions
        pred_orientation = ops.cat((pred_regression_pois[:, self.key2channel('ori_cls')],
                                    pred_regression_pois[:, self.key2channel('ori_offset')]), axis=1)  # pred_orientation shape: (num_objs, 16)
        info_dict = {'target_centers': target_points, 'offset_3D': target_offset_3D, 'pad_size': targets['pad_size'],
                     'calib': targets['calib'], 'batch_idxs': None}
        pred_rotys, _ = self.anno_encoder.decode_axes_orientation(pred_orientation, dict_for_3d_center=info_dict)

        if self.depth_with_uncertainty:
            pred_direct_uncertainty = pred_regression_pois[:, self.key2channel('depth_uncertainty')].exp()
        if self.keypoint_depth_with_uncertainty:
            pred_keypoint_uncertainty = pred_regression_pois[:, self.key2channel('corner_uncertainty')].exp()
        if self.GRM_with_uncertainty:
            if 'GRM_uncern' in self.key2channel.keys:
                GRM_uncern = pred_regression_pois[:,
                             self.key2channel('GRM_uncern')]  # GRM_uncern shape: (valid_objs, 20)
            elif 'GRM1_uncern' in self.key2channel.keys:
                uncern_GRM1 = pred_regression_pois[:,
                              self.key2channel('GRM1_uncern')]  # uncern_GRM1 shape: (valid_objs, 8)
                uncern_GRM2 = pred_regression_pois[:,
                              self.key2channel('GRM2_uncern')]  # uncern_GRM1 shape: (valid_objs, 8)
                uncern_Mono_Direct = pred_regression_pois[:, self.key2channel('Mono_Direct_uncern')]  # uncern_Mono_Direct shape: (valid_objs, 1)
                uncern_Mono_Keypoint = pred_regression_pois[:, self.key2channel('Mono_Keypoint_uncern')]  # uncern_Mono_Keypoint shape: (valid_objs, 3)
                GRM_uncern = ops.cat((uncern_GRM1.unsqueeze(2), uncern_GRM2.unsqueeze(2)), axis=2).view(-1, 16)
                GRM_uncern = ops.cat((GRM_uncern, uncern_Mono_Direct, uncern_Mono_Keypoint),axis=1)  # GRM_uncern shape: (valid_objs, 20)
            GRM_uncern = GRM_uncern.exp()

        # dimension predictions
        target_clses = targets['cls_ids'].view(-1)[reg_mask]
        pred_dimensions_offsets = pred_regression_pois[:, self.key2channel('3d_dim')]
        pred_dimensions = self.anno_encoder.decode_dimension(target_clses,pred_dimensions_offsets,)  # pred_dimensions shape: (num_objs, 3)
        # direct
        pred_direct_depths = self.anno_encoder.decode_depth(pred_depths_offset.squeeze(-1))  # pred_direct_depths shape: (num_objs,)
        # three depths from keypoints
        pred_keypoints_depths = self.anno_encoder.decode_depth_from_keypoints_batch(
            pred_keypoint_offset.view(-1, 10, 2), pred_dimensions,targets['calib'])  # pred_keypoints_depths shape: (num_objs, 3)
        # The depth solved by original MonoFlex.
        pred_combined_depths = ops.cat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths),axis=1)  # pred_combined_depths shape: (num_objs, 4)

        MonoFlex_depth_flag = self.depth_with_uncertainty and self.keypoint_depth_with_uncertainty
        if MonoFlex_depth_flag:
            # combined uncertainty
            pred_combined_uncertainty = ops.cat((pred_direct_uncertainty, pred_keypoint_uncertainty), axis=1)
            # min-uncertainty
            pred_uncertainty_min_depth = pred_combined_depths[
                ops.arange(pred_combined_depths.shape[0]), pred_combined_uncertainty.argmin(axis=1)]  # Select the depth with the smallest uncertainty.
            # inv-uncertainty weighted
            pred_uncertainty_weights = 1 / pred_combined_uncertainty
            pred_uncertainty_weights = pred_uncertainty_weights / pred_uncertainty_weights.sum(axis=1, keepdims=True)

            pred_uncertainty_softmax_depth = ops.sum(pred_combined_depths * pred_uncertainty_weights, axis=1)  # Depth produced by soft weighting.

        # Decode depth based on geometric constraints.
        if self.output_depth == 'GRM':
            pred_location_GRM, _, _ = self.anno_encoder.decode_from_GRM(pred_rotys.unsqueeze(1),
                                                                        pred_dimensions,
                                                                        pred_keypoint_offset.reshape(-1, 20),
                                                                        ops.expand_dims(pred_direct_depths,1),
                                                                        targets_dict=info_dict,
                                                                        GRM_uncern=GRM_uncern)
            pred_depth_GRM = pred_location_GRM[:, 2]
        elif self.output_depth == 'soft_GRM':
            pred_keypoint_offset = pred_keypoint_offset[:, 0:16]
            SoftGRM_depth, separate_depths = self.anno_encoder.decode_from_SoftGRM(pred_rotys.unsqueeze(1),
                                                                                   # separate_depths shape: (val_objs, 20)
                                                                                   pred_dimensions,
                                                                                   pred_keypoint_offset.reshape(-1, 16),
                                                                                   pred_combined_depths,
                                                                                   targets_dict=info_dict,
                                                                                   GRM_uncern=GRM_uncern)

            # Uncertainty guided pruning to filter the unreasonable estimation results.
            if self.cfg.TEST.UNCERTAINTY_GUIDED_PRUNING:
                SoftGRM_depth, _ = uncertainty_guided_prune(separate_depths, GRM_uncern, self.cfg,
                                                            depth_range=self.anno_encoder.depth_range)

        # 3. get ground-truth
        target_locations = targets['locations'].view(-1, 3)[reg_mask]
        target_depths = target_locations[:, -1]

        Mono_pred_combined_error = (pred_combined_depths - target_depths[:, None]).abs()
        Mono_pred_direct_error = Mono_pred_combined_error[:, 0]
        Mono_pred_keypoints_error = Mono_pred_combined_error[:, 1:]
        pred_mean_depth = pred_combined_depths.mean(axis=1)
        pred_mean_error = (pred_mean_depth - target_depths).abs()
        # upper-bound
        pred_min_error = Mono_pred_combined_error.min(axis=1)[0]

        pred_errors = {
            'Mono direct error': Mono_pred_direct_error,
            'Mono keypoint_center error': Mono_pred_keypoints_error[:, 0],
            'Mono keypoint_02 error': Mono_pred_keypoints_error[:, 1],
            'Mono keypoint_13 error': Mono_pred_keypoints_error[:, 2],
            'Mono mean error': pred_mean_error,
            'Mono min error': pred_min_error,
        }

        # abs error
        if MonoFlex_depth_flag:
            pred_uncertainty_min_error = (pred_uncertainty_min_depth - target_depths).abs()
            pred_uncertainty_softmax_error = (pred_uncertainty_softmax_depth - target_depths).abs()

            # ops.clamp(estimated_depth_error, min=0.01, max=1)
            pred_errors.update({
                'Mono sigma_min error': pred_uncertainty_min_error,
                'Mono sigma_weighted error': pred_uncertainty_softmax_error,
                'target depth': target_depths,

                'Mono direct_sigma': pred_direct_uncertainty[:, 0],

                'Mono keypoint_center_sigma': pred_keypoint_uncertainty[:, 0],
                'Mono keypoint_02_sigma': pred_keypoint_uncertainty[:, 1],
                'Mono keypoint_13_sigma': pred_keypoint_uncertainty[:, 2]
            })

        if self.output_depth == 'GRM':
            pred_GRM_error = (pred_depth_GRM - target_depths).abs()
            pred_errors.update({
                'pred_GRM_error': pred_GRM_error
            })

        if self.output_depth == 'soft_GRM':
            separate_error = (
                        separate_depths - target_depths.unsqueeze(1)).abs()  # separate_error shape: (val_objs, 20)

            height_depths = ops.cat((separate_depths[:, 1:16:2], separate_depths[:, 16:20]), axis=1)
            height_uncern = ops.cat((GRM_uncern[:, 1:16:2], GRM_uncern[:, 16:20]), axis=1)
            height_w = 1 / height_uncern
            height_w = height_w / height_w.sum(axis=1, keepdims=True)
            height_depth = (height_depths * height_w).sum(axis=1)

            SoftGRM_uncertainty_min_error = separate_error[
                ops.arange(separate_error.shape[0]), GRM_uncern.argmin(axis=1)]
            for i in range(separate_depths.shape[1]):
                error_key = 'SoftGRM error' + str(i)
                uncern_key = 'SoftGRM uncern' + str(i)
                pred_errors.update({error_key: separate_error[:, i].abs(),
                                    uncern_key: GRM_uncern[:, i]})
            pred_errors.update({'SoftGRM weighted error': (SoftGRM_depth - target_depths).abs()})
            pred_errors.update({'SoftGRM sigma_min error': SoftGRM_uncertainty_min_error})
            pred_errors.update({'SoftGRM height solved error': (height_depth - target_depths).abs()})
            pred_errors.update({'SoftGRM min error': separate_error.min(axis=1)[0]})

        return pred_errors

    def survey_depth_statistics(self, targets, pred_regression, image_ids):
        ID_TYPE_CONVERSION = {
            0: 'Car',
            1: 'Pedestrian',
            2: 'Cyclist',
        }
        batch, channel = pred_regression.shape[:2]
        output_path = self.cfg.OUTPUT_DIR

        # 1. extract prediction in points of interest
        target_points = targets['target_centers']  # target_points shape: (B, objs_per_batch, 2). (x, y)
        pred_regression_pois = select_point_of_interest(batch, target_points, pred_regression)

        pred_regression_pois = pred_regression_pois.view(-1, channel)
        reg_mask = ops.cast(targets['reg_mask'].view(-1),ms.bool_)  # reg_mask_gt shape: (B * num_objs)
        pred_regression_pois = pred_regression_pois[reg_mask]
        target_points = target_points.view(-1, 2)[reg_mask]  # Left target_points shape: (num_objs, 2)

        pred_offset_3D = pred_regression_pois[:, self.key2channel('3d_offset')]  # pred_offset_3D shape: (num_objs, 2)
        pred_orientation = ops.cat((pred_regression_pois[:, self.key2channel('ori_cls')],
                                      pred_regression_pois[:, self.key2channel('ori_offset')]), axis=1)  # pred_orientation shape: (num_objs, 16)

        # depth predictions
        pred_depths_offset = pred_regression_pois[:, self.key2channel('depth')]  # pred_direct_depths shape: (num_objs,)
        pred_keypoint_offset = pred_regression_pois[:, self.key2channel('corner_offset')]  # pred_keypoint_offset: (num_objs, 20)

        if self.depth_with_uncertainty:
            pred_direct_uncertainty = pred_regression_pois[:, self.key2channel( 'depth_uncertainty')].exp()  # pred_direct_uncertainty shape: (num_objs, 1)
        else:
            pred_direct_uncertainty = None
        if self.keypoint_depth_with_uncertainty:
            pred_keypoint_uncertainty = pred_regression_pois[:, self.key2channel( 'corner_uncertainty')].exp()  # pred_keypoint_uncertainty shape: (num_objs, 3)
        else:
            pred_keypoint_uncertainty = None

        # dimension predictions
        target_clses = targets['cls_ids'].view(-1)[reg_mask]  # target_clses shape: (num_objs,)
        pred_dimensions_offsets = pred_regression_pois[:, self.key2channel('3d_dim')]
        pred_dimensions = self.anno_encoder.decode_dimension(target_clses, pred_dimensions_offsets,)  # pred_dimensions shape: (num_objs, 3)

        dimension_residual = (pred_dimensions - targets['dimensions'][0, reg_mask, :]) / targets['dimensions'][0, reg_mask, :]  # dimension_residual shape: (num_objs, 3). (l, h, w)
        target_keypoint_offset = targets['keypoints'][0, :, :, 0:2]
        target_keypoint_offset = target_keypoint_offset[reg_mask].view(-1,20)  # Offset from target centers to keypoints.
        target_vertex_offset = target_keypoint_offset[:, 0:16]

        # direct
        pred_direct_depths = ops.expand_dims(self.anno_encoder.decode_depth(pred_depths_offset.squeeze(-1)),1)  # pred_direct_depths shape: (num_objs, 1)
        # three depths from keypoints
        pred_keypoints_depths = self.anno_encoder.decode_depth_from_keypoints_batch(
            pred_keypoint_offset.view(-1, 10, 2), pred_dimensions, targets['calib'])  # pred_keypoints_depths shape: (num_objs, 3)

        target_locations = targets['locations'].view(-1, 3)[reg_mask, :]
        target_depth = target_locations[:, 2]  # target_depth shape: (num_objs)

        # Decode depth with GRM.
        pred_combined_depths = ops.cat((pred_direct_depths, pred_keypoints_depths), axis=1)  # pred_combined_depths shape: (valid_objs, 4)
        info_dict = {'target_centers': target_points, 'offset_3D': pred_offset_3D, 'pad_size': targets['pad_size'],
                     'calib': targets['calib'], 'batch_idxs': None}
        GRM_rotys, GRM_alphas = self.anno_encoder.decode_axes_orientation(pred_orientation, dict_for_3d_center=info_dict)

        # For debug
        '''info_dict['target_centers'] = targets['target_centers'][0, reg_mask, :].float()
        info_dict['offset_3D'] = targets['offset_3D'][0, reg_mask, :]
        GRM_rotys = targets['rotys'][0, reg_mask]
        pred_dimensions = targets['dimensions'][0, reg_mask, :]
        pred_keypoint_offset = targets['keypoints'][0, reg_mask, :, 0:2].view(-1, 20)'''

        pred_vertex_offset = pred_keypoint_offset[:, 0:16]  # Do not use the top center and bottom center.
        pred_depths, separate_depths = self.anno_encoder.decode_from_SoftGRM(GRM_rotys.unsqueeze(1),
                                                                             pred_dimensions, pred_vertex_offset,
                                                                             pred_combined_depths,
                                                                             targets_dict=info_dict)  # pred_depths shape: (total_num_objs,). separate_depths shape: (num_objs, 20)

        if 'GRM_uncern' in self.key2channel.keys:
            GRM_uncern = pred_regression_pois[:, self.key2channel('GRM_uncern')]  # GRM_uncern shape: (valid_objs, 20)
        elif 'GRM1_uncern' in self.key2channel.keys:
            uncern_GRM1 = pred_regression_pois[:, self.key2channel('GRM1_uncern')]  # uncern_GRM1 shape: (valid_objs, 8)
            uncern_GRM2 = pred_regression_pois[:, self.key2channel('GRM2_uncern')]  # uncern_GRM1 shape: (valid_objs, 8)
            uncern_Mono_Direct = pred_regression_pois[:, self.key2channel( 'Mono_Direct_uncern')]  # uncern_Mono_Direct shape: (valid_objs, 1)
            uncern_Mono_Keypoint = pred_regression_pois[:, self.key2channel( 'Mono_Keypoint_uncern')]  # uncern_Mono_Keypoint shape: (valid_objs, 3)
            GRM_uncern = ops.cat((uncern_GRM1.unsqueeze(2), uncern_GRM2.unsqueeze(2)), axis=2).view(-1, 16)
            GRM_uncern = ops.cat((GRM_uncern, uncern_Mono_Direct, uncern_Mono_Keypoint), axis=1)  # GRM_uncern shape: (valid_objs, 20)

        GRM_uncern = ops.clamp(GRM_uncern, min=self.uncertainty_range[0], max=self.uncertainty_range[1]).exp()

        f = open(os.path.join(self.depth_statistics_path, image_ids[0] + '.txt'), 'w')
        for i in range(target_points.shape[0]):
            Mono_depth = ops.cat((pred_direct_depths[i], pred_keypoints_depths[i]), axis=0)
            category = target_clses[i].item()
            category = ID_TYPE_CONVERSION[category]
            output_str = "Object: {}, Category: {}, Depth label: {}\nDirect depth: {}, Keypoint depth 1: {}, " \
                         "keypoint depth 2: {} keypoint depth 3: {}, Mono Depth mean: {}, Mono Depth std: {}\n".format(i,
                category,target_depth[i].item(),pred_direct_depths[i, 0].item(),pred_keypoints_depths[i, 0].item(),
                pred_keypoints_depths[i, 1].item(),pred_keypoints_depths[i, 2].item(),Mono_depth.mean().item(),
                Mono_depth.std().item())
            for j in range(separate_depths.shape[1]):
                if j < target_vertex_offset.shape[1]:
                    output_str += 'Keypoint prediction bias: {} '.format(
                        target_vertex_offset[i][j].item() - pred_vertex_offset[i][j].item())
                output_str += 'GRM_depth {}: {}, GRM_uncern {}: {}\n'.format(j, separate_depths[i][j].item(), j,
                                                                             GRM_uncern[i][j].item())
            output_str += 'GRM final depth: {}'.format(pred_depths[i].item())
            f.write(output_str)
            f.write('\n\n')
        f.close()
