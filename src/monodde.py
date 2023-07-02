import mindspore as ms
from mindspore import nn
from mindspore import ops
import mindspore.numpy as mnp
from .backbone import *
from .predictor import *
from .loss import *


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

        # if type(features) == list:
        #     features = features[0]

        if self.training:
            edge_count=edge_infor[0]
            edge_indices=edge_infor[-1]
            output = self.heads(features, edge_count,edge_indices, iteration=iteration, istraining=self.training)
            return output
        else:
            output = self.heads(features, targets, test=self.test)
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
        edge_infor=[data[-2],data[-1]]
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
            calibs.append(dict(P=data[22][i,:,:],R0=data[23][i,:,:],C2V=data[24][i,:,:],c_u=data[25][i],c_v=data[26][i],f_u=data[27][i],
                        f_v=data[28][i],b_x=data[29][i],b_y=data[30][i]))
        reg_mask=data[9]
        reg_weight=data[10]
        ori_imgs=data[14]
        trunc_mask=data[16]
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
        GRM_valid_items = None
        pred_regression = predictions['reg']
        batch, channel, feat_h, feat_w = pred_regression.shape

        # 1. get the representative points
        targets_bbox_points = targets_variables["target_centers"]  # representative points

        reg_mask_gt = targets_variables["reg_mask"] # reg_mask_gt shape: (B, num_objs)
        flatten_reg_mask_gt = ops.Tensor(reg_mask_gt.view(-1).astype(ms.bool_))# flatten_reg_mask_gt shape: (B * num_objs)
        # the corresponding image_index for each object, used for finding pad_size, calib and so on

        batch_idxs = ops.arange(batch).view(-1,1).expand_as(reg_mask_gt).reshape(-1) # batch_idxs shape: (B * num_objs)
        batch_idxs=ms.Tensor(batch_idxs[flatten_reg_mask_gt].asnumpy())
        valid_targets_bbox_points = ms.Tensor(targets_bbox_points.view(-1, 2)[flatten_reg_mask_gt].asnumpy())# valid_targets_bbox_points shape: (valid_objs, 2)

        # fcos-style targets for 2D
        target_bboxes_2D = ops.Tensor(targets_variables['bboxes'].view(-1, 4)[flatten_reg_mask_gt].asnumpy())# target_bboxes_2D shape: (valid_objs, 4). 4 -> (x1, y1, x2, y2)
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

        target_corners_3D = self.anno_encoder.encode_box3d(target_rotys_3D, target_dimensions_3D,
                                                           target_locations_3D)  # target_corners_3D shape: (valid_objs, 8, 3)
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

        # 2. extract corresponding predictions
        pred_regression_pois_3D = select_point_of_interest(batch, targets_bbox_points, pred_regression).view(-1, channel)[flatten_reg_mask_gt] # pred_regression_pois_3D shape: (valid_objs, C)

        pred_regression_2D = self.relu(pred_regression_pois_3D[mask_regression_2D,self.key2channel('2d_dim')])  # pred_regression_2D shape: (valid_objs, 4)
        pred_offset_3D = pred_regression_pois_3D[:,self.key2channel('3d_offset')]  # pred_offset_3D shape: (valid_objs, 2)
        pred_dimensions_offsets_3D = pred_regression_pois_3D[:, self.key2channel('3d_dim')]  # pred_dimensions_offsets_3D shape: (valid_objs, 3)
        pred_orientation_3D = self.concat((pred_regression_pois_3D[:, self.key2channel('ori_cls')],
                                         pred_regression_pois_3D[:, self.key2channel('ori_offset')]))  # pred_orientation_3D shape: (valid_objs, 16)

        # decode the pred residual dimensions to real dimensions
        pred_dimensions_3D = self.anno_encoder.decode_dimension(target_clses, pred_dimensions_offsets_3D.astype(ms.float32))

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
            preds['depth_uncertainty'] = pred_regression_pois_3D[:, self.key2channel('depth_uncertainty')].squeeze(
                -1)  # preds['depth_uncertainty'] shape: (val_objs,)

            if self.uncertainty_range is not None:
                preds['depth_uncertainty'] = ops.clip_by_value(preds['depth_uncertainty'],self.uncertainty_range[0],
                                                         self.uncertainty_range[1])

        # else:
        # 	print('depth_uncertainty: {:.2f} +/- {:.2f}'.format(
        # 		preds['depth_uncertainty'].mean().item(), preds['depth_uncertainty'].std().item()))

        # predict the keypoints
        if self.compute_keypoint_corner:
            # targets for keypoints
            target_corner_keypoints = targets_variables["keypoints"].view(flatten_reg_mask_gt.shape[0], -1, 3)[flatten_reg_mask_gt]  # target_corner_keypoints shape: (val_objs, 10, 3)
            targets['keypoints'] = target_corner_keypoints[:,:, :2]  # targets['keypoints'] shape: (val_objs, 10, 2)
            targets['keypoints_mask'] = target_corner_keypoints[
                ..., -1]  # targets['keypoints_mask'] shape: (val_objs, 10)
            reg_nums['keypoints'] = targets['keypoints_mask'].sum()
            # mask for whether depth should be computed from certain group of keypoints
            target_corner_depth_mask = targets_variables["keypoints_depth_mask"].view(-1, 3)[flatten_reg_mask_gt]
            targets['keypoints_depth_mask'] = target_corner_depth_mask.astype(ms.int32)  # target_corner_depth_mask shape: (val_objs, 3)

            # predictions for keypoints
            pred_keypoints_3D = pred_regression_pois_3D[:, self.key2channel('corner_offset')]

            pred_keypoints_3D = pred_keypoints_3D.view((int(flatten_reg_mask_gt.sum()), -1, 2))
            preds['keypoints'] = pred_keypoints_3D  # pred_keypoints_3D shape: (val_objs, 10, 2)

            pred_keypoints_depths_3D = ops.transpose(self.anno_encoder.decode_depth_from_keypoints_batch(pred_keypoints_3D,
                                                                                           pred_dimensions_3D,
                                                                                           targets_variables['calib'],
                                                                                           batch_idxs),(1,0))
            preds['keypoints_depths'] = pred_keypoints_depths_3D.astype(ms.int32)  # pred_keypoints_depths_3D shape: (val_objs, 3)

        # Optimize keypoint offset with uncertainty.
        if self.corner_offset_uncern:
            corner_offset_uncern = pred_regression_pois_3D[:, self.key2channel('corner_offset_uncern')]
            preds['corner_offset_uncern'] = ops.exp(ops.clip_by_value(corner_offset_uncern, self.uncertainty_range[0],
                                                        self.uncertainty_range[1]))

        # Optimize dimension with uncertainty.
        if self.dim_uncern:
            dim_uncern = pred_regression_pois_3D[:, self.key2channel('3d_dim_uncern')]
            preds['dim_uncern'] = ops.exp(ops.clip_by_value(dim_uncern, self.uncertainty_range[0],
                                              self.uncertainty_range[1]))

        # Optimize combined_depth with uncertainty
        if self.combined_depth_uncern:
            combined_depth_uncern = pred_regression_pois_3D[:, self.key2channel('combined_depth_uncern')]
            preds['combined_depth_uncern'] = ops.exp(ops.clip_by_value(combined_depth_uncern, self.uncertainty_range[0],
                                                         self.uncertainty_range[1]))

        # Optimize corner coordinate loss with uncertainty
        if self.corner_loss_uncern:
            corner_loss_uncern = pred_regression_pois_3D[:, self.key2channel('corner_loss_uncern')]
            preds['corner_loss_uncern'] = ops.exp(ops.clip_by_value(corner_loss_uncern, self.uncertainty_range[0],
                                                      self.uncertainty_range[1]))

        # predict the uncertainties of the solved depths from groups of keypoints
        if self.corner_with_uncertainty:
            preds['corner_offset_uncertainty'] = pred_regression_pois_3D[:, self.key2channel(
                'corner_uncertainty')]  # preds['corner_offset_uncertainty'] shape: (val_objs, 3)

            if self.uncertainty_range is not None:
                preds['corner_offset_uncertainty'] = ops.clip_by_value(preds['corner_offset_uncertainty'],
                                                                 self.uncertainty_range[0],
                                                                 self.uncertainty_range[1])

        if self.corner_loss_depth == 'GRM':
            GRM_uncern = pred_regression_pois_3D[:, self.key2channel('GRM_uncern')]  # GRM_uncern shape: (num_objs, 25)
            GRM_uncern = ops.clip_by_value(GRM_uncern, self.uncertainty_range[0], self.uncertainty_range[1]).exp()
            # Decode rot_y
            # Verify the correctness of orientation decoding.
            '''gt_ori_code = torch.zeros_like(pred_orientation_3D).to(pred_orientation_3D.device)	# gt_ori_code shape: (num_objs, 16)
            gt_ori_code[:, 0:8:2] = 0.1
            gt_ori_code[:, 1:8:2] = target_orientation_3D[:, 0:4]
            gt_ori_code[:, 8::2] = torch.sin(target_orientation_3D[:, 4:8])
            gt_ori_code[:, 9::2] = torch.cos(target_orientation_3D[:, 4:8])
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
                GRM_uncern = pred_regression_pois_3D[:,
                             self.key2channel('GRM_uncern')]  # GRM_uncern shape: (num_objs, 20)
            elif 'GRM1_uncern' in self.keys:
                uncern_GRM1 = pred_regression_pois_3D[:,
                              self.key2channel('GRM1_uncern')]  # uncern_GRM1 shape: (num_objs, 8)
                uncern_GRM2 = pred_regression_pois_3D[:,
                              self.key2channel('GRM2_uncern')]  # uncern_GRM1 shape: (num_objs, 8)
                uncern_Mono_Direct = pred_regression_pois_3D[:,
                                     self.key2channel('Mono_Direct_uncern')]  # uncern_Mono_Direct shape: (num_objs, 1)
                uncern_Mono_Keypoint = pred_regression_pois_3D[:, self.key2channel(
                    'Mono_Keypoint_uncern')]  # uncern_Mono_Keypoint shape: (num_objs, 3)
                GRM_uncern = ops.Concat(axis=2)((ops.expand_dims(uncern_GRM1,2), ops.expand_dims(uncern_GRM2,2))).view(
                    -1, 16)
                GRM_uncern = ops.Concat(axis=1)((GRM_uncern, uncern_Mono_Direct, uncern_Mono_Keypoint))  # GRM_uncern shape: (num_objs, 20)
            GRM_uncern = ops.exp(ops.clip_by_value(GRM_uncern, self.uncertainty_range[0], self.uncertainty_range[1]))
            assert GRM_uncern.shape[1] == 20

            pred_combined_depths = ops.Concat(axis=1)((ops.expand_dims(pred_direct_depths_3D,1), pred_keypoints_depths_3D))  # pred_combined_depths shape: (valid_objs, 4)
            info_dict = {'target_centers': valid_targets_bbox_points.astype(ms.float32), 'offset_3D': target_offset_3D,
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
        pred_corners_3D=pred_corners_3D.astype(ms.float64)
        pred_dimensions_3D=pred_dimensions_3D.astype(ms.float64)
        pred_locations_3D=pred_locations_3D.astype(ms.float64)
        # concatenate all predictions
        pred_bboxes_3D = ops.Concat(axis=1)((pred_locations_3D, pred_dimensions_3D, pred_rotys_3D[:, None]))  # pred_bboxes_3D shape: (val_objs, 7)

        preds.update({'corners_3D': pred_corners_3D, 'rotys_3D': pred_rotys_3D, 'cat_3D': pred_bboxes_3D})

        return targets, preds, reg_nums, weights


    def construct(self,data,iteration):
        images,edge_infor,targets_heatmap, targets_variables=self.prepare_targets(data)
        predictions=self.mono_network(images, targets_variables, edge_infor, iteration)
        pred_targets, preds, reg_nums, weights = self.prepare_predictions(targets_variables, predictions)
        loss=self.loss_block(targets_heatmap, predictions, pred_targets, preds, reg_nums, weights, iteration)
        return loss