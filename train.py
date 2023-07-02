import time

import mindspore as ms
import mindspore.nn as nn
import mindspore.communication as comm
from config import cfg
import datetime
import logging
import warnings
from model_utils.config import config,default_setup
# from utils.backup_files import sync_root
from src.kitti_dataset import create_kitti_dataset
from model_utils.utils import *
from src.monodde import *
from src.optimizer import *


ms.set_seed(1)
# numpy.random.seed(1)
# random.seed(1)


def init_distribute():
    if cfg.is_distributed:
        comm.init()
        config.rank = comm.get_rank()   #获取当前进程的排名
        config.group_size = comm.get_group_size()  #获取当前通信组大小
        config.local_rank=comm.get_local_rank()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                     device_num=cfg.group_size)   #配置自动并行计算
    else:
        cfg.MODEL.USE_SYNC_BN = False


def setup(args):
    '''load default config from config\defaults'''
    cfg.merge_from_file(args.config_path)
    # cfg.merge_from_list(args.opts)

    cfg.SEED = args.seed
    cfg.DATASETS.DATASET = args.dataset
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.DATALOADER.NUM_WORKERS = args.num_work
    cfg.TEST.EVAL_DIS_IOUS = args.eval_iou
    cfg.TEST.EVAL_DEPTH = args.eval_depth
    cfg.TEST.SURVEY_DEPTH = args.survey_depth

    cfg.MODEL.COOR_ATTRIBUTE = args.Coor_Attribute
    cfg.MODEL.COOR_UNCERN = args.Coor_Uncern
    cfg.MODEL.GRM_ATTRIBUTE = args.GRM_Attribute
    cfg.MODEL.GRM_UNCERN = args.GRM_Uncern
    cfg.MODEL.BACKBONE.CONV_BODY = args.backbone
    cfg.MODEL.PRETRAIN=args.pretrained

    if args.vis_thre > 0:
        cfg.TEST.VISUALIZE_THRESHOLD = args.vis_thre

    if args.output is not None:
        cfg.OUTPUT_DIR = args.output

    if args.test:
        cfg.DATASETS.TEST_SPLIT = 'test'
        cfg.DATASETS.TEST = ("kitti_test",)
    cfg.is_training=args.is_training

    if args.demo:
        cfg.DATASETS.TRAIN = ("kitti_demo",)
        cfg.DATASETS.TEST = ("kitti_demo",)

    if args.data_root is not None:
        cfg.DATASETS.DATA_ROOT = args.data_root

    if args.debug:
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.TEST.DEBUG = args.debug

    cfg.START_TIME = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d %H:%M:%S')
    default_setup(cfg, args)
    return cfg


def train_preprocess():
    cfg=setup(config)
    if cfg.MODEL.DEVICE=='Ascend':
        device_id = get_device_id()
        ms.set_context(mode=ms.GRAPH_MODE, device_target=cfg.MODEL.DEVICE, device_id=device_id)
    else:
        ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target=cfg.MODEL.DEVICE, device_id=0)
    device=ms.get_context("device_target")
    init_distribute()  # init distributed

# @moxing_wrapper(pre_process=modelarts_pre_process, post_process=modelarts_post_process, pre_args=[config])
def train():
    train_preprocess()
    dataset=create_kitti_dataset(cfg)
    data_loader = dataset.create_tuple_iterator(do_copy=False)
    meters = MetricLogger(delimiter=" ",)

    network = Mono_net(cfg)
    val_network = Mono_net(cfg)
    network = MonoddeWithLossCell(network,cfg)
    opt=get_optim(cfg,network)
    network = nn.TrainOneStepCell(network, opt)

    network.set_train()
    logger = logging.getLogger("monoflex.trainer")
    logger.info("Start training")
    max_iter = cfg.SOLVER.MAX_ITERATION
    start_training_time = time.time()
    end = time.time()
    ckpt_queue = deque()

    default_depth_method = cfg.MODEL.HEAD.OUTPUT_DEPTH
    if cfg.local_rank == 0:
        best_mAP = 0
        best_result_str = None
        best_iteration = 0
        eval_iteration = 0
        record_metrics = ['Car_bev_', 'Car_3d_']

    iter_per_epoch=cfg.SOLVER.IMS_PER_BATCH

    for iteration in range(0, max_iter):
        for i, data in enumerate(data_loader):
            data_time = time.time() - end
            loss=network(data,iteration)
            meters.update(loss=loss.asnumpy())
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            print(loss)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            if iteration % 10 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.8f} \n",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=cfg.SOLVER.BASE_LR,
                    )
                )

            if cfg.rank == 0 and (iteration % cfg.SOLVER.SAVE_CHECKPOINT_INTERVAL == 0):
                logger.info('iteration = {}, saving checkpoint ...'.format(iteration))
                ckpt_name = os.path.join(cfg.OUTPUT_DIR, "MonoDDE_{}_{}.ckpt".format(iteration + 1, cfg.SOLVER.IMS_PER_BATCH))
                ms.save_checkpoint(network, ckpt_name)
                if len(ckpt_queue) == cfg.SOLVER.SAVE_CHECKPOINT_MAX_NUM:
                    ckpt_to_remove = ckpt_queue.popleft()
                    os.remove(ckpt_to_remove)
                ckpt_queue.append(ckpt_name)
            if iteration == max_iter and cfg.rank == 0:
                ckpt_name = os.path.join(cfg.OUTPUT_DIR,
                                         "MonoDDE_{}_{}.ckpt".format(iteration + 1, iter_per_epoch))
                ms.save_checkpoint(network, ckpt_name)

            if iteration % cfg.SOLVER.EVAL_INTERVAL == 0:
                if cfg.SOLVER.EVAL_AND_SAVE_EPOCH:
                    cur_epoch = iteration // iter_per_epoch
                    logger.info('epoch = {}, evaluate model on validation set with depth {}'.format(cur_epoch,
                                                                                                    default_depth_method))
                else:
                    logger.info('iteration = {}, evaluate model on validation set with depth {}'.format(iteration,
                                                                                                        default_depth_method))

                # result_dict, result_str, dis_ious = do_eval(cfg, model, data_loaders_val, iteration)

                # if comm.get_local_rank() == 0:
                #     # only record more accurate R40 results
                #     result_dict = result_dict[0]
                #
                #     # record the best model according to the AP_3D, Car, Moderate, IoU=0.7
                #     important_key = '{}_3d_{:.2f}/moderate'.format('Car', 0.7)
                #     eval_mAP = float(result_dict[important_key])
                #     if eval_mAP >= best_mAP:
                #         # save best mAP and corresponding iterations
                #         best_mAP = eval_mAP
                #         best_iteration = iteration
                #         best_result_str = result_str
                #         ckpt_name = os.path.join(cfg.OUTPUT_DIR,
                #                                  "model_moderate_best_{}.ckpt".format(default_depth_method))
                #         ms.save_checkpoint(network, ckpt_name)
                #
                #         if cfg.SOLVER.EVAL_AND_SAVE_EPOCH:
                #             logger.info(
                #                 'epoch = {}, best_mAP = {:.2f}, updating best checkpoint for depth {} \n'.format(
                #                     cur_epoch, eval_mAP, default_depth_method))
                #         else:
                #             logger.info(
                #                 'iteration = {}, best_mAP = {:.2f}, updating best checkpoint for depth {} \n'.format(
                #                     iteration, eval_mAP, default_depth_method))
                #
                #     eval_iteration += 1
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    if cfg.rank == 0:
        logger.info(
            "Total training time: {} ({:.4f} s / it), best model is achieved at iteration = {}".format(
                total_time_str, total_training_time / (max_iter), best_iteration,
            )
        )

        logger.info('The best performance is as follows')
        logger.info('\n' + best_result_str)


if __name__ == '__main__':
    # ms.set_context(save_graphs=True, save_graphs_path="src")
    train()