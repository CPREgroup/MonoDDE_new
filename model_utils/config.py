import ast
import argparse
from pprint import pformat
import os
import pdb

import yaml
from matplotlib.pyplot import streamplot


# from utils import comm
from .miscellaneous import mkdir
from .logger import setup_logger
from .envs import seed_all_rng



def parse_cli_to_yaml(parser, cfg, helper=None, choices=None, cfg_path="default_config.yaml"):
    """
    Parse command line arguments to the configuration according to the default yaml.

    Args:
        parser: Parent parser.
        cfg: Base configuration.
        helper: Helper description.
        cfg_path: Path to the default yaml config.
    """
    parser = argparse.ArgumentParser(description="[REPLACE THIS at config.py]",
                                     parents=[parser])
    helper = {} if helper is None else helper
    choices = {} if choices is None else choices
    for item in cfg:
        if not isinstance(cfg[item], list) and not isinstance(cfg[item], dict):
            help_description = helper[item] if item in helper else "Please reference to {}".format(cfg_path)
            choice = choices[item] if item in choices else None
            if isinstance(cfg[item], bool):
                parser.add_argument("--" + item, type=ast.literal_eval, default=cfg[item], choices=choice,
                                    help=help_description)
            else:
                parser.add_argument("--" + item, type=type(cfg[item]), default=cfg[item], choices=choice,
                                    help=help_description)
    args = parser.parse_args()
    return args


class Config:
    """
    Configuration namespace. Convert dictionary to members.
    """
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


def parse_yaml(yaml_path):
    """
    Parse the yaml config file.

    Args:
        yaml_path: Path to the yaml config.
    """
    with open(yaml_path, 'r') as fin:
        try:
            cfgs = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
            cfgs = [x for x in cfgs]
            if len(cfgs) == 1:
                cfg_helper = {}
                cfg = cfgs[0]
                cfg_choices = {}
            elif len(cfgs) == 2:
                cfg, cfg_helper = cfgs
                cfg_choices = {}
            elif len(cfgs) == 3:
                cfg, cfg_helper, cfg_choices = cfgs
            else:
                raise ValueError("At most 3 docs (config, description for help, choices) are supported in config yaml")
            print(cfg_helper)
        except:
            raise ValueError("Failed to parse yaml")
    return cfg, cfg_helper, cfg_choices


def merge(args, cfg):
    """
    Merge the base config from yaml file and command line arguments.

    Args:
        args: Command line arguments.
        cfg: Base configuration.
    """
    args_var = vars(args)
    for item in args_var:
        cfg[item] = args_var[item]
    return cfg


def default_argument_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Training",add_help=False)
    parser.add_argument("--config", dest='config_path', default="runs/MonoDDE_nuscenes2.yaml",
                        metavar="FILE", help="path to config file")

    parser.add_argument("--dataset", type=str, default='kitti',
                        help='The dataset used for training.')
    parser.add_argument("--data_root", default=None, type=str,
                        help="Root path of dataset.")

    parser.add_argument("--eval", dest='eval_only', action="store_true", help="perform evaluation only")
    parser.add_argument("--eval_iou", action="store_true", help="evaluate disentangling IoU")
    parser.add_argument("--eval_depth", action="store_true", help="evaluate depth errors")
    parser.add_argument("--eval_all_depths", action="store_true")

    parser.add_argument("--eval_score_iou", action="store_true",
                        help="evaluate the relationship between scores and IoU")
    parser.add_argument("--survey_depth", action="store_true",
                        help="evaluate the relationship between scores and IoU")

    parser.add_argument("--test", action="store_true", help="test mode")
    parser.add_argument("--demo", action="store_true", help="Use kitti demo dataset to test the code.")
    parser.add_argument("--vis", action="store_true", help="visualize when evaluating")
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument('--debug', action="store_true", help="The debug mode.")
    parser.add_argument('--Coor_Attribute', action="store_true",
                        help="Whether to use Coordinate loss to train attributes.")
    parser.add_argument('--Coor_Uncern', action="store_true",
                        help="Whether to use Coordinate loss to train uncertainty.")
    parser.add_argument('--GRM_Attribute', action="store_true", help="Whether to use GRM loss to train attributes.")
    parser.add_argument('--GRM_Uncern', action="store_true", help="Whether to use GRM loss to train uncertainty.")

    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpu")
    parser.add_argument("--seed", type=int, default=-1, help="For not fixing seed, set it as -1.")
    parser.add_argument("--batch_size", type=int, default=8, help="number of batch_size")
    parser.add_argument("--num_work", type=int, default=8, help="number of workers for dataloader")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--backbone", type=str, default='dla34_noDCN', help="dla34 or dla34_noDCN")

    parser.add_argument("--vis_thre", type=float, default=0.25, help="threshold for visualize results of detection")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--is_training", type=bool,default=True,help='whether train')
    parser.add_argument("--pretrained", type=bool,default=False,help='whether pretrain dla34')
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    # port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 13
    # parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument("--dist-url", default="auto")
    # parser.add_argument(
    #     "opts",
    #     help="Modify config options using the command-line",
    #     default=None,
    #     nargs=argparse.REMAINDER,
    # )
    path_args, _ = parser.parse_known_args()
    default, helper, choices = parse_yaml(path_args.config_path)
    args = parse_cli_to_yaml(parser=parser, cfg=default, helper=helper, choices=choices, cfg_path=path_args.config_path)
    final_config = merge(args, default)
    return Config(final_config)
    # return parser
config=default_argument_parser()


def default_setup(cfg, args):
    cfg.SEED = args.seed
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    # rank = comm.get_rank()
    logger = setup_logger(output_dir, file_name="log_{}.txt".format(cfg.START_TIME))
    logger.info("Using {} GPUs".format(args.num_gpus))
    logger.info("Collecting environment info")
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_path))
    with open(args.config_path, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)

    logger.info("Running with config:\n{}".format(cfg))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED)

