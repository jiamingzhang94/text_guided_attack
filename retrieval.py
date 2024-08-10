import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# os.environ['TORCH_HOME'] = '/new_data/yifei2/junhong/AttackVLM-main/model/blip-cache'
import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *
from lavis.common.registry import registry


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg_path", default="lavis_tool/blip/ret_coco_eval.yaml", help="path to configuration file.")
    parser.add_argument("--cache_path", default="/new_data/yifei2/junhong/dataset", help="path to dataset cache")
    parser.add_argument("--data_path", help="test data path")
    parser.add_argument("--image_path",help="path to image dataset")
    parser.add_argument("--output_dir",help="path where to save result")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    args = parse_args()
    # 修改缓存路径，默认的缓存路径没有访问权限，修改缓存到指定位置
    registry.mapping["paths"]["cache_root"] = args.cache_path
    job_id = now()

    cfg = Config(args)
    if args.image_path:
        cfg.config['datasets']['coco_retrieval']['build_info']['images']=args.image_path
    if args.output_dir:
        cfg.config['run']['output_dir'] = args.output_dir
    if args.data_path:
        cfg.config['datasets']['coco_retrieval']['build_info']['annotations']['test']['storage'] = args.data_path
    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    # task_key = list(datasets.keys())[0]
    # new_datasets = {task_key: {"test": datasets[task_key]["test"]}}  # 只用测试集当中的五千个样本
    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )

    # 默认的保存路径为registry.get_path("library_root")+output_dir/evaluate.txt 此处修改为配置文件中路径
    output_dir =os.path.join(cfg.run_cfg["output_dir"], job_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    registry.mapping["paths"]["output_dir"] = output_dir

    runner.evaluate(skip_reload=True)


if __name__ == "__main__":
    main()
