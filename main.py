import _init_paths
import os
import argparse
import json
import logging
import torch
import numpy as np
import random

from algorithms.cd_regression_v1.src.Trainer_cd_regression_v1 import Trainer_cd_regression_v1

def fix_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def parse(opt, bash_args):
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
    opt.save_point = [int(i) for i in opt.save_point.split(',')]
    opt.test_scales = [1.0]

    opt.pre_img = False

    print('Running tracking')
    opt.out_thresh = opt.track_thresh
    opt.pre_thresh = opt.track_thresh
    opt.new_thresh = opt.track_thresh
    opt.pre_img = True
    print('Using tracking threshold for out threshold!', opt.track_thresh)

    opt.fix_res = True
    print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')

    opt.head_conv = 256 if 'dla' in opt.model else 64

    opt.pad = 127 if 'hourglass' in opt.model else 31
    opt.num_stacks = 2 if opt.model == 'hourglass' else 1

    # log dirs
    opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    opt.data_dir = os.path.join(opt.root_dir, 'data')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', 'tracking')
    opt.save_dir = os.path.join(opt.exp_dir, bash_args.exp_id)
    
    if opt.resume and opt.load_model == '':
      opt.load_model = os.path.join(opt.save_dir, 'model_last.pth')
    return opt

algorithms_map = {
    'cd_regression_v1': Trainer_cd_regression_v1
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help = "Path to configuration file")
    parser.add_argument("--exp_id", help = "Index of experiment")
    parser.add_argument("--gpu_id", help = "Index of GPU")
    bash_args = parser.parse_args()
    with open(bash_args.config, "r") as inp:
        args = argparse.Namespace(**json.load(inp))

    args = parse(args, bash_args)
        
    os.environ["CUDA_VISIBLE_DEVICES"] = bash_args.gpu_id  
        
    # fix_random_seed(args.seed_value)
    logging.basicConfig(filename = "algorithms/" + args.algorithm + "/results/logs/" + args.exp_name + "_" + bash_args.exp_id + '.log', filemode = 'w', level = logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = algorithms_map[args.algorithm](args, device, bash_args.exp_id)
    trainer.train()
    # trainer.test()
    print("Finished!")