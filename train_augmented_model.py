from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models.classify import *
from engine import train_augmodel
from argparse import ArgumentParser

# 命令行解析器（描述：Train Augmented model）
parser = ArgumentParser(description='Train Augmented model')
# 定义命令行参数
# configs参数，字符串类型
# general_gan：通用GAN
# specific_gan：半监督GAN【特定于反演的GAN】
parser.add_argument('--configs', type=str, default='./config/celeba/training_augmodel/ffhq.json')
args = parser.parse_args()


if __name__ == '__main__':
    # 加载JSON配置文件
    cfg = load_json(json_file=args.configs)

    train_augmodel(cfg)