import os
import sys
import json
import yaml
import torch
import datetime
import argparse
from utils import *
import trainer
from utils import metrics
from models import *
from dataset import *
from torch.utils.data import DataLoader


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # model_config =

    # dataset index -> input

    # scalar

    # model = getattr model-config

    # dataloader: model - dataset

    # trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path', type=str, default='./config/train_config.yaml')
    # .....

    args = parser.parse_args()
    main(args)
