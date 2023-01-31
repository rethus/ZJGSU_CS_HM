import os
import sys
import yaml
import torch
import argparse
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
    train_config = load_config(args.model_config_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path', type=str, default='./config/train_config.yaml')
    # .....
    