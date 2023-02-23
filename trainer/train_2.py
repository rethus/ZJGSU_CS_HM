import os
import yaml
import json
import datetime
import argparse

import torch
import numpy as np
from torch.utils import data

from utils import transforms
from utils import train_eval_utils as teutils
from models import VisionTransformer, HighResolutionNet
from dataset.read_data import CocoKeypoint


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

"""
torch.Size([4, 3, 224, 224])
torch.Size([4, 17])

torch.Size([4, 3, 256, 192])
torch.Size([4, 17, 64, 48])
"""

def create_model(num_joints, load_pretrain_weights=False, pre_weight=""):
    # model = HighResolutionNet(base_channel=32, num_joints=num_joints)
    model = VisionTransformer()

    if load_pretrain_weights:
        # 载入预训练模型权重 √
        weights_dict = torch.load(pre_weight, map_location='cpu')

        for k in list(weights_dict.keys()):
            # 如果载入的是imagenet权重，就删除无用权重 √
            if ("head" in k) or ("fc" in k):
                del weights_dict[k]

            # 如果载入的是coco权重，对比下num_joints，如果不相等就删除 √
            if "final_layer" in k:
                if weights_dict[k].shape[0] != num_joints:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0:
            print("missing_keys: ", missing_keys)

    return model


def main(args):
    model_config = load_config(args.model_config_path)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(model_config['path']['output']):
        os.makedirs(model_config['path']['output'])

    device = torch.device(model_config['device'] if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    with open(model_config['path']['keypoints'], "r") as f:
        person_kps_info = json.load(f)

    fixed_size = model_config['fixed_size']
    heatmap_hw = (fixed_size[0] // 4, fixed_size[1] // 4)
    kps_weights = np.array(person_kps_info["kps_weights"],
                           dtype=np.float32).reshape((model_config['num_joints'],))

    # print(person_kps_info["kps_weights"], "------------------------------")
    # 数据增强 √
    data_transform = {
        "train": transforms.Compose([
            # 随机裁剪半身
            transforms.HalfBody(0.3, person_kps_info["upper_body_ids"], person_kps_info["lower_body_ids"]),
            # 2D仿射变换
            transforms.AffineTransform(scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=fixed_size),
            # 随机水平翻转
            transforms.RandomHorizontalFlip(0.5, person_kps_info["flip_pairs"]),
            # 转化热力图
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            # 检测框放大1.25倍
            transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    # 数据的根目录 √
    data_root = model_config['path']['data']
    # 加载训练集 √
    # coco2017 -> annotations -> person_keypoints_train2017.json
    train_dataset = CocoKeypoint(data_root, "train", transforms=data_transform["train"], fixed_size=fixed_size)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = model_config['batch_size']
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=nw,
                                        collate_fn=train_dataset.collate_fn)


    # 加载测试集 √
    # coco2017 -> annotations -> person_keypoints_val2017.json
    val_dataset = CocoKeypoint(data_root, "val", transforms=data_transform["val"], fixed_size=fixed_size,
                               det_json_path=args.person_det)
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=nw,
                                      collate_fn=val_dataset.collate_fn)

    # 模型初始化 √
    model = create_model(num_joints=model_config['num_joints'], load_pretrain_weights=False,
                         pre_weight=model_config['path']['pre_weight'])
    # print(model)
    # 加载GPU或CPU √
    model.to(device)

    # define optimizer √
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,
                                  lr=model_config['optimizer']['learning_rate'],
                                  weight_decay=model_config['optimizer']['weight_decay'])

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler √
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=model_config['lr_scheduler']['lr_steps'],
                                                        gamma=model_config['lr_scheduler']['lr_gamma'])

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    # 使用中断前的参数 √
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        model_config['start_epoch'] = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(model_config['start_epoch']))

    train_loss = []
    learning_rate = []
    val_map = []

    # 训练epoch次 √
    for epoch in range(model_config['start_epoch'], model_config['epoch_num']):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = teutils.train_one_epoch(model, optimizer, train_data_loader,
                                                device=device, epoch=epoch,
                                                print_freq=50, warmup=True,
                                                scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        coco_info = teutils.evaluate(model, val_data_loader, device=device,
                                     flip=True, flip_pairs=person_kps_info["flip_pairs"])

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # @0.5 mAP

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        torch.save(save_files, model_config['path']['output']+"model-{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from utils.plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from utils.plot_curve import plot_map
        plot_map(val_map)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    # 训练配置文件
    parser.add_argument('--model_config_path', type=str, default='../config/train_config.yaml')
    # 原项目提供的验证集person检测信息，如果要使用GT信息，直接将该参数置为None，建议设置成None
    parser.add_argument('--person-det', type=str, default=None)
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)


    img = f"'../data/coco2017/train2017/000000237814.jpg'"
    main(args)
