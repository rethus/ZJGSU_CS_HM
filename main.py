import os
import sys
import json
import time
import yaml
import torch
import trainer
import datetime
import argparse
from utils import *
from models import *
from models import *
from dataset import *
import torch.nn as nn
from torch import optim
from utils import metrics
from dataset.read_data import Mydata, CocoKeypoint
from torch.utils.data import DataLoader


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    model_config = load_config(args.model_config_path)
    model = VGG16Net().to(device)

    # mydata = Mydata("data/data/train")
    mydata = CocoKeypoint()
    train_loader = torch.utils.data.DataLoader(mydata,
                                               batch_size=model_config['batch_size'],
                                               shuffle=model_config['train_shuffle'],
                                               drop_last=model_config['drop_last'])

    get_loss = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(),
                          lr=model_config['learning_rate'],
                          momentum=model_config['momentum'],
                          weight_decay=model_config['weight_decay'])

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=model_config['step_size'],
                                          gamma=0.5,
                                          last_epoch=-1)

    # trainer

    # loss_list=[]
    # start=time.time()
    #
    # for epoch in range(model_config['epuch_num']):
    #     running_loss = 0.0
    #     # enumerate()是python自带的函数，用于迭代字典。参数1，是需要迭代的对象，第二参数是迭代的起始位置
    #     for i,(inputs,labels) in enumerate(train_loader,0):
    #         inputs,labels = inputs.to(device), labels.to(device)
    #
    #         optimizer.zero_grad()# 将梯度初始化为0
    #         outputs = model(inputs)# 前向传播求出预测的值
    #         loss = get_loss(outputs,labels).to(device)# 求loss,对应loss +=
    #
    #         loss.backward() # 反向传播求梯度
    #         optimizer.step() # 更新所有参数
    #
    #         running_loss += loss.item()# loss是张量，访问值时需要使用item()
    #         loss_list.append(loss.item())
    #
    #         num_print = model_config['num_print']
    #
    #         if i % num_print == num_print - 1:
    #             print('[%d epoch, %d] loss: %.6f' % (epoch + 1, i + 1, running_loss / num_print))
    #             running_loss = 0.0
    #     lr = optimizer.param_groups[0]['lr']# 查看目前的学习率
    #     print('learn_rate : %.15f'% lr)
    #     scheduler.step()# 根据迭代epoch更新学习率
    #
    # end = time.time()
    # print('time:{}'.format(end-start))
    #
    # torch.save(model, 'dataset/model.pth')

def main2(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    with open(args.keypoints_path, "r") as f:
        person_kps_info = json.load(f)

    fixed_size = args.fixed_size
    heatmap_hw = (args.fixed_size[0] // 4, args.fixed_size[1] // 4)
    kps_weights = np.array(person_kps_info["kps_weights"],
                           dtype=np.float32).reshape((args.num_joints,))
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
    data_root = args.data_path
    # 加载训练集 √
    # coco2017 -> annotations -> person_keypoints_train2017.json
    train_dataset = CocoKeypoint(data_root, "train", transforms=data_transform["train"], fixed_size=args.fixed_size)


    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
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
    val_dataset = CocoKeypoint(data_root, "val", transforms=data_transform["val"], fixed_size=args.fixed_size,
                               det_json_path=args.person_det)
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=nw,
                                      collate_fn=val_dataset.collate_fn)

    # 模型初始化 √
    model = create_model(num_joints=args.num_joints)
    # print(model)
    # 加载GPU或CPU √
    model.to(device)

    # define optimizer √
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler √
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    # 使用中断前的参数 √
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []

    # 训练epoch次 √
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True,
                                              scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        coco_info = utils.evaluate(model, val_data_loader, device=device,
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
        torch.save(save_files, "./save_weights/model-{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    # 训练配置文件
    parser.add_argument('--model_config_path', type=str, default='./config/train_config.yaml')
    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录(coco2017)
    parser.add_argument('--data-path', default='dataset/COCO2017', help='dataset')
    # COCO数据集人体关键点信息
    parser.add_argument('--keypoints-path', default="./person_keypoints.json", type=str,
                        help='person_keypoints.json path')
    # 原项目提供的验证集person检测信息，如果要使用GT信息，直接将该参数置为None，建议设置成None
    parser.add_argument('--person-det', type=str, default=None)
    parser.add_argument('--fixed-size', default=[256, 192], nargs='+', type=int, help='input size')
    # keypoints点数
    parser.add_argument('--num-joints', default=17, type=int, help='num_joints')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=210, type=int, metavar='N',
                        help='number of total epochs to run')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[170, 200], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # 学习率
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # AdamW的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 训练的batch size
    parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                        help='batch size when training.')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    # .....

    args = parser.parse_args()
    main2(args)
