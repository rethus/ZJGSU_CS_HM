import os
import json

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models import HighResolutionNet
from utils.draw_utils import draw_keypoints
from utils import transforms
from dataset.read_data import CocoKeypoint

from torch.utils import data
def predict_all_person():
    train_dataset = CocoKeypoint(f"../data/coco2017/", dataset="train")
    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=1,
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=8,
                                        collate_fn=train_dataset.collate_fn)
    valid_person_list = train_dataset.valid_person_list
    person_1 = valid_person_list[0]
    person_1_keypoints = person_1['keypoints']
    print(person_1_keypoints)
    print(1)



def predict_single_person():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    # 翻转图片再测一次，取平均 √
    flip_test = True
    resize_hw = (256, 192)
    img_path = "../data/ochuman/val2017/000010.jpg"
    weights_path = "../weights/pre_train/pose_coco/pose_hrnet_w32_256x192.pth"
    keypoint_json_path = "person_keypoints.json"

    assert os.path.exists(img_path), f"file: {img_path} does not exist."
    assert os.path.exists(weights_path), f"file: {weights_path} does not exist."
    assert os.path.exists(keypoint_json_path), f"file: {keypoint_json_path} does not exist."

    # 数据增强 √
    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # read json file √
    with open(keypoint_json_path, "r") as f:
        person_info = json.load(f)

    # read single-person image √
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})

    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    # create model √
    # HRNet-W32: base_channel=32
    # HRNet-W48: base_channel=48
    model = HighResolutionNet(base_channel=32)
    weights = torch.load(weights_path, map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    with torch.inference_mode():
        outputs = model(img_tensor.to(device))

        if flip_test:
            flip_tensor = transforms.flip_images(img_tensor)
            flip_outputs = torch.squeeze(
                transforms.flip_back(model(flip_tensor.to(device)), person_info["flip_pairs"]),
            )
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flip_outputs[..., 1:] = flip_outputs.clone()[..., 0: -1]
            outputs = (outputs + flip_outputs) * 0.5

        keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
        keypoints = np.squeeze(keypoints)
        scores = np.squeeze(scores)

        plot_img = draw_keypoints(img, keypoints, scores, thresh=0.2, r=3)
        plt.imshow(plot_img)
        plt.show()
        plot_img.save("test_result.jpg")


if __name__ == '__main__':
    # predict_single_person()
    predict_all_person()