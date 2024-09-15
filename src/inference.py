from __future__ import print_function

import argparse
import os
import random
import shutil
import time
import warnings
import wandb
import cv2
import numpy as np
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from mobilenet import get_model


@torch.no_grad()
def inference(weight, name, img_dir):
    net = get_model(name, num_classes=2)
    state_dict = torch.load(weight)
    net.load_state_dict(state_dict)
    net.eval()
    
    output_path = "./outputs_iotcamera"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            img_path = os.path.join(root, name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224,224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0).float()
            img.div_(255)
    
            out = net(img)
            logits = nn.Softmax()(out).numpy()
            class_id = np.argmax(logits)
            print(name, class_id)
            
            result_dir = os.path.join(output_path, "mask" if class_id == 0 else "nomask")
            if not os.path.isdir(result_dir):
                os.mkdir(result_dir)
            shutil.copy(img_path, os.path.join(result_dir,name))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch MobileNet Inference')
    parser.add_argument('--network', type=str, default='mobilenetv3_large', help='backbone network')
    parser.add_argument('--weight', type=str, default='./weights/mobilenetv3_large_1.0.pth')
    parser.add_argument('--img_dir', type=str, \
                        default='/home/hunght21/projects/insightface/alignment/iot_camera')
    args = parser.parse_args()
    inference(args.weight, args.network, args.img_dir)
    