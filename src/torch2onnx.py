from __future__ import print_function

import argparse
import os
import time
import warnings
import cv2
import numpy as np
warnings.filterwarnings("ignore")

import onnx
import torch
import torch.nn as nn

from mobilenet import get_model


def convert_onnx(net, weight, output, opset=9, dynamic=False, dynamic_batch=False, simplify=False):
    assert isinstance(net, torch.nn.Module)
    img = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.int32)
    img = img.astype(float)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()

    state_dict = torch.load(weight)
    net.load_state_dict(state_dict)

    net.eval()
    
    input_names = ["input"]
    output_names = ["classes"]
    dynamic_axes = {"input":{0:"batch_size"}, "classes":{0:"batch_size"}}
        
    torch.onnx.export(net, img, output, input_names=input_names, output_names=output_names, \
                      export_params=True, verbose=False, opset_version=opset, dynamic_axes=dynamic_axes)
    model = onnx.load(output)

    
    if simplify:
        print("sim")
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.checker.check_model(model)
    onnx.save(model, output)
    print("Model was successfully converted to ONNX format.")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MobileNet PyTorch to onnx')
    parser.add_argument('--input', type=str, \
                        default="./weights/mobilenetv3_large_1.0.pth", \
                        help='input backbone.pth file or path')
    parser.add_argument('--output', type=str, default="./weights/mobilenetv3_large_1.0.onnx", help='output onnx path')
    parser.add_argument('--network', type=str, default="mobilenetv3_large", help='backbone network')
    parser.add_argument('--simplify', type=bool, default=False, help='onnx simplify')
    parser.add_argument('--dynamic', type=bool, default=True, help='onnx dynamic input shape')
    parser.add_argument('--dynamic_batch', type=bool, default=True, help='onnx dynamic batch')
    
    args = parser.parse_args()
    
    input_file = args.input
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "model.pth")
    assert os.path.exists(input_file)

    print(args)
    backbone_onnx = get_model(args.network, num_classes=2)
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "mobilenetv3_large_1.0.onnx")
    convert_onnx(backbone_onnx, input_file, args.output, simplify=args.simplify, dynamic=args.dynamic, \
                 dynamic_batch=args.dynamic_batch)