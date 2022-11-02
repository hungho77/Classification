from __future__ import division
import datetime
import numpy as np
import onnx
import onnxruntime
import os
import os.path as osp
import cv2
import sys
import time
from tqdm import tqdm

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

class FaceMaskClassifier:
    def __init__(self, model_file=None, session=None):
        self.model_file = model_file
        self.session = session
        self.batched = False
        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_name = self.session.get_inputs()[0].name

    def prepare(self, ctx_id, **kwargs):
        if ctx_id<0:
            self.session.set_providers(['CPUExecutionProvider'])
        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.input_size is not None:
                print('warning: det_size is already set in scrfd model, ignore')
            else:
                self.input_size = input_size

    def forward(self, img):
        input_size = tuple(img.shape[1:3][::-1])
        st_ = time.time()
        blob = cv2.dnn.blobFromImages(img, 1.0/56, input_size, (0.485 * 255, 0.456 * 255, 0.406 * 255), swapRB=True)
        print("Time: ", time.time()-st_)
        net_outs = self.session.run(self.output_names, {self.input_name : blob})
        
        logits = softmax(net_outs[0])
        class_id = np.argmax(logits, axis=1)
                
        return class_id

    def detect(self, img, input_size = None):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size
            
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        img_input = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
        img_input[:new_height, :new_width, :] = resized_img
        img_input = np.tile(img_input,(16,1,1,1))
        class_id = self.forward(img_input)
        
        return class_id
    

if __name__ == '__main__':
    detector = FaceMaskClassifier(model_file='./weights/mobilenetv3_large_1.0.onnx')
    detector.prepare(1)
    img = cv2.imread("./test/8315.png")
    class_id = detector.detect(img, input_size = (224, 224))
    print(class_id)