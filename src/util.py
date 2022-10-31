import os
import platform
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import importlib
import os.path as osp


def get_config(config_file):
    assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    temp_config_name = osp.basename(config_file)
    temp_module_name = osp.splitext(temp_config_name)[0]
    config = importlib.import_module("configs.%s" % temp_module_name)
    cfg = config.config
    if cfg.output is None:
        cfg.output = osp.join('work_dirs', temp_module_name)
    return cfg