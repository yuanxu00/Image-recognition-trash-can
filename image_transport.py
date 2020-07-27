import cv2
import datetime
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import random
import datetime
from random import choice

# 图像增强

data_root_orig = '/home/y/Documents/train_dateset/'
# pathlib.Path() == os.path.join()
# Join various path components
data_root = pathlib.Path(data_root_orig)

# load image
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]


l = [-1 , 1, 0]

for filename in all_image_paths:
    image = cv2.imread(filename)
    hv_flip = cv2.flip(image, choice(l))
    fn,ext = os.path.splitext(os.path.split(os.path.basename(filename))[1])
    trans_image = os.path.dirname(filename) + "/mirror"  + fn + ext
    print(trans_image)
    cv2.imwrite(trans_image, hv_flip)


