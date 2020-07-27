import cv2
import numpy as np

def preprocess_image(image):
    image = cv2.resize(image, (192, 192)).astype(np.float32)
    image = image/127.5 - 1

    return image

def load_and_preprocess_image(path):
    image = cv2.imread(path)
    #print(image)
    return preprocess_image(image)

image_path1 = '/home/y/workspace/banana.jpeg'
image_path2 = '/home/y/workspace/4.jpg'

image1 = load_and_preprocess_image(image_path1)
print(image1)

