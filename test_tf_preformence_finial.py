import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import time
import cv2
import numpy as np
start = time.time()

camera = cv2.VideoCapture(0)
camera.set(3, 1280)
camera.set(4, 720)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


model = tf.keras.models.load_model('tf_savedmodel')
def preprocess_image(image):
    image = image[0:720,110:1050]
    image = cv2.resize(image, (224, 224)).astype(np.float32)
    image = image/127.5 - 1

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

class_indict=['blackground', 'milk_box', 'peel', 'plastic_bottle', 'shopping_bags']

start = time.time()
temp_class = 6
predict_right_count = 0
print('start!!!')
strat = time.time()
while True:
    ret, imagec = camera.read()
    #print("test2")
    image = preprocess_image(imagec)
    image4D = tf.expand_dims(image, 0)
    result = np.squeeze(model.predict(image4D))
    prediction = tf.keras.layers.Softmax()(result).numpy()
    predict_class = np.argmax(result)
    if temp_class == predict_class:
        i = i + 1
    else:
        temp_class = predict_class
        i = 0

    if (i == 5) and predict_class != 0 :
        print(class_indict[predict_class], prediction[predict_class])
        print(prediction)
        print(time.time() -strat)
    time.sleep(0.1)


print(time.time() - start)
