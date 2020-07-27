import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import time
import cv2
import numpy as np
import serial

# 使用摄像头捕捉目标后
# 对目标进行识别
# 识别成功后，发送指令让电机转动到指定的位置

serial_port = serial.Serial(
    port="/dev/ttyUSB0",
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)

camera = cv2.VideoCapture(1)
camera.set(3, 1280)
camera.set(4, 720)


def preprocess_image(image):
    image = cv2.resize(image, (224, 224)).astype(np.float32)
    image = image/127.5 - 1

    return image

def load_and_preprocess_image(path):
    image = cv2.imread(path)
    image = image[0:780,150:1050] 
    return preprocess_image(image)

#读取模型
saved_model_loaded = tf.saved_model.load(
    "trt_savedmodel", tags=[trt.tag_constants.SERVING])

signature_keys = list(saved_model_loaded.signatures.keys())
infer = saved_model_loaded.signatures['serving_default']


print(infer.structured_outputs)


# control stepper
# 步进电机先转过去再转回来
bag1 = 'm 00001@'
bag2 = 'm 00001@'

milk1 = 'm 5000@'
milk2 = 'm 15000@'

peel1 = 'm 10000@'
peel2 = 'm 10000@'

pb1 = 'm 15000@'
pb2 = 'm 5000@'


# class name
class_indict=['blackground', 'milk_box', 'peel', 'plastic_bottle', 'shopping_bags']


start = time.time()
temp_class = 6
predict_right_count = 0

while True:
    ret, imagec = camera.read()
    print("test2")
    print(time.time() - start)

    image_c = preprocess_image(imagec)
    imagec4D = tf.expand_dims(image_c,0)
    labeling = infer(imagec4D)

    preds = labeling['dense'].numpy()
    result = np.squeeze(preds)
    prediction = tf.keras.layers.Softmax()(result).numpy()
    predict_class = np.argmax(result)

    # filter
    if temp_class == predict_class:
        i = i + 1
    else:
        temp_class = predict_class
        i = 0

    if (i == 5) and predict_class != 0 :
        print(class_indict[predict_class], prediction[predict_class])
        print(prediction)
        # milk 
        if predict_class == 1:
            for i in range(7):
                serial_port.write(milk1[i].encode())
                print(milk1[i])
                time.sleep(0.1)
            time.sleep(5)
            for i in range(8):
                serial_port.write(milk2[i].encode())
                print(milk2[i])
                time.sleep(0.1)
        # peel
        elif predict_class == 2:
            for i in range(8):
                serial_port.write(peel1[i].encode())
                print(peel1[i])
                time.sleep(0.1)
            time.sleep(5)
            for i in range(8):
                serial_port.write(peel2[i].encode())
                print(peel2[i])
                time.sleep(0.1)
        # pb
        elif predict_class == 3:
            for i in range(8):
                serial_port.write(pb1[i].encode())
                print(pb1[i])
                time.sleep(0.1)
            time.sleep(5)
            for i in range(7):
                serial_port.write(pb2[i].encode())
                print(pb2[i])
                time.sleep(0.1)

        # bag
        elif predict_class == 4:
            for i in range(8):
                serial_port.write(bag1[i].encode())
                print(bag1[i])
                time.sleep(0.1)
            time.sleep(5)
            for i in range(8):
                serial_port.write(bag2[i].encode())
                print(bag2[i])
                time.sleep(0.1)

        print(time.time() - start)
    time.sleep(0.2)


print("test3")
camera.release()
cv2.destroyAllWindows()








