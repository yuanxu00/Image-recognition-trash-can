import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import time
import cv2
import numpy as np

# 使用摄像头捕捉目标，再对其进行预测

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

image_path1 = '/home/y/workspace/banana.jpeg'
image_path2 = '/home/y/workspace/4.jpg'




image1 = load_and_preprocess_image(image_path1)

image4D1 = tf.expand_dims(image1, 0)


saved_model_loaded = tf.saved_model.load(
    "trt_savedmodel", tags=[trt.tag_constants.SERVING])#读取模型

signature_keys = list(saved_model_loaded.signatures.keys())
infer = saved_model_loaded.signatures['serving_default']

print(infer.structured_outputs)

labeling = infer(image4D1)
preds = labeling['dense'].numpy()
print(preds.argmax(-1))


class_indict=['blackground', 'milk_box', 'peel', 'plastic_bottle', 'shopping_bags']

start = time.time()
temp_class = 6
predict_right_count = 0

while True:
    ret, imagec = camera.read()
    print("test2")
    image_c = preprocess_image(imagec)
    imagec4D = tf.expand_dims(image_c,0)
    labeling = infer(imagec4D)
    preds = labeling['dense'].numpy()
    result = np.squeeze(preds)
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
        print(time.time() - start)
        #time.sleep(0.5)


print("test3")
camera.release()
cv2.destroyAllWindows()








