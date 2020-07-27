import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import time
import cv2
start = time.time()

print("test start")
camera = cv2.VideoCapture(1)
print("test open")
camera.set(3, 1280)
camera.set(4, 720)



def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image = image/127.5 - 1

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

image_path1 = '/home/y/workspace/banana.jpeg'
image_path2 = '/home/y/workspace/4.jpg'


image1 = load_and_preprocess_image(image_path1)
image2 = load_and_preprocess_image(image_path2)
image4D1 = tf.expand_dims(image1, 0)
image4D2 = tf.expand_dims(image2, 0)

saved_model_loaded = tf.saved_model.load(
    "trt_savedmodel", tags=[trt.tag_constants.SERVING])#读取模型

signature_keys = list(saved_model_loaded.signatures.keys())
infer = saved_model_loaded.signatures['serving_default']

print(infer.structured_outputs)

labeling = infer(image4D1)
preds = labeling['dense'].numpy()
print('banana')
print(preds.argmax(-1))
print(time.time() - start)

time.sleep(10)

print(time.time() - start)
print('\n')
print('\n')


labeling = infer(image4D2)
preds = labeling['dense'].numpy()
print('kuangquanshui')
print(preds)
print(signature_keys)



t=time.time()
print(time.time()-start)


while True:
    ret, imagec = camera.read()
    print("test2")
    imagec = load_and_preprocess_image(imagec)
    imagec4D = tf.expand_dims(imagec, 0)
    labeling = infer(imagec4D)
    preds = labeling['dense'].numpy()
    print('kuangquanshui')
    print(preds)
    print(time.time() -start)
    
camera.release()
cv2.destroyAllWindows()



