import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import time
strat = time.time()
input_saved_model_dir = 'tf_savedmodel'
output_saved_model_dir = 'trt_savedmodel'

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image/127.5 - 1

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

image_path = '/home/y/workspace/4.jpg'

image = load_and_preprocess_image(image_path)
image4D = tf.expand_dims(image, 0)

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(
max_workspace_size_bytes=(1<<33))
conversion_params = conversion_params._replace(precision_mode="FP32")
conversion_params = conversion_params._replace(
maximum_cached_engines = 100)

converter = trt.TrtGraphConverterV2(
                input_saved_model_dir=input_saved_model_dir,
                conversion_params=conversion_params)
converter.convert()
def my_input_fn():
    yield (image4D),
converter.build(input_fn=my_input_fn)
converter.save(output_saved_model_dir)
print(time.time() - strat)
