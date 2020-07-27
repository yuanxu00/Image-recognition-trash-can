from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image = image/127.5 - 1

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

image_path = '/home/y/workspace/m.jpg'

image = load_and_preprocess_image(image_path)
image4D = tf.expand_dims(image, 0)

# 测试Tensor RT

params=trt.DEFAULT_TRT_CONVERSION_PARAMS
params._replace(precision_mode=trt.TrtPrecisionMode.FP16)
params._replace(is_dynamic_op=True)
converter = trt.TrtGraphConverterV2(input_saved_model_dir='tf_savedmodel',conversion_params=params)
converter.convert()

def input_fn():
   yield (image4D),
converter.build(input_fn)
converter.save('trt_savedmodel')


