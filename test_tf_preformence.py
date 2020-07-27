import tensorflow as tf
import time
start = time.time()
model = tf.keras.models.load_model('tf_savedmodel')
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
preds = model.predict(image4D)
print(preds)
print(time.time() - start)
