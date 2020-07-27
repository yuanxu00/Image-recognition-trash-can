from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import random
import datetime

#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

AUTOTUNE = tf.data.experimental.AUTOTUNE

IMG_SIZE = 224
IMG_CHANNEL = 3
BATCH_SIZE = 64
TRAIN_PERCENTAGE = 0.8
TRAIN_EPOCHS = 200
VALIDATION_STEPS = 20

# ~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5.old
data_root_orig = '/home/y/Documents/train_dateset/'
# pathlib.Path() == os.path.join()
# Join various path components
data_root = pathlib.Path(data_root_orig)

# load image
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]

random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print(image_count)

# load labels
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print(label_names)

# 将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
label_to_index = dict((name, index) for index,name in enumerate(label_names))
print(label_to_index)

# 创建一个列表，包含每个文件的标签索引
# get image path to index,index use parent.name get the class name  
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
print(all_image_labels[:10])


# split dataset to (train, validation, tes)
t_offset = int(image_count * TRAIN_PERCENTAGE)
v_offset = int(image_count * (TRAIN_PERCENTAGE + (1-TRAIN_PERCENTAGE)/2))
train_image_paths =all_image_paths[:t_offset]
vaildation_image_paths = all_image_paths[t_offset : v_offset]
test_image_paths = all_image_paths[v_offset:]

train_image_labels =all_image_labels[:t_offset]
vaildation_image_labels = all_image_labels[t_offset:v_offset]
test_image_labels = all_image_labels[v_offset:]



# load and preprocess iamge
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels = IMG_CHANNEL)
    image = tf.image.resize(image, [IMG_SIZE,IMG_SIZE])
    # Resize the images to a fixed input size, and rescale the input channels to a range of [-1,1]
    # In this case, [-1, 1] rang, but normalize to [0,1] range
    image = (image/127.5) - 1
    return image

def load_and_preprocess_image(path):
    print(path)
    image = tf.io.read_file(path)
    return preprocess_image(image)

# creat a TF Dataset
# creat a tf.data.Dataset use from_tensor_slices mothods
train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths,train_image_labels))

vaildation_ds = tf.data.Dataset.from_tensor_slices((vaildation_image_paths , vaildation_image_labels))

test_ds = tf.data.Dataset.from_tensor_slices((test_image_paths,test_image_labels))



# 元组被解压缩到映射函数的位置参数中
def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path),label

train_image_label_ds = train_ds.map(load_and_preprocess_from_path_label)
vaildation_image_label_ds = vaildation_ds.map(load_and_preprocess_from_path_label)
test_image_label_ds = test_ds.map(load_and_preprocess_from_path_label)

train_count = len(train_image_paths)

train_batches = train_image_label_ds.shuffle(train_count).batch(BATCH_SIZE)
vaildation_batches = vaildation_image_label_ds.batch(BATCH_SIZE)
test_batches = test_image_label_ds.batch(BATCH_SIZE)

#Create the base model from the pre-trained convnets
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_CHANNEL)

# Create the base model from the pre-trained model MobileNet V2
# instantiate a MobileNet V2 model pre-loaded with weights trained on ImageNet
# By specifying the include_top=False argument
# load a network that doesn't include the classification layers at the top
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


for image_batch, label_batch in train_batches.take(1):
    pass

feature_batch = base_model(image_batch)
print(feature_batch.shape)


# Fine tuning
# only training a few layers on top of an MobileNet V2 base model
# The weights of the pre-trained network were not updated during training
# In most convolutional networks
# the higher up a layer is the more specialized it is
# The goal of fine-tuning is to adapt these specialized features to work
# with the new dataset
# rather than overwrite the generic learning

# Un-freeze the top layers of the model
# unfreeze the base_model and set the bottom layers to be un-trainable
base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 135

# Freeze all layers before the 'fine_tune_at' layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

tf.keras.backend.set_learning_phase(True)

# To generate predictions from the block of features
# average over the spatial 5x5 spatial locations
# using a tf.keras.layers.GlobalAveragePooling2D layer
# to convert the features to a single 1280-element vector per image
# (32, 5, 5, 1280) -> (32, 1280)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

# Apply a tf.keras.layers.Dense layer to convert these features into a single prediction per image
# this prediction will be treated as a logit,or a raw prediction value
# Dense == fc_layer
prediction_layer = tf.keras.layers.Dense(len(label_names))
prediction_batch = prediction_layer(feature_batch_average)

# Now stack the feature extractor, and these two layers using a tf.keras.Sequential model
model = tf.keras.Sequential([
         base_model,
         global_average_layer,
         prediction_layer])

# Compile the model
base_learning_rate = 0.0001
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

print(len(model.trainable_variables))



initial_epochs = TRAIN_EPOCHS
validation_steps = VALIDATION_STEPS

fine_tune_epochs = TRAIN_EPOCHS
total_epochs = initial_epochs + fine_tune_epochs

loss0, accuracy0 = model.evaluate(vaildation_batches, steps = validation_steps)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history_fine = model.fit(train_batches,
                  epochs = total_epochs,
                  validation_data=vaildation_batches,
                  callbacks=[tensorboard_callback])

model.save('tf_savedmodel',save_format='tf')




















