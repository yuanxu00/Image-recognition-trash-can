from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import random

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

AUTOTUNE = tf.data.experimental.AUTOTUNE

IMG_SIZE = 192
IMG_CHANNEL = 3
BATCH_SIZE = 32
TRAIN_PERCENTAGE = 0.8
TRAIN_EPOCHS = 100
VALIDATION_STEPS = 20

data_root_orig = '/home/y/Documents/training_dataset'
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

# Feature extraction

# Freeze the convolutional base
# Freezing (by setting layer.trainable = False) prevents the weights
# in a given layer from being updated during training
base_model.trainable = False

# Let's take a look at the base model architecture
base_model.summary()

# To generate predictions from the block of features
# average over the spatial 5x5 spatial locations
# using a tf.keras.layers.GlobalAveragePooling2D layer
# to convert the features to a single 1280-element vector per image
# (32, 5, 5, 1280) -> (32, 1280)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

# Apply a tf.keras.layers.Dense layer to convert these features into a single prediction per image
# this prediction will be treated as a logit,or a raw prediction value
prediction_layer = tf.keras.layers.Dense(len(label_names))
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

# Now stack the feature extractor, and these two layers using a tf.keras.Sequential model
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer])

# Compile the model
# You must compile the model before training it
# Since more than two classes
# use sparse_catagorical_crossentropy
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=["accuracy"])

model.summary()
len(model.trainable_variables)

initial_epochs = TRAIN_EPOCHS
validation_steps = VALIDATION_STEPS

loss0, accuracy0 = model.evaluate(vaildation_batches, steps = validation_steps)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_batches,
                    epochs = initial_epochs,
                    validation_data = vaildation_batches)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


