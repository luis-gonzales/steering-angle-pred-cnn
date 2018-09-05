from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D
from keras.models import Sequential

# CNN model borrowed from NVIDIA at:
# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

# Preprocess including resizing and converting from RGB to YUV
def preprocess(img):
  import tensorflow as tf
  new_img = tf.image.resize_images(img, size=[66,200])
  new_img = tf.divide(new_img, 255)
  new_img = tf.image.rgb_to_yuv(new_img)
  return new_img


def get_model():
  act = 'relu'
  init = 'he_normal'

  model = Sequential()

  #Preprocess
  model.add(Cropping2D(cropping=((55,25), (0,0)), input_shape=(160,320,3)))
  model.add(Lambda(preprocess))

  # Convolutional layers
  model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), padding='valid', activation=act, kernel_initializer=init))
  model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), padding='valid', activation=act, kernel_initializer=init))
  model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), padding='valid', activation=act, kernel_initializer=init))
  model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid', activation=act, kernel_initializer=init))
  model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid', activation=act, kernel_initializer=init))
  model.add(Flatten())

  # Fully-connected layers, returning single output
  model.add(Dense(100, activation=act, kernel_initializer=init))
  model.add(Dense(50, activation=act, kernel_initializer=init))
  model.add(Dense(10, activation=act, kernel_initializer=init))
  model.add(Dense(1, kernel_initializer=init))
  return model
