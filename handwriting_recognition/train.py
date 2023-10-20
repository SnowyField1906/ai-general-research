import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'data/train'
test_data_dir = 'data/test'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(28, 28),
    batch_size=32,
    class_mode='sparse', 
    color_mode='grayscale'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(28, 28),
    batch_size=32,
    class_mode='sparse', 
    color_mode='grayscale'
)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(32, (3, 3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation('softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=32, validation_data=(x_test, y_test))
model.fit(train_generator, epochs=32, steps_per_epoch=len(train_generator))

accuracy = model.evaluate(test_generator, steps=len(test_generator))
print("Custom Data Accuracy:", accuracy[1])

model.save('digits.model')