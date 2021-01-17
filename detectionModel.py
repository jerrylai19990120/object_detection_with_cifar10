from cv2 import cv2
import numpy as np
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


#x_train = x_train.reshape(50000, 28, 28, 1)
#x_test = x_test.reshape(10000, 28, 28, 1)


y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, 3, input_shape=(32, 32, 3), activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=2))

model.add(tf.keras.layers.Conv2D(64, 3, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(50, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)

model.save("detectionModel.h5")

