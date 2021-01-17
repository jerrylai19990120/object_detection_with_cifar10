from cv2 import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("detectionModel.h5")

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

img = cv2.imread('DATA/cat.jpg')

img = cv2.resize(img, (32, 32))
img = np.array(img)
img = np.reshape(img, (1, 32, 32, 3))

res = model.predict(img)
print(f"Prediction: {labels[np.argmax(res)]}")


