from cv2 import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("detectionModel.h5")

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    resized = cv2.resize(frame, (32, 32))
    np_array = np.array(resized)
    reshaped = np.reshape(np_array, (1, 32, 32, 3))

    res = model.predict(reshaped)
    print(f"Prediction: {labels[np.argmax(res)]}")

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


