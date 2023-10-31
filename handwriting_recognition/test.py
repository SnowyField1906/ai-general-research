import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

model = tf.keras.models.load_model('digits.model')

test_data_dir = 'data/test/'  # Path to your test data directory

for label in os.listdir(test_data_dir):
    label_dir = os.path.join(test_data_dir, label)
    for img_filename in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_filename)
        
        raw_img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        img = cv.resize(raw_img, (28, 28))
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        predicted_label = np.argmax(prediction)
        confidences = [f'{prediction[0][i] * 100}%' for i in range(len(prediction[0]))]
        
        print(f"Image: {img_filename}")
        print(f"Predicted Label: {predicted_label}")
        print(f"Confidences: {confidences}")
        
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
