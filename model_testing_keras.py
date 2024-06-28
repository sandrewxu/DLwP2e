# Python script that opens a GUI to test a machine learning model
# Inspired by RT_GestureRecognition

# Run script to start, press 'q' to exit

import cv2
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os, pathlib

# Extract gesture list
original_dir = pathlib.Path("data/hagrid_dataset_512")
gestures = []
for gesture in os.listdir(original_dir):
    gestures.append(gesture)
gestures.remove(".DS_Store")

# Capture video from computer camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS, 1)

# Load Keras model
print('Loading model...')
model = keras.models.load_model('convnet_H6k2k2k_061724_best.keras')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Transformation pipeline
def transform(image):
    # CenterCrop and Normalize
    image = image.resize((180,180))  # Center crop
    img_array = image.img_to_array(image)
    img_tensor = np.expand_dims(img_array, axis=0)
    img_tensor /= 255.0
    return img_tensor

print('Starting prediction')

plt.ion()
fig, ax = plt.subplots()
cooldown = 0
num_classes = 18

while True:
    # Capture frame-by-frame
    ret, frame = cam.read()
    pre_img = Image.fromarray(frame.astype('uint8'), 'RGB')
    img = transform(pre_img)

    # Prepare the image for the model
    output = model.predict(img)
    print(output)
    out = tf.nn.softmax(output[0]).numpy()

    value = np.max(out)
    indices = np.argmax(out)
    top_3 = out.argsort()[-3:][::-1]

    if cooldown > 0:
        cooldown -= 1
    if value > 0.6 and indices < 25 and cooldown == 0:
        print('Gesture:', gestures[indices], '\t\t\t\t\t\t Value: {:.2f}'.format(value))
        cooldown = 16
    pred = indices

    bg = np.full((480, 1200, 3), 15, np.uint8)
    resized_frame_for_bg = cv2.resize(frame, (640, 480))
    bg[:480, :640] = resized_frame_for_bg

    font = cv2.FONT_HERSHEY_SIMPLEX
    if value > 0.6:
        cv2.putText(bg, gestures[pred], (40, 40), font, 1, (0, 0, 0), 2)
    cv2.rectangle(bg, (128, 48), (640-128, 480-48), (0, 255, 0), 3)
    for i, top in enumerate(top_3):
        cv2.putText(bg, gestures[top], (700, 200-70*i), font, 1, (255, 255, 255), 1)
        cv2.rectangle(bg, (700, 225-70*i), (int(700 + out[top] * 170), 205-70*i), (255, 255, 255), 3)

    cv2.imshow('preview', bg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
