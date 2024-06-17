# Python script that opens a GUI to test a machine learning model
# Inspired by RT_GestureRecognition

# Run script to start, press 'q' to exit

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras import models

# Load the saved model
model = models.load_model("convnet_H6k2k2k_061724_best.keras")

# Extract gesture list
gestures = []
for gesture in os.listdir(original_dir):
    gestures.append(gesture)
gestures.remove(".DS_Store")

# Start video capture
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert the captured frame into RGB
    im = Image.fromarray(frame, 'RGB')

    # Resizing into 180x180 because we trained the model with this image size.
    im = im.resize((180, 180))
    img_array = np.array(im)

    # Expand dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image data
    img_array = img_array / 255.0

    # Make predictions
    predictions = model.predict(img_array)[0]

    # Get the top 3 predictions
    top_indices = predictions.argsort()[-3:][::-1]
    top_gestures = [(gestures[i], predictions[i]) for i in top_indices]

    # Display the top 3 predictions on the frame
    for i, (gesture, prob) in enumerate(top_gestures):
        text = f"{gesture}: {prob:.2f}"
        cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Gesture Recognition", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the video capture and destroy all windows
video.release()
cv2.destroyAllWindows()