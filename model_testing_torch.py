# Python script that opens a GUI to test a machine learning model
# Inspired by RT_GestureRecognition

# Run script to start, press 'q' to exit

import cv2
import numpy as np
from PIL import Image
import torch
import json
from torchvision import transforms
import torch.nn.functional as F
import timm

# Extract gesture list
with open('cjm_image_classifier/2024-06-28_01-04-01/hagrid-classification-512p-no-gesture-150k-zip-classes.json') as data_file:
	config = json.load(data_file)

gestures=config['classes']

# Load the saved model
model = timm.create_model('fastvit_s12.apple_dist_in1k', pretrained=False, num_classes=len(gestures))

state_dict = torch.load("cjm_image_classifier/2024-06-28_01-04-01/fastvit_s12.apple_dist_in1k.pth")
model.load_state_dict(state_dict)
model.eval()

# Start video capture
video = cv2.VideoCapture(0)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert the captured frame into RGB
    frame = cv2.flip(frame, 1)
    im = Image.fromarray(frame, 'RGB')

    # Resizing into 180x180 because we trained the model with this image size.
    img_tensor = transform(im).unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        outputs = model(img_tensor)    
        predictions = F.softmax(outputs, dim=1)[0]
    
    predictions = predictions.numpy()

    # Get the top 3 predictions
    # top_indices = predictions.argsort(descending=True)[:3]
    # top_gestures = [(gestures[i], predictions[i]) for i in top_indices]

    # Display the top 3 predictions on the frame
    #for i, (gesture, prob) in enumerate(top_gestures):
    #    text = f"{gesture}: {prob:.2f}"
    #    cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    #cv2.imshow("Gesture Recognition", frame)

    #key = cv2.waitKey(1)
    #if key == ord('q'):
    #    break

    value = np.max(predictions)
    indices = np.argmax(predictions)
    top_3 = predictions.argsort()[-3:][::-1]

    # if value > 0.6 and indices < len(gestures):
    #    print('Gesture:', gestures[indices], '\t\t\t\t\t\t Value: {:.2f}'.format(value))
    pred = indices

    bg = np.full((480, 1200, 3), 15, np.uint8)
    resized_frame_for_bg = cv2.resize(frame, (640, 480))
    bg[:480, :640] = resized_frame_for_bg

    font = cv2.FONT_HERSHEY_SIMPLEX
    if value > 0.6:
        cv2.putText(bg, gestures[pred], (700, 100), font, 1, (0, 0, 0), 2)
    cv2.rectangle(bg, (128, 48), (640-128, 480-48), (0, 255, 0), 3)
    for i, top in enumerate(top_3):
        cv2.putText(bg, gestures[top], (700, 200+70*i), font, 1, (255, 255, 255), 1)
        cv2.rectangle(bg, (700, 225+70*i), (int(700 + predictions[top] * 170), 205+70*i), (255, 255, 255), 3)

    cv2.imshow('Gesture Recognition', bg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
video.release()
cv2.destroyAllWindows()