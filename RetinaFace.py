import cv2
import numpy as np
from retinaface import RetinaFace

# Load the RetinaFace face detection model
detector = RetinaFace(quality="normal")

# Load the image
img = cv2.imread(
    r"test_img.jpeg")

# Convert image to RGB format
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect faces in the image
faces = detector.predict(img)

# Draw rectangles around the detected faces
for face in faces:
    x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the result
cv2.imshow('Image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
