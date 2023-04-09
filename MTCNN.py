
from mtcnn import MTCNN
import cv2

# Load the MTCNN face detection model
detector = MTCNN()

# Load the image
img = cv2.imread(
    r"test_img.jpeg")

# Detect faces in the image
faces = detector.detect_faces(img)

# Draw rectangles around the detected faces
for face in faces:
    x, y, w, h = face['box']
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the result
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
