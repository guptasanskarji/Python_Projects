# import tensorflow as tf
# import tensorflow_hub as hub
# import numpy as np
# import cv2

# # Load the pre-trained model from TensorFlow Hub
# model_url = "https://tfhub.dev/tensorflow/efficientdet/d0/1"
# model = hub.load(model_url).signatures['default']

# # Load and preprocess the image (replace 'image_path' with your image file)
# image_path = "lung_ct_scan.jpg"
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]

# # Perform inference
# results = model(image)

# # Process the detection results
# scores = results['detection_scores'][0]
# boxes = results['detection_boxes'][0]

# # Set a confidence threshold
# confidence_threshold = 0.5
# selected_indices = np.where(scores > confidence_threshold)

# # Display detected boxes
# for i in selected_indices:
#     ymin, xmin, ymax, xmax = boxes[i][0].numpy()
#     h, w, _ = image.shape
#     x1, y1, x2, y2 = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
#     cv2.rectangle(image[0].numpy(), (x1, y1), (x2, y2), (0, 255, 0), 2)

# # Display the result image
# cv2.imshow("Organ Detection Result", image[0].numpy())
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load an image for face detection
image_path = 'sanskar3.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# mpHands = mp.solutions.hands
# hands = mpHands.Hands()
# mpDraw = mp.solutions.drawing_utils

# while True:
#     success, image = cap.read()
#     imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands.process(imageRGB)

# if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks: # working with each hand
#             for id, lm in enumerate(handLms.landmark):
#                 h, w, c = image.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)

#             if id == 20 :
#                 cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

#         mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

# cv2.imshow("Output", image)
# cv2.waitKey(1)

# Perform face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Draw rectangles around detected hands


# Save the output image
output_image_path = 'output_image.jpg'
cv2.imwrite(output_image_path, image)

# Display the output image
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
