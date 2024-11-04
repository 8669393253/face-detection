import os
import cv2
import numpy as np
from fer import FER

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the DNN model for face detection
model_file =r"C:\Users\tanse\Downloads\res10_300x300_ssd_iter_140000.caffemodel"
config_file ="D:\Desktop\Python\projects\deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

# Initialize the FER emotion detector
emotion_detector = FER()

# Specify the path to your video file
video_path =r"D:\Desktop\All Folders\Tanmay Videos\VID_20230329_170958.mp4" # Change this to your video file path
video_capture = cv2.VideoCapture(video_path)

# Load the Haar Cascade for smile detection
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Prepare the frame for DNN
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Perform face detection
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw rectangle around the face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

            # Region of Interest for smile detection
            face_region = frame[startY:endY, startX:endX]
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            smiles = smile_cascade.detectMultiScale(gray_face, scaleFactor=1.8, minNeighbors=20)

            # Check for smiles
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(face_region, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
                cv2.putText(frame, 'Smile!', (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Emotion detection
            emotions = emotion_detector.detect_emotions(face_region)
            if emotions:
                emotion, score = max(emotions[0]['emotions'].items(), key=lambda x: x[1])
                cv2.putText(frame, f'Emotion: {emotion} ({score:.2f})', (startX, startY + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
