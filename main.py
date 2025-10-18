import cv2
from deepface import DeepFace
from time import sleep
import json

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Get settings
try:
    print("Attempting to read settings file")
    with open('settings.json', 'r') as file:
        settings = json.load(file)
except:
    #If a settings file has not been created. Use the default settings.
    print("Settings file does not exist. Falling back to default settings")
    settings = dict(sadness_detection = True, anger_detection = True, sadness_music = "Yoga is good for your mental health!", angry_music = "You should touch grass :)")

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        
        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']

        print(emotion)#For development purposes
        
        if emotion == 'sad' and settings['sadness_detection'] == True:
            print("play music")
        elif emotion == 'angry' or emotion == 'fear' and settings['anger_detection'] == True:
            print("play music!")

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)#For development purposes

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    sleep(1)

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()