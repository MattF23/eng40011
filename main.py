import cv2
from picamera2 import Picamera2
from deepface import DeepFace
from time import sleep
import json
from os import system
from playsound import playsound

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
cap = Picamera2()
cap.start()

#Set volume
system('amixer sset Master ' + settings['Volume'])

while True:
    # Capture frame-by-frame
    frame = cap.capture_array()

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
        try:
            if emotion == 'sad' and settings['sadness_detection'] == True:
                #playsound(settings['sadness_music'] + ".mp3")
                print("Sad detected")
            elif emotion == 'angry' or emotion == 'fear' and settings['anger_detection'] == True:
                #playsound(settings["angry_music"] + ".mp3")
                print("Anger detected")
            elif emotion == 'happy' and settings['happiness_detection'] == True:
                #playsound(settings["happy_music"] + ".mp3")
                print("Happiness detected")
        except:
            print("Can't connect to your speaker, check your cabeling!")
        finally:
            continue

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