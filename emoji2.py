import os
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128,kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128,kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024,activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7,activation='softmax'))
emotion_model.load_weights('prj.h5')

emotion_dict = {0: 'Angry', 1: 'Sad', 2: 'Surprise', 3: 'Happy', 4: 'Neutral', 5: 'Disgusted', 6: 'Fear'}

# Map emotion labels to emoji characters
cur_path = os.path.dirname(os.path.abspath(_file_))
emoji_dist={0:cur_path+"/emojis/angry.png",
            1:cur_path+"/emojis/sad.png", 
            2:cur_path+"/emojis/surprise.png",
            3:cur_path+"/emojis/happy.png", 
            4:cur_path+"/emojis/neutral.png",
            5:cur_path+"/emojis/disgust.png",
            6:cur_path+"/emojis/fear.png"}

# Initialize webcam
cap = cv2.VideoCapture(1)

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("C:/Users/KIIT/Desktop/Emojify/haar cascade. xml")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Process each face in the image
    for (x, y, w, h) in faces:
        # Extract the face region from the image
        face = gray[y:y+h, x:x+w]

        # Resize the face image to 48x48 pixels (same size as the input of the model)
        face = cv2.resize(face, (48, 48))

        # Reshape the face image to a 4D tensor with shape (1, 48, 48, 1)
        face = face.reshape(1, 48, 48, 1) / 255.0

        # Use the model to predict the emotion of the face
        prediction = emotion_model.predict(face)
        emotion_label = np.argmax(prediction)
        emotion_emoji_path = emoji_dist[emotion_label]
        emoji_img = cv2.imread(emotion_emoji_path)

        # Resize the emoji image to a smaller size and display it on the frame
        emoji_img = cv2.resize(emoji_img, (h, w))
        frame[y:y+h, x:x+w] = emoji_img
        
    # Display the frame with the detected faces and predicted emotions
    cv2.imshow('Emotion-Emoji',frame)
    
    # Check for the 'q' key to exit the program
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()