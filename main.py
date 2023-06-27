from cvzone.FaceDetectionModule import FaceDetector
import cv2
import numpy as np
from keras.models import load_model
import os

path = 'Pneumothorax-New-Dataset'

img = cv2.imread("Pneumothorax-New-Dataset/1_image_100.jpg")

# Load the age detection model
model = load_model('model_10epochs.h5', compile=False)

try:    
    resizedImg = cv2.resize(img, (200, 200))
    resizedImg = np.array([resizedImg])

    # Pridict the age using age detection model and save it in variable named prediction
    prediction = model.predict(resizedImg)

   
    img = cv2.rectangle(img, (10, 10), (300, 100), (255, 255, 255), -1)
    if prediction[0][0] < 0.5:
        text = "infected"
    else:
        text ="Uninfected"
    
    print(prediction)
    # Add prediction[0][0] i.e age of the detected face on the screen  
    img = cv2.resize(img, (300, 400))  
    img = cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.4, (225, 0, 0), 1)
      
    cv2.imshow("Image", img)
except Exception as e:
    print(e)
cv2.waitKey(0)