import cv2 as cv
import mediapipe as mp
import time

import HandTrackingModule as htm


# Capture video from the webcam
cap = cv.VideoCapture(0)

# For frame Rate Calulcation
previous_time = 0
current_time = 0
detector = htm.handDetector()

while True:
    #read the camera image from the webcam
    success, img = cap.read()
    img = detector.findHand(img)   
    lmlist = detector.findPosition(img, draw = False)

    # if len(lmlist)!= 0:
    #     print(lmlist[4])
    

    # FPS Calculations
    current_time = time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time

    # Display it on image
    cv.putText(img, str(int(fps)), (10,75), cv.FONT_HERSHEY_COMPLEX, 3, (255,0,0), 2)
    
    cv.imshow("Image", img)
    cv.waitKey(1)