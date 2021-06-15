import cv2


import cv2 as cv
import mediapipe as mp
import time as time


mpPose = mp.solutions.pose

pose = mpPose.Pose(static_image_mode=False, # Checking when we are detecting and when we are tracking- True w
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) # tracking only happens after detection. Once conf drop below then it will go back to cheacking detection.

cap = cv.VideoCapture(0)
# cap = cv.VideoCapture("PoseVideos.mp4")


previous_time = 0

# run a while to continuously run the video till we quit
while True:

    # get fram image
    success, img = cap.read()
    


    # Calculate FPS
    current_time = time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time
    cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_COMPLEX, 2, (255,0,0), 3)

    cv.imshow('Image', img)
    cv.waitKey(1)