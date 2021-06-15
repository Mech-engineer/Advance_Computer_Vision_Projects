import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

# Hand Tracking Class from Mediapipe
mpHands = mp.solutions.hands

#hands = mpHands.Hands(static_image_mode = False, max_num_hands = 2, min_detection_confidence = 0.5, min_tracking_confidence = 0.5 )
hands = mpHands.Hands()


# A method to draw lines between points on hand
mpDraw = mp.solutions.drawing_utils

# For frame Rate Calulcation
previous_time = 0
current_time = 0


while True:
    #read the camera image from the webcam
    success, img = cap.read()
    # send rgb image to the object but first convert it into RGB
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # assign the image to hand object
    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks)

    # Check if there is a hand in the vide and we have landmarks
    if results.multi_hand_landmarks:

        #loop hand land marks
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, landmark)

                # this gives ids from 0 to 20
                # and the landmarks are in x,y,z but in decimal which we need in pixel that we can calculate

                h, w, c = img.shape

                # find the position
                # Center
                cx, cy =   int(lm.x*w), int(lm.y*h)

                # print(id, cx, cy)

                # if id == 0: # looking at the first landmark for testing
                #     cv.circle(img, (cx,cy),25,(255,0,255),cv.FILLED)

                # move the extracted information on list



            # Calling the hand tracking function and connection function.
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # we dont want to draw on RGB we wanna do it on BGR
            
    # FPS Calculations
    current_time = time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time

    # Display it on image

    cv.putText(img, str(int(fps)), (10,75), cv.FONT_HERSHEY_COMPLEX, 3, (255,0,0), 2)
    

    # Now to extract the values





    cv.imshow("Image", img)
    cv.waitKey(1) 