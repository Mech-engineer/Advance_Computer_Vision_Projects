import cv2 as cv
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon =0.5, trackCon = 0.5 ):

        #initialize variables for the mediapipe hand module
        self.mode = mode
        self.MaxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon


        # Hand Tracking Class from Mediapipe
        self.mpHands = mp.solutions.hands
        #hands = mpHands.Hands(static_image_mode = False, max_num_hands = 2, min_detection_confidence = 0.5, min_tracking_confidence = 0.5 )
        self.hands = self.mpHands.Hands(self.mode, self.MaxHands, self.detectionCon, self.trackCon)
        # A method to draw lines between points on hand
        self.mpDraw = mp.solutions.drawing_utils



    # Now define a function to find hands
    def findHand(self, img, draw = True):

        # send rgb image to the object but first convert it into RGB
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # assign the image to hand object result which is shared with all the functions (example findPosition)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        # Check if there is a hand in the vide and we have landmarks
        if self.results.multi_hand_landmarks:

            #loop hand landmarks 
            for handLms in self.results.multi_hand_landmarks:

                # If Draw is true then draw landmarks on the Image
                if draw: 
                    # Calling the hand tracking function and connection function.
                    # Note: we dont want to draw on RGB we wanna do it on BGR that we are capturing
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) 
        return img



    def findPosition(self, img, handNo = 0, draw = True ):
        
        # Create a list to store the landmarks
        lmList = []

        # Check if there is a hand in the vide and we have landmarks
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # print(id, landmark)

                # this gives ids from 0 to 20 as we have 21 landmarks for hand
                # and the landmarks are in x,y,z but in decimal which we need in pixel that we can calculate
                h, w, c = img.shape

                # find the position of the center                
                cx, cy =   int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                
                if draw:
                    # if id == 0: # looking at the first landmark for testing
                    cv.circle(img, (cx,cy),15,(255,0,255),cv.FILLED)

                lmList.append([id, cx, cy])

                # move the extracted information on list

        return lmList


def main():

    # Capture video from the webcam
    cap = cv.VideoCapture(0)
    
    # For frame Rate Calulcation
    previous_time = 0
    current_time = 0
    detector = handDetector()

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


if __name__ == "__main__":
    main()