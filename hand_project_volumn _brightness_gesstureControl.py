import cv2
import numpy as np
import mediapipe as mp
import time
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc
import pyautogui



mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
a=""
cl=()

cap = cv2.VideoCapture(0)   

with mp_hands.Hands(max_num_hands=2) as hands:  # Set max number of hands to 2
    while cap.isOpened():
        success, image = cap.read()

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            # WORK OF ONE HAND
            if num_hands==1:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Check if the hand is on the left or right side of the image
                    if hand_landmarks.landmark[0].x < hand_landmarks.landmark[20].x and hand_landmarks.landmark[4].x < hand_landmarks.landmark[0].x:
                        cl = (255, 0, 0)  # right hand
                        a="right"
                    if hand_landmarks.landmark[0].x > hand_landmarks.landmark[20].x and hand_landmarks.landmark[4].x > hand_landmarks.landmark[0].x:
                        cl = (0, 255, 0)  # left hand
                        a="left"

                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=cl, thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=cl, thickness=2))
                    
                    # get positions of landmarks 4 and 8
                    x4, y4 = int(hand_landmarks.landmark[4].x * image.shape[1]), int(hand_landmarks.landmark[4].y * image.shape[0])
                    x8, y8 = int(hand_landmarks.landmark[8].x * image.shape[1]), int(hand_landmarks.landmark[8].y * image.shape[0])

                    # draw a line between landmarks 4 and 8
                    cv2.line(image, (x4, y4), (x8, y8), (0, 0, 255), 1)
                    
                    # distance
                    d=math.sqrt(((x4-x8)**2)+((y4-y8))**2)
                    
                    
                    
                    if a=="right":
                        # setting up the volume
                        try:
                            devices = AudioUtilities.GetSpeakers()
                            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                            volume = cast(interface, POINTER(IAudioEndpointVolume))
                            r=volume.GetVolumeRange()
                            # interpolation of dataset using interp
                            y = np.interp(np.power(d, 1.275),[20,470],[r[0],r[1]])
                            volume.SetMasterVolumeLevel(y, None)
                            
                        except Exception as e:
                            pass
                    elif a=="left":
                        # setting up the brightness
                        try:
                            y = np.interp(np.power(d, 1.275),[20,450],[0,100])
                            sbc.set_brightness(y)   
                        except Exception as e:
                            pass
            else:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Check if the hand is on the left or right side of the image
                    if hand_landmarks.landmark[0].x < hand_landmarks.landmark[20].x and hand_landmarks.landmark[4].x < hand_landmarks.landmark[0].x:
                        cl = (255, 0, 0)  # right hand
                        a="right"
                    if hand_landmarks.landmark[0].x > hand_landmarks.landmark[20].x and hand_landmarks.landmark[4].x > hand_landmarks.landmark[0].x:
                        cl = (0, 255, 0)  # left hand
                        a="left"

                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=cl, thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=cl, thickness=2))
                

        
        # Display the resulting image
        
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
