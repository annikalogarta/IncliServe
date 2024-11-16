import cv2 as cv
import numpy as np
import pyautogui as pya
import mediapipe as mp

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
        index_finger_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

        if index_finger_y < thumb_y:
            hand_gesture = 'pointing up'
        elif index_finger_y > thumb_y:
            hand_gesture = 'pointing down'
        else:
            hand_gesture = 'other'

        
        cv.putText(frame, hand_gesture, (10,60), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1 )
        
    # Display the resulting frame
    cv.imshow('Live Video', frame)

    # Break the loop when 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()