import cv2 as cv
import numpy as np
import pyautogui as pya
import mediapipe as mp

cap = cv.VideoCapture(0)

global global_forward
global global_backward
global global_stop
global global_turnLeft
global global_turnRight

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

while True:

    global_forward = False
    global_backward = False
    global_stop = False
    global_turnLeft = False
    global_turnRight = False

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
    
    index_finger = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y]
    thumb = [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y]
    middle_finger = [hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y]
    ring_finger = [hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x , hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y ]

    pointingBool = any(
    (index_finger[i] < thumb[i] < middle_finger[i]) or
    (index_finger[i] > thumb[i] > middle_finger[i])
    for i in range(2))

    if pointingBool and (index_finger[1] < thumb[1]):
        global_forward = True
        hand_gesture = 'Pointing up - going forward'
    elif pointingBool and (index_finger[1] > thumb[1]):
        global_backward = True
        hand_gesture = 'pointing down - going backward'
    else:
        hand_gesture = 'No gesture detected'

    
    cv.putText(frame, hand_gesture, (10,60), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1 )
    
    # Display the resulting frame
    cv.imshow('Live Video', frame)

    # Break the loop when 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()