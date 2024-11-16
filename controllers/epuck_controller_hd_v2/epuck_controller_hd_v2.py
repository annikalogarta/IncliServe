from controller import Robot, Motor
import cv2 as cv
import numpy as np
import pyautogui as pya
import mediapipe as mp
import threading

# Webots Robot Setup
robot = Robot()

TIME_STEP = 64
MAX_SPEED = 6.28

leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# Gesture Control Setup
cap = cv.VideoCapture(0)
global_forward = False
global_backward = False
global_stop = False
global_turnLeft = False
global_turnRight = False

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# Function for camera processing and gesture detection
def process_gesture():
    global global_forward, global_backward, global_turnRight, global_turnLeft
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

                index_finger = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y]
                thumb = [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y]
                middle_finger = [hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y]

                pointingBool = any(
                    (index_finger[i] < thumb[i] < middle_finger[i]) or
                    (index_finger[i] > thumb[i] > middle_finger[i])
                    for i in range(2))

                if pointingBool and (index_finger[1] < thumb[1]):
                    global_forward = True
                    hand_gesture = 'Pointing up - going forward'
                elif pointingBool and (index_finger[1] > thumb[1]):
                    global_backward = True
                    hand_gesture = 'Pointing down - going backward'
                else:
                    hand_gesture = 'No gesture detected'

                cv.putText(frame, hand_gesture, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        # Display the resulting frame
        cv.imshow('Live Video', frame)

        # Break the loop when 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# Function for controlling robot's motion based on gestures
def control_robot():
    while True:
        if global_forward:
            leftMotor.setVelocity(0.2 * MAX_SPEED)
            rightMotor.setVelocity(0.2 * MAX_SPEED)
        elif global_backward:
            leftMotor.setVelocity(-0.2 * MAX_SPEED)
            rightMotor.setVelocity(-0.2 * MAX_SPEED)
        elif global_turnRight:
            leftMotor.setVelocity(0.2 * MAX_SPEED)
            rightMotor.setVelocity(-0.2 * MAX_SPEED)
        elif global_turnLeft:
            leftMotor.setVelocity(0.2 * MAX_SPEED)
            rightMotor.setVelocity(-0.2 * MAX_SPEED)
        else:
            leftMotor.setVelocity(0.0)
            rightMotor.setVelocity(0.0)
        robot.step(TIME_STEP)

# Start the threads
gesture_thread = threading.Thread(target=process_gesture)
control_thread = threading.Thread(target=control_robot)

gesture_thread.start()
control_thread.start()

gesture_thread.join()
control_thread.join()

cap.release()
cv.destroyAllWindows()