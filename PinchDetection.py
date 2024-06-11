import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution to improve performance
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Function to get the current volume level
def get_volume():
    output = os.popen("osascript -e 'output volume of (get volume settings)'").read().strip()
    return int(output)

# Function to set the volume level
def set_volume(volume):
    os.system(f"osascript -e 'set volume output volume {volume}'")

# Main loop with frame skipping
frame_skip = 2  # Process every 2nd frame
frame_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    start_time = time.time()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            thumb_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                                  hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y])
            index_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                  hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])

            # Calculate the distance between thumb and index finger
            distance = calculate_distance(thumb_tip, index_tip)

            # Get the current volume
            current_volume = get_volume()

            # Check if the distance is less than a certain threshold (indicating a pinch gesture)
            if distance < 0.05:
                cv2.putText(frame, 'Volume Up', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                set_volume(min(current_volume + 5, 100))
            elif distance > 0.1:
                cv2.putText(frame, 'Volume Down', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                set_volume(max(current_volume - 5, 0))

    # Calculate and display FPS
    end_time = time.time()
    fps = int(1 / (end_time - start_time))
    cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
