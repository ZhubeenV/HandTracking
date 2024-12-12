import cv2
import mediapipe as mp
import pyautogui as pag
import numpy as np

# based on patreon

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mouseDown = False

hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.1,
                        min_tracking_confidence=0.1)  

mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

screen_width, screen_height = pag.size()

while True:

    sucess, frame = cap.read()
    if not sucess:
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1) # flip the frame horizontally if needed

    # Convert the image to RGB from BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    frame_height, frame_width, _ = frame.shape
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            midpoint_x = (index_finger_tip.x + thumb_tip.x) / 2
            midpoint_y = (index_finger_tip.y + thumb_tip.y) / 2

            distance = np.sqrt((index_finger_tip.x - thumb_tip.x)**2 + (index_finger_tip.y - thumb_tip.y)**2) 

            if distance < 0.1 and not mouseDown:
                pag.mouseDown()
                mouseDown = True
            if distance > 0.2 and mouseDown:
                pag.mouseUp()
                mouseDown = False

            if mouseDown:
                cv2.circle(frame, (int(midpoint_x * frame_width), int(midpoint_y * frame_height)), 10, (0, 255, 0), -1)

            else:
                cv2.circle(frame, (int(midpoint_x * frame_width), int(midpoint_y * frame_height)), 10, (0, 0, 255), 1)

            # Map position to screen rexs
            x_mapped = np.interp(midpoint_x, (0, 1), (0, screen_width))
            y_mapped = np.interp(midpoint_y, (0, 1), (0, screen_height))

            pag.moveTo(x_mapped, y_mapped, duration=0.1) # this should close out the app if i move the mouse to the top left corner of the screen

    cv2.imshow("Mediapipe Hands", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()


            # for id, lm in enumerate(hand_landmarks.landmark):
            #     h, w, c = frame.shape
            #     cx, cy = int(lm.x * w), int(lm.y * h)
            #     cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)