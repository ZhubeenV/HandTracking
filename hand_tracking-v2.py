import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = None
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set higher resolution (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Gesture state variables
detected_gesture = None
gesture_start_time = 0
hysteresis_time = 0.5  # seconds

def set_max_hands(num_hands):
    global hands
    hands = mp_hands.Hands(max_num_hands=num_hands)

def detect_gesture(hand_landmarks):
    # Thumb: landmarks 1, 2, 3, 4
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]
    thumb_cmc = hand_landmarks.landmark[1]

    # Other fingers: landmarks 5, 6, 7, 8 (index), 9, 10, 11, 12 (middle), 13, 14, 15, 16 (ring), 17, 18, 19, 20 (pinky)
    index_mcp = hand_landmarks.landmark[5]
    index_tip = hand_landmarks.landmark[8]

    middle_mcp = hand_landmarks.landmark[9]
    middle_tip = hand_landmarks.landmark[12]

    ring_mcp = hand_landmarks.landmark[13]
    ring_tip = hand_landmarks.landmark[16]

    pinky_mcp = hand_landmarks.landmark[17]
    pinky_tip = hand_landmarks.landmark[20]

    # Check if thumb is above the thumb MCP and thumb IP is above thumb CMC
    is_thumb_up = (thumb_tip.y < thumb_ip.y < thumb_mcp.y < thumb_cmc.y)

    # Check if other fingers are folded
    is_index_folded = index_tip.y > index_mcp.y
    is_middle_folded = middle_tip.y > middle_mcp.y
    is_ring_folded = ring_tip.y > ring_mcp.y
    is_pinky_folded = pinky_tip.y > pinky_mcp.y

    if is_thumb_up and is_index_folded and is_middle_folded and is_ring_folded and is_pinky_folded:
        return "Thumbs Up"
    return None

def main():
    global detected_gesture, gesture_start_time, hands
    while True:
        success, img = cap.read()
        if not success:
            break

        # Lazy initialization of MediaPipe Hands
        if hands is None:
            hands = mp_hands.Hands(max_num_hands=2)

        # Convert the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        results = hands.process(img_rgb)

        # Draw hand landmarks and detect gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = detect_gesture(hand_landmarks)
                if gesture:
                    current_time = time.time()
                    if gesture == detected_gesture:
                        # Check if the gesture persists for the hysteresis time
                        if current_time - gesture_start_time >= hysteresis_time:
                            h, w, c = img.shape
                            cx, cy = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
                            cv2.putText(img, gesture, (cx, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        detected_gesture = gesture
                        gesture_start_time = current_time
                else:
                    detected_gesture = None

        # Display the image
        cv2.imshow('Hand Tracking', img)

        # Break the loop on 'q' key press or window close
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Hand Tracking', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        num_hands = input("Enter the number of hands to track (default is 2): ")
        if num_hands.strip() == "":
            num_hands = 2
        else:
            num_hands = int(num_hands)
    except ValueError:
        print("Invalid input. Defaulting to 2 hands.")
        num_hands = 2

    set_max_hands(num_hands)
    main()
