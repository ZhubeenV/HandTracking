import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # Default to tracking 2 hands
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set higher resolution (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def set_max_hands(num_hands):
    global hands
    hands = mp_hands.Hands(max_num_hands=num_hands)

def main():
    while True:
        success, img = cap.read()
        if not success:
            break

        # Convert the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        results = hands.process(img_rgb)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the image
        cv2.imshow('Hand Tracking', img)

        # Break the loop on 'q' key press or window close
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Check if the window is closed
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

