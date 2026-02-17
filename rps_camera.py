import cv2
import mediapipe as mp
import random
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

CHOICES = ["rock", "paper", "scissors"]

def computer_choice():
    return random.choice(CHOICES)

def winner(user, comp):
    if user == comp:
        return "Tie"
    if (user == "rock" and comp == "scissors") or \
       (user == "paper" and comp == "rock") or \
       (user == "scissors" and comp == "paper"):
        return "You Win"
    return "Computer Wins"

def finger_states(hand_landmarks):
    # tip ids in MediaPipe Hands
    tips = [4, 8, 12, 16, 20]

    lm = hand_landmarks.landmark

    # Thumb (different axis)
    thumb_open = lm[tips[0]].x < lm[tips[0] - 1].x  # for right hand approx

    # Other fingers: tip above pip => open
    fingers = []
    fingers.append(thumb_open)
    for i in range(1, 5):
        fingers.append(lm[tips[i]].y < lm[tips[i] - 2].y)

    return fingers  # [thumb, index, middle, ring, pinky]

def detect_rps(fingers):
    # fingers = [thumb, index, middle, ring, pinky] booleans

    # Rock: all closed (sometimes thumb may vary)
    if (not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]):
        return "rock"

    # Paper: all open
    if fingers[1] and fingers[2] and fingers[3] and fingers[4]:
        return "paper"

    # Scissors: index + middle open, ring + pinky closed
    if fingers[1] and fingers[2] and (not fingers[3]) and (not fingers[4]):
        return "scissors"

    return None

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not opening. Try closing other camera apps.")
        return

    user_score = 0
    comp_score = 0

    last_play_time = 0
    play_cooldown = 2.0  # seconds between rounds

    comp_move = "-"
    user_move = "-"
    result = "Show gesture"

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = hands.process(rgb)

            move = None

            if out.multi_hand_landmarks:
                handLms = out.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                fingers = finger_states(handLms)
                move = detect_rps(fingers)

            now = time.time()

            # Play a round when gesture is clearly detected and cooldown passed
            if move is not None and (now - last_play_time) > play_cooldown:
                user_move = move
                comp_move = computer_choice()
                result = winner(user_move, comp_move)

                if result == "You Win":
                    user_score += 1
                elif result == "Computer Wins":
                    comp_score += 1

                last_play_time = now

            # UI text
            cv2.rectangle(frame, (0, 0), (640, 140), (0, 0, 0), -1)
            cv2.putText(frame, f"Your Move: {user_move}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Computer: {comp_move}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Result: {result}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.putText(frame, f"Score You:{user_score}  Computer:{comp_score}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.putText(frame, "Press Q to quit", (10, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Rock Paper Scissors - Camera", frame)

            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
