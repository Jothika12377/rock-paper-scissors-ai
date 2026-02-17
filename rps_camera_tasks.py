import time
import random
from collections import deque, Counter
import cv2
import mediapipe as mp

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

def fingers_open_states(lm):
    """
    Returns booleans for index/middle/ring/pinky open.
    (Thumb is skipped because it is unreliable with rotation.)
    """
    # Tip ids: index 8, middle 12, ring 16, pinky 20
    # PIP ids: index 6, middle 10, ring 14, pinky 18
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    open_fingers = []
    for tip_id, pip_id in zip(tips, pips):
        # finger open if tip is higher (smaller y) than PIP by a small margin
        open_fingers.append(lm[tip_id].y < lm[pip_id].y - 0.02)
    return open_fingers  # [index, middle, ring, pinky]

def detect_rps_from_open(open4):
    """
    open4 = [index, middle, ring, pinky] booleans
    """
    idx, mid, ring, pinky = open4

    # Paper: all 4 open
    if idx and mid and ring and pinky:
        return "paper"

    # Scissors: index & middle open, ring & pinky closed
    if idx and mid and (not ring) and (not pinky):
        return "scissors"

    # Rock: all 4 closed (or mostly closed)
    if (not idx) and (not mid) and (not ring) and (not pinky):
        return "rock"

    return None

def draw_landmarks(frame, lm):
    h, w = frame.shape[:2]
    for p in lm:
        cx, cy = int(p.x * w), int(p.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

def main():
    model_path = r"models\hand_landmarker.task"

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Camera not opening.")
        return

    user_score = 0
    comp_score = 0

    # Countdown control
    counting = False
    start_time = 0

    # Buffer of detected moves during countdown
    move_buffer = deque(maxlen=30)  # about ~1 sec buffer depending on fps

    user_move = "-"
    comp_move = "-"
    result_text = "Press SPACE to start"
    live_detect = "-"

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("❌ Could not read camera frame.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)

            res = landmarker.detect_for_video(mp_image, timestamp_ms)

            move = None
            if res.hand_landmarks:
                lm = res.hand_landmarks[0]
                draw_landmarks(frame, lm)

                open4 = fingers_open_states(lm)
                move = detect_rps_from_open(open4)

            live_detect = move if move else "-"

            # If counting, collect moves into buffer
            if counting and move:
                move_buffer.append(move)

            # Countdown handling
            countdown_number = None
            if counting:
                elapsed = time.time() - start_time
                if elapsed >= 3.0:
                    counting = False

                    # pick most common move from buffer
                    if len(move_buffer) > 0:
                        user_move = Counter(move_buffer).most_common(1)[0][0]
                        comp_move = computer_choice()
                        result_text = winner(user_move, comp_move)

                        if result_text == "You Win":
                            user_score += 1
                        elif result_text == "Computer Wins":
                            comp_score += 1
                    else:
                        result_text = "Gesture not detected (try again)"

                    move_buffer.clear()
                else:
                    # show 3,2,1
                    countdown_number = 3 - int(elapsed)

            # UI
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 190), (0, 0, 0), -1)

            cv2.putText(frame, f"Your Move: {user_move}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Computer: {comp_move}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Result: {result_text}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.putText(frame, f"Live Detected: {live_detect}", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            cv2.putText(frame, f"Score You:{user_score}  Computer:{comp_score}", (10, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if countdown_number is not None:
                cv2.putText(frame, str(countdown_number), (w//2 - 25, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6)

            cv2.putText(frame, "Press SPACE to play | Q to quit", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("RPS AI Game", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key == ord(' '):
                counting = True
                start_time = time.time()
                result_text = "Get Ready! Show your gesture and hold steady"
                move_buffer.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
