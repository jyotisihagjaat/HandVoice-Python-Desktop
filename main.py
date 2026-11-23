import cv2
import mediapipe as mp
import math
import time
from collections import deque, Counter

# HandVoice Desktop: webcam -> hand landmarks -> phrase on screen.
# Tested design target: Python 3.11/3.12, MediaPipe 0.10.21, OpenCV 4.10.

PHRASES = {
    "OPEN_PALM": ("👋", "HELLO!", "Open palm"),
    "FIST": ("✊", "BYE! SEE YOU SOON.", "Closed fist"),
    "PEACE": ("✌️", "PEACE!", "Index + middle fingers"),
    "THUMB_UP": ("👍", "GREAT JOB!", "Thumbs up"),
    "THUMB_DOWN": ("👎", "I DISAGREE.", "Thumbs down"),
    "I_LOVE_YOU": ("🤟", "I LOVE YOU!", "I love you sign"),
    "CALL_ME": ("🤙", "CALL ME LATER.", "Call me sign"),
    "POINT": ("☝️", "I HAVE A QUESTION.", "Index finger"),
    "THREE": ("3", "ONE MOMENT, PLEASE.", "Three fingers"),
    "FOUR": ("4", "I AM ALMOST READY.", "Four fingers"),
    "ROCK": ("🤘", "LET'S GO!", "Rock sign"),
    "PINKY": ("🤙", "PROMISE!", "Pinky finger"),
    "THUMB_SIDE": ("👈", "MAYBE.", "Thumb sideways"),
    "SPREAD": ("🖐", "PLEASE STOP.", "Wide open hand"),
    "INDEX_MIDDLE_RING": ("3", "THANK YOU!", "Three fingers (thumb tucked)"),
    "INDEX_RING": ("✌", "I NEED HELP.", "Index + ring (approx.)"),
    "HANG_LOOSE": ("🤙", "TAKE IT EASY.", "Thumb + pinky"),
    "FINGERS_CROSSED": ("🤞", "GOOD LUCK!", "Fingers crossed (approx.)"),
    "HAND_HERE": ("✋", "I AM HERE.", "Hand held steady"),
    "ASK": ("☝", "PLEASE LISTEN.", "Index pointing up"),
}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def classify_hand(lm):
    # Landmark-based single-hand pose estimate. Hand should face camera.
    # For four fingers, compare fingertip y with the middle joint.
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    extended = [lm[t].y < lm[p].y - 0.025 for t, p in zip(tips, pips)]
    index, middle, ring, pinky = extended
    thumb_out = distance(lm[4], lm[2]) > 0.16
    thumb_up = lm[4].y < lm[3].y - 0.045 and thumb_out
    thumb_down = lm[4].y > lm[3].y + 0.045 and thumb_out
    count = sum(extended)

    # Order matters: specific combinations before broad finger counts.
    if count == 4 and thumb_out:
        return "OPEN_PALM"
    if count == 0 and not thumb_out:
        if thumb_up: return "THUMB_UP"
        if thumb_down: return "THUMB_DOWN"
        return "FIST"
    if thumb_up and count == 0:
        return "THUMB_UP"
    if thumb_down and count == 0:
        return "THUMB_DOWN"
    if index and middle and not ring and not pinky:
        return "PEACE"
    if index and not middle and not ring and not pinky:
        return "POINT"
    if thumb_out and pinky and not index and not middle and not ring:
        return "CALL_ME"
    if thumb_out and index and pinky and not middle and not ring:
        return "I_LOVE_YOU"
    if index and not middle and not ring and pinky:
        return "ROCK"
    if pinky and not index and not middle and not ring:
        return "PINKY"
    if index and middle and ring and not pinky:
        return "THREE"
    if count == 4 and not thumb_out:
        return "FOUR"
    if count == 4:
        return "SPREAD"
    if count == 3:
        return "INDEX_MIDDLE_RING"
    if count == 2:
        return "PEACE"
    if count == 1:
        return "POINT"
    return "HAND_HERE"

def draw_hud(frame, phrase, label, fps, detected):
    h, w = frame.shape[:2]
    # Bottom message card
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-145), (w, h), (12, 15, 31), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (0, h-145), (w, h-141), (180, 135, 255), -1)
    cv2.putText(frame, "HANDVOICE  /  GESTURE TO TEXT", (24, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (190, 205, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:0.1f}", (w-145, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (115, 235, 205), 2, cv2.LINE_AA)
    cv2.putText(frame, "DETECTED MEANING", (25, h-111),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 160, 190), 1, cv2.LINE_AA)
    # OpenCV's built-in font does not render emoji, so use clear words.
    cv2.putText(frame, phrase, (25, h-68),
                cv2.FONT_HERSHEY_DUPLEX, 0.92, (245, 245, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (27, h-35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 180, 210), 1, cv2.LINE_AA)
    if detected:
        cv2.circle(frame, (w-35, h-112), 7, (90, 235, 160), -1)
        cv2.putText(frame, "HAND FOUND", (w-155, h-106),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 240, 180), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "SHOW ONE HAND TO CAMERA", (w-260, h-106),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 195), 1, cv2.LINE_AA)

def main():
    print("\\nHANDVOICE — Gesture to Text")
    print("Starting webcam... Press Q or ESC in the camera window to quit.")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\\nERROR: Could not open webcam.")
        print("Close Zoom/Meet/Camera apps, check Windows camera permission, then run again.")
        input("\\nPress Enter to close...")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    history = deque(maxlen=7)
    current_id = None
    stable_id = None
    stable_frames = 0
    last_change = 0.0
    last_time = time.time()
    fps = 0.0
    current_phrase = "SHOW YOUR HAND TO BEGIN"
    current_label = "Place one hand in the camera frame"

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,  # faster webcam inference
            min_detection_confidence=0.60,
            min_tracking_confidence=0.55
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame could not be read.")
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)
                rgb.flags.writeable = True

                detected = False
                if results.multi_hand_landmarks:
                    detected = True
                    hand_landmarks = results.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(180, 135, 255), thickness=2, circle_radius=3),
                        mp_draw.DrawingSpec(color=(95, 225, 215), thickness=2)
                    )
                    raw_id = classify_hand(hand_landmarks.landmark)
                    history.append(raw_id)
                    most_common, amount = Counter(history).most_common(1)[0]
                    if most_common == stable_id:
                        stable_frames += 1
                    else:
                        stable_id = most_common
                        stable_frames = 1
                    # A short stability gate prevents flickering text.
                    if amount >= 4 and stable_frames >= 3 and (most_common != current_id or time.time()-last_change > 1.8):
                        current_id = most_common
                        emoji, phrase, label = PHRASES[current_id]
                        current_phrase = phrase
                        current_label = f"{label}  |  Hold another sign to change"
                        last_change = time.time()
                else:
                    history.clear()
                    stable_id = None
                    stable_frames = 0

                now = time.time()
                instant = 1.0 / max(now-last_time, 1e-6)
                fps = instant if fps == 0 else fps * 0.88 + instant * 0.12
                last_time = now
                draw_hud(frame, current_phrase, current_label, fps, detected)
                cv2.imshow("HandVoice — Gesture Communicator", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
    except Exception as exc:
        print(f"\\nSomething went wrong: {exc}")
        print("Try reinstalling requirements with the included RUN.bat.")
        input("\\nPress Enter to close...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
