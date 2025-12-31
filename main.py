import cv2
import time
import numpy as np
import keyboard

# MediaPipe using the tasks API
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# Volume control using (pycaw)
from pycaw.pycaw import AudioUtilities

# Configure the vido input
CAM_WIDTH, CAM_HEIGHT = 640, 480
MIN_PINCH_DIST = 30
MAX_PINCH_DIST = 200
SMOOTHING = 0.2
MIRROR_VIEW = True

PLAY_PAUSE_COOLDOWN = 1.2
NEXT_TRACK_COOLDOWN = 1.5

# Audio Setup
device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
prev_vol = volume.GetMasterVolumeLevelScalar()

# Mediapipe setup to use the library for video functions and more
model_path = "hand_landmarker.task"

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(options)

# -------------- Camera Setup ------------
cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)

pTime = 0
last_play_pause_time = 0
last_next_time = 0
media_state = "Playing"

try:
    while True:
    
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        display = cv2.flip(frame, 1) if MIRROR_VIEW else frame
        h, w, _ = frame.shape

        vol_percent = None

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]

            # -------------- Volume logic, which is done by pinch ------------
            thumb = lm[4]
            index = lm[8]

            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)

            if MIRROR_VIEW:
                x1 = w - x1
                x2 = w - x2

            cv2.circle(display, (x1, y1), 10, (255, 0, 255), -1)
            cv2.circle(display, (x2, y2), 10, (255, 0, 255), -1)
            cv2.line(display, (x1, y1), (x2, y2), (255, 0, 255), 3)

            dist = np.hypot(x2 - x1, y2 - y1)

            vol_scalar = np.interp(dist, [MIN_PINCH_DIST, MAX_PINCH_DIST], [0.0, 1.0])
            vol_scalar = np.clip(vol_scalar, 0.0, 1.0)

            smooth_vol = prev_vol + SMOOTHING * (vol_scalar - prev_vol)
            prev_vol = smooth_vol
            volume.SetMasterVolumeLevelScalar(smooth_vol, None)

            vol_percent = int(smooth_vol * 100)

            # -------------- Play and Pause logic, where Fist is used to do this ------------
            tips = [8, 12, 16, 20]
            pips = [6, 10, 14, 18]

            fingers_closed = all(lm[t].y > lm[p].y for t, p in zip(tips, pips))

            current_time = time.time()
            if fingers_closed and (current_time - last_play_pause_time) > PLAY_PAUSE_COOLDOWN:
                keyboard.send("play/pause media")
                media_state = "Paused" if media_state == "Playing" else "Playing"
                last_play_pause_time = current_time

            # -------------- Next Track logic, where it uses a peace sign ------------
            index_up = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[10].y
            ring_down = lm[16].y > lm[14].y
            pinky_down = lm[20].y > lm[18].y

            peace_sign = index_up and middle_up and ring_down and pinky_down

            if peace_sign and (current_time - last_next_time) > NEXT_TRACK_COOLDOWN:
                keyboard.send("next track")
                last_next_time = current_time

        # -------------- The ui componenet below ------------
        cTime = time.time()
        fps = int(1 / (cTime - pTime)) if pTime != 0 else 0
        pTime = cTime

        cv2.putText(display, f"FPS: {fps}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if vol_percent is not None:
            cv2.rectangle(display, (20, 60), (50, 300), (200, 200, 200), 2)
            vol_bar = int(np.interp(vol_percent, [0, 100], [300, 60]))
            cv2.rectangle(display, (20, vol_bar), (50, 300), (0, 255, 0), -1)
            cv2.putText(display, f"{vol_percent}%", (60, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(display, f"Media: {media_state}",
                    (20, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Gesture Media Control", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n[INFO] Program interrupted by user (Ctrl+C). Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released and windows closed.")