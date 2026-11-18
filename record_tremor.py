"""
TremorVision recorder v2 (research prototype, NOT a diagnostic tool)

Usage:
    python tremor_recorder_v2.py --condition rest --duration 20
    python tremor_recorder_v2.py --condition postural --duration 20 --camera 0

Records wrist position + hand_size (wrist -> middle-finger MCP distance, in image-height units)
so the analyzer can convert amplitude to cm. Press Q to stop early.
"""
import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

PROJECT_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = PROJECT_DIR / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
MAIN_FILE = PROJECT_DIR / "tremor_data.csv"

REQUESTED_WIDTH, REQUESTED_HEIGHT = 1280, 720
COUNTDOWN_S = 3

CSV_COLUMNS = [
    "time_seconds", "wrist_x", "wrist_y", "hand_detected",
    "hand_size", "aspect_ratio", "condition",
    "handedness", "handedness_score",
    "frame_index", "frame_interval_seconds", "elapsed_fps",
]

PROMPTS = {
    "rest": "REST: hand relaxed and supported on your lap/armrest, palm down. Do not move it.",
    "postural": "POSTURAL: arm stretched out in front, palm down, hold still against gravity.",
    "kinetic": "KINETIC: slowly touch your nose then the camera with your index finger, repeat.",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=list(PROMPTS), default="rest")
    ap.add_argument("--duration", type=float, default=20.0, help="seconds to record (0 = until Q)")
    ap.add_argument("--camera", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_file = RECORDINGS_DIR / f"tremor_{args.condition}_{session_id}.csv"

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1,
                           min_detection_confidence=0.6, min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        hands.close()
        raise SystemExit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect = width / height if height else 16 / 9

    print("\nTremorVision recorder v2")
    print("=" * 44)
    print(f"Camera: {width}x{height}  | condition: {args.condition}")
    print(PROMPTS[args.condition])
    print(f"Main file   : {MAIN_FILE}\nSession file: {session_file}")

    # ---- countdown (not recorded) so the person can settle into position ----
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < COUNTDOWN_S:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        left = COUNTDOWN_S - (time.perf_counter() - t0)
        cv2.putText(frame, f"Get ready: {left:.0f}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        cv2.putText(frame, args.condition.upper(), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("TremorVision - Recording", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release(); cv2.destroyAllWindows(); hands.close()
            return

    frame_count = detected = 0
    prev_t = None
    stopped_by_user = False
    start = time.perf_counter()

    try:
        with open(MAIN_FILE, "w", newline="", encoding="utf-8") as fm, \
             open(session_file, "w", newline="", encoding="utf-8") as fs:
            wm = csv.DictWriter(fm, fieldnames=CSV_COLUMNS)
            ws = csv.DictWriter(fs, fieldnames=CSV_COLUMNS)
            wm.writeheader(); ws.writeheader()

            while True:
                ok, frame = cap.read()
                if not ok:
                    print("ERROR: Could not read webcam frame.")
                    break
                frame = cv2.flip(frame, 1)

                now = time.perf_counter()
                elapsed = now - start
                interval = "" if prev_t is None else round(now - prev_t, 6)
                prev_t = now
                frame_count += 1

                res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                is_det = 0
                wx = wy = hs = hand = hscore = ""
                if res.multi_hand_landmarks:
                    lm = res.multi_hand_landmarks[0].landmark
                    wrist, mcp = lm[0], lm[9]          # 0 = wrist, 9 = middle-finger MCP
                    wx, wy = round(float(wrist.x), 6), round(float(wrist.y), 6)
                    # distance in image-height units (x is normalized by width -> multiply by aspect)
                    hs = round(math.hypot((mcp.x - wrist.x) * aspect, mcp.y - wrist.y), 6)
                    is_det = 1
                    detected += 1
                    mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                    if res.multi_handedness:
                        c = res.multi_handedness[0].classification[0]
                        hand, hscore = c.label, round(float(c.score), 4)

                row = {
                    "time_seconds": round(elapsed, 6), "wrist_x": wx, "wrist_y": wy,
                    "hand_detected": is_det, "hand_size": hs, "aspect_ratio": round(aspect, 6),
                    "condition": args.condition, "handedness": hand, "handedness_score": hscore,
                    "frame_index": frame_count, "frame_interval_seconds": interval,
                    "elapsed_fps": round(frame_count / elapsed, 2) if elapsed > 0 else 0,
                }
                wm.writerow(row); ws.writerow(row)
                if frame_count % 30 == 0:
                    fm.flush(); fs.flush()

                rate = detected / frame_count * 100
                color = (0, 220, 0) if is_det else (0, 0, 255)
                cv2.putText(frame, "HAND DETECTED" if is_det else "HAND NOT DETECTED",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                remaining = f"/{args.duration:.0f}" if args.duration > 0 else ""
                cv2.putText(frame, f"Time: {elapsed:.1f}{remaining} s", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Detection: {rate:.1f}%", (20, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"{args.condition.upper()}  | Q = stop", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("TremorVision - Recording", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stopped_by_user = True
                    break
                if args.duration > 0 and elapsed >= args.duration:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

    total = time.perf_counter() - start
    print("\nRecording summary")
    print("=" * 44)
    print(f"Frames: {frame_count}  | with hand: {detected}  | without: {frame_count - detected}")
    if frame_count and total > 0:
        print(f"Average FPS: {frame_count / total:.2f}  | detection rate: {detected / frame_count * 100:.1f}%")
    print(f"Duration: {total:.1f} s  | stopped by user: {stopped_by_user}")
    if total < 10:
        print("WARNING: recording shorter than 10 s - the analyzer may mark it INCONCLUSIVE.")
    print(f"\nNext: python tremor_analysis_v2.py --condition {args.condition}")
    print("Missing wrist coordinates are never treated as zero movement.")


if __name__ == "__main__":
    main()