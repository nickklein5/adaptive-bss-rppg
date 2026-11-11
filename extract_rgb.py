"""Turn a recorded face video into R, G, B skin traces.

For every frame: MediaPipe finds 478 face landmarks, we build a mask over
the forehead and both cheeks, and average the pixel colors inside it.
Those three averages per frame are the raw signals the heart-rate code uses.

Input:   data/raw/<name>.avi  and  data/raw/<name>_timestamps.csv
Output:  data/traces/<name>_rgb.csv   frame, t_s, R, G, B, n_pixels
         data/traces/<name>_roi.png   a frame with the skin mask drawn on;
                                      open it and check the mask is on skin

Usage (PowerShell, with .venv active):
  python extract_rgb.py still_01
"""
import argparse
import csv
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/traces")
MODEL_PATH = Path("models/face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# Landmark indices (MediaPipe face mesh) outlining each skin region. The
# convex hull of each set becomes the mask, so the order doesn't matter.
# "right" and "left" are the subject's own sides (right cheek = image left).
ROIS = {
    "forehead":    [107, 66, 69, 109, 10, 338, 299, 296, 336, 9],
    "right_cheek": [117, 118, 101, 36, 205, 187, 123, 50],
    "left_cheek":  [346, 347, 330, 266, 425, 411, 352, 280],
}


def get_model():
    """Download the face landmark model once (about 4 MB)."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def load_timestamps(path):
    with open(path, newline="") as f:
        return [float(row["t_s"]) for row in csv.DictReader(f)]


def skin_mask(landmarks, h, w):
    """Filled mask (255 = skin) covering every region in ROIS."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for idx in ROIS.values():
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in idx],
                       dtype=np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
    return mask


def save_roi_preview(frame, mask, path):
    """Tint the masked area green and outline it, so you can eyeball the ROI."""
    view = frame.copy()
    green = np.zeros_like(view)
    green[:, :, 1] = 255
    tinted = cv2.addWeighted(view, 0.6, green, 0.4, 0)
    view[mask > 0] = tinted[mask > 0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(str(path), view)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="recording name, e.g. still_01")
    args = parser.parse_args()

    video_path = RAW_DIR / f"{args.name}.avi"
    ts_path = RAW_DIR / f"{args.name}_timestamps.csv"
    for p in (video_path, ts_path):
        if not p.exists():
            sys.exit(f"Missing {p}. Record it first with capture.py.")

    timestamps = load_timestamps(ts_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{args.name}_rgb.csv"
    png_path = OUT_DIR / f"{args.name}_roi.png"

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(get_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    video = cv2.VideoCapture(str(video_path))
    rows, missing, preview_saved, last_ms = [], 0, False, -1

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for i, t in enumerate(timestamps):
            ok, frame = video.read()
            if not ok:
                print(f"Video ended at frame {i}; timestamps list {len(timestamps)}.")
                break

            # VIDEO mode needs strictly increasing integer milliseconds
            ms = max(int(round(t * 1000)), last_ms + 1)
            last_ms = ms
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)

            if not result.face_landmarks:
                missing += 1
                rows.append([i, t, np.nan, np.nan, np.nan, 0])
                continue

            h, w = frame.shape[:2]
            mask = skin_mask(result.face_landmarks[0], h, w)
            n = int(cv2.countNonZero(mask))
            b, g, r, _ = cv2.mean(frame, mask=mask)      # OpenCV frames are BGR
            rows.append([i, t, r, g, b, n])

            if not preview_saved:
                save_roi_preview(frame, mask, png_path)
                preview_saved = True

            if i % 300 == 0:
                print(f"  frame {i}/{len(timestamps)}")
    video.release()

    with open(csv_path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["frame", "t_s", "R", "G", "B", "n_pixels"])
        for i, t, r, g, b, n in rows:
            out.writerow([i, f"{t:.6f}", f"{r:.4f}", f"{g:.4f}", f"{b:.4f}", n])

    found = len(rows) - missing
    pixels = [row[5] for row in rows if row[5] > 0]
    print()
    print(f"Frames processed:  {len(rows)}")
    print(f"Face found in:     {found} ({100 * found / max(len(rows), 1):.1f}%)")
    if pixels:
        print(f"Skin pixels/frame: {int(np.median(pixels))} (median)")
    print(f"Traces:            {csv_path}")
    if preview_saved:
        print(f"ROI preview:       {png_path}  <- open this and check the mask")
    else:
        print("No face was found in any frame, so no ROI preview was saved.")


if __name__ == "__main__":
    main()
