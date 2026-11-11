"""Record a face video for heart-rate extraction.

Saves two files in data/raw/:
  <name>.avi              lossless video (FFV1), so compression doesn't
                          smear the tiny color changes we're measuring
  <name>_timestamps.csv   the real capture time of every frame

Face tracking happens later, in extract_rgb.py. Doing it during recording
would slow the loop and make the frame timing uneven again.

Usage (PowerShell, with .venv active):
  python capture.py                          # 60 s, auto-named
  python capture.py --seconds 30 --name still_01
  python capture.py --lock                   # freeze exposure + white balance

In the window: SPACE starts recording, Q quits (or stops early).
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

CAMERA_INDEX = 0           # 0 = default camera; 1 for an external webcam
WIDTH, HEIGHT, FPS = 640, 480, 30
WARMUP_FRAMES = 30         # let the camera settle before anything else
OUT_DIR = Path("data/raw")
WINDOW = "capture (SPACE = record, Q = quit)"


def open_camera():
    # DirectShow + MJPG: the combination that gave a steady ~30 fps in fps_test
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {CAMERA_INDEX}. "
                 "Close other apps using the camera, or change CAMERA_INDEX.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    return cap


def lock_camera(cap):
    """Freeze exposure and white balance at the values auto mode settled on.

    Auto-exposure and auto-white-balance change pixel brightness far more
    than blood flow does, so they can swamp the pulse. Not every webcam
    honors these settings; watch the preview, and if the image looks wrong,
    run again without --lock.
    """
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    wb = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # DirectShow: 0.25 = manual
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    if wb > 0:
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)
    print(f"Locked: exposure {cap.get(cv2.CAP_PROP_EXPOSURE):g}, "
          f"white balance {cap.get(cv2.CAP_PROP_WB_TEMPERATURE):g}, "
          f"auto-exposure now {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):g}")


def open_writer(path, size):
    """Lossless FFV1 if available; otherwise MJPG with a warning."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), FPS, size)
    if writer.isOpened():
        return writer, "FFV1 (lossless)"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, size)
    if writer.isOpened():
        print("WARNING: FFV1 not available, saving MJPG. Compression adds noise "
              "to the pulse signal; fine for testing, not for final results.")
        return writer, "MJPG (lossy)"
    sys.exit(f"Could not create a video file at {path}.")


def preview_until_space(cap):
    """Show the live camera so you can position your face. True = start."""
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        view = frame.copy()
        cv2.putText(view, "Center your face. SPACE = start, Q = quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            return True
        if key in (ord("q"), 27):        # Q or Esc
            return False


def record(cap, writer, seconds):
    """Grab frames for `seconds`; return the timestamp of each saved frame."""
    timestamps = []
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= seconds:
            break
        ok, frame = cap.read()
        if not ok:
            continue
        timestamps.append(time.perf_counter() - start)   # right after the read
        writer.write(frame)                              # save the clean frame

        view = frame.copy()                              # draw only on a copy
        cv2.putText(view, f"REC {seconds - elapsed:4.1f} s  (Q = stop)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(WINDOW, view)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            print("Stopped early.")
            break
    return timestamps


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=60, help="recording length")
    parser.add_argument("--name", default=None, help="file name (default: date_time)")
    parser.add_argument("--lock", action="store_true",
                        help="freeze exposure and white balance before recording")
    args = parser.parse_args()

    name = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = OUT_DIR / f"{name}.avi"
    ts_path = OUT_DIR / f"{name}_timestamps.csv"
    if video_path.exists():
        sys.exit(f"{video_path} already exists. Pick another --name.")

    cap = open_camera()
    writer = None
    try:
        for _ in range(WARMUP_FRAMES):
            cap.read()
        if args.lock:
            lock_camera(cap)

        if not preview_until_space(cap):
            print("Quit without recording.")
            return

        ok, frame = cap.read()
        if not ok:
            sys.exit("Camera stopped delivering frames.")
        h, w = frame.shape[:2]
        writer, codec = open_writer(video_path, (w, h))
        print(f"Recording {args.seconds:g} s to {video_path} [{codec}]. Sit still.")

        timestamps = record(cap, writer, args.seconds)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if len(timestamps) < 2:
        video_path.unlink(missing_ok=True)
        sys.exit("Fewer than 2 frames recorded; nothing saved.")

    with open(ts_path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["frame", "t_s"])
        for i, t in enumerate(timestamps):
            out.writerow([i, f"{t:.6f}"])

    t = np.array(timestamps)
    dt = np.diff(t)
    fps = (len(t) - 1) / (t[-1] - t[0])
    size_mb = video_path.stat().st_size / 1e6
    print()
    print(f"Saved {len(t)} frames ({t[-1]:.1f} s)")
    print(f"Measured fps:   {fps:.2f}")
    print(f"Frame interval: mean {dt.mean() * 1000:.1f} ms, std {dt.std() * 1000:.1f} ms, "
          f"max {dt.max() * 1000:.1f} ms")
    print(f"Video:          {video_path} ({size_mb:.0f} MB)")
    print(f"Timestamps:     {ts_path}")
    print(f"Next:           python extract_rgb.py {name}")


if __name__ == "__main__":
    main()
