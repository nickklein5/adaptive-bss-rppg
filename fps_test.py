"""Measure the webcam's real frame rate.

The camera reports a nominal fps, but the rate it actually delivers is often
lower (dim light, auto-exposure, USB bandwidth). The real rate is the sampling
rate for every heart-rate calculation later, so we measure it instead of
trusting the label.
"""
import sys
import time

import cv2
import numpy as np

CAMERA_INDEX = 0           # 0 = default camera; try 1 for an external webcam
WIDTH, HEIGHT = 640, 480   # same resolution as the UBFC-rPPG dataset
DURATION_S = 10.0          # how long to measure
WARMUP_FRAMES = 30         # skip the first frames while the camera settles


def main():
    # DirectShow opens faster on Windows and later lets us lock exposure
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {CAMERA_INDEX}. "
                 "Close other apps using the camera, or try another CAMERA_INDEX.")

    # Ask for MJPG at 30 fps. Many webcams default to an uncompressed format
    # over DirectShow that tops out far below 30 fps.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        for _ in range(WARMUP_FRAMES):
            cap.read()

        timestamps = []
        failed_reads = 0
        frame = None
        print(f"Measuring for {DURATION_S:.0f} s... sit still and keep the lighting steady.")

        start = time.perf_counter()
        while time.perf_counter() - start < DURATION_S:
            ok, img = cap.read()
            if ok:
                timestamps.append(time.perf_counter())
                frame = img
            else:
                failed_reads += 1

        reported_fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()

    if len(timestamps) < 2:
        sys.exit("Got fewer than 2 frames. The camera opened but isn't delivering images.")

    t = np.array(timestamps)
    intervals = np.diff(t)                       # seconds between frames
    measured_fps = (len(t) - 1) / (t[-1] - t[0])
    median_dt = np.median(intervals)
    late_frames = int(np.sum(intervals > 1.5 * median_dt))
    h, w = frame.shape[:2]

    print()
    print(f"Resolution:        {w} x {h}")
    print(f"Reported fps:      {reported_fps:.1f}  (what the camera claims)")
    print(f"Measured fps:      {measured_fps:.2f}  (what you actually get)")
    print(f"Frames captured:   {len(t)}  ({failed_reads} failed reads)")
    print(f"Frame interval:    mean {intervals.mean() * 1000:.1f} ms, "
          f"std {intervals.std() * 1000:.1f} ms, max {intervals.max() * 1000:.1f} ms")
    print(f"Late frames:       {late_frames}  (gap > 1.5x the usual interval)")


if __name__ == "__main__":
    main()