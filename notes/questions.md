# M0 Notes

## Webcam fps test

Laptop webcam, 640 x 480, DirectShow backend, 10 s per run.

| Run | fps | Jitter (std) | Late frames |
| --- | --- | --- | --- |
| Default format, dim light | 9.9 | 15.4 ms | 1 |
| Default format, bright light | 21.1 | 23.6 ms | 37 |
| MJPG, bright light | 29.7 | 5.0 ms | 7 |

What I learned:
- Two things capped the frame rate. Dim light makes auto-exposure lengthen each frame, and the default (uncompressed) format tops out far below 30 fps. Bright light plus requesting MJPG gives a steady ~30 fps.
- Frames still don't arrive perfectly evenly. The FFT assumes even spacing, so capture must save a timestamp per frame, and processing must resample onto an even time grid (np.interp) before the FFT.
- If the FFT is told the wrong fps, every bpm reading is scaled: reported bpm = true bpm x (assumed fps / real fps). Example: real 21 fps, assumed 30 fps, true 72 bpm reads ~103 bpm.

## Questions

### 1. Highest measurable frequency at 29.7 fps, and is it enough?

The Nyquist frequency is half the sampling rate: 29.7 / 2 = 14.85 Hz = 891 bpm.
The fastest realistic heart rate is about 240 bpm (4 Hz), so this is ~3.7x more than needed.
Even the old 9.9 fps (Nyquist 4.95 Hz = 297 bpm) would have been enough.
Conclusion: frame rate is not the limit. Timing accuracy (even spacing) is.

### 2. Frequency resolution with a 30 s window?

FFT bin spacing is 1 / window length.
- 30 s window: 1/30 = 0.033 Hz = 2 bpm between bins, so about +/-1 bpm.
- 10 s window: 1/10 = 0.1 Hz = 6 bpm between bins, so about +/-3 bpm.
- 1 bpm between bins needs a 60 s window (1/60 Hz x 60 = 1 bpm).

Trade-off: a longer window gives finer resolution but reacts slowly to heart-rate changes and has more chances to be corrupted by motion. Interpolating between FFT bins can sharpen short windows (improvement #5).

### 3. Why does the green channel carry the strongest pulse?

- Hemoglobin in blood absorbs green light (~540-580 nm) much more strongly than red light, so each heartbeat's change in blood volume changes green reflectance the most.
- Green light penetrates the skin just deep enough to reach the small blood vessels (capillaries) in the upper dermis. Blue is absorbed too close to the surface; red goes deeper but is absorbed only weakly by blood, so its pulse signal is small.
- Camera sensors (Bayer pattern) have twice as many green pixels as red or blue, so the green channel is also the least noisy.

Red and blue still carry useful information: motion and lighting changes affect all three channels similarly, which is exactly what ICA and later methods (CHROM, POS) use to separate the pulse from noise.
