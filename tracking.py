"""Kalman-filter heart-rate tracking across sliding windows.

Picking the tallest spectral peak in each window independently lets one noisy
window jump 20-40 bpm. A real heart can't do that in a second. This tracker
keeps a running estimate of heart rate and its uncertainty, and uses them to
judge each new window:

  state       x = heart rate (bpm)
  uncertainty P = variance of x (bpm^2)

  1. Predict:  heart rate stays about the same, but we grow less sure
               x stays,  P <- P + Q * dt           (Q: how fast HR can drift)
  2. Choose:   among the window's top spectral peaks, pick the one that best
               balances "tall" and "close to the prediction"
  3. Gate:     if even the best peak is implausibly far (> GATE standard
               deviations), ignore this window. P keeps growing, so after a
               few rejected windows a real change (e.g. after exercise) is
               accepted again.
  4. Update:   blend prediction and measurement, trusting the measurement
               more when its SNR is high (small R)
               K = P / (P + R);  x <- x + K * (z - x);  P <- (1 - K) * P

Two safeguards against locking onto a wrong rhythm:
  Warm-up:     start from the median of the first WARMUP windows' tallest
               peaks, not from window 1 alone (one bad first window would
               otherwise steer the whole recording).
  Re-lock:     if the last WARMUP windows' tallest peaks agree with each
               other (within AGREE bpm) but sit more than RELOCK bpm away
               from the tracked value, the tracker is following the wrong
               peak (or HR really changed fast), so jump to them.

Every constant below is a physical guess, not tuned to our data, so the
evaluation stays honest.
"""
from collections import deque

import numpy as np
from scipy.signal import find_peaks

from green_hr import BAND

Q = 4.0            # process noise, bpm^2 per second: HR drifts ~2 bpm/s at most
R0 = 9.0           # measurement variance at 0 dB SNR (std 3 bpm)
R_MIN, R_MAX = 1.0, 400.0
P0 = 100.0         # initial uncertainty (std 10 bpm)
GATE = 3.0         # reject measurements more than 3 std devs from prediction
N_CANDIDATES = 5   # how many spectral peaks to consider per window
WARMUP = 5         # windows used to start (and to check for re-locking)
AGREE = 6.0        # bpm: tallest peaks this close count as "agreeing"
RELOCK = 12.0      # bpm: re-lock if agreeing peaks are this far from x


def candidate_peaks(freqs, power, n=N_CANDIDATES):
    """Top-n local maxima inside the heart-rate band.

    Returns a list of (bpm, relative_height, snr_db), tallest first.
    """
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    f, p = freqs[band], power[band]
    idx, _ = find_peaks(p)
    if len(idx) == 0:
        idx = np.array([np.argmax(p)])
    idx = idx[np.argsort(p[idx])[::-1][:n]]
    out = []
    for i in idx:
        near = np.abs(f - f[i]) <= 0.1
        snr = 10 * np.log10(p[near].sum() / max(p[~near].sum(), 1e-12))
        out.append((f[i] * 60, p[i] / p.max(), snr))
    return out


def measurement_variance(snr_db):
    """Noisier peaks (low SNR) get a larger variance, so they move x less."""
    return float(np.clip(R0 * 10 ** (-snr_db / 10), R_MIN, R_MAX))


class HeartRateTracker:
    def __init__(self):
        self.x = None
        self.P = None
        self.recent = deque(maxlen=WARMUP)       # last few tallest peaks

    def step(self, freqs, power, dt=1.0):
        """Feed one window's spectrum; returns the tracked heart rate in bpm."""
        cands = candidate_peaks(freqs, power)
        self.recent.append(cands[0][0])

        if self.x is None:                       # warm-up
            if len(self.recent) < WARMUP:
                return float(np.median(self.recent))
            self.x, self.P = float(np.median(self.recent)), P0
            return self.x

        self.P += Q * dt                         # 1. predict

        best = None                              # 2. choose
        for bpm, height, snr in cands:
            r = measurement_variance(snr)
            s = self.P + r                       # expected spread of z - x
            score = np.log(height) - 0.5 * (bpm - self.x) ** 2 / s - 0.5 * np.log(s)
            if best is None or score > best[0]:
                best = (score, bpm, r, s)
        _, z, r, s = best

        if (z - self.x) ** 2 <= GATE ** 2 * s:   # 3. gate (skip if implausible)
            k = self.P / (self.P + r)            # 4. update
            self.x += k * (z - self.x)
            self.P *= 1 - k

        recent = np.array(self.recent)           # re-lock check
        if (len(recent) == WARMUP and np.ptp(recent) <= AGREE
                and abs(np.median(recent) - self.x) > RELOCK):
            self.x, self.P = float(np.median(recent)), P0
        return self.x
