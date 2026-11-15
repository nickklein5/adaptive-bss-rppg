"""Estimate heart rate from the green channel (M2 baseline, no ICA yet).

Steps:
  1. Resample R, G, B onto an even time grid using the real frame timestamps
     (the FFT assumes evenly spaced samples).
  2. Detrend: remove the slow linear drift.
  3. Normalize each channel: subtract the mean, divide by the std.
  4. Band-pass 0.75-4 Hz (45-240 bpm) with a zero-phase Butterworth filter.
  5. FFT each channel and take the strongest peak inside the band.

Usage (PowerShell, with .venv active):
  python green_hr.py still_02
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt

TRACE_DIR = Path("data/traces")
RESULTS_DIR = Path("results")
FS = 30.0                 # resample rate (Hz)
BAND = (0.75, 4.0)        # heart-rate band (Hz) = 45-240 bpm
FILTER_ORDER = 3
NFFT = 16384              # zero-padding: interpolates the spectrum so the
                          # peak can be read between FFT bins


def load(name):
    data = np.genfromtxt(TRACE_DIR / f"{name}_rgb.csv", delimiter=",", names=True)
    good = np.isfinite(data["G"])                 # drop frames with no face
    if good.sum() < len(data):
        print(f"Skipping {len(data) - good.sum()} frames with no face.")
    return data["t_s"][good], {ch: data[ch][good] for ch in "RGB"}


def preprocess(t, x):
    """Resample to FS, detrend, normalize, band-pass. Returns (t_even, x_clean)."""
    t_even = np.arange(t[0], t[-1], 1 / FS)
    x = np.interp(t_even, t, x)                   # step 1
    x = detrend(x)                                # step 2
    x = (x - x.mean()) / x.std()                  # step 3
    sos = butter(FILTER_ORDER, BAND, btype="bandpass", fs=FS, output="sos")
    return t_even, sosfiltfilt(sos, x)            # step 4 (zero-phase)


def spectrum(x):
    """Power spectrum of x (Hann window to reduce leakage)."""
    win = np.hanning(len(x))
    power = np.abs(np.fft.rfft(x * win, NFFT)) ** 2
    freqs = np.fft.rfftfreq(NFFT, 1 / FS)
    return freqs, power


def peak_bpm(freqs, power):
    """Strongest in-band peak, plus SNR in dB: power within +/-0.1 Hz of the
    peak vs. the rest of the band. Above ~0 dB the peak clearly stands out."""
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    f_peak = freqs[band][np.argmax(power[band])]
    near = band & (np.abs(freqs - f_peak) <= 0.1)
    snr_db = 10 * np.log10(power[near].sum() / power[band & ~near].sum())
    return f_peak * 60, snr_db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="recording name, e.g. still_02")
    args = parser.parse_args()

    t, raw = load(args.name)
    duration = t[-1] - t[0]
    colors = {"R": "tab:red", "G": "tab:green", "B": "tab:blue"}

    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(10, 7))
    print(f"\n{args.name}: {duration:.1f} s, "
          f"FFT resolution {60 / duration:.1f} bpm between bins\n")
    for ch in "RGB":
        t_even, clean = preprocess(t, raw[ch])
        freqs, power = spectrum(clean)
        bpm, snr_db = peak_bpm(freqs, power)
        print(f"  {ch}: {bpm:5.1f} bpm   SNR {snr_db:+5.1f} dB")

        main_ch = ch == "G"
        ax_t.plot(t_even - t_even[0], clean, color=colors[ch],
                  linewidth=1.2 if main_ch else 0.6, alpha=1 if main_ch else 0.4,
                  label=ch)
        band = (freqs >= 0.5) & (freqs <= 4.5)
        ax_f.plot(freqs[band] * 60, power[band] / power[band].max(), color=colors[ch],
                  linewidth=1.5 if main_ch else 0.8, alpha=1 if main_ch else 0.5,
                  label=f"{ch}: {bpm:.0f} bpm")
        if main_ch:
            ax_f.axvline(bpm, color=colors[ch], linestyle="--", linewidth=1)
            green_bpm = bpm

    ax_t.set_xlim(0, min(10, duration))
    ax_t.set_xlabel("time (s), first 10 s")
    ax_t.set_ylabel("filtered, normalized")
    ax_t.set_title("Cleaned signal (0.75-4 Hz)")
    ax_t.legend(loc="upper right")
    ax_t.grid(alpha=0.3)
    ax_f.axvspan(0.5 * 60, BAND[0] * 60, color="gray", alpha=0.1)
    ax_f.axvspan(BAND[1] * 60, 4.5 * 60, color="gray", alpha=0.1)
    ax_f.set_xlabel("heart rate (bpm)")
    ax_f.set_ylabel("relative power")
    ax_f.set_title(f"Spectrum: green peak at {green_bpm:.1f} bpm")
    ax_f.legend(loc="upper right")
    ax_f.grid(alpha=0.3)
    fig.suptitle(f"Green-channel heart rate: {args.name}")
    fig.tight_layout()

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.name}_green_hr.png"
    fig.savefig(out, dpi=120)
    print(f"\nGreen-channel estimate: {green_bpm:.1f} bpm")
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
