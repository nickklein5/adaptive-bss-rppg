"""Estimate heart rate with ICA, the method of Poh, McDuff & Picard (2010).

Idea: the camera's R, G, B traces are three different mixtures of the same
underlying sources (the pulse, motion, lighting changes). Independent
Component Analysis finds the un-mixing that makes the outputs as
statistically independent as possible, so the pulse ends up mostly in one
component instead of being smeared across all three channels.

Steps:
  1. Same cleaning as green_hr.py: resample, detrend, normalize, band-pass.
  2. FastICA on the three cleaned channels -> three components.
  3. FFT each component; pick the one with the highest spectral peak
     (the paper's selection rule; improvement #3 targets this rule).
  4. Compare against the green-channel baseline on the same recording.

ICA returns components in random order and with random sign, so
"component 2" means nothing on its own; only the spectra matter.

Usage (PowerShell, with .venv active):
  python ica_hr.py still_05
  python ica_hr.py still_05 --true 66      # adds the error vs a reference
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA

from green_hr import BAND, load, peak_bpm, preprocess, spectrum

RESULTS_DIR = Path("results")
SEED = 0                  # fixes ICA's random start so runs are repeatable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="recording name, e.g. still_05")
    parser.add_argument("--true", type=float, default=None,
                        help="reference heart rate from a sensor, in bpm")
    args = parser.parse_args()

    t, raw = load(args.name)
    clean = {}
    for ch in "RGB":
        t_even, clean[ch] = preprocess(t, raw[ch])
    X = np.column_stack([clean["R"], clean["G"], clean["B"]])   # samples x 3

    ica = FastICA(n_components=3, whiten="unit-variance", random_state=SEED,
                  max_iter=1000)
    S = ica.fit_transform(X)                                     # samples x 3

    # Green baseline, for comparison
    freqs, g_power = spectrum(clean["G"])
    g_bpm, g_snr = peak_bpm(freqs, g_power)

    # Score each component; the paper picks the one with the tallest peak
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    comps = []
    for k in range(3):
        _, power = spectrum(S[:, k])
        bpm, snr = peak_bpm(freqs, power)
        comps.append({"k": k, "power": power, "bpm": bpm, "snr": snr,
                      "peak": power[band].max() / power[band].sum()})
    chosen = max(comps, key=lambda c: c["peak"])

    print(f"\n{args.name}")
    print(f"  Green baseline:  {g_bpm:5.1f} bpm   SNR {g_snr:+5.1f} dB")
    for c in comps:
        mark = "  <- chosen" if c is chosen else ""
        print(f"  ICA comp {c['k'] + 1}:      {c['bpm']:5.1f} bpm   SNR {c['snr']:+5.1f} dB{mark}")
    if args.true is not None:
        print(f"\n  Reference:       {args.true:5.1f} bpm")
        print(f"  Green error:     {g_bpm - args.true:+5.1f} bpm")
        print(f"  ICA error:       {chosen['bpm'] - args.true:+5.1f} bpm")

    # ---- plot -----------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    ax_t, ax_f = axes
    show = (t_even - t_even[0]) <= 10
    colors = ["tab:purple", "tab:orange", "tab:cyan"]
    for c, color in zip(comps, colors):
        is_chosen = c is chosen
        ax_t.plot(t_even[show] - t_even[0], S[show, c["k"]] + 4 * c["k"], color=color,
                  linewidth=1.3 if is_chosen else 0.8,
                  label=f"comp {c['k'] + 1}" + (" (chosen)" if is_chosen else ""))
        zoom = (freqs >= 0.5) & (freqs <= 4.5)
        ax_f.plot(freqs[zoom] * 60, c["power"][zoom] / c["power"][zoom].max(),
                  color=color, linewidth=1.6 if is_chosen else 0.8,
                  label=f"comp {c['k'] + 1}: {c['bpm']:.0f} bpm")
    zoom = (freqs >= 0.5) & (freqs <= 4.5)
    ax_f.plot(freqs[zoom] * 60, g_power[zoom] / g_power[zoom].max(), color="tab:green",
              linestyle=":", linewidth=1, label=f"green baseline: {g_bpm:.0f} bpm")
    if args.true is not None:
        ax_f.axvline(args.true, color="black", linestyle="--", linewidth=1,
                     label=f"reference: {args.true:.0f} bpm")

    ax_t.set_xlabel("time (s), first 10 s")
    ax_t.set_ylabel("component (offset for clarity)")
    ax_t.set_yticks([])
    ax_t.set_title("ICA components")
    ax_t.legend(loc="upper right", fontsize=8)
    ax_t.grid(alpha=0.3)
    ax_f.axvspan(30, BAND[0] * 60, color="gray", alpha=0.1)
    ax_f.axvspan(BAND[1] * 60, 270, color="gray", alpha=0.1)
    ax_f.set_xlabel("heart rate (bpm)")
    ax_f.set_ylabel("relative power")
    ax_f.set_title(f"Spectra: ICA picks {chosen['bpm']:.1f} bpm, green says {g_bpm:.1f} bpm")
    ax_f.legend(loc="upper right", fontsize=8)
    ax_f.grid(alpha=0.3)
    fig.suptitle(f"ICA heart rate: {args.name}")
    fig.tight_layout()

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.name}_ica_hr.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved {out}")
    plt.show()


if __name__ == "__main__":
    main()
