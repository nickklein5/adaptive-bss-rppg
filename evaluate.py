"""Sliding-window evaluation of every method, grouped by recording condition.

Each recording is cut into overlapping windows (10 s long, moving 1 s at a
time), so one 30 s clip gives ~21 heart-rate estimates per method instead of 1.
Each window is scored against the sensor reading interpolated to its center.

Reads notes/recordings.csv:
  name,condition,bpm_start,bpm_mid,bpm_end
  bench_01,still,68,66,65

Methods (all on the same windows):
  green    green channel only
  k_size   G - k*R, k = std(G)/std(R) per window    (same as G - R after
           z-normalizing, so it replaces the old g_minus_r)
  k_least  G - k*R, k = cov(G,R)/var(R) per window
  ica      FastICA, component with the tallest peak (Poh 2010 baseline)

Usage (PowerShell, with .venv active):
  python evaluate.py              # all recordings in notes/recordings.csv
  python evaluate.py --plot       # also save a per-recording plot
  python evaluate.py --win 20     # try a different window length
"""
import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

# On short windows FastICA sometimes stops before fully converging. It still
# returns usable components, so hide the warning instead of flooding the screen.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from green_hr import BAND, FILTER_ORDER, FS, load, peak_bpm, spectrum

RECORDINGS = Path("notes/recordings.csv")
RESULTS_DIR = Path("results")
METHODS = ["green", "k_size", "k_least", "ica"]
STEP_S = 1.0


def prepare(t, raw):
    """Whole recording: resample to FS, relative change, detrend, band-pass."""
    t_even = np.arange(t[0], t[-1], 1 / FS)
    sos = butter(FILTER_ORDER, BAND, btype="bandpass", fs=FS, output="sos")
    out = {}
    for ch in "RGB":
        x = np.interp(t_even, t, raw[ch])
        out[ch] = sosfiltfilt(sos, detrend(x / x.mean() - 1))
    return t_even - t_even[0], out


def estimate(seg):
    """All methods on one window. seg = {"R","G","B": arrays}. Returns {method: (bpm, snr)}."""
    R, G, B = seg["R"], seg["G"], seg["B"]
    res = {"green": peak_bpm(*spectrum(G))}

    k = G.std() / R.std()
    res["k_size"] = peak_bpm(*spectrum(G - k * R))

    k = np.cov(G, R)[0, 1] / R.var(ddof=1)
    res["k_least"] = peak_bpm(*spectrum(G - k * R))

    X = np.column_stack([(c - c.mean()) / c.std() for c in (R, G, B)])
    S = FastICA(n_components=3, whiten="unit-variance", random_state=0,
                max_iter=1000).fit_transform(X)
    best = None
    for i in range(3):
        freqs, power = spectrum(S[:, i])
        band = (freqs >= BAND[0]) & (freqs <= BAND[1])
        tallest = power[band].max() / power[band].sum()
        if best is None or tallest > best[0]:
            best = (tallest, peak_bpm(freqs, power))
    res["ica"] = best[1]
    return res


def reference_at(t_center, duration, start, mid, end):
    """Sensor bpm at a time, interpolated between the start/middle/end readings."""
    return np.interp(t_center, [0, duration / 2, duration], [start, mid, end])


def plot_recording(name, condition, rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tc = [r["t_center"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(tc, [r["reference"] for r in rows], color="black", linewidth=2, label="sensor")
    for m, color in zip(METHODS, ["tab:green", "tab:orange", "tab:purple", "tab:blue"]):
        ax.plot(tc, [r[f"{m}_bpm"] for r in rows], marker="o", markersize=3,
                linewidth=1, color=color, label=m)
    ax.set_xlabel("window center (s)")
    ax.set_ylabel("heart rate (bpm)")
    ax.set_title(f"{name} ({condition})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summarize(label, errs):
    e = np.array(errs)
    return (f"{label:<14}{len(e):>5}  " + "".join(
        f"{np.abs(e[:, i]).mean():>7.1f}{(np.abs(e[:, i]) <= 3).mean():>7.0%}"
        for i in range(len(METHODS))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--win", type=float, default=10.0, help="window length in seconds")
    parser.add_argument("--plot", action="store_true", help="save a plot per recording")
    args = parser.parse_args()

    if not RECORDINGS.exists():
        raise SystemExit(f"Missing {RECORDINGS}. Columns: name,condition,bpm_start,bpm_mid,bpm_end")
    with open(RECORDINGS, newline="", encoding="utf-8-sig") as f:
        recs = [r for r in csv.DictReader(f) if r["name"].strip()]

    all_rows, by_cond = [], {}
    for rec in recs:
        name, cond = rec["name"].strip(), rec["condition"].strip()
        if not Path(f"data/traces/{name}_rgb.csv").exists():
            print(f"Skipping {name}: no traces (run extract_rgb.py {name})")
            continue
        refs = [float(rec[k]) for k in ("bpm_start", "bpm_mid", "bpm_end")]
        t, raw = load(name)
        t_even, sig = prepare(t, raw)
        duration = t_even[-1]
        n_win = int(round(args.win * FS))

        rows = []
        start = 0
        while start + n_win <= len(t_even):
            seg = {ch: sig[ch][start:start + n_win] for ch in "RGB"}
            tc = t_even[start] + args.win / 2
            ref = reference_at(tc, duration, *refs)
            row = {"name": name, "condition": cond, "t_center": round(tc, 2),
                   "reference": round(float(ref), 1)}
            for m, (bpm, snr) in estimate(seg).items():
                row[f"{m}_bpm"] = round(bpm, 1)
                row[f"{m}_err"] = round(bpm - ref, 1)
                row[f"{m}_snr"] = round(snr, 1)
            rows.append(row)
            start += int(round(STEP_S * FS))

        all_rows += rows
        by_cond.setdefault(cond, []).extend(
            [[r[f"{m}_err"] for m in METHODS] for r in rows])
        if args.plot:
            (RESULTS_DIR / "windows").mkdir(parents=True, exist_ok=True)
            plot_recording(name, cond, rows, RESULTS_DIR / "windows" / f"{name}.png")

    if not all_rows:
        raise SystemExit("No recordings to evaluate.")

    print(f"\nWindow {args.win:g} s, step {STEP_S:g} s "
          f"(FFT bin spacing {60 / args.win:.0f} bpm before zero-padding)")
    print("Each cell: MAE in bpm, then share of windows within 3 bpm\n")
    header = f"{'condition':<14}{'wins':>5}  " + "".join(f"{m:>14}" for m in METHODS)
    print(header)
    print("-" * len(header))
    for cond in sorted(by_cond):
        print(summarize(cond, by_cond[cond]))
    print("-" * len(header))
    print(summarize("ALL", [e for errs in by_cond.values() for e in errs]))

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"windows_{args.win:g}s.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {out}" + ("  and plots in results/windows/" if args.plot else ""))


if __name__ == "__main__":
    main()
