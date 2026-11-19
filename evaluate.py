"""Sliding-window evaluation of every method, with and without tracking.

Each recording is cut into overlapping windows (10 s long, moving 1 s at a
time), so one 30 s clip gives ~21 heart-rate estimates per method instead of 1.
Each window is scored against the sensor reading interpolated to its center.

Every method is reported two ways:
  raw      tallest spectral peak in each window, judged on its own
  tracked  Kalman-filter tracking across windows (see tracking.py)

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
  python evaluate.py --name talk_01   # one recording, even without sensor readings
"""
import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

from green_hr import BAND, FILTER_ORDER, FS, load, peak_bpm, spectrum
from tracking import HeartRateTracker

# On short windows FastICA sometimes stops before fully converging. It still
# returns usable components, so hide the warning instead of flooding the screen.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

RECORDINGS = Path("notes/recordings.csv")
RESULTS_DIR = Path("results")
METHODS = ["green", "k_size", "k_least", "ica"]
VARIANTS = ["raw", "tracked"]
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


def method_spectra(seg):
    """Pulse-signal spectrum for every method on one window.

    seg = {"R","G","B": arrays}. Returns {method: (freqs, power)}.
    """
    R, G, B = seg["R"], seg["G"], seg["B"]
    out = {"green": spectrum(G)}

    k = G.std() / R.std()
    out["k_size"] = spectrum(G - k * R)

    k = np.cov(G, R)[0, 1] / R.var(ddof=1)
    out["k_least"] = spectrum(G - k * R)

    X = np.column_stack([(c - c.mean()) / c.std() for c in (R, G, B)])
    S = FastICA(n_components=3, whiten="unit-variance", random_state=0,
                max_iter=1000).fit_transform(X)
    best = None
    for i in range(3):
        freqs, power = spectrum(S[:, i])
        band = (freqs >= BAND[0]) & (freqs <= BAND[1])
        tallest = power[band].max() / power[band].sum()
        if best is None or tallest > best[0]:
            best = (tallest, (freqs, power))
    out["ica"] = best[1]
    return out


def reference_at(t_center, duration, start, mid, end):
    """Sensor bpm at a time, interpolated between the start/middle/end readings."""
    return np.interp(t_center, [0, duration / 2, duration], [start, mid, end])


def plot_recording(name, condition, rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tc = [r["t_center"] for r in rows]
    colors = ["tab:green", "tab:orange", "tab:purple", "tab:blue"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, variant in zip(axes, VARIANTS):
        ax.plot(tc, [r["reference"] for r in rows], color="black", linewidth=2,
                label="sensor")
        for m, color in zip(METHODS, colors):
            ax.plot(tc, [r[f"{m}_{variant}_bpm"] for r in rows], marker="o",
                    markersize=3, linewidth=1, color=color, label=m)
        ax.set_title(variant)
        ax.set_xlabel("window center (s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("heart rate (bpm)")
    axes[0].legend(fontsize=8, ncol=5, loc="upper center",
                   bbox_to_anchor=(1.0, -0.18), frameon=False)
    fig.suptitle(f"{name} ({condition})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def print_table(title, by_cond, variant):
    header = f"{'condition':<14}{'wins':>5}  " + "".join(f"{m:>14}" for m in METHODS)
    print(f"\n{title}")
    print(header)
    print("-" * len(header))

    def line(label, rows):
        cells = ""
        for m in METHODS:
            e = np.array([r[f"{m}_{variant}_err"] for r in rows])
            cells += f"{np.abs(e).mean():>7.1f}{(np.abs(e) <= 3).mean():>7.0%}"
        return f"{label:<14}{len(rows):>5}  {cells}"

    for cond in sorted(by_cond):
        print(line(cond, by_cond[cond]))
    print("-" * len(header))
    print(line("ALL", [r for rows in by_cond.values() for r in rows]))


def run_recording(name, cond="unscored", refs=(np.nan,) * 3, win=10.0):
    """Every method, raw and tracked, on every window of one recording."""
    t, raw = load(name)
    t_even, sig = prepare(t, raw)
    duration = t_even[-1]
    n_win = int(round(win * FS))
    trackers = {m: HeartRateTracker() for m in METHODS}   # fresh per recording

    rows = []
    start = 0
    while start + n_win <= len(t_even):
        seg = {ch: sig[ch][start:start + n_win] for ch in "RGB"}
        tc = t_even[start] + win / 2
        ref = float(reference_at(tc, duration, *refs))
        row = {"name": name, "condition": cond, "t_center": round(tc, 2),
               "reference": round(ref, 1)}
        for m, (freqs, power) in method_spectra(seg).items():
            raw_bpm, snr = peak_bpm(freqs, power)
            trk_bpm = trackers[m].step(freqs, power, dt=STEP_S)
            row[f"{m}_snr"] = round(snr, 1)
            for variant, bpm in (("raw", raw_bpm), ("tracked", trk_bpm)):
                row[f"{m}_{variant}_bpm"] = round(bpm, 1)
                row[f"{m}_{variant}_err"] = round(bpm - ref, 1)
        rows.append(row)
        start += int(round(STEP_S * FS))
    return rows


def estimate_average_hr(name, win=10.0):
    """Average heart rate for a recording: median of each method's tracked bpm.

    Returns (best_bpm, {method: (bpm, snr_db)}). "Best" is k_least, our
    strongest method so far; the others are shown so you can see agreement.
    """
    rows = run_recording(name, win=win)
    if not rows:
        return None, {}
    per = {m: (float(np.median([r[f"{m}_tracked_bpm"] for r in rows])),
               float(np.median([r[f"{m}_snr"] for r in rows]))) for m in METHODS}
    return per["k_least"][0], per


def print_single(name, rows):
    """Summary for one recording: what each method says, with or without a reference."""
    has_ref = np.isfinite(rows[0]["reference"])
    print(f"\n{name}: {len(rows)} windows")
    if has_ref:
        print(f"Sensor: {np.median([r['reference'] for r in rows]):.1f} bpm (median)")
    else:
        print("No sensor readings: showing what each method says, not its error.")
    print(f"\n{'method':<10}{'raw bpm':>10}{'tracked bpm':>13}{'median SNR':>12}"
          + (f"{'tracked MAE':>13}" if has_ref else ""))
    for m in METHODS:
        raw = np.median([r[f"{m}_raw_bpm"] for r in rows])
        trk = np.median([r[f"{m}_tracked_bpm"] for r in rows])
        snr = np.median([r[f"{m}_snr"] for r in rows])
        line = f"{m:<10}{raw:>10.1f}{trk:>13.1f}{snr:>+11.1f} "
        if has_ref:
            line += f"{np.mean([abs(r[f'{m}_tracked_err']) for r in rows]):>12.1f}"
        print(line)
    print("\n(bpm columns are medians across windows)")
    print(f"Plot: results\\windows\\{name}.png   ->   start results\\windows\\{name}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--win", type=float, default=10.0, help="window length in seconds")
    parser.add_argument("--plot", action="store_true", help="save a plot per recording")
    parser.add_argument("--name", default=None,
                        help="look at one recording only; works without sensor readings")
    args = parser.parse_args()

    recs = []
    if RECORDINGS.exists():
        with open(RECORDINGS, newline="", encoding="utf-8-sig") as f:
            recs = [r for r in csv.DictReader(f) if r["name"].strip()]
    elif args.name is None:
        raise SystemExit(f"Missing {RECORDINGS}. Columns: name,condition,bpm_start,bpm_mid,bpm_end")

    if args.name is not None:
        recs = [r for r in recs if r["name"].strip() == args.name] or [
            {"name": args.name, "condition": "unscored",
             "bpm_start": "nan", "bpm_mid": "nan", "bpm_end": "nan"}]
        args.plot = True

    all_rows, by_cond = [], {}
    for rec in recs:
        name, cond = rec["name"].strip(), rec["condition"].strip()
        if not Path(f"data/traces/{name}_rgb.csv").exists():
            print(f"Skipping {name}: no traces (run extract_rgb.py {name})")
            continue
        refs = [float(rec[k]) for k in ("bpm_start", "bpm_mid", "bpm_end")]
        rows = run_recording(name, cond, refs, args.win)
        all_rows += rows
        by_cond.setdefault(cond, []).extend(rows)
        if args.plot:
            (RESULTS_DIR / "windows").mkdir(parents=True, exist_ok=True)
            plot_recording(name, cond, rows, RESULTS_DIR / "windows" / f"{name}.png")

    if not all_rows:
        raise SystemExit("No recordings to evaluate.")

    if args.name is not None:
        print_single(args.name, all_rows)
        return

    print(f"\nWindow {args.win:g} s, step {STEP_S:g} s "
          f"(FFT bin spacing {60 / args.win:.0f} bpm before zero-padding)")
    print("Each cell: MAE in bpm, then share of windows within 3 bpm")
    print_table("RAW: tallest peak per window", by_cond, "raw")
    print_table("TRACKED: Kalman filter across windows", by_cond, "tracked")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"windows_{args.win:g}s.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {out}" + ("  and plots in results/windows/" if args.plot else ""))


if __name__ == "__main__":
    main()
