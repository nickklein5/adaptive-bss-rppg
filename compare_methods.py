"""Compare heart-rate methods on recordings with a reference bpm.

Reads notes/reference.csv (columns: name,bpm), runs every method on each
recording's data/traces/<name>_rgb.csv, and prints the error of each.

Methods
  green     green channel only (M2 baseline)
  g_minus_r G - R on z-normalized channels (fixed k = 1 after normalizing)
  k_size    G - k*R, k = std(G) / std(R)          (option 1: compare sizes)
  k_least   G - k*R, k = cov(G,R) / var(R)        (option 2: minimum wobble)
  ica       FastICA, component with the tallest peak (Poh 2010)

For k_size and k_least, each channel is first turned into relative change
(x / mean(x) - 1), so k is measured on the data, not fixed in advance.

Usage (PowerShell, with .venv active):
  python compare_methods.py
"""
import csv
from pathlib import Path

import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt
from sklearn.decomposition import FastICA

from green_hr import BAND, FILTER_ORDER, FS, load, peak_bpm, preprocess, spectrum

REFERENCE = Path("notes/reference.csv")
RESULTS = Path("results/benchmark.csv")
METHODS = ["green", "g_minus_r", "k_size", "k_least", "ica"]


def relative_change(t, x):
    """Resample to FS, express as fractional change from the mean, detrend, band-pass."""
    t_even = np.arange(t[0], t[-1], 1 / FS)
    x = np.interp(t_even, t, x)
    x = detrend(x / x.mean() - 1)
    sos = butter(FILTER_ORDER, BAND, btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, x)


def run_methods(t, raw):
    """Return {method: (bpm, snr_db, k or None)}."""
    z = {ch: preprocess(t, raw[ch])[1] for ch in "RGB"}           # z-normalized
    rel = {ch: relative_change(t, raw[ch]) for ch in "RGB"}        # relative change
    out = {}

    def score(signal):
        return peak_bpm(*spectrum(signal))

    out["green"] = (*score(z["G"]), None)
    out["g_minus_r"] = (*score(z["G"] - z["R"]), None)

    k = rel["G"].std() / rel["R"].std()
    out["k_size"] = (*score(rel["G"] - k * rel["R"]), k)

    k = np.cov(rel["G"], rel["R"])[0, 1] / rel["R"].var(ddof=1)
    out["k_least"] = (*score(rel["G"] - k * rel["R"]), k)

    X = np.column_stack([z["R"], z["G"], z["B"]])
    S = FastICA(n_components=3, whiten="unit-variance", random_state=0,
                max_iter=1000).fit_transform(X)
    best = None
    for i in range(3):
        freqs, power = spectrum(S[:, i])
        band = (freqs >= BAND[0]) & (freqs <= BAND[1])
        tallest = power[band].max() / power[band].sum()
        if best is None or tallest > best[0]:
            best = (tallest, *peak_bpm(freqs, power))
    out["ica"] = (best[1], best[2], None)
    return out


def main():
    if not REFERENCE.exists():
        raise SystemExit(f"Missing {REFERENCE}. Create it with lines like: bench_01,66.3")
    with open(REFERENCE, newline="", encoding="utf-8-sig") as f:
        refs = [(row["name"].strip(), float(row["bpm"])) for row in csv.DictReader(f)]

    rows, errors = [], {m: [] for m in METHODS}
    for name, true_bpm in refs:
        if not Path(f"data/traces/{name}_rgb.csv").exists():
            print(f"Skipping {name}: no traces (run extract_rgb.py {name})")
            continue
        t, raw = load(name)
        res = run_methods(t, raw)
        row = {"name": name, "reference": true_bpm}
        for m in METHODS:
            bpm, snr, k = res[m]
            errors[m].append(bpm - true_bpm)
            row[f"{m}_bpm"] = round(bpm, 1)
            row[f"{m}_err"] = round(bpm - true_bpm, 1)
            row[f"{m}_snr"] = round(snr, 1)
            if k is not None:
                row[f"{m}_k"] = round(k, 2)
        rows.append(row)

    if not rows:
        raise SystemExit("No recordings to compare.")

    # Per-recording table: error in bpm (SNR in dB)
    header = f"{'recording':<11}{'ref':>6}  " + "".join(f"{m:>16}" for m in METHODS)
    print("\nError vs reference in bpm (SNR in dB)\n")
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = "".join(f"{row[f'{m}_err']:>+8.1f} ({row[f'{m}_snr']:+5.1f})" for m in METHODS)
        print(f"{row['name']:<11}{row['reference']:>6.1f}  {cells}")
    print("-" * len(header))

    # Summary: mean absolute error, root-mean-square error, hits within 3 bpm
    print(f"\n{'method':<11}{'MAE':>7}{'RMSE':>7}{'within 3 bpm':>15}")
    for m in METHODS:
        e = np.array(errors[m])
        print(f"{m:<11}{np.abs(e).mean():>7.1f}{np.sqrt((e ** 2).mean()):>7.1f}"
              f"{f'{(np.abs(e) <= 3).sum()}/{len(e)}':>15}")

    ks = [(r["name"], r.get("k_size_k"), r.get("k_least_k")) for r in rows]
    print("\nMeasured k per recording (k_size, k_least):")
    for name, a, b in ks:
        print(f"  {name}: {a}, {b}")

    RESULTS.parent.mkdir(exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(RESULTS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
