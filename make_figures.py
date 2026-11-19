"""Summary figures and table for the README.

Runs the full benchmark (notes/recordings.csv) at several window lengths and
saves:
  results/summary.csv               MAE and within-3-bpm per method/condition/window
  results/error_by_condition.png    tracked MAE per condition, at FINAL_WIN
  results/window_sweep.png          tracked MAE vs window length, all recordings

Usage (PowerShell, with .venv active):
  python make_figures.py            (takes a minute or two)
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate import FINAL_WIN, METHODS, RECORDINGS, RESULTS_DIR, run_recording

WINDOWS = [6.0, 10.0, 15.0, 20.0]
CONDITIONS = ["still", "talk", "turn", "dim"]

# Shown in the figures; fixed colors so each method keeps its color everywhere
SHOWN = ["fusion", "ica", "k_least", "green"]
COLORS = {"fusion": "#2a78d6", "ica": "#eb6834", "k_least": "#1baf7a", "green": "#eda100"}
LABELS = {"fusion": "Fusion (final)", "ica": "ICA (Poh 2010)",
          "k_least": "Adaptive k", "green": "Green only"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_recordings():
    with open(RECORDINGS, newline="", encoding="utf-8-sig") as f:
        recs = [r for r in csv.DictReader(f) if r["name"].strip()]
    return [r for r in recs if Path(f"data/traces/{r['name'].strip()}_rgb.csv").exists()]


def score(rows, method):
    e = np.abs([r[f"{method}_tracked_err"] for r in rows])
    return float(e.mean()), float((e <= 3).mean())


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    recs = load_recordings()
    print(f"{len(recs)} recordings with sensor readings")
    results = {}                                    # win -> list of rows
    for win in WINDOWS:
        rows = []
        for r in recs:
            refs = [float(r[k]) for k in ("bpm_start", "bpm_mid", "bpm_end")]
            rows += run_recording(r["name"].strip(), r["condition"].strip(), refs, win)
        results[win] = rows
        print(f"  {win:g} s windows: {len(rows)} windows")

    # ---- table -------------------------------------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)
    table = []
    for win, rows in results.items():
        for cond in CONDITIONS + ["all"]:
            sub = rows if cond == "all" else [r for r in rows if r["condition"] == cond]
            if not sub:
                continue
            for m in METHODS:
                mae, within = score(sub, m)
                table.append({"window_s": win, "condition": cond, "method": m,
                              "windows": len(sub), "mae_bpm": round(mae, 2),
                              "within_3bpm": round(within, 3)})
    with open(RESULTS_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    # ---- figure 1: error by condition at FINAL_WIN -------------------------
    rows = results[FINAL_WIN]
    groups = [c for c in CONDITIONS if any(r["condition"] == c for r in rows)] + ["all"]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style(ax)
    width = 0.8 / len(SHOWN)
    for i, m in enumerate(SHOWN):
        vals = [score(rows if g == "all" else [r for r in rows if r["condition"] == g], m)[0]
                for g in groups]
        x = np.arange(len(groups)) + (i - (len(SHOWN) - 1) / 2) * width
        ax.bar(x, vals, width * 0.88, color=COLORS[m], label=LABELS[m])
    counts = {g: len({r["name"] for r in rows if g == "all" or r["condition"] == g})
              for g in groups}
    ax.set_xticks(np.arange(len(groups)),
                  [f"{g}\n({counts[g]} clip{'s' if counts[g] != 1 else ''})"
                   for g in groups], color=INK)
    ax.set_ylabel("mean absolute error (bpm)", color=INK_2)
    ax.set_title(f"Heart-rate error by condition ({FINAL_WIN:g} s windows, with tracking)",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper left",
              bbox_to_anchor=(0, -0.2), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "error_by_condition.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)

    # ---- figure 2: window-length sweep ------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style(ax)
    for m in SHOWN:
        vals = [score(results[w], m)[0] for w in WINDOWS]
        ax.plot(WINDOWS, vals, color=COLORS[m], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=LABELS[m])
        if m == "fusion":                       # label only the final method;
            ax.annotate(f"{LABELS[m]}: {vals[-2]:.1f} bpm at {WINDOWS[-2]:g} s",
                        (WINDOWS[-2], vals[-2]), xytext=(0, -22),
                        textcoords="offset points", ha="center", fontsize=9,
                        color=INK)              # the others converge, see legend
    ax.set_xticks(WINDOWS, [f"{w:g} s" for w in WINDOWS], color=INK)
    ax.set_xlim(WINDOWS[0] - 1.5, WINDOWS[-1] + 1.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("window length", color=INK_2)
    ax.set_ylabel("mean absolute error (bpm)", color=INK_2)
    ax.set_title("Longer windows are more accurate but react more slowly",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper left",
              bbox_to_anchor=(0, -0.18), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "window_sweep.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)

    # ---- printout ----------------------------------------------------------
    print(f"\nTracked MAE in bpm (share within 3 bpm), {FINAL_WIN:g} s windows")
    print(f"{'method':<10}" + "".join(f"{g:>16}" for g in groups))
    for m in METHODS:
        cells = ""
        for g in groups:
            mae, within = score(rows if g == "all" else
                                [r for r in rows if r["condition"] == g], m)
            cells += f"{mae:>9.1f} ({within:>4.0%})"
        print(f"{m:<10}{cells}")
    print(f"\nTracked MAE (all conditions) by window length")
    print(f"{'method':<10}" + "".join(f"{w:>8g} s" for w in WINDOWS))
    for m in METHODS:
        print(f"{m:<10}" + "".join(f"{score(results[w], m)[0]:>10.1f}" for w in WINDOWS))
    print(f"\nSaved {RESULTS_DIR}/summary.csv, error_by_condition.png, window_sweep.png")


if __name__ == "__main__":
    main()
