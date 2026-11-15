"""Plot the raw R, G, B skin traces from extract_rgb.py.

Usage (PowerShell, with .venv active):
  python plot_traces.py still_02            # first 10 s
  python plot_traces.py still_02 --all      # whole recording
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TRACE_DIR = Path("data/traces")
RESULTS_DIR = Path("results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="recording name, e.g. still_02")
    parser.add_argument("--all", action="store_true", help="plot the whole recording")
    args = parser.parse_args()

    data = np.genfromtxt(TRACE_DIR / f"{args.name}_rgb.csv", delimiter=",", names=True)
    t = data["t_s"]
    keep = slice(None) if args.all else t <= 10.0

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    for ax, ch, color in zip(axes, ["R", "G", "B"], ["tab:red", "tab:green", "tab:blue"]):
        ax.plot(t[keep], data[ch][keep], color=color, linewidth=1)
        ax.set_ylabel(f"{ch} (mean pixel value)")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Raw skin color traces: {args.name}")
    fig.tight_layout()

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.name}_raw_traces.png"
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
