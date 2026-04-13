"""
Generate monitoring data overview figures for thesis chapters 2, 3, 4.
Each chapter uses a tailored set of displacement monitoring points.

Chapter 2: MJ9, MJ1, MJ3          — LightGBM/SHAP feature analysis targets
Chapter 3: MJ9, MJ1, MJ3, ATU4    — LSTM model inputs; MJ1 highlighted
Chapter 4: MJ9, MJ1, MJ3          — warning verification points
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "monitoring data.xlsx"
df = pd.read_excel(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.unicode_minus": False,
})

# Per-chapter configuration
CHAPTER_CFG = {
    "chapter2": {
        "cols":    ["MJ9/mm", "MJ1/mm", "MJ3/mm"],
        "labels":  ["MJ9",    "MJ1",    "MJ3"],
        "colors":  ["#e41a1c", "#377eb8", "#4daf4a"],
        "widths":  [1.2, 1.2, 1.2],
        "styles":  ["-", "-", "-"],
    },
    "chapter3": {
        "cols":    ["MJ9/mm", "MJ1/mm",  "MJ3/mm",  "ATU4/mm"],
        "labels":  ["MJ9",    "MJ1",     "MJ3",     "ATU4"],
        "colors":  ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"],
        "widths":  [1.0, 2.0, 1.0, 1.0],   # MJ1 thicker to highlight
        "styles":  ["-", "-", "-", "--"],
    },
    "chapter4": {
        "cols":    ["MJ9/mm", "MJ1/mm", "MJ3/mm"],
        "labels":  ["MJ9",    "MJ1",    "MJ3"],
        "colors":  ["#e41a1c", "#377eb8", "#4daf4a"],
        "widths":  [1.2, 1.2, 1.2],
        "styles":  ["-", "-", "-"],
    },
}


def make_figure(out_path: Path, cfg: dict):
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax2 = ax1.twinx()

    # ── Rainfall bars (right axis, inverted) ────────────────────────────────
    ax2.bar(df["Date"], df["Rainfall/mm"], color="#4CAF50", alpha=0.5,
            width=1.0, label="降雨量", zorder=1)
    ax2.set_ylim(300, 0)
    ax2.set_ylabel("降雨量 (mm)", fontsize=10)

    # ── Reservoir water level (third axis, offset right) ────────────────────
    rwl = df["RWL/m"]
    rwl_min, rwl_max = rwl.min(), rwl.max()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 55))
    ax3.set_ylim(rwl_min - 5, rwl_max + 5)
    ax3.set_ylabel("库水位 (m)", fontsize=10, color="#1565C0")
    ax3.tick_params(axis="y", colors="#1565C0", labelsize=8)
    ax3.plot(df["Date"], rwl, color="#1565C0", linewidth=1.2,
             linestyle="--", label="库水位", zorder=3)
    ax3.set_yticks(np.arange(int(rwl_min // 5) * 5,
                             int(rwl_max // 5 + 1) * 5 + 1, 5))

    # ── Displacement lines (left axis) ──────────────────────────────────────
    for col, label, color, lw, ls in zip(
            cfg["cols"], cfg["labels"], cfg["colors"],
            cfg["widths"], cfg["styles"]):
        ax1.plot(df["Date"], df[col], color=color, linewidth=lw,
                 linestyle=ls, label=label, zorder=4)

    ax1.set_xlabel("日期", fontsize=10)
    ax1.set_ylabel("累积位移 (mm)", fontsize=10)
    ax1.set_xlim(df["Date"].iloc[0], df["Date"].iloc[-1])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # ── Legend below x-axis ──────────────────────────────────────────────────
    lines1, labels1 = ax1.get_legend_handles_labels()
    bar_handle, bar_label = ax2.get_legend_handles_labels()
    rwl_handle, rwl_label = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + bar_handle + rwl_handle,
               labels1 + bar_label + rwl_label,
               loc="upper center", bbox_to_anchor=(0.5, -0.18),
               ncol=6, fontsize=9, framealpha=0.8)

    ax1.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)   # make room for legend below x-axis
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


BASE = Path(__file__).parent.parent / "docs" / "latex" / "figures"
for ch, cfg in CHAPTER_CFG.items():
    make_figure(BASE / ch / "monitoring_data_overview.pdf", cfg)
