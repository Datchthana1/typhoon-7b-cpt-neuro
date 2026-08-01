"""Plot ผล benchmark สากล (จาก run_benchmark.py) เทียบ step แบบ small-multiples

อ่าน benchmark_history.jsonl (step, task, metric, value ต่อบรรทัด) แล้ว plot
กราฟแยกตาม task หนึ่งช่องต่อหนึ่ง task (คนละ scale กัน เลยไม่ยัดแกน y เดียวกัน)
พร้อม % เปลี่ยนแปลงจาก step แรกของแต่ละ task กำกับไว้ที่จุดสุดท้าย

การใช้งาน:
    python src/plot_benchmark.py
    python src/plot_benchmark.py --path outputs_large_v2/run1/benchmark_history.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GOOD = "#006300"
BAD = "#d03b3b"
# ลำดับ categorical คงที่ตาม palette (ผ่านการตรวจ CVD แล้ว) — ไม่สลับสีตาม task
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def load_history(path: Path) -> dict[str, list[dict]]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_task[rec["task"]].append(rec)
    for records in by_task.values():
        records.sort(key=lambda r: r["step"])
    return dict(by_task)


def _strip_chrome(ax) -> None:
    for name, spine in ax.spines.items():
        spine.set_visible(name == "bottom")
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def plot(by_task: dict[str, list[dict]], out_path: Path, title: str) -> None:
    tasks = list(by_task.keys())
    n = len(tasks)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Leelawadee UI", "Tahoma", "Segoe UI", "sans-serif"]

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4 * nrows), facecolor=SURFACE, squeeze=False,
    )
    fig.suptitle(title, color=INK_PRIMARY, fontsize=15, fontweight="bold", x=0.02, ha="left")

    for i, task in enumerate(tasks):
        ax = axes[i // ncols][i % ncols]
        records = by_task[task]
        steps = [r["step"] for r in records]
        values = [r["value"] for r in records]
        metric_name = records[0]["metric"]
        color = CATEGORICAL[i % len(CATEGORICAL)]

        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
        ax.plot(steps, values, color=color, linewidth=2, solid_capstyle="round", zorder=3)
        ax.scatter(steps, values, s=64, color=color, edgecolors=SURFACE, linewidths=2, zorder=4)

        if len(values) >= 2 and values[0] != 0:
            pct = (values[-1] - values[0]) / abs(values[0]) * 100
            sign = "+" if pct > 0 else ""
            delta_color = GOOD if pct > 0 else (BAD if pct < 0 else INK_MUTED)
            label = f"{values[-1]:.4f}  ({sign}{pct:.2f}%)"
        else:
            delta_color = INK_PRIMARY
            label = f"{values[-1]:.4f}"

        ax.annotate(
            label,
            xy=(steps[-1], values[-1]), xytext=(8, 0), textcoords="offset points",
            va="center", ha="left", color=delta_color, fontsize=10, fontweight="bold",
        )
        ax.set_title(f"{task}  ({metric_name})", color=INK_PRIMARY, fontsize=11, loc="left")
        ax.set_xlabel("training step", color=INK_SECONDARY, fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        _strip_chrome(ax)

    # ปิดช่องที่เหลือถ้า task ไม่พอเต็มกริด
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="outputs_large_v2/run1/benchmark_history.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="ผล Benchmark สากล ต่อ training step")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists() or path.stat().st_size == 0:
        print(f"ยังไม่มีผล benchmark — รัน src/run_benchmark.py ก่อน ({path})")
        return 1

    by_task = load_history(path)
    out_path = Path(args.out) if args.out else path.parent / "benchmark_progress.png"
    plot(by_task, out_path, args.title)

    print(f"บันทึกกราฟที่ {out_path}")
    for task, records in by_task.items():
        print(f"  {task}: {len(records)} จุด, ล่าสุด step {records[-1]['step']} = {records[-1]['value']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
