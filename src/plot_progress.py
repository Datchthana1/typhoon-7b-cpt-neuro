"""Plot eval_loss / perplexity ความคืบหน้าจากไฟล์ eval_metrics.jsonl

MetricHistoryCallback ใน train_cpt.py เขียนไฟล์นี้ทันทีทุกครั้งที่ eval เสร็จ
(flush + fsync กันเลขหายเวลาเครื่องแครช) สคริปต์นี้อ่านมา plot ได้ทุกเมื่อ
แม้ระหว่างที่ยังเทรนอยู่

การใช้งาน:
    python src/plot_progress.py
    python src/plot_progress.py --path outputs_large_v2/run1/eval_metrics.jsonl --out progress.png
"""
from __future__ import annotations

import argparse
import json
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
BLUE = "#2a78d6"
GOOD = "#006300"
BAD = "#d03b3b"


def load_history(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["step"])
    return records


def _strip_chrome(ax) -> None:
    for name, spine in ax.spines.items():
        spine.set_visible(name == "bottom")
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def plot(records: list[dict], out_path: Path, title: str) -> None:
    steps = [r["step"] for r in records]
    losses = [r["eval_loss"] for r in records]
    baseline = losses[0]
    pct = [(l - baseline) / baseline * 100 for l in losses]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Leelawadee UI", "Tahoma", "Segoe UI", "sans-serif"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, facecolor=SURFACE,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.15},
    )

    # --- panel 1: eval_loss ดิบ ---
    ax1.set_facecolor(SURFACE)
    ax1.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax1.plot(steps, losses, color=BLUE, linewidth=2, solid_capstyle="round", zorder=3)
    ax1.scatter(steps, losses, s=64, color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=4)
    ax1.annotate(
        f"{losses[-1]:.4f}",
        xy=(steps[-1], losses[-1]), xytext=(10, 0), textcoords="offset points",
        va="center", ha="left", color=INK_PRIMARY, fontsize=11, fontweight="bold",
    )
    ax1.set_ylabel("eval_loss", color=INK_SECONDARY, fontsize=11)
    ax1.set_title(title, color=INK_PRIMARY, fontsize=14, fontweight="bold", loc="left", pad=12)
    _strip_chrome(ax1)

    # --- panel 2: % เปลี่ยนแปลงเทียบ step แรก (indexed to baseline) ---
    ax2.set_facecolor(SURFACE)
    ax2.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax2.axhline(0, color=AXIS, linewidth=1, zorder=1)
    ax2.fill_between(steps, pct, 0, color=BLUE, alpha=0.10, zorder=2)
    ax2.plot(steps, pct, color=BLUE, linewidth=2, solid_capstyle="round", zorder=3)
    ax2.scatter(steps, pct, s=64, color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=4)
    delta_color = GOOD if pct[-1] < 0 else (BAD if pct[-1] > 0 else INK_MUTED)
    sign = "+" if pct[-1] > 0 else ""
    ax2.annotate(
        f"{sign}{pct[-1]:.2f}%",
        xy=(steps[-1], pct[-1]), xytext=(10, 0), textcoords="offset points",
        va="center", ha="left", color=delta_color, fontsize=11, fontweight="bold",
    )
    ax2.set_ylabel("% จาก step แรก (ลบ = ดีขึ้น)", color=INK_SECONDARY, fontsize=11)
    ax2.set_xlabel("training step", color=INK_SECONDARY, fontsize=11)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _strip_chrome(ax2)

    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="outputs_large_v2/run1/eval_metrics.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="ความคืบหน้าการเทรน — eval_loss ต่อ step")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists() or path.stat().st_size == 0:
        print(f"ยังไม่มีข้อมูล eval — รอ eval รอบแรกก่อน ({path})")
        return 1

    records = load_history(path)
    if len(records) < 2:
        print(f"มีข้อมูลแค่ {len(records)} จุด ต้องมีอย่างน้อย 2 จุดถึงจะเทียบ % ได้ (รอ eval รอบถัดไป)")
        return 1

    out_path = Path(args.out) if args.out else path.parent / "eval_progress.png"
    plot(records, out_path, args.title)
    print(f"บันทึกกราฟที่ {out_path}")
    print(
        f"จุดข้อมูล: {len(records)} | step ล่าสุด: {records[-1]['step']} | "
        f"eval_loss ล่าสุด: {records[-1]['eval_loss']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
