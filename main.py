"""
CLI รวมของไปป์ไลน์ CPT — typhoon-7b × คลังบทความ neuroplasticity ภาษาไทย

    python main.py prepare      # ทำความสะอาด + dedup + pack
    python main.py baseline     # วัด PPL ของโมเดลตั้งต้น
    python main.py hpo          # ค้น hyperparameter ด้วย Optuna
    python main.py train        # เทรนจริง (--from-study เพื่อใช้ค่าที่ Optuna หาได้)
    python main.py eval         # เทียบก่อน/หลัง
    python main.py merge        # รวม adapter เข้า base แล้ว export
    python main.py status       # ดูว่าไปถึงขั้นไหนแล้ว
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib  import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils import LOG, Config, resolve  # noqa: E402


def run(script: str, extra: list[str]) -> int:
    cmd = [sys.executable, str(ROOT / "src" / script), *extra]
    LOG.info("$ %s", " ".join(cmd[1:]))
    return subprocess.call(cmd)


def status(cfg: Config) -> int:
    raw = resolve(cfg.paths.raw_dir)
    proc = resolve(cfg.paths.processed_dir)
    out = resolve(cfg.paths.output_dir)

    checks = [
        ("① คลังข้อมูลดิบ", raw, lambda p: len(list(p.glob("*.md"))) + len(list(p.glob("*.jsonl")))),
        ("② clean.jsonl", proc / "clean.jsonl", lambda p: 1 if p.exists() else 0),
        ("③ dedup.jsonl", proc / "dedup.jsonl", lambda p: 1 if p.exists() else 0),
        ("④ dataset ที่ pack แล้ว", proc / "train", lambda p: 1 if p.exists() else 0),
        ("⑤ replay corpus", resolve(cfg.paths.replay_dir), lambda p: len(list(p.glob("*.jsonl")))),
        ("⑥ baseline.json", out / "baseline.json", lambda p: 1 if p.exists() else 0),
        ("⑦ optuna.db", out / "optuna.db", lambda p: 1 if p.exists() else 0),
        ("⑧ adapter สุดท้าย", out / "final" / "adapter", lambda p: 1 if p.exists() else 0),
    ]
    print(f"\n{'='*62}\nสถานะไปป์ไลน์\n{'-'*62}")
    for label, path, counter in checks:
        n = counter(path) if path.exists() else 0
        mark = "\033[32m✓\033[0m" if n else "\033[90m·\033[0m"
        detail = f"({n} ไฟล์)" if n > 1 else ""
        print(f"  {mark} {label:<28} {detail}")

    if (proc / "train").exists():
        from datasets import load_from_disk

        tr = load_from_disk(str(proc / "train"))
        va = load_from_disk(str(proc / "val"))
        n_tok = (len(tr) + len(va)) * cfg.data.seq_len
        print(f"{'-'*62}\n  train={len(tr)} บล็อก | val={len(va)} บล็อก | รวม {n_tok:,} token")
    print(f"{'='*62}\n")
    return 0


COMMANDS = {
    "prepare": ("prepare_data.py", ["--stage", "all"]),
    "clean": ("prepare_data.py", ["--stage", "clean"]),
    "pack": ("prepare_data.py", ["--stage", "pack"]),
    "stats": ("prepare_data.py", ["--stats"]),
    "baseline": ("evaluate.py", ["--baseline"]),
    "hpo": ("hpo_optuna.py", []),
    "report": ("hpo_optuna.py", ["--report"]),
    "train": ("train_cpt.py", []),
    "eval": ("evaluate.py", ["--compare"]),
    "probe": ("evaluate.py", ["--probe"]),
    "merge": ("merge_export.py", []),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=[*COMMANDS, "status"])
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="อาร์กิวเมนต์ที่ส่งต่อให้สคริปต์ย่อย")
    args = ap.parse_args()

    if args.command == "status":
        return status(Config.load("configs/cpt.yaml"))

    script, preset = COMMANDS[args.command]
    return run(script, preset + args.rest)


if __name__ == "__main__":
    raise SystemExit(main())
