"""
เตรียมข้อมูล CPT: clean → dedup → tokenize → pack → split

ทำไมต้อง "pack" :
  CPT ที่ดีต้องให้ทุก token มี loss และไม่มี padding เปล่า ๆ
  วิธีคือเอาเอกสารทั้งหมดมาต่อกันโดยคั่นด้วย EOS แล้วหั่นเป็นบล็อกความยาวเท่ากัน
  → GPU ทำงานเต็มประสิทธิภาพ 100% ไม่มี token ทิ้ง

การใช้งาน:
    python src/prepare_data.py --stage all
    python src/prepare_data.py --stage clean       # ทำความสะอาดอย่างเดียว
    python src/prepare_data.py --stage pack        # tokenize + pack
    python src/prepare_data.py --stats             # ดูสถิติ dataset ที่ทำเสร็จแล้ว
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    LOG,
    Config,
    load_tokenizer,
    normalize_thai,
    read_jsonl,
    resolve,
    set_seed,
    thai_char_ratio,
    write_jsonl,
)

# --------------------------------------------------------------------------- #
# 1) CLEAN
# --------------------------------------------------------------------------- #
# artifact ที่โมเดล generative มักหลุดมา — ต้องเอาออกก่อนเทรน
# ไม่งั้นโมเดลจะเรียนที่จะพูดว่า "ในบทนี้เราจะ..." ซึ่งเป็นสำนวน assistant ไม่ใช่สำนวนหนังสือ
META_PATTERNS = [
    re.compile(r"^\s*(ตกลง|แน่นอน|ได้เลย|นี่คือ).{0,80}$", re.M),
    re.compile(r"^\s*(หมายเหตุ|Note)\s*[:：].*$", re.M),
    re.compile(r"^\s*---+\s*$", re.M),
    re.compile(r"^\s*(บทที่ \d+ จบ|จบบทที่ \d+)\s*$", re.M),
    re.compile(r"<\/?(chapter|thinking|answer)>", re.I),
]
MD_INLINE = [
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),      # ตัวหนา → ข้อความธรรมดา
    (re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"), r"\1"),  # ตัวเอียง
    (re.compile(r"`{1,3}([^`]+)`{1,3}"), r"\1"),
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),  # ลิงก์ → เก็บแต่ข้อความ
]


def clean_text(text: str, keep_heading: bool = True) -> str:
    text = normalize_thai(text)
    for pat in META_PATTERNS:
        text = pat.sub("", text)
    for pat, repl in MD_INLINE:
        text = pat.sub(repl, text)

    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            if keep_heading:
                # เก็บชื่อบทไว้เป็นข้อความธรรมดา (เป็นสัญญาณขอบเขตเอกสารที่มีประโยชน์)
                lines.append(stripped.lstrip("# ").strip())
            continue
        lines.append(line)
    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def load_raw(raw_dir: Path) -> list[dict]:
    docs: list[dict] = []
    for path in sorted(raw_dir.glob("*.md")):
        docs.append({"id": path.stem, "text": path.read_text(encoding="utf-8"), "source": "book"})
    for path in sorted(raw_dir.glob("*.jsonl")):
        for i, row in enumerate(read_jsonl(path)):
            docs.append(
                {"id": row.get("id", f"{path.stem}-{i}"), "text": row["text"], "source": row.get("source", path.stem)}
            )
    for path in sorted(raw_dir.glob("*.txt")):
        docs.append({"id": path.stem, "text": path.read_text(encoding="utf-8"), "source": "text"})
    return docs


def stage_clean(cfg: Config) -> Path:
    raw_dir = resolve(cfg.paths.raw_dir)
    out = resolve(cfg.paths.processed_dir) / "clean.jsonl"
    docs = load_raw(raw_dir)
    LOG.info("อ่านเอกสารดิบ %d ชิ้นจาก %s", len(docs), raw_dir)
    if not docs:
        LOG.error("ไม่พบไฟล์ใน %s — รัน generate_book.py ก่อน", raw_dir)
        raise SystemExit(1)

    kept, dropped = [], Counter()
    min_chars = cfg.clean.min_chars
    min_thai = cfg.clean.min_thai_ratio

    for doc in docs:
        text = clean_text(doc["text"], keep_heading=cfg.clean.keep_heading)
        if len(text) < min_chars:
            dropped["สั้นเกินไป"] += 1
            continue
        ratio = thai_char_ratio(text)
        if ratio < min_thai:
            dropped[f"ไทยน้อยกว่า {min_thai:.0%}"] += 1
            continue
        kept.append({"id": doc["id"], "text": text, "source": doc["source"], "chars": len(text)})

    n = write_jsonl(out, kept)
    LOG.info("clean: เก็บ %d / ทิ้ง %s", n, dict(dropped) or "ไม่มี")
    LOG.info("ตัวอักษรรวม %s", f"{sum(d['chars'] for d in kept):,}")
    return out


# --------------------------------------------------------------------------- #
# 2) DEDUP (exact + near-duplicate ด้วย MinHash LSH)
# --------------------------------------------------------------------------- #
def char_ngrams(text: str, n: int = 5) -> set[str]:
    """
    ใช้ character n-gram ไม่ใช่ word n-gram
    เพราะภาษาไทยไม่มีเว้นวรรคระหว่างคำ การตัดคำจะเพิ่ม dependency และ error อีกชั้น
    """
    compact = re.sub(r"\s+", "", text)
    return {compact[i : i + n] for i in range(max(0, len(compact) - n + 1))}


def stage_dedup(cfg: Config) -> Path:
    from datasketch import MinHash, MinHashLSH

    src = resolve(cfg.paths.processed_dir) / "clean.jsonl"
    out = resolve(cfg.paths.processed_dir) / "dedup.jsonl"
    docs = list(read_jsonl(src))

    # ชั้นที่ 1: exact duplicate ด้วย hash
    seen: set[str] = set()
    stage1 = []
    for doc in docs:
        h = hashlib.sha256(re.sub(r"\s+", "", doc["text"]).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        stage1.append(doc)
    LOG.info("dedup ชั้น exact: %d → %d", len(docs), len(stage1))

    # ชั้นที่ 2: near-duplicate
    threshold = cfg.dedup.jaccard_threshold
    num_perm = cfg.dedup.num_perm
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept, removed = [], []
    for i, doc in enumerate(stage1):
        m = MinHash(num_perm=num_perm)
        # update_batch เร็วกว่าเรียก update ทีละ gram หลายเท่า
        # (สำคัญมากเมื่อคลังมีหลักหมื่นเอกสาร × หลักพัน gram ต่อเอกสาร)
        m.update_batch([g.encode("utf-8") for g in char_ngrams(doc["text"], cfg.dedup.ngram)])
        if lsh.query(m):
            removed.append(doc["id"])
            continue
        lsh.insert(doc["id"], m)
        kept.append(doc)
        if len(stage1) > 2000 and (i + 1) % 2000 == 0:
            LOG.info("  dedup %6d/%d | เก็บ %d | ตัด %d", i + 1, len(stage1), len(kept), len(removed))

    LOG.info("dedup ชั้น near (jaccard>%.2f): %d → %d", threshold, len(stage1), len(kept))
    if removed:
        LOG.info("  ตัดออก: %s", ", ".join(removed[:10]) + (" ..." if len(removed) > 10 else ""))
    write_jsonl(out, kept)
    return out


# --------------------------------------------------------------------------- #
# 3) PACK — tokenize + ต่อกัน + หั่นเป็นบล็อก
# --------------------------------------------------------------------------- #
def stage_pack(cfg: Config) -> None:
    from datasets import Dataset

    set_seed(cfg.seed)
    tok = load_tokenizer(cfg.model.name, cfg.model.get("cache_dir"))
    seq_len = cfg.data.seq_len
    proc_dir = resolve(cfg.paths.processed_dir)

    docs = list(read_jsonl(proc_dir / "dedup.jsonl"))
    random.shuffle(docs)

    # --- แบ่ง train/val ระดับ "เอกสาร" ไม่ใช่ระดับบล็อก ---
    # สำคัญมาก: ถ้าแบ่งหลัง pack บล็อกจาก doc เดียวกันจะไปอยู่ทั้ง train และ val
    # → leakage → PPL ดูดีเกินจริง
    n_val = max(1, int(len(docs) * cfg.data.val_ratio))
    val_docs, train_docs = docs[:n_val], docs[n_val:]
    LOG.info("แบ่งระดับเอกสาร: train=%d val=%d", len(train_docs), len(val_docs))

    # --- replay corpus: กัน catastrophic forgetting ---
    replay_dir = resolve(cfg.paths.replay_dir)
    replay_docs: list[dict] = []
    if cfg.data.replay_ratio > 0:
        # ยกเว้น general_val*.jsonl — เป็นชุด "held-out" สำหรับวัด catastrophic
        # forgetting เท่านั้น ถ้าเผลอผสมเข้า train ด้วย จะเกิด leakage ทำให้
        # general PPL ดูดีเกินจริงเพราะโมเดลท่องจำชุดทดสอบไปแล้ว ไม่ใช่เพราะเก่งขึ้นจริง
        for path in sorted(replay_dir.glob("*.jsonl")):
            if path.name.startswith("general_val"):
                continue
            replay_docs.extend(read_jsonl(path))
        if replay_docs:
            train_chars = sum(len(d["text"]) for d in train_docs)
            budget = int(train_chars * cfg.data.replay_ratio)
            random.shuffle(replay_docs)
            picked, acc = [], 0
            for d in replay_docs:
                if acc >= budget:
                    break
                picked.append(d)
                acc += len(d["text"])
            LOG.info(
                "replay: ผสม %d เอกสาร (%s อักขระ ≈ %.0f%% ของ train)",
                len(picked), f"{acc:,}", 100 * acc / max(1, train_chars),
            )
            train_docs = train_docs + picked
            random.shuffle(train_docs)
        else:
            LOG.warning(
                "replay_ratio=%.2f แต่ %s ว่าง → ข้ามไป (ดู docs/04_PREPROCESSING.md วิธีเตรียม)",
                cfg.data.replay_ratio, replay_dir,
            )

    eos = tok.eos_token_id

    def pack(doc_list: list[dict], name: str) -> Dataset:
        """
        ต่อ token ของทุกเอกสารเข้าด้วยกัน คั่นด้วย EOS แล้วหั่นเป็นบล็อกความยาวเท่ากัน

        ใช้ numpy ไม่ใช่ list ของ int เพราะคลังขนาด 20M token
        ถ้าเก็บเป็น Python list จะกิน RAM ~700MB (int object ละ 28 bytes)
        ส่วน numpy int32 กินแค่ 80MB — ต่างกันเกือบสิบเท่า
        """
        import numpy as np

        chunks: list[np.ndarray] = []
        total = 0
        # tokenize เป็นชุด ๆ เพื่อให้ fast tokenizer ใช้ batching ได้
        BATCH = 256
        for i in range(0, len(doc_list), BATCH):
            texts = [d["text"] for d in doc_list[i : i + BATCH]]
            for ids in tok(texts, add_special_tokens=False)["input_ids"]:
                # EOS คั่นระหว่างเอกสาร = บอกโมเดลว่า "จบเรื่องหนึ่ง เริ่มอีกเรื่อง"
                arr = np.asarray(ids + [eos], dtype=np.int32)
                chunks.append(arr)
                total += arr.size
            if len(doc_list) > 5000 and (i // BATCH) % 20 == 0:
                LOG.info("  tokenize %-5s %6d/%d เอกสาร (%s token)",
                         name, min(i + BATCH, len(doc_list)), len(doc_list), f"{total:,}")

        stream = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int32)
        del chunks
        n_blocks = stream.size // seq_len
        if n_blocks == 0:
            raise ValueError(f"{name}: token น้อยเกินไป ({stream.size}) สำหรับ seq_len={seq_len}")
        blocks = stream[: n_blocks * seq_len].reshape(n_blocks, seq_len)
        LOG.info(
            "pack %-5s: %s token → %s บล็อก × %d (ทิ้งเศษ %d token)",
            name, f"{stream.size:,}", f"{n_blocks:,}", seq_len, stream.size - n_blocks * seq_len,
        )
        ids_list = blocks.tolist()
        return Dataset.from_dict(
            {
                "input_ids": ids_list,
                "attention_mask": np.ones((n_blocks, seq_len), dtype=np.int8).tolist(),
                "labels": ids_list,  # CPT: labels = input_ids (คำนวณ loss ทุก token)
            }
        )

    train_ds = pack(train_docs, "train")
    val_ds = pack(val_docs, "val")

    train_ds.save_to_disk(str(proc_dir / "train"))
    val_ds.save_to_disk(str(proc_dir / "val"))

    # --- val ภาษาไทยทั่วไป: ใช้วัด catastrophic forgetting ---
    general = list((resolve(cfg.paths.replay_dir)).glob("general_val*.jsonl"))
    if general:
        gen_docs = [r for p in general for r in read_jsonl(p)]
        pack(gen_docs, "gval").save_to_disk(str(proc_dir / "general_val"))
        LOG.info("บันทึก general_val สำหรับตรวจ forgetting แล้ว")
    else:
        LOG.warning("ไม่พบ data/replay/general_val*.jsonl → จะตรวจ forgetting ไม่ได้")

    LOG.info("บันทึกลง %s เรียบร้อย", proc_dir)


# --------------------------------------------------------------------------- #
def show_stats(cfg: Config) -> None:
    from datasets import load_from_disk

    proc = resolve(cfg.paths.processed_dir)
    tok = load_tokenizer(cfg.model.name, cfg.model.get("cache_dir"))
    for split in ("train", "val", "general_val"):
        path = proc / split
        if not path.exists():
            continue
        ds = load_from_disk(str(path))
        n_tok = len(ds) * cfg.data.seq_len
        LOG.info("%-12s | %5d บล็อก | %s token", split, len(ds), f"{n_tok:,}")
    train = load_from_disk(str(proc / "train"))
    sample = tok.decode(train[0]["input_ids"][:200])
    LOG.info("ตัวอย่างบล็อกแรก:\n%s...", sample)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cpt.yaml")
    ap.add_argument("--stage", choices=["all", "clean", "dedup", "pack"], default="all")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.stats:
        show_stats(cfg)
        return 0

    if args.stage in ("all", "clean"):
        stage_clean(cfg)
    if args.stage in ("all", "dedup"):
        stage_dedup(cfg)
    if args.stage in ("all", "pack"):
        stage_pack(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
