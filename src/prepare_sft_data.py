"""เตรียมข้อมูล SFT (Q&A) โดเมนประสาทวิทยา/จิตเวช จาก Thaweewat/thai-med-pack

ต่างจาก CPT (src/prepare_data.py) ตรงที่:
  - ข้อมูลเป็น {instruction, answer} ไม่ใช่ข้อความต่อเนื่อง
  - loss คำนวณเฉพาะ token ของคำตอบ (prompt ถูก mask ด้วย label=-100)
  - เป้าหมายคือสอน "พฤติกรรมการตอบ" ไม่ใช่ฝัง "ความรู้ใหม่" (นั่นคือหน้าที่ของ CPT)

thai-med-pack (189,190 แถว) เป็นคำถาม-คำตอบจริงระหว่างผู้ป่วยกับแพทย์ ครอบคลุมทุกสาขา
(สูตินรีเวช, ผิวหนัง, จิตเวช, ประสาทวิทยา ฯลฯ) ต้องกรองเฉพาะที่เกี่ยวกับสมอง/ระบบประสาท/จิตเวช
คำที่ใช้กรองเป็นคำที่ผู้ป่วยพูดจริง (ปวดหัว, ชัก, ลืมง่าย) ไม่ใช่ศัพท์วิชาการแบบ build_corpus.py
เพราะคนละ register ของภาษา (ผู้ป่วยไม่พูดว่า "hippocampus")

การใช้งาน:
    python src/prepare_sft_data.py --min-score 4
    python src/prepare_sft_data.py --min-score 4 --out data/sft
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# --------------------------------------------------------------------------- #
# คำที่ผู้ป่วยใช้จริงเวลาถามเรื่องสมอง/ระบบประสาท/จิตเวช
# แบ่งน้ำหนักเหมือน build_corpus.py แต่เปลี่ยนเป็นคำเชิงอาการ ไม่ใช่ศัพท์วิชาการ
# --------------------------------------------------------------------------- #
CORE = [  # น้ำหนัก 3 — อาการ/โรคทางระบบประสาทที่เฉพาะเจาะจงมาก
    "ชัก", "ลมชัก", "อัมพาต", "อัมพฤกษ์", "สมองเสื่อม", "อัลไซเมอร์", "พาร์กินสัน",
    "ไมเกรน", "เส้นเลือดในสมอง", "หลอดเลือดสมอง", "สโตรก", "งูสวัด", "ปลายประสาทอักเสบ",
    "ประสาทหลอน", "ชาปลายมือปลายเท้า", "สั่นโดยไม่ตั้งใจ", "พูดไม่ชัดกะทันหัน",
    "ซึมเศร้า", "โรคซึมเศร้า", "ไบโพลาร์", "จิตเภท", "วิตกกังวล", "แพนิค", "ตื่นตระหนก",
]
MID = [  # น้ำหนัก 2 — เกี่ยวข้องแต่กว้างกว่า
    "ปวดหัว", "ปวดศีรษะ", "เวียนหัว", "มึนหัว", "หน้ามืด", "ความจำ", "ลืมง่าย", "สมาธิสั้น",
    "นอนไม่หลับ", "อารมณ์แปรปรวน", "เครียด", "กังวล", "หดหู่", "ทำร้ายตัวเอง", "ฆ่าตัวตาย",
    "มือสั่น", "ตัวสั่น", "ชาแขนขา", "อ่อนแรง", "เกร็ง", "หมดสติ", "เป็นลม",
]
WEAK = [  # น้ำหนัก 1 — ทั่วไป อาจไม่ใช่ทางประสาท แต่มักปนกัน
    "นอน", "อารมณ์", "จิตใจ", "สมอง", "ประสาท", "โรคจิต", "จิตแพทย์", "จิตวิทยา",
]

INST_RE = re.compile(r"<s>\s*\[INST\](.*?)\[/INST\](.*?)</s>", re.S)


def score(instruction: str, answer: str) -> tuple[int, int]:
    text = instruction + " " + answer
    n_core = sum(1 for k in CORE if k in text)
    total = 3 * n_core + 2 * sum(1 for k in MID if k in text) + sum(1 for k in WEAK if k in text)
    return total, n_core


def parse_row(text: str) -> tuple[str, str] | None:
    m = INST_RE.search(text)
    if not m:
        return None
    instruction, answer = m.group(1).strip(), m.group(2).strip()
    if not instruction or not answer:
        return None
    return instruction, answer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Thaweewat/thai-med-pack")
    ap.add_argument("--min-score", type=int, default=4)
    ap.add_argument("--min-core-hits", type=int, default=1)
    ap.add_argument("--min-answer-chars", type=int, default=80)
    ap.add_argument("--max-answer-chars", type=int, default=3000)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"โหลด {args.dataset} ...")
    ds = load_dataset(args.dataset, split="train")
    print(f"ทั้งหมด {len(ds):,} แถว")

    kept = []
    n_parse_fail = n_too_short = n_too_long = n_low_score = 0
    for row in ds:
        parsed = parse_row(row["text"])
        if parsed is None:
            n_parse_fail += 1
            continue
        instruction, answer = parsed
        if len(answer) < args.min_answer_chars:
            n_too_short += 1
            continue
        if len(answer) > args.max_answer_chars:
            n_too_long += 1
            continue
        total, n_core = score(instruction, answer)
        if n_core < args.min_core_hits or total < args.min_score:
            n_low_score += 1
            continue
        kept.append({"instruction": instruction, "answer": answer, "score": total, "core_hits": n_core})

    print(f"เก็บ {len(kept):,} คู่ | ทิ้ง: parse_fail={n_parse_fail:,} สั้นเกิน={n_too_short:,} "
          f"ยาวเกิน={n_too_long:,} คะแนนต่ำ={n_low_score:,}")

    random.seed(args.seed)
    random.shuffle(kept)
    n_val = max(1, int(len(kept) * args.val_ratio))
    val, train = kept[:n_val], kept[n_val:]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"บันทึก {name}: {len(rows):,} คู่ -> {path}")

    n_chars = sum(len(r["instruction"]) + len(r["answer"]) for r in kept)
    print(f"รวมตัวอักษร (instruction+answer): {n_chars:,} (~{n_chars // 3:,} token โดยประมาณ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
