"""
สร้างคลังข้อมูลขนาดใหญ่ในโดเมน "ระบบประสาท / จิตวิทยา / ชีววิทยา / การเรียนรู้"
จาก Thai Wikipedia เพื่อดัน perplexity ลงให้ถึงเป้า

เหตุผลที่ต้องมีสคริปต์นี้:
  คลังหนังสือที่เขียนเอง 8 บท = 16K token ซึ่งน้อยเกินกว่าจะขยับน้ำหนักโมเดลได้จริง
  CPT ที่เห็นผลต้องการอย่างน้อยระดับ 10M token ขึ้นไป
  Thai Wikipedia มีบทความในโดเมนนี้ราว 61,000 บทความ / 436M อักขระ ≈ 145M token

วิธีคัดเลือก:
  ให้คะแนนบทความจากความหนาแน่นของคำสำคัญในโดเมน แล้วเก็บเฉพาะบทความที่ผ่านเกณฑ์
  ไม่ใช้ "มีคำใดคำหนึ่ง" เพราะคำอย่าง "ยา" หรือ "เซลล์" โผล่ในบทความทั่วไปเยอะเกินไป
  (การกรองหลวมทำให้ได้ 38% ของทั้ง wiki ซึ่งไม่ใช่โดเมนเฉพาะอีกต่อไป)

การใช้งาน:
    python src/build_corpus.py --target-chars 120000000
    python src/build_corpus.py --target-chars 50000000 --min-score 4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import LOG, normalize_thai, resolve, thai_char_ratio

# --------------------------------------------------------------------------- #
# คำสำคัญแบ่งตามน้ำหนัก — คำเฉพาะทางให้คะแนนสูงกว่าคำกว้าง
# --------------------------------------------------------------------------- #
CORE = [  # น้ำหนัก 3 — เจอคำใดคำหนึ่งก็เกือบการันตีว่าอยู่ในโดเมน
    "เซลล์ประสาท", "ระบบประสาท", "สมองส่วน", "ฮิปโปแคมปัส", "คอร์เทกซ์", "ไซแนปส์",
    "สารสื่อประสาท", "โดพามีน", "เซโรโทนิน", "อะเซทิลโคลีน", "ไมอีลิน", "แอกซอน",
    "เดนไดรต์", "นิวรอน", "ประสาทวิทยา", "จิตวิทยา", "พฤติกรรมศาสตร์",
    "การเรียนรู้", "ความทรงจำ", "ความจำระยะ", "สมองกลีบ", "ไขสันหลัง",
    "ต่อมใต้สมอง", "ระบบลิมบิก", "อะมิกดาลา", "ซีรีเบลลัม", "สมองน้อย",
]
MID = [  # น้ำหนัก 2
    "สมอง", "ประสาท", "ฮอร์โมน", "พฤติกรรม", "การรับรู้", "สติปัญญา", "อารมณ์",
    "การนอนหลับ", "ความเครียด", "สรีรวิทยา", "กายวิภาค", "ชีววิทยา", "พันธุกรรม",
    "วิวัฒนาการ", "การทดลอง", "งานวิจัย", "นักวิทยาศาสตร์", "การศึกษาวิจัย",
]
WEAK = [  # น้ำหนัก 1
    "เซลล์", "โปรตีน", "ยีน", "เอนไซม์", "โมเลกุล", "อวัยวะ", "โรค", "การรักษา",
    "กล้ามเนื้อ", "เลือด", "ระบบภูมิคุ้มกัน", "การแพทย์", "สุขภาพ", "ทฤษฎี",
]

# ตัดส่วนท้ายที่ไม่ใช่เนื้อความ — พวกนี้เป็น list ล้วนซึ่งทำลายเป้าหมาย "ประโยคยาว"
TAIL_SECTIONS = re.compile(
    r"\n\s*(อ้างอิง|ดูเพิ่ม|แหล่งข้อมูลอื่น|บรรณานุกรม|หนังสืออ่านเพิ่มเติม|ระเบียงภาพ|เชิงอรรถ)\s*\n"
)
# บรรทัดที่เป็นรายการ/ตาราง/ลิงก์ — ต้องกรองทิ้งตาม STYLE_GUIDE
LIST_LINE = re.compile(r"^\s*([-*•·]|\d+[.)]|\|)")


def score(text: str) -> int:
    head = text[:6000]
    return (
        3 * sum(1 for k in CORE if k in head)
        + 2 * sum(1 for k in MID if k in head)
        + sum(1 for k in WEAK if k in head)
    )


def clean_wiki(text: str) -> str:
    """เก็บเฉพาะย่อหน้าร้อยแก้วยาว ตัดรายการ/ตาราง/ส่วนอ้างอิงทิ้ง"""
    m = TAIL_SECTIONS.search(text)
    if m:
        text = text[: m.start()]

    kept = []
    for para in text.split("\n"):
        p = para.strip()
        if len(p) < 120:            # สั้นเกิน = หัวข้อ/คำบรรยายภาพ
            continue
        if LIST_LINE.match(p):      # เป็นรายการ
            continue
        if p.count("|") > 2:        # เศษตาราง
            continue
        if thai_char_ratio(p) < 0.5:
            continue
        kept.append(p)
    return normalize_thai("\n\n".join(kept))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-chars", type=int, default=120_000_000)
    ap.add_argument("--min-score", type=int, default=6, help="คะแนนขั้นต่ำ (สูง = โดเมนแคบลง)")
    ap.add_argument("--min-chars", type=int, default=2500, help="ความยาวขั้นต่ำหลัง clean")
    ap.add_argument("--out", default="data/domain")
    args = ap.parse_args()

    from datasets import load_dataset

    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "wiki_domain.jsonl"
    val_path = out_dir / "wiki_domain_val.jsonl"

    LOG.info("สตรีม Thai Wikipedia | เป้าหมาย %s อักขระ | min_score=%d",
             f"{args.target_chars:,}", args.min_score)

    ds = load_dataset("wikimedia/wikipedia", "20231101.th", split="train", streaming=True)

    n_seen = n_kept = n_chars = 0
    n_val = 0
    stats = Counter()
    ftrain = open(train_path, "w", encoding="utf-8")
    fval = open(val_path, "w", encoding="utf-8")

    try:
        for row in ds:
            n_seen += 1
            raw = row["text"]
            if len(raw) < args.min_chars:
                stats["สั้นเกิน"] += 1
                continue
            s = score(raw)
            if s < args.min_score:
                stats["คะแนนโดเมนต่ำ"] += 1
                continue
            text = clean_wiki(raw)
            if len(text) < args.min_chars:
                stats["เหลือน้อยหลัง clean"] += 1
                continue

            rec = {"id": f"wiki-{row['id']}", "text": text, "source": "wiki_domain", "score": s}
            # ทุกเอกสารที่ 40 กันไว้เป็น val — สุ่มแบบ deterministic ไม่ต้องโหลดทั้งหมดก่อน
            if n_kept % 40 == 0 and n_val < 400:
                fval.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_val += 1
            else:
                ftrain.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_chars += len(text)
            n_kept += 1

            if n_kept % 2000 == 0:
                LOG.info("  สแกน %s | เก็บ %s | %s อักขระ (%.0f%% ของเป้า)",
                         f"{n_seen:,}", f"{n_kept:,}", f"{n_chars:,}",
                         100 * n_chars / args.target_chars)
            if n_chars >= args.target_chars:
                LOG.info("ถึงเป้าหมายแล้ว")
                break
    finally:
        ftrain.close()
        fval.close()

    LOG.info("=" * 64)
    LOG.info("สแกนทั้งหมด : %s บทความ", f"{n_seen:,}")
    LOG.info("เก็บเข้า train: %s เอกสาร / %s อักขระ", f"{n_kept - n_val:,}", f"{n_chars:,}")
    LOG.info("เก็บเข้า val  : %s เอกสาร", f"{n_val:,}")
    LOG.info("ประมาณ token : %s  (≈3.0 อักขระ/token)", f"{n_chars // 3:,}")
    LOG.info("ที่คัดออก    : %s", dict(stats))
    LOG.info("ไฟล์: %s | %s", train_path, val_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
