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

# --------------------------------------------------------------------------- #
# สร้าง thinking จาก reasoning ที่หมอเขียนไว้ในคำตอบอยู่แล้ว
#
# ⚠️ ข้อจำกัดที่ต้องเข้าใจ: นี่คือการ **จัดโครงสร้างใหม่จากคำตอบจริง** (extractive)
# ไม่ใช่การสร้าง reasoning ใหม่ด้วย LLM ที่เก่งกว่า (distillation) — เพราะเครื่องนี้
# ไม่มี API key ของโมเดลใดเลย ข้อดีคือไม่มีความเสี่ยงว่า thinking จะขัดกับคำตอบ
# (ทุกประโยคมาจากคำตอบจริง) ข้อเสียคือรูปแบบจะค่อนข้างตายตัว โมเดลจะเรียน
# "ฟอร์แมตของการคิด" ได้ แต่ไม่ได้เรียนการให้เหตุผลที่หลากหลายเท่า distillation จริง
# ถ้าต้องการคุณภาพระดับนั้น ต้องมี API key แล้วเขียน distillation script เพิ่ม
# --------------------------------------------------------------------------- #
GREETING_RE = re.compile(r"^\s*(สวัสดี(ค่ะ|ครับ|คะ)?|ฉันเข้าใจ[^ก-๙]{0,20})\s*")

# คำตอบในชุดนี้เป็น run-on text ล้วน — ไม่มีขึ้นบรรทัด ไม่มีเว้นวรรคคู่ ไม่มีจุดจบประโยค
# (ตรวจจากข้อมูลจริง: newlines=0, doublespace=0 ในเกือบทุกแถว)
# จึงตัดที่ "discourse marker" ที่หมอใช้เปลี่ยนประเด็นแทนการตัดประโยคแบบปกติ
CUES: list[tuple[str, re.Pattern]] = [
    ("flag", re.compile(
        r"(ควรรีบไปพบแพทย์|รีบไปพบแพทย์|รีบพบแพทย์|ควรไปพบแพทย์|ควรพบแพทย์"
        r"|สังเกตอาการ|หากมีอาการ|ถ้ามีอาการ|หากอาการ|ถ้าอาการ)")),
    ("plan", re.compile(
        r"(เบื้องต้นแนะนำ|แนะนำให้|แนะนำการ|ขอแนะนำ|แนะนำ|ควรปฏิบัติ"
        r"|การดูแลตัวเอง|การดูแลเบื้องต้น|วิธีแก้|เบื้องต้นควร)")),
    # "สาเหตุ" เดี่ยว ๆ ตะกละเกินไป (ไปจับกลางวลีอย่าง "สาเหตุ หรืออาจหลบซ่อนตัว")
    # จึงบังคับให้ต้องมีบริบทที่บ่งว่ากำลังจะแจกแจงจริงตามหลัง
    ("cause", re.compile(
        r"(อาจเกิดจาก|มักเกิดจาก|เกิดได้จาก|น่าจะเกิดจาก|เกิดจาก"
        r"|สาเหตุ(อื่น|ที่พบบ่อย|ได้แก่|เช่น|จาก|คือ|หลัก)"
        r"|เข้าได้กับ|อาจเป็น|มักเป็น|อาจจะเป็น)")),
]


# thai-med-pack ผสมคำตอบ 2 แบบ: หมอไทยเขียนเอง (ภาษาเป็นธรรมชาติ) กับที่แปลจากอังกฤษ
# ด้วยเครื่อง (สำนวนแข็ง แปลศัพท์ผิด เช่น "โซเดียมต่ำ (ภาวะโพแทสเซียมในเลือดต่ำ)")
# ตัวหลังทำให้ thinking ที่สกัดออกมาเสียคุณภาพ จึงคัดออกก่อน
MT_ARTIFACTS = re.compile(
    r"(หวังว่าฉันจะ|เรายินดีที่จะช่วยเหลือคุณต่อไป|ขอให้คุณมีสุขภาพที่ดี"
    r"|ฉันเข้าใจสถานการณ์|หวังว่าสิ่งนี้จะช่วย|ขอบคุณ\.|โปรดอย่าลังเลที่จะ"
    r"|ด้วยความเคารพ|ปรึกษาแพทย์ผู้เชี่ยวชาญด้าน)"
)
# หมอไทยในชุดนี้ลงท้ายด้วย ค่ะ/ครับ เสมอ — ใช้เป็นสัญญาณว่าเป็นภาษาไทยต้นฉบับ
NATIVE_THAI = re.compile(r"(ค่ะ|ครับ|นะคะ|นะครับ)")


def looks_native_thai(answer: str) -> bool:
    """คัดเฉพาะคำตอบที่หมอไทยเขียนเอง ไม่ใช่ที่แปลจากอังกฤษด้วยเครื่อง"""
    if MT_ARTIFACTS.search(answer):
        return False
    return bool(NATIVE_THAI.search(answer))


def _trim_at_boundary(seg: str, limit: int) -> str:
    """ตัดข้อความไม่ให้ยาวเกิน limit โดยตัดที่ช่องว่างสุดท้าย ไม่ตัดกลางคำ/กลางวลี"""
    if len(seg) <= limit:
        return seg
    cut = seg[:limit]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > limit * 0.6 else cut).rstrip()


def _segments(body: str) -> list[tuple[str, str]]:
    """หาตำแหน่ง cue ทั้งหมด แล้วตัดข้อความจาก cue หนึ่งไปถึง cue ถัดไป

    คืน [(ชนิด, ข้อความ), ...] เรียงตามตำแหน่งที่ปรากฏจริงในคำตอบ
    """
    hits: list[tuple[int, str]] = []
    for kind, pat in CUES:
        for m in pat.finditer(body):
            hits.append((m.start(), kind))
    if not hits:
        return []
    hits.sort()

    # กัน cue ที่ซ้อนใกล้กันเกินไป (เช่น "แนะนำ" ซ้อนใน "เบื้องต้นแนะนำ")
    dedup: list[tuple[int, str]] = []
    for pos, kind in hits:
        if dedup and pos - dedup[-1][0] < 12:
            continue
        dedup.append((pos, kind))

    out = []
    for i, (pos, kind) in enumerate(dedup):
        end = dedup[i + 1][0] if i + 1 < len(dedup) else len(body)
        seg = _trim_at_boundary(body[pos:end].strip(), 320)
        if len(seg) > 25:
            out.append((kind, seg))
    return out


def build_thinking(instruction: str, answer: str) -> str | None:
    """ดึง reasoning ที่มีอยู่จริงในคำตอบออกมาเรียงเป็นขั้นตอน

    คืน None ถ้าหาโครงสร้างไม่เจอ — กรณีนั้นจะเทรนเป็น Q&A ธรรมดาแทน
    ดีกว่าใส่ thinking กลวง ๆ ที่ไม่ได้สะท้อนการคิดจริง
    """
    if not looks_native_thai(answer):
        return None
    body = GREETING_RE.sub("", answer).strip()
    segs = _segments(body)
    if not segs:
        return None

    causes = [s for k, s in segs if k == "cause"]
    plans = [s for k, s in segs if k == "plan"]
    flags = [s for k, s in segs if k == "flag"]

    # ต้องมีอย่างน้อย "สาเหตุที่เป็นไปได้" หรือ "แนวทาง" ถึงจะถือว่ามี reasoning พอ
    if not causes and not plans:
        return None

    lines = []
    q_head = re.sub(r"\s+", " ", instruction).strip()
    lines.append(f"ผู้ถามเล่าอาการ: {q_head[:220]}{'...' if len(q_head) > 220 else ''}")
    if causes:
        lines.append("แยกสาเหตุที่เป็นไปได้:")
        lines += [f"- {c[:300]}" for c in causes[:5]]
    if flags:
        lines.append("อาการที่ต้องเฝ้าระวัง:")
        lines += [f"- {f[:250]}" for f in flags[:2]]
    if plans:
        lines.append("แนวทางที่จะแนะนำ:")
        lines += [f"- {p[:250]}" for p in plans[:3]]
    return "\n".join(lines)


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
    ap.add_argument("--no-thinking", action="store_true", help="ไม่ต้องสร้าง thinking (เทรน Q&A ล้วน)")
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
        rec = {"instruction": instruction, "answer": answer, "score": total, "core_hits": n_core}
        if not args.no_thinking:
            thinking = build_thinking(instruction, answer)
            if thinking:
                rec["thinking"] = thinking
        kept.append(rec)

    n_think = sum(1 for r in kept if "thinking" in r)
    print(f"เก็บ {len(kept):,} คู่ | ทิ้ง: parse_fail={n_parse_fail:,} สั้นเกิน={n_too_short:,} "
          f"ยาวเกิน={n_too_long:,} คะแนนต่ำ={n_low_score:,}")
    print(f"สร้าง thinking ได้ {n_think:,} คู่ ({100*n_think/max(1,len(kept)):.1f}%) "
          f"— ที่เหลือเทรนเป็น Q&A ธรรมดา (หาโครงสร้าง reasoning ในคำตอบไม่เจอ)")

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
