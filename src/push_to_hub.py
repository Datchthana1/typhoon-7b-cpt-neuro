"""
อัปโหลด adapter หรือโมเดลที่ merge แล้วขึ้น Hugging Face Hub

ค่าเริ่มต้นเป็น **private** โดยเจตนา — การเผยแพร่โมเดลสู่สาธารณะเป็นการกระทำที่ย้อนกลับยาก
ต้องระบุ --public อย่างชัดเจนเท่านั้นจึงจะเป็น repo สาธารณะ

สคริปต์จะ **ปฏิเสธการอัปโหลด** ถ้าโมเดลยังไม่ผ่านเกณฑ์คุณภาพใน metrics.json
เว้นแต่จะระบุ --skip-quality-gate

การใช้งาน:
    hf auth login                                        # ครั้งแรกเท่านั้น
    python src/push_to_hub.py --repo user/typhoon-7b-neuro-cpt
    python src/push_to_hub.py --repo user/model --merged  # อัปโหลดโมเดลเต็มแทน adapter
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import LOG, Config, resolve

CARD = """---
license: apache-2.0
language:
- th
base_model: typhoon-ai/typhoon-7b
tags:
- continued-pretraining
- cpt
- qlora
- thai
- neuroscience
library_name: peft
---

# {repo}

LoRA adapter จากการทำ **Continued Pre-Training (CPT)** บน `typhoon-ai/typhoon-7b`
ด้วยคลังบทความภาษาไทยแนวสารคดีวิทยาศาสตร์ เรื่องความยืดหยุ่นของระบบประสาทและการฝึกฝน

## ผลการประเมิน

| ตัวชี้วัด | ก่อน CPT | หลัง CPT | เปลี่ยนแปลง |
|---|---|---|---|
| PPL โดเมนใหม่ | {base_domain:.3f} | {tuned_domain:.3f} | {d_domain:+.1f}% |
| PPL ภาษาไทยทั่วไป (Wikipedia) | {base_general:.3f} | {tuned_general:.3f} | {d_general:+.1f}% |

ตัวเลขที่สองคือการตรวจ catastrophic forgetting — ยิ่งเปลี่ยนแปลงน้อยยิ่งดี

## รายละเอียดการเทรน

| | |
|---|---|
| วิธี | QLoRA (NF4 4-bit + double quant) |
| LoRA rank / alpha | {lora_r} / {lora_alpha} |
| Target modules | q, k, v, o, gate, up, down (ทุก linear layer) |
| Learning rate | {lr} |
| Scheduler | {sched} |
| Sequence length | {seq_len} |
| Effective batch | {eff_batch} บล็อก ({eff_tokens:,} token/step) |
| GPU | {gpu} |

## ⚠️ ข้อจำกัดที่ต้องอ่านก่อนใช้

- **คลังข้อมูล**: เทรนบน {n_blocks} บล็อก (~{n_tokens:,} token)
  CPT ที่เห็นผลชัดมักต้องการระดับ 10M token ขึ้นไป — ต่ำกว่านั้นผลที่ได้จะเป็นระดับ
  *สำนวนและคำศัพท์เฉพาะทาง* มากกว่าการฝังความรู้ทั้งก้อน
- **ยังไม่ได้ตรวจสอบความถูกต้องเชิงข้อเท็จจริงของข้อความที่โมเดลสร้าง** สำหรับ checkpoint นี้
  โมเดลภาษาโดยทั่วไปสามารถแต่งชื่อทฤษฎี ปีงานวิจัย หรือตัวเลขที่ไม่มีอยู่จริงได้
  (ตรวจได้ด้วย `python main.py probe`)
- **PPL ที่ดีขึ้นไม่ได้รับประกันความถูกต้องของเนื้อหา** — วัดแค่ว่าโมเดลทำนายข้อความในโดเมนได้ดีขึ้น
- **ห้ามใช้เป็นแหล่งอ้างอิงทางการแพทย์หรือวิชาการโดยเด็ดขาด**
- นี่คือ **base model ที่ผ่าน CPT** ไม่ใช่ instruct model — ใช้ต่อประโยค ไม่ใช่ตอบคำถาม
- ยังไม่ได้ผ่าน SFT หรือ alignment ใด ๆ

เนื้อหาที่ใช้เทรนถูกตรวจสอบข้อเท็จจริงเทียบกับงานวิจัยต้นทางแล้ว (ตาราง fact 48 รายการ)
แต่นั่นรับประกันคุณภาพของ *ข้อมูลนำเข้า* เท่านั้น ไม่ได้รับประกัน *ผลลัพธ์ที่โมเดลสร้าง*

## วิธีใช้

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("typhoon-ai/typhoon-7b", dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "{repo}")
tok = AutoTokenizer.from_pretrained("typhoon-ai/typhoon-7b")

prompt = "ความสามารถของสมองในการปรับเปลี่ยนโครงสร้างตัวเองตามประสบการณ์"
out = model.generate(**tok(prompt, return_tensors="pt").to(model.device), max_new_tokens=200)
print(tok.decode(out[0], skip_special_tokens=True))
```
"""


def check_auth() -> str:
    from huggingface_hub import whoami

    try:
        return whoami()["name"]
    except Exception as exc:
        LOG.error("ยืนยันตัวตนกับ Hugging Face ไม่สำเร็จ: %s", exc)
        LOG.error("รัน `hf auth login` ก่อน แล้วลองใหม่")
        raise SystemExit(1)


def quality_gate(cfg: Config, adapter: Path, skip: bool) -> tuple[dict, dict]:
    """
    ตรวจว่าโมเดลผ่านเกณฑ์ก่อนอนุญาตให้ push
      • domain PPL ต้องลดลงอย่างน้อย 5%
      • general PPL ต้องเพิ่มไม่เกิน 5%  (catastrophic forgetting)

    หมายเหตุ: เกณฑ์นี้ดู PPL อย่างเดียว ซึ่ง **ไม่พอ**
    ต้องรัน `python main.py probe` แล้วอ่านด้วยตาว่าข้อเท็จจริงถูกต้องด้วย
    เพราะโมเดลที่ PPL ดีขึ้นยังอาจแต่งชื่องานวิจัยขึ้นมาเองได้ (ดู RESULTS.md §6.5)
    """
    out = resolve(cfg.paths.output_dir)
    base_path = out / "baseline.json"
    if not base_path.exists():
        LOG.error("ไม่พบ baseline.json — รัน `python main.py baseline` ก่อน")
        raise SystemExit(1)
    base = json.loads(base_path.read_text(encoding="utf-8"))

    LOG.info("ประเมิน %s เพื่อตรวจเกณฑ์...", adapter)
    from src.evaluate import evaluate_model

    tuned = evaluate_model(cfg, adapter)

    d_domain = 100 * (tuned["domain_ppl"] - base["domain_ppl"]) / base["domain_ppl"]
    d_general = (
        100 * (tuned["general_ppl"] - base["general_ppl"]) / base["general_ppl"]
        if "general_ppl" in base and "general_ppl" in tuned
        else 0.0
    )
    LOG.info("domain %+.1f%% | general %+.1f%%", d_domain, d_general)

    problems = []
    if d_domain > -5:
        problems.append(f"domain PPL ลดเพียง {-d_domain:.1f}% (ต้องลดอย่างน้อย 5%)")
    if d_general > 5:
        problems.append(f"general PPL เพิ่ม {d_general:.1f}% (catastrophic forgetting)")

    if problems:
        LOG.error("ไม่ผ่านเกณฑ์คุณภาพ:")
        for p in problems:
            LOG.error("  • %s", p)
        if not skip:
            LOG.error("ยกเลิกการอัปโหลด — ใช้ --skip-quality-gate ถ้าต้องการ push อยู่ดี")
            raise SystemExit(1)
        LOG.warning("ข้ามเกณฑ์ตามที่ระบุ — กำลัง push โมเดลที่ยังไม่ผ่านเกณฑ์")
    else:
        LOG.info("✓ ผ่านเกณฑ์คุณภาพ")
    return base, tuned


def build_card(cfg: Config, repo: str, base: dict, tuned: dict, adapter: Path) -> str:
    import torch

    # hp.json ถูกเขียนตอนเทรน "จบครบทุก epoch" เท่านั้น ถ้า push จาก checkpoint กลางทาง
    # (เช่นหยุดเพราะ overfit แล้วเอา checkpoint ที่ดีที่สุด) จะไม่มีไฟล์นี้ →
    # เดิม fallback เป็น "?" กับ grad_accum=16 ทำให้ model card แสดงค่าผิด
    # ตอนนี้ถอยไปอ่านจาก cfg.hp ซึ่งเป็นค่าที่ใช้เทรนจริงเสมอ
    hp_path = adapter.parent / "hp.json"
    if hp_path.exists():
        hp = json.loads(hp_path.read_text(encoding="utf-8"))
    else:
        hp = cfg.hp.to_dict().copy()
        LOG.info("ไม่พบ %s (น่าจะ push จาก checkpoint กลางทาง) → ใช้ค่า hp จาก config แทน", hp_path)
    eff = cfg.train.micro_batch_size * hp.get("grad_accum", cfg.hp.get("grad_accum", 16))
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"

    def pct(a: float, b: float) -> float:
        return 100 * (b - a) / a

    n_blocks = 0
    train_dir = resolve(cfg.paths.processed_dir) / "train"
    if train_dir.exists():
        from datasets import load_from_disk

        n_blocks = len(load_from_disk(str(train_dir)))

    return CARD.format(
        n_blocks=n_blocks,
        n_tokens=n_blocks * cfg.data.seq_len,
        repo=repo,
        base_domain=base["domain_ppl"],
        tuned_domain=tuned["domain_ppl"],
        d_domain=pct(base["domain_ppl"], tuned["domain_ppl"]),
        base_general=base.get("general_ppl", float("nan")),
        tuned_general=tuned.get("general_ppl", float("nan")),
        d_general=pct(base.get("general_ppl", 1), tuned.get("general_ppl", 1)),
        lora_r=hp.get("lora_r", "?"),
        lora_alpha=hp.get("lora_alpha", "?"),
        lr=hp.get("learning_rate", "?"),
        sched=hp.get("lr_scheduler", "?"),
        seq_len=cfg.data.seq_len,
        eff_batch=eff,
        eff_tokens=eff * cfg.data.seq_len,
        gpu=gpu,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cpt.yaml")
    ap.add_argument("--repo", required=True, help="เช่น username/typhoon-7b-neuro-cpt")
    ap.add_argument("--adapter", default=None, help="path ของ adapter (ค่าเริ่มต้น outputs/final/adapter)")
    ap.add_argument("--merged", action="store_true", help="อัปโหลดโมเดลเต็มแทน adapter")
    ap.add_argument("--public", action="store_true", help="ทำเป็น repo สาธารณะ (ค่าเริ่มต้นคือ private)")
    ap.add_argument("--skip-quality-gate", action="store_true")
    ap.add_argument("--message", default="Upload CPT adapter")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    cfg = Config.load(args.config)
    user = check_auth()
    LOG.info("เข้าสู่ระบบในชื่อ: %s", user)

    out = resolve(cfg.paths.output_dir)
    folder = (
        out / "merged"
        if args.merged
        else (Path(args.adapter) if args.adapter else out / "final" / "adapter")
    )
    if not folder.exists():
        LOG.error("ไม่พบ %s", folder)
        raise SystemExit(1)

    base, tuned = quality_gate(cfg, folder, args.skip_quality_gate)

    card_path = folder / "README.md"
    card_path.write_text(build_card(cfg, args.repo, base, tuned, folder), encoding="utf-8")
    LOG.info("สร้าง model card แล้ว")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    LOG.info("อัปโหลด %s → %s (%s)", folder, args.repo, "public" if args.public else "private")
    api.upload_folder(folder_path=str(folder), repo_id=args.repo, commit_message=args.message)
    LOG.info("เสร็จ: https://huggingface.co/%s", args.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
