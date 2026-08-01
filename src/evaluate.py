"""
ประเมินผล CPT

วัด 3 อย่าง:
  1. Perplexity บนโดเมนใหม่ (val)          — ยิ่งต่ำยิ่งดี = เรียนรู้เนื้อหาแล้ว
  2. Perplexity บนภาษาไทยทั่วไป (general_val) — ต้อง "ไม่แย่ลงมาก" = ไม่ลืมของเดิม
  3. Generation probe — ต่อประโยคจาก prompt ปลายเปิด แล้วดูด้วยตาว่าสำนวนเปลี่ยนไหม
     (CPT ที่สำเร็จ = ต่อประโยคเป็นร้อยแก้ววิชาการต่อเนื่อง ไม่ใช่ตอบเป็น Q&A)

การใช้งาน:
    python src/evaluate.py --baseline                       # โมเดลตั้งต้น
    python src/evaluate.py --adapter outputs/final/adapter  # หลังเทรน
    python src/evaluate.py --compare                        # เทียบก่อน/หลังในตารางเดียว
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from src.utils import LOG, Config, load_tokenizer, resolve, vram_report

PROBES = [
    "ความสามารถของสมองในการปรับเปลี่ยนโครงสร้างตัวเองตามประสบการณ์",
    "เมื่อพฤติกรรมหนึ่งถูกทำซ้ำจนกลายเป็นนิสัย วงจรประสาทที่รับผิดชอบ",
    "งานวิจัยเรื่องคนขับแท็กซี่ในกรุงลอนดอนแสดงให้เห็นว่า",
    "ไมอีลินคือปลอกไขมันที่ห่อหุ้มแอกซอน ซึ่งทำหน้าที่",
    "ความเชื่อที่ว่าการสร้างนิสัยใหม่ต้องใช้เวลายี่สิบเอ็ดวันนั้น",
]


def load_model(cfg: Config, adapter: str | Path | None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
        cache_dir=cfg.model.get("cache_dir"),
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
        LOG.info("โหลด adapter จาก %s", adapter)
    model.eval()
    model.config.use_cache = True
    return model


@torch.no_grad()
def perplexity(model, dataset, batch_size: int = 1) -> float:
    """
    PPL = exp( ผลรวม NLL / จำนวน token )

    ต้องถ่วงน้ำหนักด้วยจำนวน token ไม่ใช่เฉลี่ย loss ต่อ batch เฉย ๆ
    (ที่นี่ทุกบล็อกยาวเท่ากันจึงเท่ากัน แต่เขียนให้ถูกไว้ก่อน)
    """
    total_nll, total_tokens = 0.0, 0
    for i in range(0, len(dataset), batch_size):
        chunk = dataset[i : i + batch_size]
        ids = torch.tensor(chunk["input_ids"], device=model.device)
        out = model(input_ids=ids, labels=ids)
        n_tok = ids.numel() - ids.shape[0]  # shift ทำให้ token สุดท้ายของแต่ละแถวไม่มี label
        total_nll += out.loss.item() * n_tok
        total_tokens += n_tok
    mean_nll = total_nll / max(1, total_tokens)
    return math.exp(min(20.0, mean_nll))


def evaluate_model(cfg: Config, adapter: str | Path | None) -> dict:
    proc = resolve(cfg.paths.processed_dir)
    model = load_model(cfg, adapter)
    result: dict = {"adapter": str(adapter) if adapter else None}

    val = load_from_disk(str(proc / "val"))
    result["domain_ppl"] = perplexity(model, val)
    result["domain_loss"] = math.log(result["domain_ppl"])
    LOG.info("domain PPL  = %.3f  (%d บล็อก)", result["domain_ppl"], len(val))

    gval_path = proc / "general_val"
    if gval_path.exists():
        gval = load_from_disk(str(gval_path))
        result["general_ppl"] = perplexity(model, gval)
        result["general_loss"] = math.log(result["general_ppl"])
        LOG.info("general PPL = %.3f  (%d บล็อก)", result["general_ppl"], len(gval))
    else:
        LOG.warning("ไม่มี general_val → ตรวจ catastrophic forgetting ไม่ได้")

    del model
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def probe(cfg: Config, adapter: str | Path | None, max_new_tokens: int = 160) -> list[dict]:
    tok = load_tokenizer(cfg.model.name, cfg.model.get("cache_dir"))
    model = load_model(cfg, adapter)
    rows = []
    for prompt in PROBES:
        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tok.pad_token_id,
        )
        text = tok.decode(out[0], skip_special_tokens=True)
        rows.append({"prompt": prompt, "continuation": text[len(prompt) :].strip()})
        print(f"\n\033[36m{prompt}\033[0m{rows[-1]['continuation']}")
    del model
    torch.cuda.empty_cache()
    return rows


def compare(cfg: Config, adapter: Path) -> None:
    out_dir = resolve(cfg.paths.output_dir)
    base_path = out_dir / "baseline.json"
    base = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else evaluate_model(cfg, None)
    tuned = evaluate_model(cfg, adapter)

    def delta(a: float, b: float) -> str:
        pct = 100 * (b - a) / a
        color = "\033[32m" if pct < 0 else "\033[31m"
        return f"{color}{pct:+.1f}%\033[0m"

    print(f"\n{'='*70}")
    print(f"{'ตัวชี้วัด':<22}{'ก่อน CPT':>13}{'หลัง CPT':>13}{'เปลี่ยนแปลง':>18}")
    print("-" * 70)
    print(f"{'PPL โดเมนใหม่':<22}{base['domain_ppl']:>13.3f}{tuned['domain_ppl']:>13.3f}"
          f"{delta(base['domain_ppl'], tuned['domain_ppl']):>27}")
    if "general_ppl" in base and "general_ppl" in tuned:
        print(f"{'PPL ไทยทั่วไป':<22}{base['general_ppl']:>13.3f}{tuned['general_ppl']:>13.3f}"
              f"{delta(base['general_ppl'], tuned['general_ppl']):>27}")
    print("=" * 70)
    print("เกณฑ์ตัดสิน: PPL โดเมนใหม่ควรลด >15% และ PPL ไทยทั่วไปควรเพิ่มไม่เกิน 5%")
    if "general_ppl" in base and "general_ppl" in tuned:
        forget = 100 * (tuned["general_ppl"] - base["general_ppl"]) / base["general_ppl"]
        if forget > 5:
            print(f"\033[31m⚠ forgetting {forget:+.1f}% — ลด learning_rate, เพิ่ม replay_ratio, หรือลด epochs\033[0m")
        else:
            print(f"\033[32m✓ forgetting {forget:+.1f}% อยู่ในเกณฑ์\033[0m")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cpt.yaml")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    out_dir = resolve(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)   # กันกรณีใช้ config ที่ชี้ไป output_dir ใหม่
    adapter = Path(args.adapter) if args.adapter else out_dir / "final" / "adapter"

    if args.baseline:
        res = evaluate_model(cfg, None)
        (out_dir / "baseline.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        LOG.info("บันทึก baseline แล้ว")
        if args.probe:
            probe(cfg, None)
        return 0

    if args.compare:
        compare(cfg, adapter)
        if args.probe:
            probe(cfg, adapter)
        return 0

    res = evaluate_model(cfg, adapter if adapter.exists() else None)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.probe:
        probe(cfg, adapter if adapter.exists() else None)
    LOG.info(vram_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
