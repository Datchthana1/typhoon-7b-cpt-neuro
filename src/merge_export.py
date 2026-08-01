"""
รวม LoRA adapter เข้ากับ base model แล้ว export เป็นโมเดลเดี่ยว

ข้อควรระวังสำคัญ:
  ห้าม merge adapter เข้ากับ base ที่โหลดแบบ 4-bit
  เพราะ dequantize → บวก ΔW → quantize ใหม่ จะสะสม error จนคุณภาพตก
  วิธีที่ถูกคือโหลด base เป็น fp16/bf16 บน CPU แล้วค่อย merge
  → ใช้ RAM ~15GB แต่ไม่ใช้ VRAM เลย (เครื่อง 8GB ทำได้)

การใช้งาน:
    python src/merge_export.py
    python src/merge_export.py --adapter outputs/final/adapter --out outputs/merged
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from src.utils import LOG, Config, load_tokenizer, resolve


def merge(cfg: Config, adapter: Path, out: Path, dtype: str = "bfloat16") -> Path:
    if not adapter.exists():
        LOG.error("ไม่พบ adapter ที่ %s — รัน train ก่อน", adapter)
        raise SystemExit(1)

    torch_dtype = getattr(torch, dtype)
    LOG.info("โหลด base model เป็น %s บน CPU (ใช้ RAM ~15GB, ไม่ใช้ VRAM)", dtype)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        cache_dir=cfg.model.get("cache_dir"),
    )

    LOG.info("ผนวก adapter จาก %s", adapter)
    model = PeftModel.from_pretrained(base, str(adapter), dtype=torch_dtype)
    model = model.merge_and_unload()
    model.config.use_cache = True

    out.mkdir(parents=True, exist_ok=True)
    LOG.info("บันทึกโมเดลรวมที่ %s", out)
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size="4GB")

    tok = load_tokenizer(cfg.model.name, cfg.model.get("cache_dir"))
    tok.save_pretrained(str(out))

    # เก็บ hp/metrics ไว้ข้าง ๆ โมเดล เพื่อให้ย้อนรอยได้ว่าโมเดลนี้มาจากการเทรนแบบไหน
    for name in ("hp.json", "metrics.json"):
        src = adapter.parent / name
        if src.exists():
            shutil.copy2(src, out / name)

    total = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 1024**3
    LOG.info("เสร็จ — ขนาดรวม %.1f GB", total)
    return out


def print_next_steps(out: Path) -> None:
    print(
        f"""
{'='*72}
โมเดลพร้อมใช้งานที่: {out}

  ▸ ใช้กับ transformers
      from transformers import AutoModelForCausalLM, AutoTokenizer
      m = AutoModelForCausalLM.from_pretrained(r"{out}", dtype="bfloat16", device_map="auto")

  ▸ เสิร์ฟด้วย vLLM (ต้องการ VRAM ~16GB สำหรับ fp16)
      vllm serve "{out}" --dtype bfloat16 --max-model-len 4096

  ▸ แปลงเป็น GGUF เพื่อรันบน CPU / llama.cpp
      python llama.cpp/convert_hf_to_gguf.py "{out}" --outfile typhoon-cpt-f16.gguf
      llama.cpp/llama-quantize typhoon-cpt-f16.gguf typhoon-cpt-q4_k_m.gguf Q4_K_M

  ▸ ทางเลือกที่ไม่ต้อง merge (ประหยัดดิสก์ 15GB)
      โหลด base 4-bit แล้วแนบ adapter ตอน runtime — ดู src/evaluate.py:load_model()
{'='*72}
"""
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cpt.yaml")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()

    cfg = Config.load(args.config)
    out_root = resolve(cfg.paths.output_dir)
    adapter = Path(args.adapter) if args.adapter else out_root / "final" / "adapter"
    out = Path(args.out) if args.out else out_root / "merged"

    merge(cfg, adapter, out, args.dtype)
    print_next_steps(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
