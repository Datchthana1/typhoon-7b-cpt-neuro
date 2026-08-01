"""
QLoRA Continued Pre-Training สำหรับ typhoon-ai/typhoon-7b บน GPU 8 GB

ประเด็นออกแบบสำคัญ (อธิบายละเอียดใน docs/05_TRAINING_CPT.md):
  • โหลด base เป็น NF4 4-bit → น้ำหนัก 7B เหลือ ~3.9 GB
  • LoRA ครอบ **ทุก linear layer** ไม่ใช่แค่ q/v — CPT ต้องขยับ MLP ด้วย
    เพราะความรู้เชิงข้อเท็จจริงส่วนใหญ่เก็บอยู่ใน feed-forward ไม่ใช่ attention
  • gradient checkpointing เปิดตลอด — แลกความเร็ว ~30% กับ VRAM ที่ประหยัดได้ ~40%
  • paged_adamw_8bit — กัน OOM ตอน optimizer step พุ่ง
  • ไม่มี chat template ไม่มี prompt masking — loss ทุก token คือหัวใจของ CPT

การใช้งาน:
    python src/train_cpt.py                          # ใช้ค่าจาก configs/cpt.yaml
    python src/train_cpt.py --from-study              # ใช้ best params จาก Optuna
    python src/train_cpt.py --override learning_rate=1e-4 lora_r=64
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

from src.utils import LOG, Config, count_trainable, load_tokenizer, resolve, set_seed, vram_report

# ครอบทุก linear ของ Mistral architecture
ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def pick_attn_impl() -> str:
    """flash-attn ถ้ามี ไม่งั้น sdpa (เร็วกว่า eager มากและมากับ torch อยู่แล้ว)"""
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def build_model(cfg: Config, hp: dict, for_hpo: bool = False):
    """สร้าง 4-bit base + LoRA adapter"""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NF4 แม่นกว่า fp4 สำหรับน้ำหนักที่แจกแจงแบบปกติ
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,     # quantize ตัว quantization constant อีกชั้น ประหยัด ~0.4 GB
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=pick_attn_impl(),
        cache_dir=cfg.model.get("cache_dir"),
    )
    model.config.use_cache = False           # ต้องปิดคู่กับ gradient checkpointing
    model.config.pretraining_tp = 1

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora = LoraConfig(
        r=hp["lora_r"],
        lora_alpha=hp["lora_alpha"],
        lora_dropout=hp["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=ALL_LINEAR,
        use_rslora=hp.get("use_rslora", True),  # rank-stabilized: alpha/sqrt(r) ทำให้ปรับ r ได้อิสระ
    )
    model = get_peft_model(model, lora)

    if not for_hpo:
        trainable, total = count_trainable(model)
        LOG.info(
            "พารามิเตอร์ที่เทรน: %s / %s (%.3f%%)",
            f"{trainable:,}", f"{total:,}", 100 * trainable / total,
        )
        LOG.info(vram_report("after-load"))
    return model


def build_args(cfg: Config, hp: dict, out_dir: Path, for_hpo: bool = False) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        # --- batch ---
        per_device_train_batch_size=cfg.train.micro_batch_size,
        per_device_eval_batch_size=cfg.train.micro_batch_size,
        gradient_accumulation_steps=hp["grad_accum"],
        # --- schedule ---
        num_train_epochs=hp.get("epochs", cfg.train.epochs),
        max_steps=hp.get("max_steps", -1),
        learning_rate=hp["learning_rate"],
        lr_scheduler_type=hp["lr_scheduler"],
        warmup_ratio=hp["warmup_ratio"],
        weight_decay=hp["weight_decay"],
        max_grad_norm=hp["max_grad_norm"],
        # --- precision / memory ---
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        # --- eval / logging ---
        # ให้ hp ทับค่าจาก config ได้ เพื่อให้ปรับจังหวะ eval ตามขนาด dataset ได้
        # (corpus เล็กมี step น้อย ถ้า eval_steps ใหญ่กว่าจำนวน step ทั้งหมดจะไม่ eval เลย)
        eval_strategy="steps",
        eval_steps=hp.get("eval_steps", cfg.train.eval_steps),
        logging_steps=hp.get("logging_steps", cfg.train.logging_steps),
        save_strategy="no" if for_hpo else "steps",
        save_steps=hp.get("save_steps", cfg.train.save_steps),
        # เครื่องแครช (nvlddmkm BSOD) ตรงกับจังหวะเซฟ optimizer.pt ของ paged_adamw_8bit
        # ทุกครั้ง (step 1000, 1000, 2000) — ไม่เคยแครชกลาง step ปกติเลย เพราะงั้นข้าม
        # การเซฟ optimizer/scheduler/rng ไปเลย ในเมื่อไม่เคยโหลดสำเร็จอยู่แล้วสักครั้ง
        save_only_model=True,
        save_total_limit=2,
        load_best_model_at_end=not for_hpo,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # --- misc ---
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to=[],
        dataloader_num_workers=0,   # Windows: >0 ทำให้ช้าลงเพราะ spawn ใหม่ทุกครั้ง
        dataloader_pin_memory=True,
        group_by_length=False,      # ไม่จำเป็น — ทุกบล็อกยาวเท่ากันอยู่แล้ว
        disable_tqdm=for_hpo,
        remove_unused_columns=False,
    )


def default_hp(cfg: Config) -> dict:
    h = cfg.hp.to_dict().copy()
    h.setdefault("lora_alpha", h["lora_r"] * 2)
    return h


def load_best_hp(cfg: Config) -> dict:
    """ดึง best params จาก Optuna study"""
    import optuna

    storage = f"sqlite:///{resolve(cfg.paths.output_dir) / 'optuna.db'}"
    study = optuna.load_study(study_name=cfg.optuna.study_name, storage=storage)
    hp = default_hp(cfg)
    hp.update(study.best_params)
    hp["lora_alpha"] = hp["lora_r"] * hp.pop("alpha_ratio", 2)
    LOG.info("โหลด best params จาก trial #%d (value=%.4f)", study.best_trial.number, study.best_value)
    LOG.info("  %s", json.dumps(study.best_params, ensure_ascii=False))
    return hp


def find_resumable_checkpoint(out_dir: Path) -> str | None:
    """หา checkpoint ล่าสุดที่ resume ได้จริง

    ถ้าเครื่องแครช (BSOD/ไฟดับ) กลางการเซฟ checkpoint จะได้โฟลเดอร์ที่ไม่มี
    trainer_state.json ค้างอยู่ — Trainer.train(resume_from_checkpoint=True)
    เดิมจะหยิบโฟลเดอร์ล่าสุดแบบไม่ตรวจสอบ (get_last_checkpoint แค่ดูเลข step
    มากสุด) แล้ว error FileNotFoundError ตอนอ่าน trainer_state.json ทันที
    ฟังก์ชันนี้ข้าม checkpoint ที่เสีย (เปลี่ยนชื่อเป็น .broken ไว้ดูภายหลัง)
    แล้วถอยไปใช้อันก่อนหน้าที่สมบูรณ์แทน
    """
    candidates = sorted(
        out_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
        reverse=True,
    )
    for ckpt in candidates:
        if (ckpt / "trainer_state.json").is_file():
            return str(ckpt)
        LOG.warning(
            "checkpoint เสีย (ไม่มี trainer_state.json — น่าจะแครชกลางการเซฟ) ข้าม: %s", ckpt
        )
        ckpt.rename(ckpt.with_name(ckpt.name + ".broken"))
    return None


class MetricHistoryCallback(TrainerCallback):
    """บันทึก eval metric ทุกจุดลงไฟล์ทันที (flush + fsync) กันเลขหายเวลาเครื่องแครช

    เคยเจอ eval_loss หายเพราะค้างอยู่ใน stdout buffer ตอนเครื่องแครชกลางคัน
    (nvlddmkm BSOD) ไฟล์นี้แยกจาก log/checkpoint โดยสิ้นเชิง เขียนทันทีทุกครั้ง
    ที่ eval เสร็จ ใช้ src/plot_progress.py อ่านมา plot กราฟความคืบหน้าได้
    """

    def __init__(self, path: Path):
        self.path = path

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_world_process_zero or metrics is None or "eval_loss" not in metrics:
            return
        record = {
            "step": state.global_step,
            "epoch": metrics.get("epoch", state.epoch),
            "eval_loss": metrics["eval_loss"],
            "eval_perplexity": math.exp(min(20, metrics["eval_loss"])),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


def train(cfg: Config, hp: dict, out_dir: Path) -> dict:
    set_seed(cfg.seed)
    proc = resolve(cfg.paths.processed_dir)
    train_ds = load_from_disk(str(proc / "train"))
    val_ds = load_from_disk(str(proc / "val"))
    LOG.info("train=%d บล็อก | val=%d บล็อก | seq_len=%d", len(train_ds), len(val_ds), cfg.data.seq_len)

    eff_batch = cfg.train.micro_batch_size * hp["grad_accum"]
    steps_per_epoch = max(1, len(train_ds) // eff_batch)
    LOG.info(
        "effective batch = %d × %d = %d token/step | %d step/epoch",
        eff_batch, cfg.data.seq_len, eff_batch * cfg.data.seq_len, steps_per_epoch,
    )

    tok = load_tokenizer(cfg.model.name, cfg.model.get("cache_dir"))
    model = build_model(cfg, hp)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        args=build_args(cfg, hp, out_dir),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
    )
    trainer.add_callback(MetricHistoryCallback(out_dir / "eval_metrics.jsonl"))

    # กู้คืนจาก checkpoint ล่าสุดอัตโนมัติถ้ามี — ทำให้รันคำสั่งเดิมซ้ำได้ปลอดภัย
    # ทั้งตอนเริ่มใหม่และตอนกู้คืนหลังไฟดับ/เครื่องรีสตาร์ท ไม่ต้องจำ flag พิเศษ
    resume_checkpoint = find_resumable_checkpoint(out_dir)
    if resume_checkpoint:
        LOG.info("พบ checkpoint เดิมที่ %s → กู้คืนและเทรนต่อ", resume_checkpoint)
        # save_only_model=True ไม่เซฟ scheduler.pt เลย ถ้าไม่ทำอะไรเพิ่ม cosine LR
        # schedule จะรีเซ็ตกลับไปเริ่มใหม่ทุกครั้งที่ resume (LambdaLR เริ่ม last_epoch=-1
        # เสมอถ้าไม่ได้ load state) ทั้งที่ global_step เดินหน้าถูกต้อง — fast-forward
        # scheduler เองให้ตรงตำแหน่งจริงก่อนเริ่มเทรนต่อ
        resume_state = json.loads((Path(resume_checkpoint) / "trainer_state.json").read_text(encoding="utf-8"))
        trainer.create_optimizer_and_scheduler(num_training_steps=resume_state["max_steps"])
        for _ in range(resume_state["global_step"]):
            trainer.lr_scheduler.step()
        LOG.info(
            "fast-forward LR scheduler ไปที่ step %d แล้ว (lr=%.3e)",
            resume_state["global_step"], trainer.lr_scheduler.get_last_lr()[0],
        )
    else:
        LOG.info("เริ่มเทรน...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    LOG.info(vram_report("peak"))

    metrics = trainer.evaluate()
    metrics["perplexity"] = math.exp(min(20, metrics["eval_loss"]))
    LOG.info("eval_loss=%.4f | perplexity=%.2f", metrics["eval_loss"], metrics["perplexity"])

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(out_dir / "adapter"))
    tok.save_pretrained(str(out_dir / "adapter"))
    (out_dir / "hp.json").write_text(json.dumps(hp, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("บันทึก adapter ที่ %s", out_dir / "adapter")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cpt.yaml")
    ap.add_argument("--from-study", action="store_true", help="ใช้ best params จาก Optuna")
    ap.add_argument("--override", nargs="*", default=[], help="เช่น learning_rate=1e-4 lora_r=64")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    hp = load_best_hp(cfg) if args.from_study else default_hp(cfg)

    for item in args.override:
        key, _, raw = item.partition("=")
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        hp[key] = val
        LOG.info("override %s = %r", key, val)
    if "lora_r" in [i.split("=")[0] for i in args.override] and "lora_alpha" not in hp:
        hp["lora_alpha"] = hp["lora_r"] * 2

    out_dir = resolve(args.out or (Path(cfg.paths.output_dir) / "final"))
    train(cfg, hp, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
