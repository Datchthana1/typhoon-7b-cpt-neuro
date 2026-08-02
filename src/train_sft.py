"""SFT (Supervised Fine-Tuning) — สอนให้โมเดล "ตอบคำถาม" ต่อจาก CPT ที่สอน "ความรู้"

ต่างจาก src/train_cpt.py 3 จุดที่เป็นหัวใจ:

  1. **loss เฉพาะคำตอบ** — token ของคำถาม (prompt) ถูก mask เป็น -100
     ถ้าคิด loss ทั้งหมดเหมือน CPT โมเดลจะเรียน "วิธีเขียนคำถาม" ไปด้วย ซึ่งไม่ใช่เป้าหมาย

  2. **dynamic padding** — คู่ Q&A ส่วนใหญ่สั้น (p50 = 306 token) ถ้า pad เต็ม 1024
     ทุกแถวเหมือน CPT จะเสีย compute ไปกับ padding ~70% จึง pad เท่าที่ยาวสุดในแต่ละ batch

  3. **เริ่มจาก adapter ของ CPT** — โหลด LoRA ที่ผ่าน CPT มาแล้วเทรนต่อ (is_trainable=True)
     เพื่อไม่ทิ้งความรู้โดเมนที่ได้มา ถ้าอยากเริ่มจาก base เปล่า ๆ ใช้ --no-init-adapter

การใช้งาน:
    python src/train_sft.py --config configs/sft.yaml
    python src/train_sft.py --config configs/sft.yaml --out outputs_sft/run1
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
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.utils import LOG, Config, count_trainable, load_tokenizer, resolve, set_seed, vram_report

ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# typhoon-7b เป็น Mistral architecture — ใช้ template เดียวกับที่ thai-med-pack เก็บข้อมูลมา
PROMPT_TMPL = "<s>[INST] {instruction} [/INST]"


def build_example(rec: dict) -> tuple[str, str]:
    """คืน (prompt, response) — response คือส่วนที่จะคิด loss เท่านั้น"""
    prompt = PROMPT_TMPL.format(instruction=rec["instruction"].strip())
    if rec.get("thinking"):
        response = f"<thinking>\n{rec['thinking'].strip()}\n</thinking>\n\n{rec['answer'].strip()}</s>"
    else:
        response = f"{rec['answer'].strip()}</s>"
    return prompt, response


class SFTDataset(torch.utils.data.Dataset):
    """tokenize แล้ว mask prompt ทิ้ง — เก็บเป็น list ความยาวไม่เท่ากัน (pad ตอน collate)"""

    # เผื่อระยะห่างจาก max_len เสมอ กันกรณี prompt สั้นกว่า max_len แค่นิดเดียว
    # จน truncate แล้วเหลือ response ไม่กี่ token (สัญญาณ loss จะอ่อนเกินจะใช้ได้จริง)
    MIN_RESPONSE_TOKENS = 8

    def __init__(self, path: Path, tok, max_len: int):
        self.rows = []
        n_trunc = n_skipped = 0
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            prompt, response = build_example(rec)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            r_ids = tok(response, add_special_tokens=False)["input_ids"]

            # ⚠️ บั๊กที่เคยเกิดจริง: ถ้า prompt เพียงอย่างเดียวยาวใกล้/เกิน max_len
            # การตัดท้ายจะทำให้เหลือ response 0 token → labels ทั้งแถวเป็น -100 หมด
            # → cross-entropy หารด้วยศูนย์ (ไม่มี token ให้คิด loss เลย) → NaN
            # แล้ว NaN แพร่เข้า gradient ทำให้น้ำหนักโมเดลพังถาวรตั้งแต่ step แรกที่เจอ
            # (เกิดขึ้นจริงกับ 30/21,526 แถวตอนรันจริง ทำให้เทรนพังตั้งแต่ step ~100)
            # ทางแก้ที่ถูกต้องคือ "ข้ามแถวนี้ไปเลย" ไม่ใช่ตัดแล้วฝืนใช้
            if len(p_ids) + self.MIN_RESPONSE_TOKENS > max_len:
                n_skipped += 1
                continue

            ids = p_ids + r_ids
            if len(ids) > max_len:
                ids = ids[:max_len]        # ตัดจากท้าย (คำตอบ) เท่านั้น การันตีแล้วว่าเหลือ >= MIN_RESPONSE_TOKENS
                n_trunc += 1
            # -100 = ไม่คิด loss (ค่าที่ PyTorch cross-entropy ใช้เป็น ignore_index)
            labels = [-100] * len(p_ids) + ids[len(p_ids):]
            self.rows.append({"input_ids": ids, "labels": labels})

        if n_skipped:
            LOG.warning("ข้าม %d แถวที่ prompt เดียวก็เกือบ/เกิน max_len=%d แล้ว (จะทำให้ response เหลือ token ไม่พอ)",
                        n_skipped, max_len)
        if n_trunc:
            LOG.warning("ตัดท้าย %d แถว (%.1f%%) ที่ยาวเกิน max_len=%d",
                        n_trunc, 100 * n_trunc / len(self.rows), max_len)
        # กันไว้อีกชั้น: assert ว่าไม่มีแถวไหนหลุดรอดมาแบบ fully-masked จริง ๆ
        assert all(any(l != -100 for l in r["labels"]) for r in self.rows), \
            "เจอแถวที่ labels เป็น -100 ทั้งแถว — ตรวจ logic ข้างบนใหม่"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch: list[dict], pad_id: int) -> dict:
    """pad เท่าที่ยาวสุดใน batch นี้ ไม่ใช่ pad เต็ม max_len — เร็วกว่ามากเมื่อข้อมูลสั้น"""
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)        # padding ไม่คิด loss
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


class MetricHistoryCallback(TrainerCallback):
    """บันทึก eval metric ทันที (flush+fsync) กันเลขหายถ้าเครื่องแครช — เหมือนใน train_cpt.py"""

    def __init__(self, path: Path):
        self.path = path

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_world_process_zero or metrics is None or "eval_loss" not in metrics:
            return
        rec = {
            "step": state.global_step,
            "epoch": metrics.get("epoch", state.epoch),
            "eval_loss": metrics["eval_loss"],
            "eval_perplexity": math.exp(min(20, metrics["eval_loss"])),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


def find_resumable_checkpoint(out_dir: Path) -> str | None:
    """ข้าม checkpoint ที่เสีย (แครชกลางเซฟ) แล้วถอยไปใช้อันก่อนหน้า — เหมือน train_cpt.py"""
    for ckpt in sorted(out_dir.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]), reverse=True):
        if (ckpt / "trainer_state.json").is_file():
            return str(ckpt)
        LOG.warning("checkpoint เสีย ข้าม: %s", ckpt)
        ckpt.rename(ckpt.with_name(ckpt.name + ".broken"))
    return None


def build_model(cfg: Config, no_init_adapter: bool = False):
    """สร้าง base (4-bit) + ต่อ adapter ของ CPT (หรือ LoRA ใหม่) — แยกออกมาจาก main()
    ให้ src/hpo_optuna_sft.py เรียกใช้ซ้ำได้ ไม่ต้องก็อปโค้ดโหลดโมเดล"""
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
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    init_adapter = cfg.model.get("init_adapter")
    if init_adapter and not no_init_adapter:
        LOG.info("ต่อจาก adapter ของ CPT: %s", init_adapter)
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        LOG.info("เริ่ม LoRA ใหม่จาก base (ไม่ได้ต่อจาก CPT)")
        model = get_peft_model(model, LoraConfig(
            r=cfg.hp.lora_r,
            lora_alpha=cfg.hp.lora_alpha,
            lora_dropout=cfg.hp.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=ALL_LINEAR,
            use_rslora=cfg.hp.get("use_rslora", True),
        ))
    return model


def build_args(cfg: Config, hp: dict, out_dir: Path, for_hpo: bool = False) -> TrainingArguments:
    """แยกออกมาจาก main() เพื่อให้ HPO ตั้งค่า hp ต่อ trial ได้ — เหมือน train_cpt.py"""
    return TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.train.micro_batch_size,
        per_device_eval_batch_size=cfg.train.micro_batch_size,
        gradient_accumulation_steps=hp.get("grad_accum", cfg.hp.grad_accum),
        num_train_epochs=1 if for_hpo else cfg.train.epochs,
        max_steps=hp.get("max_steps", -1),
        learning_rate=hp["learning_rate"],
        lr_scheduler_type=hp.get("lr_scheduler", cfg.hp.lr_scheduler),
        warmup_ratio=hp.get("warmup_ratio", cfg.hp.warmup_ratio),
        weight_decay=hp.get("weight_decay", cfg.hp.weight_decay),
        max_grad_norm=cfg.hp.max_grad_norm,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        eval_strategy="steps",
        # eval บนชุดเต็ม (1,137 ตัวอย่าง) ใช้เวลา ~12 นาทีต่อรอบ (วัดจากรันจริง) ถ้า eval
        # ถี่แบบตอนเทรนจริง (ทุก 5 step) trial สั้น ๆ 80 step จะเสียเวลากับ eval 16 รอบ
        # (~3 ชม.ต่อ trial!) — HPO จึงต้องทั้ง eval ห่างขึ้น (ทุก 20 step) และใช้ val subset
        # เล็กลง (ดู make_objective ใน hpo_optuna_sft.py) พร้อมกันทั้งสองทาง
        eval_steps=20 if for_hpo else cfg.train.eval_steps,
        logging_steps=cfg.train.logging_steps,
        save_strategy="no" if for_hpo else "steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=2,
        save_only_model=True,
        load_best_model_at_end=False if for_hpo else True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        group_by_length=True,
        disable_tqdm=for_hpo,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-init-adapter", action="store_true",
                    help="เริ่ม LoRA ใหม่จาก base เปล่า ๆ แทนที่จะต่อจาก adapter ของ CPT")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    out_dir = resolve(args.out or cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(cfg.model.name, cfg.model.get("cache_dir"))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    data_dir = resolve(cfg.paths.sft_dir)
    train_ds = SFTDataset(data_dir / "train.jsonl", tok, cfg.data.max_len)
    val_ds = SFTDataset(data_dir / "val.jsonl", tok, cfg.data.max_len)
    LOG.info("train=%d คู่ | val=%d คู่ | max_len=%d", len(train_ds), len(val_ds), cfg.data.max_len)

    model = build_model(cfg, no_init_adapter=args.no_init_adapter)

    trainable, total = count_trainable(model)
    LOG.info("พารามิเตอร์ที่เทรน: %s / %s (%.3f%%)", f"{trainable:,}", f"{total:,}",
             100 * trainable / total)
    LOG.info(vram_report("after-load"))

    targs = build_args(cfg, cfg.hp.to_dict(), out_dir)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.add_callback(MetricHistoryCallback(out_dir / "eval_metrics.jsonl"))

    resume = find_resumable_checkpoint(out_dir)
    if resume:
        LOG.info("กู้คืนจาก %s", resume)
        # save_only_model=True ไม่เซฟ scheduler.pt → cosine schedule จะรีเซ็ตกลับไปเริ่มใหม่
        # ทุกครั้งที่ resume ถ้าไม่ fast-forward เอง (เจอปัญหานี้จริงในรอบ CPT)
        state = json.loads((Path(resume) / "trainer_state.json").read_text(encoding="utf-8"))
        trainer.create_optimizer_and_scheduler(num_training_steps=state["max_steps"])
        for _ in range(state["global_step"]):
            trainer.lr_scheduler.step()
        LOG.info("fast-forward LR ไป step %d (lr=%.3e)",
                 state["global_step"], trainer.lr_scheduler.get_last_lr()[0])
    else:
        LOG.info("เริ่มเทรน SFT...")

    trainer.train(resume_from_checkpoint=resume)
    LOG.info(vram_report("peak"))

    metrics = trainer.evaluate()
    metrics["perplexity"] = math.exp(min(20, metrics["eval_loss"]))
    LOG.info("eval_loss=%.4f | perplexity=%.2f", metrics["eval_loss"], metrics["perplexity"])

    trainer.model.save_pretrained(str(out_dir / "adapter"))
    tok.save_pretrained(str(out_dir / "adapter"))
    (out_dir / "hp.json").write_text(json.dumps(cfg.hp.to_dict(), ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    LOG.info("บันทึก adapter ที่ %s", out_dir / "adapter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
