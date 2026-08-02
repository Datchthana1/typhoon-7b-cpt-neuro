"""ค้นหา Hyperparameter (โดยเฉพาะ learning_rate) ด้วย Optuna สำหรับ SFT

ต่างจาก src/hpo_optuna.py (สำหรับ CPT) ตรงที่:
  - ไม่มี forgetting_penalty เพราะ SFT ต่อยอดจาก adapter ของ CPT ที่ rank ล็อกไว้แล้ว
    สอนพฤติกรรมการตอบ ไม่ได้เขียนทับความรู้โดเมนทั้งก้อนแบบ CPT จึงไม่เสี่ยง
    catastrophic forgetting ในลักษณะเดียวกัน — objective จึงเป็น eval_loss ตรง ๆ
  - ไม่ค้น lora_r/alpha เพราะต่อยอดจาก adapter เดิมที่ rank ล็อกไว้แล้ว
  - เหตุที่ต้องมีสคริปต์นี้แยก: เลือก lr=1e-4 เองจากค่าอ้างอิงตอนแรก (QLoRA/Alpaca-LoRA
    ใช้ 2e-4/3e-4 แต่โปรเจกต์นี้ไวต่อ lr สูงผิดปกติจาก RESULTS.md §11 จึงลดลงมา) รันจริงแล้ว
    eval_loss แกว่งแบนไม่ลงชัดใน 500 step แรก จึงกลับมาค้นอย่างเป็นระบบแทนการเดาต่อ

การใช้งาน:
    python src/hpo_optuna_sft.py --trials 15
    python src/hpo_optuna_sft.py --report
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import Subset
from transformers import Trainer

from src.hpo_optuna import PruningCallback, suggest
from src.train_sft import SFTDataset, build_args, build_model, collate
from src.utils import LOG, Config, load_tokenizer, resolve, set_seed, vram_report

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_objective(cfg: Config, ocfg: Config, train_ds: SFTDataset, val_ds: SFTDataset, pad_id: int):
    frac = ocfg.subset_ratio
    n = max(8, int(len(train_ds) * frac))
    g = torch.Generator().manual_seed(cfg.seed)
    idx = torch.randperm(len(train_ds), generator=g)[:n].tolist()
    train_sub = Subset(train_ds, idx)
    LOG.info("HPO ใช้ train subset %d/%d คู่ (%.0f%%)", len(train_sub), len(train_ds), frac * 100)

    # eval บน val เต็ม (1,137 ตัวอย่าง) ใช้เวลา ~12 นาที/รอบ (วัดจากรันจริง) — เร็วเกินไป
    # ที่จะ eval ซ้ำหลายรอบต่อ trial แบบนั้น ใช้ subset เล็กแทนสำหรับ HPO เท่านั้น
    # (การเทรนจริงหลัง HPO ยังคง eval บน val เต็มตามปกติ ไม่กระทบส่วนนี้)
    n_val = min(len(val_ds), 100)
    val_idx = torch.randperm(len(val_ds), generator=torch.Generator().manual_seed(cfg.seed))[:n_val].tolist()
    val_sub = Subset(val_ds, val_idx)
    LOG.info("HPO ใช้ val subset %d/%d คู่ (กัน eval กินเวลาทั้ง trial)", len(val_sub), len(val_ds))

    space = ocfg.search_space.to_dict()
    out_root = resolve(cfg.paths.output_dir) / "hpo_sft"

    def objective(trial: optuna.Trial) -> float:
        set_seed(cfg.seed)
        hp = suggest(trial, space)
        hp["max_steps"] = ocfg.max_steps_per_trial

        model = None
        trainer = None
        try:
            model = build_model(cfg)
            trainer = Trainer(
                model=model,
                args=build_args(cfg, hp, out_root / f"trial_{trial.number}", for_hpo=True),
                train_dataset=train_sub,
                eval_dataset=val_sub,
                data_collator=lambda b: collate(b, pad_id),
                callbacks=[PruningCallback(trial)],
            )
            trainer.train()
            metrics = trainer.evaluate()
            loss = metrics["eval_loss"]

            trial.set_user_attr("eval_loss", loss)
            LOG.info(
                "trial %2d | lr=%.2e warmup=%.3f wd=%.3f | eval_loss=%.4f",
                trial.number, hp["learning_rate"], hp["warmup_ratio"], hp["weight_decay"], loss,
            )
            return loss

        except torch.cuda.OutOfMemoryError:
            LOG.warning("trial %d OOM → prune", trial.number)
            raise optuna.TrialPruned("OOM")
        finally:
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()

    return objective


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--optuna-config", default="configs/optuna_sft.yaml")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    ocfg = Config.load(args.optuna_config).optuna
    out_dir = resolve(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{out_dir / 'optuna_sft.db'}"

    if args.report:
        study = optuna.load_study(study_name=ocfg.study_name, storage=storage)
        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"\n{'='*72}\nOptuna study: {ocfg.study_name}")
        print(f"เสร็จ {len(done)} | ทั้งหมด {len(study.trials)}")
        if done:
            print(f"\nดีที่สุด: trial #{study.best_trial.number}  eval_loss={study.best_value:.4f}")
            for k, v in study.best_params.items():
                print(f"    {k:20s} = {v}")
            print(f"\n{'='*72}\n5 อันดับแรก")
            for t in sorted(done, key=lambda t: t.value)[:5]:
                print(f"  #{t.number:2d} eval_loss={t.value:.4f}  {t.params}")
            if len(done) >= 4:
                print(f"\n{'='*72}\nความสำคัญของแต่ละพารามิเตอร์ (fANOVA)")
                try:
                    for k, v in sorted(
                        optuna.importance.get_param_importances(study).items(), key=lambda x: -x[1]
                    ):
                        print(f"  {k:20s} {v:6.3f} {'█' * int(v * 40)}")
                except Exception as exc:
                    print(f"  คำนวณไม่ได้: {exc}")
        print()
        return 0

    tok = load_tokenizer(cfg.model.name, cfg.model.get("cache_dir"))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    data_dir = resolve(cfg.paths.sft_dir)
    train_ds = SFTDataset(data_dir / "train.jsonl", tok, cfg.data.max_len)
    val_ds = SFTDataset(data_dir / "val.jsonl", tok, cfg.data.max_len)

    study = optuna.create_study(
        study_name=ocfg.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=TPESampler(seed=cfg.seed, n_startup_trials=ocfg.n_startup_trials, multivariate=True),
        pruner=MedianPruner(n_startup_trials=ocfg.n_startup_trials, n_warmup_steps=ocfg.pruner_warmup_evals),
    )
    if args.resume:
        LOG.info("ทำต่อจาก study เดิม (มีอยู่แล้ว %d trials)", len(study.trials))

    n_trials = args.trials or ocfg.n_trials
    LOG.info("เริ่ม HPO (SFT) %d trials | %s", n_trials, vram_report())

    study.optimize(
        make_objective(cfg, ocfg, train_ds, val_ds, tok.pad_token_id),
        n_trials=n_trials,
        gc_after_trial=True,
        catch=(RuntimeError,),
    )

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if done:
        LOG.info("ดีที่สุด: trial #%d eval_loss=%.4f | %s",
                 study.best_trial.number, study.best_value, study.best_params)
    LOG.info("ดูรายงานเต็ม: python src/hpo_optuna_sft.py --report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
