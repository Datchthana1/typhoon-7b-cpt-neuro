"""
ค้นหา Hyperparameter ด้วย Optuna สำหรับ QLoRA-CPT

จุดที่ต่างจาก HPO ทั่วไป และเป็นหัวใจของงานนี้:

  objective ไม่ได้วัดแค่ "PPL บนโดเมนใหม่ต่ำสุด"
  เพราะค่า lr สูง ๆ จะทำให้ PPL โดเมนใหม่ต่ำได้จริง แต่โมเดล "ลืม" ภาษาไทยทั่วไป
  (catastrophic forgetting) — ซึ่งพังทั้งโมเดล

  score = log(PPL_domain) + λ · max(0, log(PPL_general) − log(PPL_general_base))
                             └── ลงโทษเฉพาะเมื่อ "แย่ลงกว่าเดิม" เท่านั้น ────┘

  λ (forgetting_penalty) ปรับได้ใน configs/optuna.yaml
    λ = 0   → สนใจโดเมนใหม่อย่างเดียว
    λ = 2   → สมดุล (ค่าเริ่มต้น)
    λ = 5   → หวงความสามารถเดิมมาก

การใช้งาน:
    python src/hpo_optuna.py --trials 20
    python src/hpo_optuna.py --resume            # ทำต่อจาก study เดิม
    python src/hpo_optuna.py --report            # สรุปผล + ความสำคัญของแต่ละพารามิเตอร์
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna
import torch
from datasets import load_from_disk
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from transformers import Trainer, TrainerCallback, default_data_collator

from src.train_cpt import build_args, build_model
from src.utils import LOG, Config, resolve, set_seed, vram_report

optuna.logging.set_verbosity(optuna.logging.WARNING)


class PruningCallback(TrainerCallback):
    """รายงาน eval_loss ให้ Optuna ทุกครั้งที่ eval → ตัด trial ที่ไม่มีอนาคตทิ้งเร็ว"""

    def __init__(self, trial: optuna.Trial):
        self.trial = trial
        self.step = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return
        self.step += 1
        self.trial.report(metrics["eval_loss"], self.step)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"ตัดที่ eval#{self.step} loss={metrics['eval_loss']:.4f}")


def suggest(trial: optuna.Trial, space: dict) -> dict:
    """
    สร้าง hyperparameter จาก search space ที่นิยามใน configs/optuna.yaml
    รองรับ 3 แบบ: float (log/linear), int, categorical
    """
    hp: dict = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "float":
            hp[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif kind == "int":
            hp[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
        elif kind == "categorical":
            hp[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"ไม่รู้จัก type '{kind}' ของ {name}")
    return hp


@torch.no_grad()
def eval_loss(trainer: Trainer, dataset, prefix: str) -> float:
    m = trainer.evaluate(eval_dataset=dataset, metric_key_prefix=prefix)
    return m[f"{prefix}_loss"]


def make_objective(cfg: Config, ocfg: Config, baseline: dict):
    proc = resolve(cfg.paths.processed_dir)
    train_full = load_from_disk(str(proc / "train"))
    val_ds = load_from_disk(str(proc / "val"))
    gval_path = proc / "general_val"
    gval_ds = load_from_disk(str(gval_path)) if gval_path.exists() else None

    # HPO ใช้ subset เพื่อให้แต่ละ trial จบเร็ว — ค่าที่ดีบน subset มัก transfer ไป full ได้
    frac = ocfg.subset_ratio
    n = max(8, int(len(train_full) * frac))
    train_ds = train_full.shuffle(seed=cfg.seed).select(range(min(n, len(train_full))))
    LOG.info("HPO ใช้ train subset %d/%d บล็อก (%.0f%%)", len(train_ds), len(train_full), frac * 100)

    lam = ocfg.forgetting_penalty
    base_gen = baseline.get("general_loss")
    space = ocfg.search_space.to_dict()

    def objective(trial: optuna.Trial) -> float:
        set_seed(cfg.seed)
        hp = suggest(trial, space)
        hp["lora_alpha"] = hp["lora_r"] * hp.pop("alpha_ratio", 2)
        hp["max_steps"] = ocfg.max_steps_per_trial
        hp["epochs"] = 1

        model = None
        trainer = None
        try:
            model = build_model(cfg, hp, for_hpo=True)
            trainer = Trainer(
                model=model,
                args=build_args(cfg, hp, resolve(cfg.paths.output_dir) / f"hpo/trial_{trial.number}", for_hpo=True),
                train_dataset=train_ds,
                eval_dataset=val_ds,
                data_collator=default_data_collator,
                callbacks=[PruningCallback(trial)],
            )
            trainer.train()

            domain_loss = eval_loss(trainer, val_ds, "domain")
            score = domain_loss  # = log(PPL_domain)

            gen_loss = None
            if gval_ds is not None and base_gen is not None:
                gen_loss = eval_loss(trainer, gval_ds, "general")
                penalty = lam * max(0.0, gen_loss - base_gen)
                score += penalty
                trial.set_user_attr("general_loss", gen_loss)
                trial.set_user_attr("forget_penalty", penalty)

            trial.set_user_attr("domain_loss", domain_loss)
            trial.set_user_attr("domain_ppl", math.exp(min(20, domain_loss)))
            LOG.info(
                "trial %2d | lr=%.2e r=%d ga=%d | domain_ppl=%.2f%s | score=%.4f",
                trial.number, hp["learning_rate"], hp["lora_r"], hp["grad_accum"],
                math.exp(min(20, domain_loss)),
                f" gen_ppl={math.exp(min(20, gen_loss)):.2f}" if gen_loss is not None else "",
                score,
            )
            return score

        except torch.cuda.OutOfMemoryError:
            LOG.warning("trial %d OOM (r=%d, ga=%d) → prune", trial.number, hp["lora_r"], hp["grad_accum"])
            raise optuna.TrialPruned("OOM")
        finally:
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()

    return objective


def compute_baseline(cfg: Config) -> dict:
    """PPL ของโมเดลตั้งต้น (ยังไม่เทรน) — ใช้เป็นเส้นอ้างอิงของ forgetting penalty"""
    from src.evaluate import evaluate_model

    cache = resolve(cfg.paths.output_dir) / "baseline.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        LOG.info("ใช้ baseline ที่แคชไว้: domain_ppl=%.2f general_ppl=%s",
                 data["domain_ppl"], f"{data.get('general_ppl', float('nan')):.2f}")
        return data
    LOG.info("คำนวณ baseline ของโมเดลตั้งต้น (ทำครั้งเดียว)...")
    data = evaluate_model(cfg, adapter=None)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def report(cfg: Config, ocfg: Config) -> None:
    storage = f"sqlite:///{resolve(cfg.paths.output_dir) / 'optuna.db'}"
    study = optuna.load_study(study_name=ocfg.study_name, storage=storage)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\n{'='*72}\nOptuna study: {ocfg.study_name}")
    print(f"เสร็จ {len(done)} | ถูกตัด {len(pruned)} | ทั้งหมด {len(study.trials)}")
    print(f"\nดีที่สุด: trial #{study.best_trial.number}  score={study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"    {k:20s} = {v}")
    for k, v in study.best_trial.user_attrs.items():
        print(f"    [attr] {k:14s} = {v:.4f}" if isinstance(v, float) else f"    [attr] {k} = {v}")

    print(f"\n{'='*72}\n5 อันดับแรก")
    for t in sorted(done, key=lambda t: t.value)[:5]:
        print(f"  #{t.number:2d} score={t.value:.4f}  ppl={t.user_attrs.get('domain_ppl', 0):.2f}  {t.params}")

    if len(done) >= 4:
        print(f"\n{'='*72}\nความสำคัญของแต่ละพารามิเตอร์ (fANOVA)")
        try:
            for k, v in sorted(
                optuna.importance.get_param_importances(study).items(), key=lambda x: -x[1]
            ):
                bar = "█" * int(v * 40)
                print(f"  {k:20s} {v:6.3f} {bar}")
        except Exception as exc:
            print(f"  คำนวณไม่ได้: {exc}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cpt.yaml")
    ap.add_argument("--optuna-config", default="configs/optuna.yaml")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    ocfg = Config.load(args.optuna_config).optuna
    out_dir = resolve(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.report:
        report(cfg, ocfg)
        return 0

    baseline = compute_baseline(cfg)
    storage = f"sqlite:///{out_dir / 'optuna.db'}"
    study = optuna.create_study(
        study_name=ocfg.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=TPESampler(seed=cfg.seed, n_startup_trials=ocfg.n_startup_trials, multivariate=True),
        pruner=MedianPruner(
            n_startup_trials=ocfg.n_startup_trials,
            n_warmup_steps=ocfg.pruner_warmup_evals,
        ),
    )
    if args.resume:
        LOG.info("ทำต่อจาก study เดิม (มีอยู่แล้ว %d trials)", len(study.trials))

    n_trials = args.trials or ocfg.n_trials
    LOG.info("เริ่ม HPO %d trials | penalty λ=%.1f | %s", n_trials, ocfg.forgetting_penalty, vram_report())

    study.optimize(
        make_objective(cfg, ocfg, baseline),
        n_trials=n_trials,
        gc_after_trial=True,
        catch=(RuntimeError,),
    )
    report(cfg, ocfg)
    LOG.info("ต่อไป: python src/train_cpt.py --from-study")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
