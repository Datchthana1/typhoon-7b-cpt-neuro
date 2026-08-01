"""วัดความแม่นยำของ checkpoint เทียบ benchmark สากล (ผ่าน lm-evaluation-harness)

lm-evaluation-harness คือเครื่องมือมาตรฐานที่อยู่เบื้องหลัง HuggingFace Open LLM
Leaderboard และ paper ส่วนใหญ่ที่รายงานตัวเลข benchmark ของ LLM

⚠️ ห้ามรันพร้อมกับ train_cpt.py บน GPU เดียวกัน — VRAM เต็มอยู่แล้วระหว่างเทรน
รันสคริปต์นี้หลังเทรนจบ หรือหยุดเทรนไว้ชั่วคราวก่อน (Ctrl+C ปลอดภัย เพราะมี
checkpoint กู้คืนได้ทุก 500 step อยู่แล้ว)

การใช้งาน:
    # benchmark checkpoint ล่าสุดใน outputs_large_v2/run1 อัตโนมัติ
    python src/run_benchmark.py

    # ระบุ checkpoint เอง
    python src/run_benchmark.py --checkpoint outputs_large_v2/run1/checkpoint-2000

    # เพิ่ม MMLU ภาษาไทยเต็มชุด (57 วิชา — ช้ากว่ามาก แนะนำรันแยกทีหลัง)
    python src/run_benchmark.py --tasks mmlu_th_llama --limit 20

การหา task อื่นๆ (ยืนยันชื่อจริงก่อนใช้ ชื่อเดามักไม่ตรง):
    python -c "from lm_eval.tasks import TaskManager; tm=TaskManager(); \
        print([t for t in tm.all_tasks if 'thai' in t.lower() or '_th' in t.lower()])"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ตรวจสอบแล้วว่ามีจริงใน lm-eval task registry (2026-07-30):
#   belebele_tha_Thai — Belebele: อ่านจับใจความหลายภาษา รวมไทย (มาตรฐานสากล)
#   xcopa_th          — XCOPA: เหตุผลเชิงสามัญสำนึกแบบ causal ภาษาไทย
# ไม่ใส่ mmlu_th_llama ในค่า default เพราะเป็น group ที่มี ~57 วิชาย่อย ช้ามาก
# บน GPU 8 GB เครื่องนี้ — รันแยกเองด้วย --tasks mmlu_th_llama ถ้าต้องการ
DEFAULT_TASKS = ["belebele_tha_Thai", "xcopa_th"]
BASE_MODEL = "typhoon-ai/typhoon-7b"


def latest_checkpoint(out_dir: Path) -> Path:
    candidates = [
        p for p in out_dir.glob("checkpoint-*")
        if p.is_dir() and (p / "trainer_state.json").is_file()
    ]
    if not candidates:
        raise SystemExit(f"ไม่เจอ checkpoint ที่ใช้งานได้ใน {out_dir}")
    return max(candidates, key=lambda p: int(p.name.split("-")[1]))


def step_of(checkpoint: Path) -> int:
    return int(checkpoint.name.split("-")[1])


def primary_metric(task_metrics: dict) -> tuple[str, float] | None:
    """เลือก metric หลักของ task นั้นๆ แบบ heuristic (acc_norm > acc > exact_match > อันแรกที่เจอ)"""
    for prefer in ("acc_norm,none", "acc,none", "exact_match,none"):
        if prefer in task_metrics:
            return prefer, task_metrics[prefer]
    for key, val in task_metrics.items():
        if isinstance(val, (int, float)) and not key.startswith("alias") and "stderr" not in key:
            return key, val
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs_large_v2/run1")
    ap.add_argument("--checkpoint", default=None, help="ระบุ path ตรงๆ แทนการหา checkpoint ล่าสุด")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--limit", type=int, default=200, help="จำกัดจำนวนตัวอย่างต่อ task (เร็วขึ้น ผลอาจ noisy กว่าเต็มชุด)")
    ap.add_argument("--batch-size", default="1")
    ap.add_argument("--history", default=None, help="ไฟล์ jsonl เก็บผลสะสม (default: <out-dir>/benchmark_history.jsonl)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(out_dir)
    step = step_of(checkpoint)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    history_path = Path(args.history) if args.history else out_dir / "benchmark_history.jsonl"

    print(f"Checkpoint: {checkpoint} (step {step})")
    print(f"Tasks: {tasks}")
    print(f"Limit ต่อ task: {args.limit}")

    import lm_eval  # นำเข้าตรงนี้ ไม่ใช่หัวไฟล์ — กัน error ตอน --help ถ้ายังไม่ได้ลง lm-eval

    model_args = (
        f"pretrained={BASE_MODEL},peft={checkpoint},"
        "load_in_4bit=True,bnb_4bit_quant_type=nf4,bnb_4bit_compute_dtype=bfloat16,"
        "bnb_4bit_use_double_quant=True,dtype=bfloat16"
    )

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=0,
        batch_size=args.batch_size,
        device="cuda:0",
        limit=args.limit,
    )

    history_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n=== ผลลัพธ์ ===")
    with open(history_path, "a", encoding="utf-8") as f:
        for task_name, task_metrics in results["results"].items():
            picked = primary_metric(task_metrics)
            if picked is None:
                print(f"  {task_name}: ไม่มี metric ตัวเลขให้ดึง ข้าม")
                continue
            metric_name, value = picked
            print(f"  {task_name:24s} {metric_name:16s} = {value:.4f}")
            record = {
                "step": step,
                "checkpoint": str(checkpoint),
                "task": task_name,
                "metric": metric_name,
                "value": value,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nบันทึกผลสะสมที่ {history_path}")
    print("Plot กราฟเทียบได้ด้วย: python src/plot_benchmark.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
