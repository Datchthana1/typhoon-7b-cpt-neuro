# 09 — แก้ปัญหา

หัวข้อที่มีเครื่องหมาย ⚡ คือปัญหาที่**เจอจริง**ระหว่างรันโปรเจกต์นี้ และแก้ไปแล้วในโค้ด

---

## ⚡ `UnicodeEncodeError: 'charmap' codec can't encode character '→'`

```
File "...\encodings\cp874.py", line 19, in encode
    return codecs.charmap_encode(input, self.errors, encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '→'
```

**สาเหตุ:** Windows console เริ่มต้นเป็น code page **cp874** (Thai) ซึ่งไม่มีอักขระ `→`
ทำให้ `logging` พังทั้งบรรทัด (แต่โปรแกรมยังทำงานต่อ — เลยดูเหมือน log หาย)

**แก้แล้วใน `src/utils.py`:**
```python
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
```

**ถ้ายังเจอในสคริปต์อื่น:**
```powershell
$env:PYTHONIOENCODING = "utf-8"
```

---

## ⚡ รันจบด้วย exit code 0 แต่ไม่มี adapter ถูกบันทึก

**สาเหตุ:** `load_best_model_at_end=True` บังคับว่า `save_steps` ต้องหารลงตัวด้วย `eval_steps`
เมื่อตั้ง `eval_steps=3, save_steps=1000` → transformers raise `ValueError` หลังโหลดโมเดลเสร็จ
ถ้า pipe output ผ่าน grep จะไม่เห็น traceback และ exit code ที่ได้มาจาก grep ไม่ใช่ python

**แก้:** ตั้งให้หารลงตัว เช่น `eval_steps=2, save_steps=1000`

**ป้องกัน:** อย่า pipe stdout ผ่าน grep ตอนรันครั้งแรก — เก็บลงไฟล์แล้วค่อยกรอง
```powershell
python src/train_cpt.py ... > run.log 2>&1 ; Get-Content run.log -Tail 30
```

---

## ⚡ เทรนจบเร็วผิดปกติ / ไม่มี optimizer step เลย

**อาการ:** `train_runtime` สั้นมาก, ไม่มี `{'loss': ...}` ออกมาเลย

**สาเหตุ:** `จำนวนบล็อก ÷ (micro_batch × grad_accum) < 1`
เช่น 15 บล็อก ÷ (1 × 16) = 0.94 → ไม่ครบ 1 step

**ตรวจ:**
```powershell
python main.py stats
```
แล้วคำนวณ `จำนวนบล็อก ÷ grad_accum` ต้องได้อย่างน้อย 5–10 step/epoch

**แก้:** ลด `grad_accum` หรือเพิ่มข้อมูล

---

## OOM (`torch.cuda.OutOfMemoryError`)

ไล่ตามลำดับ — แต่ละข้อลด VRAM มากที่สุดก่อน

| ลำดับ | ทำอะไร | ลด VRAM ประมาณ |
|---|---|---|
| 1 | `seq_len: 2048 → 1024` ใน `configs/cpt.yaml` | 30–40% |
| 2 | `lora_r: 64 → 32 → 16` | 5–10% ต่อขั้น |
| 3 | ตรวจว่า `gradient_checkpointing=True` จริง | 40% (ถ้าเผลอปิด) |
| 4 | ตัดโมดูลออกจาก `ALL_LINEAR` (เอา `k_proj`,`o_proj` ออกก่อน) | 5% |
| 5 | ปิดโปรแกรมอื่นที่กิน VRAM (Chrome, VS Code GPU accel) | 0.5–1.5 GB |
| 6 | ใช้ `unsloth` แทน peft ตรง ๆ | ~40% |

**ตรวจว่ามีอะไรกิน VRAM อยู่:**
```powershell
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv
```

**Windows หน่วง/ค้างแทนที่จะ OOM:** ระบบสลับไปใช้ shared GPU memory (RAM) แทน ช้าลง 10–50 เท่า
ปิดได้ใน NVIDIA Control Panel → Manage 3D Settings → CUDA Sysmem Fallback Policy → *Prefer No Sysmem Fallback*

---

## `train_loss` ลงถึงเกือบ 0 แต่ `eval_loss` สูง

**นี่คือ overfitting/การท่องจำ — ไม่ใช่ความสำเร็จ**

เจอจริงในโปรเจกต์นี้: `train_loss = 0.005` คู่กับ `eval_loss = 3.80`
→ PPL โดเมนใหม่พุ่งจาก 11.07 เป็น 45.13

| แก้ | ลำดับความสำคัญ |
|---|---|
| เพิ่มข้อมูล (ทางแก้ที่แท้จริง) | 1 |
| ลด `learning_rate` (ลอง ÷ 5) | 2 |
| ลด `epochs` เหลือ 2–3 | 3 |
| ลด `lora_r` | 4 |
| เพิ่ม `lora_dropout` เป็น 0.05–0.1 | 5 |
| เพิ่ม `weight_decay` เป็น 0.01–0.05 | 6 |

---

## PPL ภาษาไทยทั่วไปพุ่งขึ้น (catastrophic forgetting)

**เกณฑ์:** เพิ่มเกิน 5% ถือว่าเสียหาย · เกิน 50% ถือว่าโมเดลพัง

| แก้ | หมายเหตุ |
|---|---|
| เพิ่ม `data.replay_ratio` เป็น 0.20–0.30 | ทางแก้ตรงจุดที่สุด |
| ลด `learning_rate` | ตัวแปรที่ส่งผลมากที่สุดจากการวัดจริง |
| ลด `epochs` | |
| เพิ่ม `forgetting_penalty` ใน `configs/optuna.yaml` เป็น 5.0 | ให้ Optuna หลีกเลี่ยงเอง |

**เตรียม replay corpus** (ถ้า `data/replay/` ว่าง):
```python
from datasets import load_dataset
import json
ds = load_dataset("wikimedia/wikipedia", "20231101.th", split="train", streaming=True)
rows, gval = [], []
for i, r in enumerate(ds):
    t = r["text"].strip()
    if len(t) < 1500: continue
    (gval if len(gval) < 40 else rows).append({"id": f"wiki-{i}", "text": t[:12000]})
    if len(rows) >= 400: break
for name, data in [("general_th", rows), ("general_val", gval)]:
    with open(f"data/replay/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in data: f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

---

## `loss = nan` หรือ `inf`

| สาเหตุ | แก้ |
|---|---|
| ใช้ `fp16` แทน `bf16` | ตั้ง `bf16=True, fp16=False` — fp16 มี exponent range แคบเกินสำหรับ LLM |
| `learning_rate` สูงเกิน | ลดลง 10 เท่า |
| `max_grad_norm` สูงเกิน | ลดเหลือ 0.3 |
| `warmup_ratio` = 0 | ตั้งเป็น 0.03–0.05 |

---

## `ValueError: Can't find 'adapter_config.json'`

adapter ไม่ถูกบันทึก — ดูหัวข้อ "รันจบด้วย exit code 0 แต่ไม่มี adapter" ข้างบน
หรือระบุ path ผิด (ต้องชี้ไปที่โฟลเดอร์ `adapter/` ไม่ใช่โฟลเดอร์แม่)

```powershell
Get-ChildItem outputs\final\adapter    # ต้องมี adapter_config.json + adapter_model.safetensors
```

---

## PPL หลังเทรนดีเกินจริง (ต่ำผิดปกติ)

น่าจะมี **data leakage** — บล็อกจากเอกสารเดียวกันกระจายไปทั้ง train และ val

โค้ดนี้แบ่งระดับเอกสาร**ก่อน** pack อยู่แล้ว (`prepare_data.py:stage_pack`)
แต่ถ้าเนื้อหามีการซ้ำกันสูง (เช่น augment หลายเวอร์ชันของบทเดียวกัน) ให้ตั้ง

```yaml
dedup:
  jaccard_threshold: 0.75   # เข้มขึ้นเพื่อตัดเวอร์ชันที่คล้ายกันออก
```

---

## โมเดลตอบเป็น Q&A แทนที่จะต่อประโยค

CPT ที่ทำถูกต้องจะได้โมเดลที่**ต่อข้อความ** ไม่ใช่ตอบคำถาม
ถ้ามันตอบเป็น Q&A แปลว่าคลังข้อมูลมีรูปแบบ Q&A ปนอยู่

**ตรวจ:**
```powershell
Select-String -Path data\raw\*.md -Pattern "คำถาม:|^Q:|^A:|^\s*[-*]\s" | Select-Object -First 20
```

**แก้:** ลบออกจาก `data/raw/` แล้วรัน `python main.py prepare` ใหม่
กฎทั้งหมดอยู่ที่ [book/STYLE_GUIDE.md](../book/STYLE_GUIDE.md)

---

## เทรนช้ามาก (> 60 วินาที/step)

| ตรวจ | คำสั่ง |
|---|---|
| GPU ถูกใช้จริงไหม | `nvidia-smi` — utilization ควร > 80% |
| ตกไปใช้ shared memory ไหม | ถ้า `nvidia-smi` แสดง memory เต็ม 8GB พอดี → ดูหัวข้อ OOM |
| `dataloader_num_workers` | บน Windows ต้องเป็น 0 — ค่ามากกว่านั้นทำให้ช้าลงเพราะ spawn process ใหม่ทุก epoch |
| attention implementation | โค้ดใช้ `sdpa` อัตโนมัติ ถ้าติดตั้ง flash-attn ได้จะเร็วขึ้นอีก ~15% |

ค่าอ้างอิงจากการรันจริงบน RTX 4060 8GB: **~17 วินาที/step** ที่ `seq_len=1024, grad_accum=4`
