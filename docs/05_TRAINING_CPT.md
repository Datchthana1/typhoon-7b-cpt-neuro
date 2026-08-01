# 05 — QLoRA Continued Pre-Training ทีละบรรทัด

## การตั้งค่า quantization

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

| ตัวเลือก | ทำไม |
|---|---|
| `nf4` ไม่ใช่ `fp4` | NormalFloat4 ออกแบบมาให้ระดับ quantization กระจายตาม normal distribution ซึ่งเป็นการกระจายที่น้ำหนักโมเดลมีจริง — แม่นกว่า fp4 ที่กระจายแบบ uniform |
| `compute_dtype=bfloat16` | น้ำหนักเก็บเป็น 4-bit แต่ตอนคูณเมทริกซ์ต้อง dequantize ขึ้นมาก่อน bf16 มี exponent range เท่า fp32 จึงไม่ overflow เหมือน fp16 (RTX 4060 เป็น Ada รองรับ bf16 native) |
| `double_quant=True` | quantize ตัว quantization constant อีกชั้น ประหยัดเพิ่มราว 0.4 GB โดยเสียความแม่นยำแทบไม่วัดได้ |

---

## LoRA config

```python
LoraConfig(
    r=hp["lora_r"],
    lora_alpha=hp["lora_alpha"],
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_rslora=True,
    task_type="CAUSAL_LM",
)
```

### ทำไมต้องครบทุก linear

ตัวอย่างที่เห็นทั่วไปใช้แค่ `["q_proj","v_proj"]` ซึ่งเหมาะกับ SFT
แต่ CPT ต้องเปลี่ยน**ความรู้** ซึ่งงาน mechanistic interpretability ชี้ว่าเก็บใน feed-forward layer เป็นหลัก

จำนวนพารามิเตอร์ที่วัดได้จริงบนเครื่องนี้ (r=16, Mistral-7B, 32 layers):

| โมดูล | มิติ | พารามิเตอร์ LoRA ต่อ layer |
|---|---|---|
| q_proj | 4096 → 4096 | 16 × (4096+4096) = 131,072 |
| k_proj | 4096 → 1024 | 16 × (4096+1024) = 81,920 |
| v_proj | 4096 → 1024 | 81,920 |
| o_proj | 4096 → 4096 | 131,072 |
| gate_proj | 4096 → 14336 | 16 × (4096+14336) = 294,912 |
| up_proj | 4096 → 14336 | 294,912 |
| down_proj | 14336 → 4096 | 294,912 |
| **รวม/layer** | | **1,310,720** |
| **× 32 layers** | | **41,943,040** = 1.098% ของโมเดล |

ตัวเลข 41,943,040 นี้ตรงกับที่โค้ดรายงานจริงทุกหลัก — ใช้ตรวจว่า `target_modules` ถูกต้อง

### ทำไมต้อง rsLoRA

LoRA มาตรฐาน scale ด้วย `alpha/r` → เพิ่ม `r` แล้ว effective LR ลดลงตาม
ทำให้ `r` กับ `learning_rate` พันกัน และ Optuna ค้นแยกกันไม่ได้

rsLoRA scale ด้วย `alpha/√r` → สองตัวแปรเป็นอิสระต่อกันมากขึ้น

---

## Gradient checkpointing

```python
prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
model.config.use_cache = False
```

**หลักการ:** ปกติ forward pass เก็บ activation ทุกชั้นไว้เพื่อใช้ตอน backward
gradient checkpointing เก็บเฉพาะบางจุด แล้วคำนวณ activation ที่เหลือใหม่ตอน backward

**ต้นทุน:** ช้าลง ~30% · **ผลตอบแทน:** VRAM activation ลด ~40%

| ประเด็น | คำอธิบาย |
|---|---|
| `use_reentrant=False` | เวอร์ชันใหม่ของ PyTorch — จำเป็นเมื่อบางพารามิเตอร์ `requires_grad=False` (ซึ่งเป็นกรณีของ LoRA ทั้งหมด) |
| `use_cache=False` | KV cache ใช้ตอน generate เท่านั้น ถ้าเปิดคู่กับ gradient checkpointing จะได้ warning และเปลืองเมมโมรีเปล่า |

---

## Optimizer

```python
optim="paged_adamw_8bit"
```

- **8-bit** — AdamW เก็บ state สองตัว (momentum, variance) ต่อพารามิเตอร์ ถ้าเก็บ fp32 = 8 bytes/param, 8-bit เหลือ 2 bytes
- **paged** — ใช้ NVIDIA unified memory ให้ optimizer state ย้ายไป RAM ชั่วคราวได้เมื่อ VRAM แน่น ป้องกัน OOM ที่จุดสูงสุด

บน 8 GB ตัวเลือกนี้ไม่ใช่ทางเลือกแต่เป็นข้อบังคับ

---

## Data collator

```python
data_collator=default_data_collator
```

**ไม่ใช้** `DataCollatorForLanguageModeling` เพราะ dataset ถูก pack มาแล้วให้ทุกบล็อกยาว 1024 เท่ากัน
และ `labels` ถูกสร้างไว้ตั้งแต่ `prepare_data.py` แล้ว:

```python
"labels": [list(b) for b in blocks]   # labels = input_ids → loss ทุก token
```

Transformers จะ shift `labels` ให้เองภายใน `MistralForCausalLM.forward()`
จึงไม่ต้อง shift เอง — การ shift ซ้ำเป็นบั๊กที่พบบ่อยและตรวจจับยาก (loss จะดูปกติแต่โมเดลเรียนผิด)

---

## ตารางค่าที่ใช้ได้จริงบน RTX 4060 8GB

| seq_len | lora_r | grad_ckpt | VRAM peak | สถานะ |
|---|---|---|---|---|
| 1024 | 16 | ✅ | **6.44 GB** | ✅ ทดสอบแล้ว |
| 1024 | 32 | ✅ | **6.96 GB** | ✅ ทดสอบแล้ว |
| 1024 | 64 | ✅ | ~7.4 GB | ⚠️ เสี่ยง |
| 2048 | 16 | ✅ | ~7.6 GB | ⚠️ เสี่ยง |
| 2048 | 32 | ✅ | OOM | ❌ |
| 1024 | 32 | ❌ | OOM | ❌ |

*(สองแถวแรกคือค่าที่วัดได้จริงจากการรัน — ดู [RESULTS.md](../RESULTS.md))*

---

## ผลจริง: hyperparameter สำคัญแค่ไหน

จากการรันจริงสองครั้งบนคลังเดียวกัน:

| | เชิงรุก | อนุรักษ์ |
|---|---|---|
| `learning_rate` | 1e-4 | 1.5e-5 |
| `epochs` | 8 | 2 |
| `lora_r` | 32 | 16 |
| PPL โดเมนใหม่ | 45.13 (+308%) | **10.05 (−9.2%)** |
| PPL ไทยทั่วไป | 38.86 (+384%) | **7.73 (−3.8%)** |

`learning_rate` ต่างกัน 6.7 เท่า → ผลลัพธ์ต่างกันระหว่าง "โมเดลพัง" กับ "โมเดลดีขึ้น"

**สัญญาณเตือนที่ต้องจับให้ได้:** `train_loss` ต่ำกว่า 0.1 ขณะที่ `eval_loss` สูงกว่า 3
คือลายเซ็นของการท่องจำ ไม่ใช่การเรียนรู้ → ลด lr, ลด epochs, ลด r, เพิ่มข้อมูล

---

## คำสั่ง

```powershell
python main.py train                              # ค่าจาก configs/cpt.yaml
python main.py train -- --from-study              # ค่าที่ Optuna หาได้
python main.py train -- --override learning_rate=2e-5 lora_r=16 epochs=3
python main.py train -- --out outputs/exp2        # แยก output directory
```

`--override` ทับค่าใน `hp` ได้ทุกตัว รวมถึง `eval_steps`, `logging_steps`, `save_steps`

> ⚠️ **`load_best_model_at_end=True` บังคับให้ `save_steps` หารลงตัวด้วย `eval_steps`**
> ถ้าไม่ตรง transformers จะ raise หลังโหลดโมเดลเสร็จ ทำให้เสียเวลาโหลดฟรี ๆ
> (บั๊กนี้เจอจริงระหว่างรัน — ดู [09_TROUBLESHOOTING.md](09_TROUBLESHOOTING.md))
