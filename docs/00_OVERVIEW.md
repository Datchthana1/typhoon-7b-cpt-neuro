# 00 — ภาพรวมสถาปัตยกรรมและการตัดสินใจเชิงออกแบบ

## CPT คืออะไร และต่างจาก SFT อย่างไร

Continued Pre-Training คือการเทรนโมเดลภาษาต่อจากจุดที่ pre-training หยุดไว้
ด้วย objective เดิมทุกประการ นั่นคือทำนาย token ถัดไป บนข้อความดิบที่ไม่มีโครงสร้างคำสั่ง

| | CPT | SFT |
|---|---|---|
| ข้อมูล | ข้อความดิบต่อเนื่อง | คู่ `{instruction, output}` |
| Loss | ทุก token | เฉพาะ token ของ output |
| Template | ไม่มี | มี |
| เปลี่ยนอะไรในโมเดล | **ความรู้ + การกระจายของภาษา** | **พฤติกรรมการตอบสนอง** |
| ลำดับที่ถูกต้อง | ทำก่อน | ทำหลัง CPT |

การเอา dataset แบบ Q&A มาเทรนเป็น CPT คือการสอนโมเดลว่า
"หลังเครื่องหมายคำถามต้องตามด้วยคำตอบ" ซึ่งได้ pseudo-instruct ที่แย่กว่าการทำ SFT จริงมาก
นี่คือเหตุผลที่ [book/STYLE_GUIDE.md](../book/STYLE_GUIDE.md) ห้ามรูปแบบ Q&A อย่างเด็ดขาด

---

## ทำไมต้อง QLoRA ไม่ใช่ full fine-tuning

งบ VRAM ของการเทรน 7B แบบ full-parameter ด้วย AdamW:

| รายการ | สูตร | ขนาด |
|---|---|---|
| น้ำหนัก (bf16) | 7.24B × 2 bytes | 14.5 GB |
| Gradient (bf16) | 7.24B × 2 | 14.5 GB |
| Optimizer state (fp32 m, v) | 7.24B × 8 | 58 GB |
| Master weights (fp32) | 7.24B × 4 | 29 GB |
| **รวมขั้นต่ำ** | | **≈ 116 GB** |

RTX 4060 Laptop มี **8 GB** → ต้องใช้ H100 80GB สองใบขึ้นไป

QLoRA เปลี่ยนสมการทั้งหมด:

| รายการ | ขนาด |
|---|---|
| น้ำหนัก NF4 4-bit + double quant | ≈ 3.9 GB |
| LoRA r=32 ทุก linear (~40M param, bf16) | 0.08 GB |
| Gradient ของ LoRA เท่านั้น | 0.08 GB |
| paged AdamW 8-bit state | 0.08 GB |
| Activation (seq 1024, batch 1, + grad ckpt) | ≈ 1.5 GB |
| **รวม** | **≈ 5.7 GB** |

**ค่าที่วัดได้จริงบนเครื่องนี้: `alloc=4.77GB peak_reserved=6.96GB` จาก 8.0GB** — ตรงตามที่ประมาณไว้

---

## การตัดสินใจเชิงออกแบบที่สำคัญ 6 ข้อ

### 1. LoRA ครอบทุก linear layer ไม่ใช่แค่ q_proj/v_proj

การตั้งค่ามาตรฐานของ LoRA ที่เห็นในบทความส่วนใหญ่คือแตะเฉพาะ attention
ซึ่งใช้ได้ดีกับ SFT เพราะ SFT เปลี่ยน*พฤติกรรม* ไม่ใช่*ความรู้*

แต่ CPT ต้องการเปลี่ยนความรู้ และหลักฐานจากงาน mechanistic interpretability
ชี้ว่าความรู้เชิงข้อเท็จจริงถูกเก็บใน **feed-forward layer** เป็นหลัก ไม่ใช่ attention
จึงต้องรวม `gate_proj`, `up_proj`, `down_proj` ด้วย

```python
ALL_LINEAR = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
```

### 2. ใช้ rsLoRA

LoRA มาตรฐาน scale ด้วย `alpha/r` ซึ่งทำให้เมื่อเพิ่ม `r` แล้ว effective learning rate ลดลงตาม
ทำให้ Optuna ค้น `r` กับ `learning_rate` แยกกันไม่ได้ (สองตัวแปรนี้พันกัน)

rsLoRA scale ด้วย `alpha/√r` แทน ทำให้ `r` กับ `lr` เป็นอิสระต่อกันมากขึ้น
→ search space ของ Optuna มีความหมายขึ้น

### 3. Pack แบบต่อเนื่องแทน padding

```
เอกสาร A </s> เอกสาร B </s> เอกสาร C </s> ...   →  หั่นทุก 1024 token
```

ทุก token มี loss ไม่มี padding ทิ้ง GPU utilization ≈ 100%

**ข้อแลกเปลี่ยนที่ต้องรู้:** บล็อกหนึ่งอาจคาบเกี่ยวสองเอกสาร ทำให้เกิด cross-document attention
วิธีแก้ที่ถูกต้องคือ block-diagonal attention mask ผ่าน FlashAttention varlen
แต่บทความที่ยาวเป็นหมื่น token ทำให้จุดคาบเกี่ยวมีสัดส่วนน้อยมาก จึงยอมรับได้ในกรณีนี้

### 4. แบ่ง train/val ระดับเอกสาร ไม่ใช่ระดับบล็อก

ถ้าแบ่งหลัง pack บล็อกจากบทเดียวกันจะกระจายไปทั้ง train และ val
→ โมเดลเห็นเนื้อหาของ val มาแล้วตอนเทรน → PPL ดูดีเกินจริง
โค้ดจึงแบ่งก่อน pack เสมอ (`prepare_data.py:stage_pack`)

### 5. Replay corpus

ปัญหาใหญ่ที่สุดของ CPT บนคลังเล็กคือ catastrophic forgetting
โมเดลปรับตัวเข้ากับการกระจายใหม่จนสูญเสียการกระจายเดิม

วิธีแก้ที่ถูกที่สุดคือผสมข้อความจากการกระจายเดิมกลับเข้าไป 10–20%
โปรเจกต์นี้ใช้ Thai Wikipedia เป็น replay corpus (`data/replay/general_th.jsonl`)

### 6. วัด forgetting เป็น first-class metric

`general_val` (Thai Wikipedia held-out) ถูกวัดทุกครั้งควบคู่กับ domain val
และเป็นส่วนหนึ่งของ objective ใน Optuna โดยตรง — ดู [06_OPTUNA_HPO.md](06_OPTUNA_HPO.md)

---

## ลำดับการทำงานทั้งหมด

```
data/raw/*.md                     ← เนื้อหาหนังสือ (เขียนเอง)
        │  prepare_data.py --stage clean
        ▼
data/processed/clean.jsonl        ← ตัด markdown / meta-text / กรองสัดส่วนอักขระไทย
        │  --stage dedup
        ▼
data/processed/dedup.jsonl        ← exact hash + MinHashLSH (char 5-gram, jaccard 0.85)
        │  --stage pack
        ▼
data/processed/{train,val,general_val}/   ← Arrow dataset, บล็อกละ 1024 token
        │
        ├─ evaluate.py --baseline  → outputs/baseline.json
        │
        ├─ hpo_optuna.py           → outputs/optuna.db
        │        objective = domain_loss + λ·max(0, general_loss − baseline)
        ▼
   train_cpt.py --from-study       → outputs/final/adapter/
        │
        ├─ evaluate.py --compare   → ตารางเทียบ + คำเตือน forgetting
        ▼
   merge_export.py                 → outputs/merged/  (fp16, พร้อม vLLM / GGUF)
```

---

## สิ่งที่โปรเจกต์นี้ **ไม่ได้** ทำ

| ไม่ได้ทำ | เหตุผล / ทางเลือก |
|---|---|
| ขยาย vocabulary | typhoon-7b มี vocab ไทย 35,219 อยู่แล้ว การขยายต้องเทรน embedding ใหม่ซึ่งเกินงบ 8GB |
| Full-parameter CPT | ต้องการ ~116 GB VRAM |
| Block-diagonal attention mask | ต้องการ flash-attn ซึ่ง build ยากบน Windows |
| SFT ต่อจาก CPT | อยู่นอกขอบเขต — แต่เป็นขั้นถัดไปที่ควรทำถ้าอยากได้โมเดลที่ตอบคำถามได้ |
| Multi-GPU / DeepSpeed | มี GPU ใบเดียว |
