# 08 — Merge, Export และการอัปโหลดขึ้น Hugging Face

## ตัวเลือก: merge หรือไม่ merge

| | เก็บเป็น adapter | merge เข้า base |
|---|---|---|
| ขนาดดิสก์ | ~80 MB | ~14.5 GB |
| ความเร็ว inference | ช้ากว่าเล็กน้อย (คำนวณ ΔW ทุก forward) | เร็วกว่า |
| ใช้กับ vLLM / GGUF | ต้อง merge ก่อน | ใช้ได้ตรง |
| สลับหลาย adapter บน base เดียว | ✅ ทำได้ | ❌ |
| แชร์ให้คนอื่น | เบา แต่คนรับต้องมี base | หนัก แต่พร้อมใช้ |

**คำแนะนำ:** ระหว่างพัฒนาให้เก็บเป็น adapter — merge เฉพาะตอนจะ deploy จริง

---

## ข้อควรระวังที่สำคัญที่สุดตอน merge

> **ห้าม merge adapter เข้ากับ base ที่โหลดแบบ 4-bit**

เพราะกระบวนการจะเป็น `dequantize → บวก ΔW → quantize ใหม่`
ซึ่งสะสม quantization error สองรอบ ทำให้คุณภาพตกโดยไม่มีสัญญาณเตือน

วิธีที่ถูกคือโหลด base เป็น **bf16 บน CPU** แล้วค่อย merge:

```python
base = AutoModelForCausalLM.from_pretrained(
    "typhoon-ai/typhoon-7b",
    dtype=torch.bfloat16,
    device_map="cpu",          # ← ไม่ใช้ VRAM เลย
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(base, adapter_path, dtype=torch.bfloat16)
model = model.merge_and_unload()
```

ใช้ RAM ~15 GB แต่ **VRAM 0 GB** → เครื่อง 8 GB ทำได้สบาย

```powershell
python main.py merge
```

---

## Export ไปยัง runtime ต่าง ๆ

### vLLM (เสิร์ฟ API, ต้องการ VRAM ~16 GB)

```powershell
vllm serve "outputs/merged" --dtype bfloat16 --max-model-len 4096
```

vLLM รองรับ LoRA adapter โดยตรงด้วย ไม่ต้อง merge:

```powershell
vllm serve typhoon-ai/typhoon-7b --enable-lora --lora-modules neuro=outputs/final/adapter
```

### GGUF (รันบน CPU / llama.cpp / LM Studio / Ollama)

```powershell
git clone https://github.com/ggerganov/llama.cpp
python llama.cpp/convert_hf_to_gguf.py outputs/merged --outfile typhoon-cpt-f16.gguf
llama.cpp/llama-quantize typhoon-cpt-f16.gguf typhoon-cpt-q4_k_m.gguf Q4_K_M
```

| Quant | ขนาด | คุณภาพ |
|---|---|---|
| `Q8_0` | ~7.7 GB | เกือบเท่า fp16 |
| `Q5_K_M` | ~5.1 GB | ดีมาก |
| `Q4_K_M` | ~4.4 GB | **จุดสมดุลที่แนะนำ** |
| `Q3_K_M` | ~3.5 GB | เริ่มเห็นคุณภาพตก |

> ⚠️ **ตรวจ tokenizer หลังแปลง GGUF เสมอ** — typhoon-7b มี vocab ขยายภาษาไทย (35,219 token)
> ถ้าตัวแปลงจัดการ vocab ผิด ข้อความไทยจะออกมาเป็นขยะโดยที่ perplexity ยังดูปกติ
> ทดสอบด้วยการ generate ข้อความไทยยาว ๆ หนึ่งย่อหน้าก่อนใช้งานจริง

---

## อัปโหลดขึ้น Hugging Face

### ขั้นตอน

```powershell
# 1. login (ครั้งแรกเท่านั้น)
hf auth login

# 2. ตรวจว่า login สำเร็จ
python -c "from huggingface_hub import whoami; print(whoami()['name'])"

# 3. push
python src/push_to_hub.py --repo <username>/typhoon-7b-neuro-cpt
```

### สิ่งที่ `push_to_hub.py` ทำให้อัตโนมัติ

| | |
|---|---|
| **Quality gate** | ปฏิเสธการอัปโหลดถ้า domain PPL ลดไม่ถึง 5% หรือ general PPL เพิ่มเกิน 5% |
| **Private โดยค่าเริ่มต้น** | ต้องระบุ `--public` อย่างชัดเจนจึงจะเป็นสาธารณะ |
| **สร้าง model card** | ใส่ตาราง PPL ก่อน/หลัง, hyperparameter, ข้อจำกัด, ตัวอย่างโค้ด |

### ตัวเลือก

```powershell
python src/push_to_hub.py --repo user/model --public              # เผยแพร่สาธารณะ
python src/push_to_hub.py --repo user/model --merged              # อัปโหลดโมเดลเต็ม
python src/push_to_hub.py --repo user/model --skip-quality-gate   # ข้ามเกณฑ์ (ไม่แนะนำ)
```

---

## เกณฑ์ก่อนเผยแพร่สู่สาธารณะ

การเผยแพร่โมเดลเป็นการกระทำที่ย้อนกลับยาก — คนดาวน์โหลดไปแล้วเรียกคืนไม่ได้
ควรผ่านทั้ง 5 ข้อก่อน

| # | เกณฑ์ | ตรวจอย่างไร |
|---|---|---|
| 1 | domain PPL ลดลง > 15% | `python main.py eval` |
| 2 | general PPL เพิ่มไม่เกิน 5% | ตารางเดียวกัน |
| 3 | val set มีอย่างน้อย 30 บล็อก | `python main.py stats` — ต่ำกว่านี้ตัวเลขไม่มีความหมายทางสถิติ |
| 4 | generation probe **ให้ข้อเท็จจริงถูกต้อง** ไม่ใช่แค่รูปแบบถูก | `python main.py probe` |
| 5 | ระบุข้อจำกัดใน model card ครบ | ตรวจ README ที่สคริปต์สร้าง |

**ข้อ 3 สำคัญกว่าที่คิด** — ในการรันจริงของโปรเจกต์นี้ val มีเพียง 1 บล็อก
ทำให้ตัวเลข PPL ±9% แยกจาก noise ไม่ได้เลย

**ข้อ 4 สำคัญที่สุด และ PPL แทนไม่ได้** — ในการรันจริง PPL "ดีขึ้น 9.2%"
แต่ probe เผยว่าโมเดลยังแต่งชื่อทฤษฎีและปีงานวิจัยขึ้นมาเอง
(เช่น อ้าง "Maslow's Habit Formation Theory ปี 1937" ที่ไม่มีอยู่จริง)

**ถ้าดูแต่ PPL จะสรุปผิดแล้ว push โมเดลที่ยังหลอนออกไป** — ดู [RESULTS.md](../RESULTS.md) §6.5

---

## ปัญหาที่พบบ่อยตอน push

| อาการ | สาเหตุ | แก้ |
|---|---|---|
| `Invalid user token` | token หมดอายุหรือถูกเพิกถอน | `hf auth login` ใหม่ |
| `403 Forbidden` | token เป็นแบบ read-only | สร้าง token ใหม่แบบ **write** ที่ huggingface.co/settings/tokens |
| อัปโหลดแล้วโหลดกลับมาใช้ไม่ได้ | ลืมอัปโหลด tokenizer ไปด้วย | `train_cpt.py` บันทึก tokenizer ลงโฟลเดอร์ adapter อยู่แล้ว — ตรวจว่ามี `tokenizer.model` |
| ช้ามาก | ไฟล์ใหญ่ | ใช้ `--merged` เฉพาะเมื่อจำเป็น — adapter อย่างเดียวแค่ ~80 MB |
