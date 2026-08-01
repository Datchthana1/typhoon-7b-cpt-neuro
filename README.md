# CPT Pipeline — typhoon-7b × คลังบทความ Neuroplasticity ภาษาไทย

ไปป์ไลน์ **Continued Pre-Training (CPT)** สำหรับ `typhoon-ai/typhoon-7b`
บนคลังข้อมูลที่เป็น **บทความยาวต่อเนื่อง (long-form prose)** — ไม่ใช่ Q&A ไม่ใช่ instruction

หัวข้อเนื้อหา: **ระบบประสาทกับการฝึกฝน / neuroplasticity** (แนวเดียวกับ *Atomic Habits* + *Livewired*)

> ## 🔴 อัปเดตล่าสุด (1 ส.ค. 2026): รันเต็มสเกลแล้ว — **ผลลบ ไม่ผ่านเกณฑ์**
> โมเดลบน Hugging Face ตอนนี้คือผลจากรอบเทรนเต็มสเกล (5,853 step / 3 epoch) ซึ่ง **แย่กว่าโมเดลดั้งเดิมที่ไม่ผ่าน CPT เลย**
> (domain PPL +6.6%) สาเหตุน่าจะเป็น `learning_rate` สูงเกินไป — รายละเอียดเต็มอยู่ที่ [RESULTS.md §11](RESULTS.md#11-รันเต็มสเกล-outputs_large_v2--29-กค1-สค-2026--ผลลบ)
> **อย่าเพิ่งใช้โมเดลนี้งานจริง** รอรอบถัดไปที่จะลด lr แล้วเทรนใหม่
>
> **สถานะ:** ไปป์ไลน์ทำงานครบทุกขั้นตอนและผ่านการรันจริงบน RTX 4060 8GB แล้ว (ทั้ง smoke test เล็กและรันเต็มสเกล)
> เนื้อหาหนังสือเขียนไปแล้ว **8 บทจาก 42 บท** → ดูตัวเลขจริงที่ [RESULTS.md](RESULTS.md)
>
> **โมเดล:** [`Datchthana/typhoon-7b-neuro-cpt`](https://huggingface.co/Datchthana/typhoon-7b-neuro-cpt) 🌐 public
> — model card ระบุผลลบของรอบล่าสุดไว้ตรงไปตรงมาแล้ว

---

## ผลการรันเต็มสเกล (1 ส.ค. 2026) — 🔴 ผลลบ ไม่ผ่านเกณฑ์

| ตัวชี้วัด | ก่อน CPT (base) | หลัง CPT (รอบนี้) | เปลี่ยนแปลง |
|---|---|---|---|
| **PPL โดเมนใหม่** | 9.858 | 10.507 | 🔴 **+6.6% (แย่ลง)** |
| PPL ภาษาไทยทั่วไป | 8.034 | ~~3.741~~ | ⚠️ ใช้ไม่ได้ — มี data leakage (แก้แล้ว ดู §11) |

สาเหตุหลัก: `learning_rate=8e-5` สูงเกินไปสำหรับคลังข้อมูลนี้ ทำให้เกิด "warmup shock" ตั้งแต่ต้นการเทรน
แล้วฟื้นตัวไม่ทันเต็มร้อย — วิเคราะห์ละเอียดพร้อมบั๊กที่พบระหว่างรัน (เครื่องแครช 3 ครั้ง, data leakage, LR schedule reset)
อยู่ที่ [RESULTS.md §11](RESULTS.md)

### ผลการรัน smoke test เล็ก (26 ก.ค. 2026) — สำเร็จแต่ noise สูง

รันแรกบนคลังข้อมูลเล็กมาก (15 บล็อกเทรน) เพื่อทดสอบว่าไปป์ไลน์ทำงานถูกต้อง ไม่ใช่ผลที่เชื่อถือได้:

| ตัวชี้วัด | ก่อน CPT | หลัง CPT | เปลี่ยนแปลง |
|---|---|---|---|
| PPL โดเมนใหม่ | 11.073 | **10.051** | 🟢 −9.2% |
| PPL ภาษาไทยทั่วไป (Wikipedia) | 8.034 | **7.725** | 🟢 −3.8% (ไม่ลืม) |
| VRAM peak | — | 6.44 GB / 8.0 GB | |
| พารามิเตอร์ที่เทรน | — | 41.9M / 3.82B (1.098%) | |

และการรันที่ **ล้มเหลว** ด้วย hyperparameter เชิงรุก (`lr=1e-4, 8 epochs`) ก็ถูกบันทึกไว้เช่นกัน:
PPL พุ่งจาก 11.07 → **45.13** และภาษาไทยทั่วไป 8.03 → **38.86** — ดูการวิเคราะห์ใน [RESULTS.md](RESULTS.md)

---

## ทำไมต้อง CPT ไม่ใช่ SFT

| | CPT (สิ่งที่เราทำ) | SFT / Instruction Tuning |
|---|---|---|
| รูปแบบข้อมูล | ข้อความดิบต่อเนื่อง | `{instruction, input, output}` |
| Loss | ทุก token | เฉพาะ token ของคำตอบ |
| Template | **ไม่มี** | มี (`<s>[INST]...`) |
| เปลี่ยนอะไร | **ความรู้ + สำนวน** | **พฤติกรรมการตอบสนอง** |
| Base model ที่เหมาะ | base (`typhoon-7b`) ✅ | instruct |

`typhoon-7b` เป็น **base model** (Mistral-7B arch, vocab 35,219, 1.098% ของพารามิเตอร์ถูกเทรน)
→ ทำ CPT ได้ตรง ๆ ไม่ต้องแปลงอะไร

---

## ลำดับขั้นตอน

```
 ① ออกแบบสเปกหนังสือ           book/OUTLINE.md · STYLE_GUIDE.md
 ② ตรวจ Fact ก่อนเขียน          book/FACT_TABLE.md   ← กันโมเดลเรียน "ความรู้ผิด"
 ③ เขียนเนื้อหา                 data/raw/chNN.md
 ④ clean + dedup + pack        python main.py prepare
 ⑤ วัด baseline                python main.py baseline
 ⑥ ค้น hyperparameter          python main.py hpo
 ⑦ เทรนจริง                    python main.py train -- --from-study
 ⑧ ประเมิน                     python main.py eval
 ⑨ merge + push                python main.py merge  ·  python src/push_to_hub.py
```

ดูสถานะปัจจุบันได้ทุกเมื่อ: `python main.py status`

---

## ติดตั้ง

```powershell
pip install -r requirements.txt
$env:PYTHONIOENCODING = "utf-8"      # จำเป็นบน Windows — console เริ่มต้นเป็น cp874
```

เตรียม replay corpus (ป้องกัน catastrophic forgetting) — สคริปต์อยู่ใน [docs/09_TROUBLESHOOTING.md](docs/09_TROUBLESHOOTING.md#ppl-ภาษาไทยทั่วไปพุ่งขึ้น-catastrophic-forgetting)

---

## แผนที่ไฟล์

### เอกสารเทคนิค

| ไฟล์ | เนื้อหา |
|---|---|
| [docs/00_OVERVIEW.md](docs/00_OVERVIEW.md) | สถาปัตยกรรม · คณิตศาสตร์ VRAM (116GB → 5.7GB) · การตัดสินใจเชิงออกแบบ 6 ข้อ |
| [docs/02_DATASET_DESIGN.md](docs/02_DATASET_DESIGN.md) | สเปก dataset · ทำไมต้องประโยคยาว · การเพิ่มปริมาณข้อมูล · replay corpus |
| [docs/05_TRAINING_CPT.md](docs/05_TRAINING_CPT.md) | QLoRA ทีละบรรทัด · ตาราง VRAM ที่วัดจริง · สัญญาณเตือน overfit |
| [docs/06_OPTUNA_HPO.md](docs/06_OPTUNA_HPO.md) | search space · objective ที่ลงโทษ forgetting · pruner · fANOVA |
| [docs/07_EVALUATION.md](docs/07_EVALUATION.md) | **Perplexity คืออะไร อ่านอย่างไร** · ทำไม PPL ดีขึ้นไม่ได้แปลว่าโมเดลใช้ได้ |
| [docs/08_DEPLOY_MERGE.md](docs/08_DEPLOY_MERGE.md) | merge (ห้าม merge บน 4-bit) · vLLM · GGUF · push ขึ้น HF |
| [docs/09_TROUBLESHOOTING.md](docs/09_TROUBLESHOOTING.md) | บั๊กจริงที่เจอ 3 ข้อ · OOM · loss=nan · forgetting |
| [RESULTS.md](RESULTS.md) | **ผลการทดลองจริงทั้งหมด พร้อมการรันที่ล้มเหลว** |

### เนื้อหาหนังสือ

| ไฟล์ | เนื้อหา |
|---|---|
| [book/OUTLINE.md](book/OUTLINE.md) | โครง 500 หน้า · 7 ภาค · 42 บท พร้อมโควตาหน้าและขอบเขตรายบท |
| [book/FACT_TABLE.md](book/FACT_TABLE.md) | **ตาราง Fact เทียบ 48 รายการ** — ความเชื่อยอดนิยม vs หลักฐานจริง + งานวิจัยอ้างอิง |
| [book/STYLE_GUIDE.md](book/STYLE_GUIDE.md) | สเปกทางเทคนิคของข้อความ · กฎห้าม Q&A/bullet · ตัวอย่างเปรียบเทียบ |
| `data/raw/*.md` | เนื้อหาบทที่เขียนแล้ว **8 บท** |

**บทที่เขียนเสร็จแล้ว**

| บท | ชื่อ | ภาค |
|---|---|---|
| 1 | เนื้อเยื่อที่ยอมเปลี่ยนรูป | สมองที่ไม่เคยหยุดนิ่ง |
| 2 | ตัวเลขที่คนเข้าใจผิดเรื่องสมอง | สมองที่ไม่เคยหยุดนิ่ง |
| 3 | หลักการเฮบบ์และประโยคที่เฮบบ์ไม่เคยเขียน | สมองที่ไม่เคยหยุดนิ่ง |
| 13 | เบซัลแกงเกลียกับความทรงจำที่ไม่ต้องนึก | วงจรของนิสัย |
| 15 | โดพามีนไม่ใช่สารแห่งความสุข | วงจรของนิสัย |
| 19 | หกสิบหกวันไม่ใช่ยี่สิบเอ็ด | วงจรของนิสัย |
| 20 | การฝึกที่จงใจ กับสิ่งที่เอริกสันไม่ได้พูด | สถาปัตยกรรมของการฝึกฝน |
| 27 | การหลับคือส่วนหนึ่งของการฝึก | ตัวปรับสภาพความยืดหยุ่น |

### โค้ด

| ไฟล์ | หน้าที่ |
|---|---|
| `main.py` | CLI รวม (`prepare` · `baseline` · `hpo` · `train` · `eval` · `merge` · `status`) |
| `src/prepare_data.py` | clean → dedup (MinHashLSH) → pack เป็นบล็อก + ผสม replay |
| `src/train_cpt.py` | QLoRA-CPT (NF4 + LoRA ทุก linear + rsLoRA + paged AdamW 8-bit) |
| `src/hpo_optuna.py` | Optuna TPE + MedianPruner + objective ที่ลงโทษ forgetting |
| `src/evaluate.py` | PPL โดเมน/ทั่วไป · ตารางเทียบ · generation probe |
| `src/merge_export.py` | merge adapter เข้า base บน CPU (ไม่ใช้ VRAM) |
| `src/push_to_hub.py` | อัปโหลดขึ้น HF พร้อม quality gate + สร้าง model card |
| `configs/cpt.yaml` · `configs/optuna.yaml` | ค่าคอนฟิกทั้งหมด พร้อมเหตุผลกำกับทุกบรรทัด |

---

## ข้อควรรู้ที่สำคัญที่สุด 4 ข้อ

1. **learning_rate คือตัวแปรที่สำคัญที่สุดอย่างขาดลอย**
   จากการรันจริง: `lr` ต่างกัน 6.7 เท่า → ผลต่างระหว่าง "โมเดลพัง (PPL +308%)" กับ "โมเดลดีขึ้น (−9.2%)"

2. **ต้องวัด catastrophic forgetting ทุกครั้ง**
   `general_val` จาก Thai Wikipedia ถูกวัดคู่กับ domain PPL เสมอ และเป็นส่วนหนึ่งของ objective ใน Optuna
   ถ้าไม่วัด จะเลือก hyperparameter ที่ทำลายโมเดลโดยไม่รู้ตัว

3. **500 หน้า ≈ 400K token ยังนับว่าเล็กสำหรับ CPT**
   ผลที่คาดหวังได้จริงคือ *สำนวน + คำศัพท์เฉพาะทาง* มากกว่าการฝังความรู้ทั้งก้อน

4. **Fact ต้องตรวจก่อน generate ไม่ใช่หลัง**
   ข้อมูลผิดที่ถูกฝังเข้าน้ำหนักโมเดลแล้วแก้ด้วย prompt ไม่ได้ — ไม่มี undo ใน CPT

---

## สิ่งที่ยังเหลือ

| ลำดับ | งาน | สถานะ |
|---|---|---|
| 1 | เขียนบทที่เหลือ 34 บท | ยังไม่ทำ (มี 8/42 บท) |
| 2 | เทรนเต็มสเกล (5,853 step) | ✅ ทำแล้ว (1 ส.ค. 2026) — **แต่ไม่ผ่านเกณฑ์** (domain PPL +6.6%) |
| 3 | แก้ data leakage bug ใน replay corpus | ✅ แก้แล้ว (`src/prepare_data.py`) |
| 4 | **ลด `learning_rate` แล้วเทรนใหม่** | ⬜ ถัดไป — ลองแถว 1.5e-5–3e-5 (ดู [RESULTS.md §11](RESULTS.md)) |
| 5 | push อัปเดตทับโมเดลเมื่อผ่านเกณฑ์ | ⬜ รอข้อ 4 |
