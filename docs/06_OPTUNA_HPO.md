# 06 — การค้นหา Hyperparameter ด้วย Optuna

## ปัญหาที่ Optuna ต้องแก้ในงาน CPT (ต่างจาก HPO ทั่วไป)

HPO ปกติมี objective เดียว: ทำให้ validation loss ต่ำที่สุด

แต่ใน CPT การไล่ตาม domain loss อย่างเดียวจะได้คำตอบที่**พังโมเดล**
เพราะ `learning_rate` สูง ๆ กับจำนวน epoch มาก ๆ ทำให้ PPL บนโดเมนใหม่ต่ำได้จริง
โดยแลกกับการที่โมเดลลืมภาษาไทยทั่วไปไปหมด

**ผลจริงที่วัดได้ในโปรเจกต์นี้** (ดู [RESULTS.md](../RESULTS.md)) พิสูจน์ข้อนี้อย่างชัดเจน:
`lr=1e-4, r=32, 8 epochs` บนคลัง 15 บล็อก → train loss ลงถึง **0.005**
แต่ domain PPL **พุ่งจาก 11.07 เป็น 45.13** และ general PPL **จาก 8.03 เป็น 38.86**

Optuna ที่มี objective เป็น domain loss อย่างเดียวจะ**เลือกค่าชุดนี้ว่าดี**
ทั้งที่มันทำลายโมเดล ดังนั้น objective ต้องมองสองด้าน

---

## Objective ที่ใช้

```
score = domain_loss + λ · max(0, general_loss − general_loss_baseline)
        └──────────┘   └──────────────────────────────────────────────┘
         เรียนได้ดีแค่ไหน            ลืมของเดิมไปแค่ไหน
```

### รายละเอียดของแต่ละส่วน

**`domain_loss`** คือ cross-entropy บน held-out ของคลังใหม่
ใช้ loss ตรง ๆ ไม่ใช่ PPL เพราะ `loss = log(PPL)` การบวกใน log-space
ทำให้สองพจน์มีสเกลเทียบกันได้ (ถ้าใช้ PPL พจน์เดียวจะครอบงำหมด)

**`max(0, ...)`** เป็น hinge — ลงโทษเฉพาะเมื่อ **แย่ลงกว่า baseline** เท่านั้น
ถ้าโมเดลบังเอิญเก่งภาษาไทยทั่วไปขึ้นด้วย จะไม่ได้รางวัลพิเศษ (ซึ่งถูกต้อง เพราะไม่ใช่เป้าหมาย)

**`λ`** ตั้งใน `configs/optuna.yaml` → `forgetting_penalty`

| λ | ความหมาย | ใช้เมื่อ |
|---|---|---|
| 0 | ไม่สนใจ forgetting | คลังใหญ่มาก และจะใช้โมเดลเฉพาะโดเมนนี้เท่านั้น |
| **2** (ค่าเริ่มต้น) | สมดุล | กรณีทั่วไป |
| 5 | หวงความสามารถเดิมมาก | โมเดลต้องใช้งานทั่วไปด้วย |

---

## Search space และเหตุผลของแต่ละตัว

| พารามิเตอร์ | ช่วง | ทำไมช่วงนี้ |
|---|---|---|
| `learning_rate` | 3e-5 – 4e-4, **log scale** | ผลของ lr เป็นเชิงคูณ การสุ่มแบบ linear จะกระจุกที่ค่าสูง ค่าต่ำกว่า 3e-5 แทบไม่ขยับ LoRA ค่าสูงกว่า 4e-4 ทำให้ divergence |
| `lora_r` | 16 / 32 / 64 | ต่ำกว่า 16 = capacity ไม่พอเก็บความรู้ใหม่ สูงกว่า 64 = OOM บน 8GB และ overfit คลังเล็ก |
| `alpha_ratio` | 1 / 2 / 4 → `alpha = r × ratio` | **ค้นอัตราส่วน ไม่ใช่ค่าสัมบูรณ์** เพราะ alpha ที่เหมาะขึ้นกับ r เสมอ ถ้าค้นแยกกันจะได้คู่ที่ไม่มีความหมาย เช่น r=64 กับ alpha=8 |
| `grad_accum` | 8 / 16 / 32 | `micro_batch` ล็อกที่ 1 เพราะ VRAM จึงปรับ effective batch ผ่านตัวนี้ (= 8K/16K/32K token ต่อ step) |
| `lr_scheduler` | cosine / linear / constant_with_warmup | cosine มักดีที่สุดเมื่อ step เยอะ constant_with_warmup ดีกว่าเมื่อ step น้อยมาก |
| `warmup_ratio` | 0.01 – 0.12 | CPT ช่วงแรก gradient แกว่งแรง warmup สั้นเกินทำให้ loss กระโดด |
| `weight_decay` | 0.0 – 0.10 | regularization ที่ช่วยเมื่อคลังเล็ก |
| `max_grad_norm` | 0.3 – 1.0 | ต่ำกว่า default (1.0) เพราะ CPT บนคลังเล็กมี gradient spike บ่อย |

---

## กลไกการตัด trial ที่ไม่มีอนาคต (Pruning)

แต่ละ trial เทรนเต็มจะกินเวลา 30–60 นาที × 25 trials = นานเกินรับได้
Optuna จึงตัด trial ที่แย่ทิ้งกลางคัน

```python
class PruningCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.step += 1
        self.trial.report(metrics["eval_loss"], self.step)
        if self.trial.should_prune():
            raise optuna.TrialPruned(...)
```

ใช้ `MedianPruner` — ตัด trial ที่ eval_loss ณ step นั้นแย่กว่ามัธยฐานของ trial ก่อนหน้า

| พารามิเตอร์ | ค่า | เหตุผล |
|---|---|---|
| `n_startup_trials` | 6 | ต้องมีข้อมูลอย่างน้อย 6 trial ก่อนจึงจะมีมัธยฐานที่เชื่อถือได้ |
| `pruner_warmup_evals` | 2 | ให้ทุก trial ได้ eval อย่างน้อย 2 ครั้ง — trial ที่ warmup ยาวจะดูแย่ตอนต้นแต่ดีตอนท้าย |

**OOM ก็ถูกจับเป็น prune ด้วย** ไม่ใช่ crash:
```python
except torch.cuda.OutOfMemoryError:
    raise optuna.TrialPruned("OOM")
```
ทำให้ Optuna เรียนรู้เองว่า `r=64 × grad_accum=32` ไม่ไหวบนเครื่องนี้ แล้วเลี่ยงไปเอง

---

## Sampler

```python
TPESampler(seed=42, n_startup_trials=6, multivariate=True)
```

`multivariate=True` สำคัญมาก — TPE แบบปกติมองแต่ละพารามิเตอร์เป็นอิสระต่อกัน
แต่ `learning_rate` กับ `grad_accum` มีปฏิสัมพันธ์กันสูง (batch ใหญ่ขึ้นรับ lr สูงขึ้นได้)
โหมด multivariate จำลอง joint distribution จึงหาคู่ที่ดีได้เร็วกว่า

---

## การใช้ subset เพื่อความเร็ว

```yaml
max_steps_per_trial: 150
subset_ratio: 0.35
```

แต่ละ trial เทรนบน 35% ของ train set เป็นเวลา 150 step
สมมติฐานคือ **การจัดอันดับ** ของ hyperparameter บน subset ใกล้เคียงกับบน full set
ซึ่งเป็นสมมติฐานที่ใช้ได้ดีในทางปฏิบัติแม้จะไม่สมบูรณ์แบบ

หลังได้ค่าที่ดีที่สุดแล้ว `train_cpt.py --from-study` จะเทรนเต็มด้วยค่านั้นอีกครั้ง

---

## คำสั่ง

```powershell
python main.py hpo                    # 25 trials ตาม config
python main.py hpo -- --trials 40     # กำหนดจำนวนเอง
python main.py hpo -- --resume        # ทำต่อจาก study เดิม (เก็บใน sqlite)
python main.py report                 # สรุปผล + fANOVA importance
python main.py train -- --from-study  # เทรนเต็มด้วยค่าที่ดีที่สุด
```

Study เก็บใน `outputs/optuna.db` (SQLite) จึงหยุดกลางคันแล้วรันต่อได้

ดูแบบ web UI:
```powershell
optuna-dashboard sqlite:///outputs/optuna.db
```

---

## การอ่านผล `--report`

```
ความสำคัญของแต่ละพารามิเตอร์ (fANOVA)
  learning_rate        0.512 ████████████████████
  lora_r               0.203 ████████
  grad_accum           0.118 ████
  warmup_ratio         0.087 ███
```

fANOVA บอกว่าพารามิเตอร์ใดอธิบายความแปรปรวนของ score ได้มากที่สุด
ถ้า `learning_rate` ครองเกิน 0.5 (ซึ่งเป็นเรื่องปกติมาก) แปลว่า
รอบถัดไปควร**แคบช่วง lr ลง**รอบค่าที่ดีที่สุด แล้วขยายช่วงตัวอื่นแทน

---

## ข้อควรระวังที่พบจากการรันจริง

1. **ถ้า val set เล็กเกินไป objective จะมี noise สูงกว่า signal**
   ในการรันจริงของโปรเจกต์นี้ val มีเพียง 1 บล็อก → PPL แกว่งแรงมาก
   ควรมี val อย่างน้อย 30–50 บล็อก จึงจะแยกความต่างระหว่าง trial ได้จริง

2. **ต้องมี `general_val` ไม่งั้น penalty term หายไปเงียบ ๆ**
   โค้ดจะเตือนใน log แต่จะรันต่อโดยใช้ objective แค่ครึ่งเดียว

3. **baseline ถูกแคชไว้ที่ `outputs/baseline.json`**
   ถ้าเปลี่ยน val set ต้องลบไฟล์นี้ทิ้ง ไม่งั้น penalty จะเทียบกับเส้นฐานผิด
