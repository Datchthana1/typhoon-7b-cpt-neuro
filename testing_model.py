import torch

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer

# NF4 keeps the 7B model within the RTX 4060 Laptop GPU's 8 GB VRAM and
# avoids the large Windows RAM/page-file allocation caused by full BF16 loading.
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("โหลด base model (typhoon-7b, NF4)...", flush=True)
base = AutoModelForCausalLM.from_pretrained(
    "typhoon-ai/typhoon-7b",
    quantization_config=quantization,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)

print("โหลด LoRA adapter (typhoon-7b-neuro-cpt)...", flush=True)
model = PeftModel.from_pretrained(base, "Datchthana/typhoon-7b-neuro-cpt")
model.eval()

tok = AutoTokenizer.from_pretrained("typhoon-ai/typhoon-7b")
streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
prompt = "สมองเราปรับเปลี่ยนได้เนื่องจาก"
inputs = tok(prompt, return_tensors="pt").to("cuda")

print("เริ่ม generate (สตรีมทีละ token ด้านล่าง)...", flush=True)
with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        streamer=streamer,
    )

print("\n=== ผลลัพธ์เต็ม ===")
print(tok.decode(out[0], skip_special_tokens=True))
