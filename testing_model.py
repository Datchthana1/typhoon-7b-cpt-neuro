from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "typhoon-ai/typhoon-7b", dtype="bfloat16", device_map="auto"
)
model = PeftModel.from_pretrained(base, "Datchthana/typhoon-7b-neuro-cpt")
tok = AutoTokenizer.from_pretrained("typhoon-ai/typhoon-7b")

prompt = "ความสามารถของสมองในการปรับเปลี่ยนโครงสร้างตัวเองตามประสบการณ์"
out = model.generate(
    **tok(prompt, return_tensors="pt").to(model.device),
    max_new_tokens=200,
    do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.15,
)
print(tok.decode(out[0], skip_special_tokens=True))