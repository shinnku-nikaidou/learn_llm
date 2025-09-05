import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "./data/qwen2-0.5b"

# 1) Load tokenizer / model (automatically select device & precision)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype="auto",  # Prefer bf16/fp16 on GPU, fp32 on CPU
    device_map="auto",  # Automatically place model on available GPU/CPU
)

# 2) Prepare a simple chat input (recommended to use chat template)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "用两句话解释量子纠缠。"},
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,  # Automatically add assistant start token
)

# 3) Encode and feed into model
inputs = tokenizer([prompt], return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

# 4) Print only the newly generated part (remove the prompt)
gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
print(tokenizer.decode(gen_ids, skip_special_tokens=True))
