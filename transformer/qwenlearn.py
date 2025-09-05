import torch
import torch.nn as nn
from transformers import AutoTokenizer

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "用两句话解释量子纠缠。"},
]

TOKENIZER_MODEL_DIR = "./data/qwen2-0.5b"

# 1) Load tokenizer / model (automatically select device & precision)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_DIR, use_fast=True)

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

inputs = tokenizer([prompt], return_tensors="pt")

