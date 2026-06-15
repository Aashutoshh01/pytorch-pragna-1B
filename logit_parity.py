import torch
from transformers import (AutoTokenizer,AutoModelForCausalLM)
from models import ModelArgs, Transformer
from convert_weights import converted

prompt = "The capital of India is"

tokenizer = AutoTokenizer.from_pretrained("soketlabs/pragna-1b")

inputs = tokenizer(prompt,return_tensors="pt")
input_ids = inputs["input_ids"]

print("Loading HF model...")

hf_model = AutoModelForCausalLM.from_pretrained("soketlabs/pragna-1b",torch_dtype=torch.float32)
hf_model.eval()

print("Loading custom model...")

custom_model = Transformer(ModelArgs())

custom_model.load_state_dict(converted,strict=True)
custom_model.eval()

with torch.no_grad():
    hf_logits = hf_model(input_ids).logits[:, -1, :]
    custom_logits = custom_model(input_ids)[:, -1, :]

diff = (hf_logits - custom_logits).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()
std_diff = diff.std().item()

print("\nLOGIT PARITY")
print(f"Max Difference  : {max_diff:.8f}")
print(f"Mean Difference : {mean_diff:.8f}")
print(f"Std Difference  : {std_diff:.8f}")

hf_top = torch.argmax(hf_logits, dim=-1).item()
custom_top = torch.argmax(custom_logits, dim=-1).item()

print("\nTOP TOKEN")

print("HF Token:", hf_top, "|", tokenizer.decode([hf_top]))
print("Custom Token:", custom_top, "|",tokenizer.decode([custom_top]))

with open("logit_parity_results.txt","w",encoding="utf-8") as f:
    f.write("LOGIT PARITY REPORT\n")
    f.write("-" * 60 + "\n\n")
    f.write(f"Prompt:\n{prompt}\n\n")
    f.write(f"Max Difference  : {max_diff:.8f}\n")
    f.write(f"Mean Difference : {mean_diff:.8f}\n")
    f.write(f"Std Difference  : {std_diff:.8f}\n\n")
    f.write("HF Top Token:\n")
    f.write(f"{hf_top} -> {tokenizer.decode([hf_top])}\n\n")
    f.write("Custom Top Token:\n")
    f.write(f"{custom_top} -> {tokenizer.decode([custom_top])}\n")
print("\nSaved logit_parity_results.txt")