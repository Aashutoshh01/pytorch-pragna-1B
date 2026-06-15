import torch
from transformers import (AutoTokenizer,AutoModelForCausalLM)
from models import ModelArgs, Transformer
from convert_weights import converted
tokenizer = AutoTokenizer.from_pretrained("soketlabs/pragna-1b")

print("Loading HF model...")

hf_model = AutoModelForCausalLM.from_pretrained("soketlabs/pragna-1b",torch_dtype=torch.float32)
hf_model.eval()

print("Loading custom model...")

custom_model = Transformer(ModelArgs())

custom_model.load_state_dict(converted,strict=True)
custom_model.eval()

prompts = [
    "भारत की राजधानी",
    "नमस्ते मेरा नाम",
    "Explain machine learning",
    "The capital of India is"
]

MAX_NEW_TOKENS = 50

with open("hf_vs_custom_results.txt","w",encoding="utf-8") as f:
    f.write("HF vs CUSTOM PRAGNA RUNTIME\n")
    f.write("-" * 80 + "\n\n")
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt,return_tensors="pt")
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            hf_out = hf_model.generate(input_ids,max_new_tokens=MAX_NEW_TOKENS,do_sample=False)

        hf_text = tokenizer.decode(hf_out[0],skip_special_tokens=True)

        custom_ids = input_ids.clone()

        for _ in range(MAX_NEW_TOKENS):
            with torch.no_grad():
                logits = custom_model(custom_ids)
            next_token = torch.argmax(logits[:, -1, :],dim=-1,keepdim=True)
            custom_ids = torch.cat([custom_ids, next_token],dim=1)

        custom_text = tokenizer.decode(custom_ids[0],skip_special_tokens=True)

        f.write(f"PROMPT:\n{prompt}\n\n")
        f.write("HF OUTPUT:\n")
        f.write(hf_text)
        f.write("\n\n")
        f.write("CUSTOM OUTPUT:\n")
        f.write(custom_text)
        f.write("\n\n")
        f.write("-" * 80)
        f.write("\n\n")

print("\nSaved to hf_vs_custom_results.txt")