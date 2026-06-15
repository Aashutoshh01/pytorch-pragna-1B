import torch
from transformers import AutoTokenizer
from models import ModelArgs, Transformer
from convert_weights import converted

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained("soketlabs/pragna-1b")

print("Loading model...")

model = Transformer(ModelArgs())
model.load_state_dict(converted,strict=True)
model.eval()

prompts = [
    "The capital of India is",
    "भारत की राजधानी",
    "Explain machine learning",
    "नमस्ते मेरा नाम"
]

MAX_NEW_TOKENS = 50

output_file = "sampling_comparison_results.txt"

with open(output_file,"w",encoding="utf-8") as f:
    f.write("PRAGNA SAMPLING COMPARISON\n")
    f.write("-" * 80 + "\n\n")
    for prompt in prompts:
        print(f"Testing: {prompt}")
        input_ids = tokenizer(prompt,return_tensors="pt")["input_ids"]
        greedy_ids = model.generate(input_ids.clone(),max_new_tokens=MAX_NEW_TOKENS,temperature=0.0)
        greedy_text = tokenizer.decode(greedy_ids[0],skip_special_tokens=True)
        topk_ids = model.generate(input_ids.clone(),max_new_tokens=MAX_NEW_TOKENS,temperature=0.8,top_k=50)
        topk_text = tokenizer.decode(topk_ids[0],skip_special_tokens=True)
        topp_ids = model.generate(input_ids.clone(),max_new_tokens=MAX_NEW_TOKENS,temperature=0.8,top_p=0.9)
        topp_text = tokenizer.decode(topp_ids[0],skip_special_tokens=True)

        f.write(f"PROMPT:\n{prompt}\n\n")
        f.write("GREEDY OUTPUT:\n")
        f.write(greedy_text)
        f.write("\n\n")
        f.write("TOP-K OUTPUT:\n")
        f.write(topk_text)
        f.write("\n\n")
        f.write("TOP-P OUTPUT:\n")
        f.write(topp_text)
        f.write("\n\n")
        f.write("-" * 80)
        f.write("\n\n")

print(f"\nSaved results to: {output_file}")