import time
import torch
from transformers import AutoTokenizer
from models import ModelArgs, Transformer
from convert_weights import converted

tokenizer = AutoTokenizer.from_pretrained("soketlabs/pragna-1b")
model = Transformer(ModelArgs())
model.load_state_dict(converted,strict=True)
model.eval()
prompts = [
    "भारत की राजधानी",
    "Explain machine learning",
    "नमस्ते मेरा नाम"
]

results = []

for prompt in prompts:
    input_ids = tokenizer(prompt,return_tensors="pt")["input_ids"]
    start = time.time()
    generated = model.generate(input_ids,max_new_tokens=50,temperature=0.8,top_k=50)
    elapsed = time.time() - start
    output_text = tokenizer.decode(generated[0],skip_special_tokens=True)
    tokens_generated = (generated.shape[1]- input_ids.shape[1])
    tps = tokens_generated / elapsed
    results.append(
        {
            "prompt": prompt,
            "tokens": tokens_generated,
            "time": elapsed,
            "tokens_per_sec": tps,
            "output": output_text,
        }
    )

with open("benchmark_results.txt","w",encoding="utf-8") as f:

    f.write("PRAGNA CUSTOM RUNTIME BENCHMARK\n")
    f.write("-" * 80 + "\n\n")
    for r in results:
        f.write(f"Prompt: {r['prompt']}\n")
        f.write(f"Tokens Generated: {r['tokens']}\n")
        f.write(f"Time: {r['time']:.2f} sec\n")
        f.write(f"Tokens/sec: {r['tokens_per_sec']:.2f}\n")
        f.write("\nOutput:\n")
        f.write(r["output"])
        f.write("\n")
        f.write("-" * 80 + "\n\n")

print("Saved benchmark_results.txt")