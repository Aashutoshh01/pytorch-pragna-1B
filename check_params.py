from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from models import ModelArgs, Transformer


weights_path = hf_hub_download("soketlabs/pragna-1b",filename="model.safetensors")
hf_sd = load_file(weights_path)
hf_total = sum(v.numel() for v in hf_sd.values())
print(f"HF params: {hf_total:,}")

model = Transformer(ModelArgs())
custom_total = sum(p.numel() for p in model.parameters())
print(f"Custom params: {custom_total:,}")

print(f"Difference: {hf_total - custom_total:,}")