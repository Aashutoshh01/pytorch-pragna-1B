from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from models import ModelArgs, Transformer

weights_path = hf_hub_download("soketlabs/pragna-1b",filename="model.safetensors")
hf_sd = load_file(weights_path)

converted = {}

for key, tensor in hf_sd.items():
    if "rotary_emb.inv_freq" in key:
        continue
    new_key = key
    new_key = new_key.replace("model.embed_tokens.weight","tok_embeddings.weight")
    new_key = new_key.replace("lm_head.weight","output.weight")
    new_key = new_key.replace("model.norm.weight","norm.weight")
    new_key = new_key.replace("input_layernorm","attention_norm")
    new_key = new_key.replace("post_attention_layernorm","ffn_norm")
    new_key = new_key.replace("self_attn.q_proj","attention.wq")
    new_key = new_key.replace("self_attn.k_proj","attention.wk")
    new_key = new_key.replace("self_attn.v_proj","attention.wv")
    new_key = new_key.replace("self_attn.o_proj","attention.wo")
    new_key = new_key.replace("mlp.gate_proj","feed_forward.w1")
    new_key = new_key.replace("mlp.down_proj","feed_forward.w2")
    new_key = new_key.replace("mlp.up_proj","feed_forward.w3")
    new_key = new_key.replace("model.","")
    converted[new_key] = tensor

model = Transformer(ModelArgs())
missing, unexpected = model.load_state_dict(converted,strict=False)
print("\nMissing Keys")
print("-" * 50)

for k in missing:
    print(k)

print("\nUnexpected Keys")
print("-" * 50)

for k in unexpected:
    print(k)

print("\nSummary")
print("-" * 50)
print("Missing:", len(missing))
print("Unexpected:", len(unexpected))