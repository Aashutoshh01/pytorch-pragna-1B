import torch
from transformers import AutoTokenizer
from models import ModelArgs, Transformer
from convert_weights import converted

tokenizer = AutoTokenizer.from_pretrained("soketlabs/pragna-1b")

model = Transformer(ModelArgs())
model.load_state_dict(converted,strict=True)
model.eval()

prompt = "भारत की राजधानी"

input_ids = tokenizer("भारत की राजधानी",return_tensors="pt")["input_ids"]

for _ in range(30):
    with torch.no_grad():
        logits = model(input_ids)
    next_token = torch.argmax(logits[:, -1, :],dim=-1,keepdim=True)
    input_ids = torch.cat([input_ids, next_token],dim=1)

print(tokenizer.decode(input_ids[0]))