import torch

from models import ModelArgs, Transformer

config = ModelArgs()

model = Transformer(config)

print("Model created")
x = torch.randint(0,config.vocab_size,(1, 16))
print("Input shape:", x.shape)

with torch.no_grad():
    logits = model(x)

print("Output shape:", logits.shape)