from models import ModelArgs, Transformer

config = ModelArgs()
model = Transformer(config)

print("MODEL PARAMETERS\n")

with open("custom_model_inventory.txt", "w", encoding="utf-8") as f:
    for name, param in model.state_dict().items():
        line = f"{name:<70} {tuple(param.shape)}"
        print(line)
        f.write(line + "\n")
print("\nSaved to custom_model_inventory.txt")