from transformers import Trainer
import matplotlib.pyplot as plt
import json

# 读取 trainer_state.json，它里面包含 log_history
state_path = "./output/Qwen2.5-7B-Instruct_lora/checkpoint-1875/trainer_state.json"

with open(state_path, "r", encoding="utf-8") as f:
    trainer_state = json.load(f)

log_history = trainer_state["log_history"]

losses = []
steps = []
for item in log_history: 
    if "loss" in item:
        losses.append(item["loss"]) 
        steps.append(item.get("step", len(steps)))

plt.figure(figsize=(8, 5))
plt.plot(steps, losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve (Restored)")
plt.grid()
plt.savefig("./result/loss_curve.png")
