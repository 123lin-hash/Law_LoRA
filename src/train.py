from datasets import Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
import matplotlib.pyplot as plt
import json
import os
import yaml

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# process_fun可以将它们统一为微调模型期望的格式
def process_func(example):
    MAX_LENGTH = 2048   
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|><|im_start|>user\n{example['instruction'] + example['input']}<|im_end|><|im_start|>assistant\n")     # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<|im_end|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1] 
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == "__main__":
    cfg = load_config("config.yaml")

    with open(cfg["data"]["system_prompt_file"], "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model"],
        device_map="auto",
        torch_dtype=torch.bfloat16 if cfg["model"]["torch_dtype"] == "bfloat16" else torch.float32,
        trust_remote_code=cfg["model"]["trust_remote_code"]
    )
    model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model"],
        use_fast=cfg["model"]["use_fast_tokenizer"],
        trust_remote_code=cfg["model"]["trust_remote_code"]
    )
    tokenizer.pad_token = tokenizer.eos_token

    df = pd.read_json(cfg["data"]["train_file"])
    ds = Dataset.from_pandas(df)
    tokenized = ds.map(process_func, remove_columns=ds.column_names)

    # LoRA 配置
    lora_cfg = cfg["lora"]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        inference_mode=False
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 训练参数
    train_cfg = cfg["training"]
    args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        logging_steps=train_cfg["logging_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        save_steps=train_cfg["save_steps"],
        learning_rate=train_cfg["learning_rate"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        save_on_each_node=train_cfg["save_on_each_node"]
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
















