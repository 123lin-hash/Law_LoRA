import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_chinese import Rouge
import jieba
from bert_score import score

mode_path = "/data/linfengyun/models/Qwen/Qwen2.5-7B-Instruct"
lora_path = "./output/Qwen2.5-7B-Instruct_lora/checkpoint-1875"
test_path = "/data/linfengyun/dataset/test.json"
save_result_path = "./result/eval_result_2.json"   # ← 只保存指标

device = "cuda:4"
rouge = Rouge()

print(">>> Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    mode_path,
    device_map={"": 4},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

print(">>> Loading LoRA...")
model = PeftModel.from_pretrained(base_model, lora_path).eval().to(device)


with open(test_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

preds = []
refs = []
bleu_scores = 0
rouge_scores = 0
bert_scores = 0
print(">>> Evaluating...")


def char_split(text):
    return " ".join(list(text))

for item in test_data:
    instruction = item["instruction"]
    reference = item["output"]

    messages = [
        {"role": "system", "content": "你是一名专业法律从业者。"},
        {"role": "user", "content": instruction}
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer([input_text], return_tensors="pt").to(device)

    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=512
    )

    outputs = outputs[:, inputs.input_ids.shape[1]:]

    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # ROUGE指标
    reference1 = ' '.join(jieba.cut(reference))
    pred1 = ' '.join(jieba.cut(pred))
    rouge_score = rouge.get_scores(pred1, reference1)
    rouge_score = rouge_score[0]['rouge-l']['f']
    rouge_scores += rouge_score
    # BLEU指标
    reference2 = list(jieba.cut(reference))
    pred2 = list(jieba.cut(pred))
    bleu_score = sentence_bleu([reference2], pred2, weights=(1,0,0,0))
    bleu_scores += bleu_score
    # bert_score指标
    P, R, F1 = score([pred], [reference], model_type="/data/linfengyun/models/OctopusMind/longbert-embedding-8k-zh", num_layers=12, lang="zh",verbose=True)
    bert_scores += F1.item()


bleu_score = bleu_scores / len(test_data)
rouge_score = rouge_scores / len(test_data)
bert_score = bert_scores / len(test_data)

result = {
    "test_samples": len(test_data),
    "BLEU": bleu_score,
    "ROUGE": rouge_score,
    "bert_score": bert_score
}

with open(save_result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print("\n>>> 最终评估结果已保存到：", save_result_path)
print(result)
