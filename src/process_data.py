import json

data_path = "/data/linfengyun/dataset/1.json"
train_path = "/data/linfengyun/dataset/train.json"
test_path = "/data/linfengyun/dataset/test2.json"

train_count = 10000
test_count = 12000

with open(data_path, "r", encoding="utf-8") as f_in:
    data = json.load(f_in)

test_data = data[train_count:test_count]

with open(test_path, "w", encoding="utf-8") as f_out:
    json.dump(test_data, f_out, indent=4, ensure_ascii=False)
