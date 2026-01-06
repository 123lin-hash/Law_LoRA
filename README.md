# 基于LoRA微调的本地法律领域大模型应用
本项目是一个基于 Python 的本地法律领域大模型，应用于智能法律检索、合同审查、法律咨询、法律文书生成以及辅助司法决策等场景，在提升法律服务效率和降低专业门槛方面展现出广阔前景。
## 所需依赖安装指南
`pip install requirements.txt`
## 数据集
本实验使用了法律微调数据集，数据集地址为<https://www.modelscope.cn/datasets/KuugoRen/chinese_law_ft_dataset>。
### 数据集下载
`python data_download.py`
## 基座模型
本实验使用了Qwen2.5-7B-Instruct。
### 模型下载
`python model_download.py`
## 嵌入模型
由于使用bert_score对模型进行评测时需要使用合适的嵌入模型，本实验使用了longbert-embedding-8k-zh作为嵌入模型，。
### 模型下载
`python embedding_download.py`
## 模型微调训练
`python train.py`
## 运行指南
`python main.py`
