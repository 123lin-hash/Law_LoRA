#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('OctopusMind/longbert-embedding-8k-zh', cache_dir='/data/linfengyun/models', revision='master')