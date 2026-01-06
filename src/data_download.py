from modelscope.msdatasets import MsDataset

model_dir = MsDataset.load('KuugoRen/chinese_law_ft_dataset', cache_dir='/data/linfengyun/dataset')