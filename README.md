## 数据制作
```
cd data_generate/generate_datasets_v2
```
#### step1
```
python chinese_public_dataset_preprocess.py
```
#### step2
```
python data_vad.py
```
#### step3
```
python audio_preprocess.py
```


#### 注意
```
1. 更换脚本中的文件路径
2. 最后的gt被存放在clean_gt_base中
3. 处理后的音频数据放在processed_datasets中
```



## 训练
```
先在configs/config_v1.py中进行训练的配置
python train/coach_v1.py
```