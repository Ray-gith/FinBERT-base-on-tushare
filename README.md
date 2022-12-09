# FinBERT-base-on-tushare
这个项目使用tfhub中预训练模型bert-base-chinese和tushare中所有的财经快讯新闻（2018年至今）微调了BERT模型，使之更加符合中文的财经新闻语境。

## 训练配置
训练过程见`分布式训练.ipynb`
- RTX3090 * 3
- tensorflow == 2.9.0
- tensorflow-hub、tensorflow-text == 2.9
- tushare所有财经快讯新闻（新浪财经，共100+万条），带有频道标签。

预训练任务即是拟合这个中文财经数据集，使BERT模型更适应中文财经新闻的语境，在下游任务中表现更出色。

模型在此数据集获得样本外AUC>=93%.

## 模型
百度网盘下载链接

https://pan.baidu.com/s/1M9qtgJJqW8eodg7qJrW6yg 

提取码: muzr 

## 使用方法
使用方法和tfhub模型的使用方法一致（见https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4）

Usage.py
```python
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)

preprocessor = hub.KerasLayer('FinBERT/preprocess_layer')
bert_model = hub.KerasLayer('FinBERT/bert_layer')

emb_vec = bert_model(preprocessor(text_input))['pooled_output']
output_layer = tf.keras.layers.Dense(len(classes), activation='sigmoid')(emb_vec)

model = tf.keras.models.Model(text_input, output_layer)
```

