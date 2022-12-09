# FinBERT-base-on-tushare
这个项目使用tfhub中预训练模型bert-base-chinese和tushare中所有的财经快讯新闻（2018年至今）微调了BERT模型，使之更加符合中文的财经新闻语境。

## 训练配置
- RTX3090 * 3
- tensorflow == 2.9.0
- tensorflow-hub、tensorflow-text
- tushare所有财经快讯新闻（共100+万条），带有频道标签。

## 模型
百度网盘下载链接

https://pan.baidu.com/s/1M9qtgJJqW8eodg7qJrW6yg 

提取码: muzr 

## 使用方法
```python
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text


```
