# https://blog.csdn.net/ziyi9663/article/details/107311169

import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, losses, optimizers
import jieba 

with open('bailuyuan.txt',encoding='utf-8') as f:
   article = f.readlines()[0]
#读取文章

cutted_article = [i for i in article]
#article变量是一个包含了文章内容的长字符串，这里用列表迭代式就能获取每个字符了，
#文章生成用的是字符级别的，没有切词，用不着，
#保留了标点，毕竟写好的文章也得有标点符号不是。
vocab = sorted(set(cutted_article))
#创建从非重复字符到索引的映射
char2idx = {u:i for i, u in enumerate(vocab)}
#官方示例从0开始映射，我习惯从1开始，0分配给稀有字符，写文章的时候出现稀有字符也不好看，这里就先不改了
idx2char = np.array(vocab)
#index到字符的映射
text_as_int = np.array([char2idx[c] for c in article])
#把整个文章用编码表示

seq_length = 100
#设定每个输入句子长度的最大值
examples_per_epoch = len(article)//seq_length
#一共能创建多少个样本
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
   input_text = chunk[:-1]
   target_text = chunk[1:]
   return input_text, target_text

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
dataset = dataset.repeat(200).shuffle(1000).batch(BATCH_SIZE, drop_remainder=True)
# dataset = dataset.shuffle(1000).batch(BATCH_SIZE, drop_remainder=True)
#测试文章就七千多字，直接repeat一下，不然都凑不够batch的


vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

class MyModel(keras.Model):
   def __init__(self, units,batch_size,input_length=None):
       super(MyModel, self).__init__()
       self.forward = Sequential([
           layers.Embedding(vocab_size,
                            embedding_dim,
                            input_length=input_length,
                            batch_input_shape=[batch_size,input_length],
                            trainable=True),
           layers.GRU(units, dropout=0.2, return_sequences=True,stateful=True),
           layers.Dense(vocab_size)
       ])
       # 想要设置stateful=True，就必须在Sequential包装的第一层里设置batch_input_shape
       # 或者在Input layer里设置batch_shape，但是在Input layer里怎么设置都是报错，所以这里只能用第一个方案了，当然这俩都一样，我个人不喜欢Sequential
       # 不设置stateful参数也行，我觉得也没啥必要，效果都差不多

   def call(self,input,training=None):
       return self.forward(input)

model = MyModel(rnn_units, BATCH_SIZE)

model.compile(optimizer=optimizers.Adam(1e-3),
             loss = losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(dataset,epochs=2)

model.save_weights('model.ckpt')
del model

model = MyModel(rnn_units,batch_size=1)
model.load_weights('model.ckpt')
model.build(input_shape=(1,None))
model.summary()
#写摘要时，是手动输入几个字，所以输入的第一个维度是1，需要重新build一下，
#但是如果不删掉模型并重新加载权重的话，build虽然不会报错，但是模型结构也没改



# def gen_article(model,start_string,length,temperature=1.0):
#    """
#    生成文章
#    :param model:           训练好的模型
#    :param start_string:    起始文字
#    :param length:          计划写的文章长度
#    :param temperature:     低温度会生成更可预测的文本，较高温度会生成更令人惊讶的文本，可以通过试验以找到最好的设定
#    :return:                写好的文章
#    """
#    # 将起始字符串转换为数字（向量化）
#    input_eval = [char2idx[s] for s in start_string]
#    input_eval = tf.expand_dims(input_eval, 0)

#    # 空字符串用于存储结果
#    text_generated = []

#    model.reset_states()
#    for i in range(length):
#        predictions = model(input_eval)
#        # 删除批次的维度
#        predictions = tf.squeeze(predictions, 0)

#        # 用分类分布预测模型返回的字符
#        predictions = predictions / temperature
#        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
#    	   # 从这个分布中抽样很重要，用argmax的话很容易使模型卡在循环中。
#        # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
#        input_eval = tf.expand_dims([predicted_id], 0)

#        text_generated.append(idx2char[predicted_id])


#    final_article = start_string + ''.join(text_generated)
#    return final_article

# with open('start_string.txt',encoding='utf-8') as l:
#    start_s = l.readlines()[0]

# final_article = gen_article(model,start_s,200)
