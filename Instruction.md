## 注意:
- Dataset:
  - ieee.txt
  - 29 abstractions

- Begin:
  - "abstract:"

- Parameter:
  - temperature = 1.0
  - EPOCHS = 50
  - BATCH_SIZE = 32

- Environment:
  - python 3.7.13
  - anaconda 4.10.3
  - tensorflow 1.15.0


- Dataset:
  - shakespeare.txt
  - 40000 lines

- Begin:
  - "ROMEO:"

- Parameter:
  - temperature = 1.0
  - EPOCHS = 50
  - BATCH_SIZE = 32

- Environment:
  - python 3.7.13
  - anaconda 4.10.3
  - tensorflow 1.15.0

---

# 步骤

## 1.数据集预处理

- 选取要求：
  1. 不含大量符号
  2. 不含latex等表达式
  3. 尽量少的缩写

- 预处理
  1. 每一段摘要以`Abstract`开始
  2. 将段落拆分成句

## 模型参数设定

---

# 总结

## 缺点
- 有很多的专有名词
- 有很多缩写省略