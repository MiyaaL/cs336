# lecture 13: Data {ignore true}

## 目录{ignore true}

[toc]

## 1 介绍

- 数据集非常重要，并且LLM公司一般都会对数据保密

训练阶段：
 
- 预训练：使用原始文本（例如，来自网络的文档）进行训练
- 训练中期：使用更多高质量数据进行训练以增强能力  
- 训练后：根据指令跟随数据进行微调（或进行强化学习）以进行指令跟随

## 2 主流模型数据集

### 2.1 bert

书籍语料、维基百科（可能会被数据投毒）

### 2.2 gpt2_webtext

WebText：来自Reddit论坛的帖子

### 2.3 Common Crawl

爬虫抓取网页文本，html->text

### 2.4 CCNet

- 目标：构建用于预训练的大型高质量数据集的自动化方法
- 从网络爬虫提取高质量数据集

### 2.5 t5_c4（T5 - Collosal Clean Crawled corpus）

对爬虫数据做筛选，例如：
- 保留以标点符号结尾且字数大于等于 5 个单词的行。
- 删除少于3句话的页面
- 删除包含任何“不雅词汇”的页面
- 使用语言检测功能过滤非英语文本（英语文本的概率为 0.99）

### 2.6 gpt3

Common Crawl + WebText2 + 维基百科 + 书籍语料

### 2.7 the pile

涵盖大量语料来源，民间发起志愿收集

### 2.8 gopher massivetext

MassiveText数据集用于训练Gopher，语料来源有：
- MassiveWeb
- C4
- 书籍
- 新闻
- GitHub
- 维基百科

### 2.9 llama

- Common Crawl
- C4
- github
- 维基百科
- arxiv
- ...

### 2.10 RefinedWeb

基于网络数据做精炼

### 2.11 dolma

- Common Crawl
- C4
- 维基百科
- ...

### 2.12 dclm

- 目标：在固定模型架构的前提下，通过系统化实验探索“什么样的训练数据能带来最佳性能”，并证明数据质量 > 数据量，推动模型发展

### 2.13 nemotron-cc

- NVIDIA发布的高质量数据集，网络爬虫而来

## 3 总结

现在LLM训练里大量使用网络数据，并进行加工精炼、筛选去重后得到高质量数据集进行训练。