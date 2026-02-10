# lecture 12: Evaluation {ignore true}

## 目录{ignore true}

[toc]

## 1 介绍

- 现阶段模型评估依赖于各种benchmark
- 各家模型排行榜可以参考[HELM](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard)

## 2 perplexity

困惑度本身是一个好的评估方式：
$$
\text{PPL} = \exp(\text{CrossEntropy}) = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid w_1, w_2, \dots, w_{i-1}) \right)
$$

因为：
- 平滑连续
- 普适性强

现在除了使用perplexity外，也会使用下游任务表现来对模型进行评估。

## 3 benchmark

- MMLU（Massive Multitask Language Understanding）：涵盖57个领域，主要考察知识水平而不是语言理解能力
- MMLU-pro：CoT评估，更加注重逻辑推理，选项从4个增加到了10个
- GPQA（Graduate-Level Google-Proof Q&A）：研究生级别，博士撰写，难度较高
- HLE（Humanity's Last Exam）：各科学界极难问题

## 4 指令跟随benchmark

评估模型是否能够准确理解并严格执行用户以自然语言给出的约束性指令。

- Chatbot Arena：用户提问，选两个匿名模型中更优的回答
- Instruction-Following Eval (IFEval)：完全自动化，对每个约束单独打分（0/1），最终报告"严格遵循率"（Strict Accuracy）和"宽松遵循率"（Loose Accuracy）
- AlpacaEval：1. 模型生成回答 → 2. GPT-4裁判盲评 → 3. 计算胜率（Win Rate）作为指标 
- WildBench：自动化评估框架，结合规则验证与LLM裁判（GPT-4）

## 5 agent benchmark

关注模型行动能力，而不仅仅是生成能力，是评估模型能否真正"干活"的关键标准

- SWEBench：给定开源Python项目代码库 + issue描述（含复现步骤），要求agent生成patch修复问题
- CyBench：自主完成网络安全攻防任务，在隔离沙箱环境中，要求agent发现漏洞、编写exploit、获取flag
- MLEBench：端到端完成Kaggle竞赛，从数据下载、EDA、特征工程、模型训练、调参到最终提交预测结果

## 6 pure reasoning benchmark

将知识与推理能力进行分离

- ARC-AGI：设计成对人类简单，但是对LLM来说非常困难的题目，例如从图形中推断隐藏规则并进行预测。

## 7 safety benchmark

评估LLM是否可能生成有害信息，在安全性上的权衡是否合理

- HarmBench
- AIR-Bench
- Jailbreaking

## 8 总结

- 没有唯一正确的评估方法
- benchmark关注点：性能、安全性、成本、现实性