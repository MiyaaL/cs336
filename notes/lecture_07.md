# lecture 7: PARALLELISM BASICS {ignore true}

## 目录{ignore true}

[toc]

## 1 Basics of networking for LLMs

### 1.1 网络拓扑

这里给了张图，介绍了现在LLM infra里的网络拓扑：

![lecture_07_1](../images/lecture_07/lecture_07_1.png)

| 通信域 | 硬件 | 软件 |
| - | - | - |
| 同节点GPU间 | NVLink + NVSwitch，假如没有NVLink，GPU间就只能走慢得多的PCIe，从CPU侧绕一圈 | NCCL |
| 同节点CPU和GPU间 | PCIe | CUDA内存搬运API |
| 同节点CPU process间 | 共享内存 | MPI |
| 不同节点CPU间 | IB（需要IB网卡，成本高）、RoCE（基于以太网，低成本高性能）、高速以太网（性能最差） | NCCL、MPI |

### 1.2 集合通信

![lecture_07_2](../images/lecture_07/lecture_07_2.png)

- all_reduce可以拆分为reduce_scatter、all_gather两步来操作

![lecture_07_3](../images/lecture_07/lecture_07_3.png)

## 2 Standard LLM parallelization primitives

### 2.1 data parallelism

这里以SGD为例，介绍了三种数据并行的stage，先来看公式：
$$
    \theta_{t+1} = \theta_{t} - \eta \sum_{i=1}^B \nabla f(x_i) \label{2.1} \tag{2.1}
$$
这里总共有几个数据，分别是：

- **activations:** $x$
- **parameters:** $\theta$
- **gradients:** $\nabla$
- **optimizer_states:** 例如Adam优化器中的动量和方差

针对这些数据的切分方法，有了几种不同的memory优化方式。这里主要介绍**ZeRO (ZeroRedundancy Optimizer)**，它是**DeepSpeed**提出的并行训练显存优化方法。根据优化程度，ZeRO可以分为三个stage：
- **ZeRO-1 (Stage 1):** 仅对优化器状态进行分区。这是最基础的内存优化。
- **ZeRO-2 (Stage 2):** 在 ZeRO-1 的基础上，进一步对梯度进行分区。
- **ZeRO-3 (Stage 3):** 在 ZeRO-2 的基础上，再对模型参数本身进行分区。在前向和反向计算时，按需从其他 GPU 收集所需的参数，计算完成后再释放。

下面分别介绍。

#### 2.1.1 ZeRO-1

下面是stage 1的计算以及通信方式：

![lecture_07_4](../images/lecture_07/lecture_07_4.png)

解释如下：
- 因为SGD梯度是全局所有batch累加的(见式$\eqref{2.1}$)，所以必须要进行一次全局的reduce，接下来解释为什么是reduce_scatter而不是all_reduce
- 因为对梯度的更新，不需要每张卡都对所有梯度进行，而是只更新自己的那部分之后再all_gather就可以了（尽管这时所有卡都拿着所有的parameters）。所以上述全局reduce并不需要all_reduce之后全量更新梯度，而是拆分为了reduce_scatter + all_gather。
- 所以总的来说stage 1的通信是：**1 reduce scatter (send gradients) + 1 all gather (collect params)**

#### 2.1.2 ZeRO-2

![lecture_07_5](../images/lecture_07/lecture_07_5.png)

解释如下：
- 如何在每张卡仍然拿着全部parameters的情况下去对gradients做shard，答案是求完梯度后立马发送到对应的卡上，本卡只留对应分区的gradients，其他梯度全部删除释放掉。这需要一次reduce_scatter
- 优化器对应分区的状态可以直接拿到，不再需要单独更新。（因为当前卡对应分区上的gradients是准的，已经reduce过的，拿着这个gradients输入到优化器中得到的优化器状态也是准的）
- 最后需要一次all_gather来合并所有卡上的parameters
- 所以总的来说stage 2的通信是：**1 reduce scatter (send gradients) + 1 all gather (collect params)**，与stage 1保持一致。stage 2的收益在于非当前分区的gradients被提前释放掉，内存占用量降低。

#### 2.1.3 ZeRO-3 (aka FSDP) 

![lecture_07_6](../images/lecture_07/lecture_07_6.png)

解释如下：
- 首先每张卡不再加载全量模型参数，所以前向和反向传播时，需要all_gather来拿全部parameters，计算完毕后释放掉对应memory（因为模型有很多层，所以这样做是有收益的）
- 同样地，类似stage 2，也需要一次reduce_scatter来完成对gradients的更新
- 所以总的来说stage 3的通信是：**1 reduce scatter (send gradients) + 2 all gather (collect params)**

#### 2.1.4 问题

- 边际效益递减：一味的增加并行度可能并不会有很大收益，因为通信成本也在不断增加
- 对memory的scale不是很好，增加卡数每张卡的算力利用率会下降较多

#### 2.1.5 总结

| 并行方式 | activations | parameters | gradients | optimizer_states | comm |
| ------- | ----------- | ---------- | --------- | ---------------- | ----- |
| **ZeRO-1** | no-shard | no-shard | no-shard | shard | 1 reduce scatter (send gradients) + 1 all gather (collect params) |
| **ZeRO-2** | no-shard | no-shard | shard | shard | 1 reduce scatter (send gradients) + 1 all gather (collect params) |
| **ZeRO-3** | no-shard | shard | shard | shard | 1 reduce scatter (send gradients) + 2 all gather (collect params) |

### 2.2 model parallelism

#### 2.2.1 Pipeline parallel

对模型分层，每个模型只去计算其中的一层或者几层，之后在多batch场景下便可以流水线化：

![lecture_07_7](../images/lecture_07/lecture_07_7.png)

PP的好处是：
- 节省内存：每张卡可能只需要保存几层的模型参数
- 通信友好：不需要集合通信，只需要跟当前卡的上下游卡去进行点对点通信即可

当网络受限又想去降低显存瓶颈时，PP是一个较好的手段。但是上述PP手段对batch size比较敏感，大batch size可以把算力打的更高，空泡更少，但是小batch size情况就很差了。

因此有了**Zero bubble pipelining**：
- **关键idea**：拆分反向传播（Backward Pass），将标准的反向传播过程拆分为两个独立的部分：
    - B-pass (Backward for activations)：计算并传递用于激活的梯度（即中间结果的梯度）。这部分依赖于下游阶段的 B-pass。
    - W-pass (Backward for weights)：计算模型参数（权重）本身的梯度。这部分只依赖于本阶段的前向结果和 B-pass 的输出，可以更早地执行。

#### 2.2.2 Tensor parallel (+ Sequence parallel)

