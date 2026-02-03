# lecture 6: Kernels, Triton {ignore true}

## 目录{ignore true}

[toc]

## 1 GPU 简介

| 组件 | 类型 | 作用域 | 容量（A100） | 关键特性 |
|------|------|--------|-------------|--------|
| **Grid** | 逻辑 | 整个 GPU | 用户定义 | 包含多个 Thread Blocks |
| **Thread Block** | 逻辑 | Block 内 | ≤1024 线程 | 可同步；调度到单个 SM |
| **Warp** | 逻辑 | Warp 内 | 32 线程 | 硬件调度单位；SIMT 执行 |
| **SM** | 物理 | 单个 SM | 108 个/卡 | 执行 Blocks；含计算单元 + 片上存储 |
| **Shared Memory** | 物理（可配） | Block 内共享 | 最大 164KB | 显式管理；用于协作通信 |
| **L1 Cache** | 物理（可配） | SM 内 | 与 Shared 共享 192KB | 缓存全局内存；可绕过 |
| **L2 Cache** | 物理 | 全 GPU 共享 | 40MB | 统一缓存；服务所有 SM |
| **Global Memory** | 物理 | 全 GPU | 40–80GB | 主存；高带宽、高延迟 |

- **关键说明**
    - **Shared Memory 与 L1 Cache**：同一块物理 SRAM，通过 `cudaFuncSetCacheConfig()` 配置比例。
    - **Block 调度**：一个 Block 固定在一个 SM 上执行；一个 SM 可驻留多个 Blocks。
    - **Warp 执行**：SM 的 Warp 调度器在就绪 Warps 间切换，隐藏内存延迟。
    - **内存路径**：Global → L2 → (L1 或 Shared) → Register；Shared 不经过 L2。
    - **无 CPU 式伪共享**：GPU 无 MESI 协议，合并写入同一 cache line 是高效且推荐的。

## 2 benchmark and profiling

### 2.1 benchmark

比较简单，就是统计预热后的算子平均运行时间

### 2.2 profiling

介绍了一下torch.profiler的用法，详细介绍参考[官网](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

## 3 kernel fusion

主要是参考了[这篇文章](https://horace.io/brrr_intro.html)

未融合的算子，很可能会花费大量的时间在调度上，如[lecture_06课件](https://stanford-cs336.github.io/spring2025-lectures/?trace=var%2Ftraces%2Flecture_06.json&step=7884)中所示

## 4 cuda kernel

在python中可以直接去打开、编译、使用cu文件中定义的函数，像[lecture_06课件](https://stanford-cs336.github.io/spring2025-lectures/?trace=var%2Ftraces%2Flecture_06.json&step=7960)这样

## 5 triton kernel

triton是openAI发展出的DSL，它的编程逻辑单元是thread block，而不像cuda kernel那样是thread

| 功能 | CUDA | Triton |
| - | - | - |
| Memory coalescing (transfer from DRAM) | manual | automatic |
| Shared memory management | manual | automatic |  
| Scheduling within SMs | manual | automatic |
| Scheduling across SMs | manual | manual |

## 6 torch compile

可以直接使用torch编译函数，生成更优化的kernel，同时代码改动量还很少。参考[官方教材](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

torch.compile默认使用的编译后端就是triton，所以在profile中看到triton字样的热点函数是正常的。

## 7 总结

当你写一个kernel的时候，可能需要考虑几种方式：
| kernel实现 | 特点 | 代码改动量 | 性能 |
| - | - | - | - |
| **manual** | 手写 | 中 | 最差 |
| **PyTorch** | 原生实现，无需手写 | 无 | 好，但不同硬件适配性能可能天差地别 |
| **compiled** | torch编译 | 小 | 好 |
| **Triton** | py内写triton kernel | 中 | 好 |
| **CUDA** | 写cu文件，手动编译注册函数 | 大 | 好，但需要深入优化 |
