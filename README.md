## Learning AI

本仓库包含 AI 学习笔记与代码示例，重点围绕 **llama.cpp** 生态。

> 📖 完整阅读顺序 PDF：[`llama_cpp_reading_order.pdf`](./llama_cpp_reading_order.pdf)
>
> 🗺️ 交互式 Roadmap（带进度追踪）：[打开 Roadmap](https://blime4.github.io/learning-ai/roadmap.html)

---

## 🗺️ llama.cpp 学习路线图

进度追踪：在 GitHub 上直接编辑此文件，勾选 `[ ]` → `[x]` 即可记录学习进度。

### 第一阶段：基础与环境搭建

- [ ] **1.** [`fundamentals/ggml/README.md`](./fundamentals/ggml/README.md) — GGML 是 llama.cpp 的底层张量库，先理解 tensor、backend、计算图等基本概念
- [ ] **2.** [`fundamentals/llama.cpp/README.md`](./fundamentals/llama.cpp/README.md) — 项目入口：子模块配置、编译方式、GDB 调试、CUDA 构建
- [ ] **3.** [`notes/llama.cpp/debugging.md`](./notes/llama.cpp/debugging.md) — 调试技巧，后续阅读源码笔记时会频繁用到

### 第二阶段：核心概念（推理流水线）

- [ ] **4.** [`fundamentals/llama.cpp/src/tokenize.cpp`](./fundamentals/llama.cpp/src/tokenize.cpp) / [`notes/llama.cpp/llama-vocab-notes.md`](./notes/llama.cpp/llama-vocab-notes.md) — 分词器 — 推理的第一步
- [ ] **5.** [`fundamentals/llama.cpp/src/simple-prompt.cpp`](./fundamentals/llama.cpp/src/simple-prompt.cpp) — 最简单的推理示例，理解整体调用流程
- [ ] **6.** [`notes/llama.cpp/llama-batch.md`](./notes/llama.cpp/llama-batch.md) — llama_batch 和 llama_ubatch 结构，token 如何组织送入模型
- [ ] **7.** [`notes/llama.cpp/process_ubatch.md`](./notes/llama.cpp/process_ubatch.md) — micro-batch 处理细节
- [ ] **8.** [`notes/llama.cpp/kv-cache.md`](./notes/llama.cpp/kv-cache.md) — KV 缓存机制 — 推理加速的核心
- [ ] **9.** [`fundamentals/llama.cpp/src/kv-cache.cpp`](./fundamentals/llama.cpp/src/kv-cache.cpp) — KV 缓存的代码实践
- [ ] **10.** [`notes/llama.cpp/graph-inputs.md`](./notes/llama.cpp/graph-inputs.md) — 计算图输入的构建
- [ ] **11.** [`notes/llama.cpp/gpu-sampling.md`](./notes/llama.cpp/gpu-sampling.md) — GPU 上的采样实现（temperature、top-k、top-p 等）
- [ ] **12.** [`notes/llama.cpp/output_ids.md`](./notes/llama.cpp/output_ids.md) — 输出 token ID 的处理
- [ ] **13.** [`fundamentals/llama.cpp/src/simple-prompt-multi.cpp`](./fundamentals/llama.cpp/src/simple-prompt-multi.cpp) — 多 prompt 批处理示例

### 第三阶段：GPU 加速与后端

- [ ] **14.** [`notes/llama.cpp/cuda.md`](./notes/llama.cpp/cuda.md) — CUDA 后端加载机制（ggml_backend_load_all）
- [ ] **15.** [`notes/llama.cpp/cuda-mul-mat.md`](./notes/llama.cpp/cuda-mul-mat.md) — CUDA 矩阵乘法实现
- [ ] **16.** [`notes/llama.cpp/cuda-fp16-release-build-issue.md`](./notes/llama.cpp/cuda-fp16-release-build-issue.md) — FP16 构建问题记录
- [ ] **17.** [`fundamentals/ggml/src/llama-att-softmax.cpp`](./fundamentals/ggml/src/llama-att-softmax.cpp) — attention softmax 的 GGML 实现
- [ ] **18.** [`notes/llama.cpp/flash-attn-misalignment-issue.md`](./notes/llama.cpp/flash-attn-misalignment-issue.md) — Flash Attention 对齐问题
- [ ] **19.** [`notes/llama.cpp/macosx.md`](./notes/llama.cpp/macosx.md) — macOS (Metal) 平台相关
- [ ] **20.** [`notes/llama.cpp/ggml-threadpool-macos-issue.md`](./notes/llama.cpp/ggml-threadpool-macos-issue.md) — 线程池问题

### 第四阶段：模型转换与量化

- [ ] **21.** [`notes/llama.cpp/convert.md`](./notes/llama.cpp/convert.md) — convert_hf_to_gguf.py 流程解析
- [ ] **22.** [`notes/llama.cpp/quantization.md`](./notes/llama.cpp/quantization.md) — 量化原理与 QAT 量化
- [ ] **23.** [`notes/llama.cpp/convert-dequantize.md`](./notes/llama.cpp/convert-dequantize.md) — 反量化过程
- [ ] **24.** [`notes/llama.cpp/devstral2-conversion.md`](./notes/llama.cpp/devstral2-conversion.md) — Devstral2 模型转换实例
- [ ] **25.** [`notes/llama.cpp/convert-mamba-issue.md`](./notes/llama.cpp/convert-mamba-issue.md) — Mamba 模型转换问题
- [ ] **26.** [`notes/llama.cpp/gemma-bos-issue.md`](./notes/llama.cpp/gemma-bos-issue.md) — Gemma BOS token 问题

### 第五阶段：Embeddings

- [ ] **27.** [`fundamentals/llama.cpp/src/embeddings.cpp`](./fundamentals/llama.cpp/src/embeddings.cpp) — embedding 生成代码
- [ ] **28.** [`notes/llama.cpp/embeddings-presets.md`](./notes/llama.cpp/embeddings-presets.md) — embedding 预设配置

### 第六阶段：Server 与部署

- [ ] **29.** [`notes/llama.cpp/llama-server.md`](./notes/llama.cpp/llama-server.md) — server 启动与 API 调用
- [ ] **30.** [`notes/llama.cpp/server-checkpoints.md`](./notes/llama.cpp/server-checkpoints.md) — checkpoint 管理
- [ ] **31.** [`notes/llama.cpp/server-logprob-issue.md`](./notes/llama.cpp/server-logprob-issue.md) — log probability 问题
- [ ] **32.** [`notes/llama.cpp/server-unit-tests.md`](./notes/llama.cpp/server-unit-tests.md) — server 测试
- [ ] **33.** [`notes/llama.cpp/llama-perplexity.md`](./notes/llama.cpp/llama-perplexity.md) — 困惑度计算
- [ ] **34.** [`notes/llama.cpp/tests.md`](./notes/llama.cpp/tests.md) — 测试框架
- [ ] **35.** [`notes/llama.cpp/sbatch.md`](./notes/llama.cpp/sbatch.md) — SLURM 批量提交

### 第七阶段：Finetuning

- [ ] **36.** [`fundamentals/llama.cpp/README.md`](./fundamentals/llama.cpp/README.md) — LoRA 微调、Shakespeare 数据集、chat 格式训练

### 第八阶段：多模态与特殊模型

- [ ] **37.** [`notes/llama.cpp/llama-3-2-vision.md`](./notes/llama.cpp/llama-3-2-vision.md) — Llama 3.2 视觉模型
- [ ] **38.** [`notes/llama.cpp/qwen-2.5VL-3B-instruct.md`](./notes/llama.cpp/qwen-2.5VL-3B-instruct.md) — Qwen 视觉模型
- [ ] **39.** [`notes/llama.cpp/vision-model-issue.md`](./notes/llama.cpp/vision-model-issue.md) — 视觉模型问题
- [ ] **40.** [`fundamentals/image-processing/src/mllama.cpp`](./fundamentals/image-processing/src/mllama.cpp) — Llama 视觉模型实现
- [ ] **41.** [`notes/llama.cpp/tts.md`](./notes/llama.cpp/tts.md) — TTS 集成

### 第九阶段：Agent 与上层应用

- [ ] **42.** [`agents/llama-cpp-agent/README.md`](./agents/llama-cpp-agent/README.md) — 基于 WASM 的 agent 框架
- [ ] **43.** [`agents/llama-cpp-agent/agent/src/main.rs`](./agents/llama-cpp-agent/agent/src/main.rs) / [`agent.rs`](./agents/llama-cpp-agent/agent/src/agent.rs) / [`tool.rs`](./agents/llama-cpp-agent/agent/src/tool.rs) — Rust agent 实现

### 第十阶段：其他语言绑定与集成

- [ ] **44.** [`fundamentals/python/src/llama-chat-format.py`](./fundamentals/python/src/llama-chat-format.py) — Python chat 格式处理
- [ ] **45.** [`fundamentals/rust/llm-chains-llama-example/README.md`](./fundamentals/rust/llm-chains-llama-example/README.md) — Rust LLM chains + Llama
- [ ] **46.** [`fundamentals/rust/llm-chains-chat-demo/src/main-llama.rs`](./fundamentals/rust/llm-chains-chat-demo/src/main-llama.rs) — Rust chat demo

### 补充：Issue 笔记（按需查阅）

- [ ] **A1.** [`notes/llama.cpp/sched-issue.md`](./notes/llama.cpp/sched-issue.md) — 调度问题
- [ ] **A2.** [`notes/llama.cpp/update_chat_msg-issue.md`](./notes/llama.cpp/update_chat_msg-issue.md) — chat 消息更新问题

---

### Topics

* [Tokenization](./notes/tokenization/README.md)
* [Architectures](./notes/architectures/README.md)
* [GGML](./notes/ggml.md)
* [Llama.cpp](./notes/llama.md)
* [Position Embeddings](./notes/position-embeddings)
* [GPUs](./gpu/README.md)
* [Vector Databases](./notes/vector-databases.md)
* [Vision](./notes/vision)

### Examples/Exploration code

* [GGML](./fundamentals/ggml) GGML C library exploration code
* [Llama.cpp](fundamentals/llama.cpp) Llama.cpp library exploration code
* [GPU](gpu/README.md) CUDA, Kompute, Metal, OpenCL, ROCm, and Vulkan exploration code
* [Embeddings](./embeddings) Word embeddings examples in Rust and Python
* [Huggingface API](./hugging-face/python) Huggingface API example written in Python
* [Qdrant Vector Database](./vector-databases/qdrant) Examples in Python and Rust
* [LanceDB Vector Database](./vector-databases/lancedb) Examples in Python and Rust
