# Qwen Parallelism Examples

这组脚本面向单机 `2` 卡 / `4` 卡并行策略学习，默认使用 `mock-data`，重点是理解 Megatron-LM 的并行维度，而不是追求真实收敛。

## 运行前提

- 当前机器已经安装好 `Transformer Engine` 与 `Apex`
- `conda` 环境名默认是 `llm`
- 脚本会自动尝试：
  - `conda activate llm`
  - 设置 `CUDA_HOME=/usr/local/cuda`
  - 设置 `CUDA_DEVICE_MAX_CONNECTIONS=1`
  - 设置 `HF_ENDPOINT=https://hf-mirror.com`

## 推荐顺序

1. `run_qwen3_8b_tp2_2gpu.sh`
2. `run_qwen3_8b_tp2_sp_2gpu.sh`
3. `run_qwen3_8b_pp2_2gpu.sh`
4. `run_qwen3_8b_cp2_2gpu.sh`
5. `run_qwen3_8b_dp2_2gpu.sh`
6. `run_qwen3_8b_tp2_sp_pp2_4gpu.sh`
7. `run_qwen3_8b_tp2_sp_cp2_4gpu.sh`
8. `run_qwen3_8b_dp2_tp2_sp_4gpu.sh`
9. `run_moe_ep2_2gpu.sh`

## 主要脚本

- `run_qwen3_8b_tp2_2gpu.sh`
  - 2 卡纯 TP。
- `run_qwen3_8b_tp2_sp_2gpu.sh`
  - 2 卡 TP + SP，对比纯 TP。
- `run_qwen3_8b_pp2_2gpu.sh`
  - 2 卡纯 PP。
- `run_qwen3_8b_cp2_2gpu.sh`
  - 2 卡纯 CP，默认把 `SEQ_LENGTH` 提高到 `8192`。
- `run_qwen3_8b_dp2_2gpu.sh`
  - 2 卡纯 DP。

4 卡对应脚本同理。

## EP 说明

`Qwen3-8B` 是 dense 模型，不能直接用于 Expert Parallelism。  
因此：

- `run_moe_ep2_2gpu.sh`
- `run_moe_ep4_4gpu.sh`

使用的是一个小型 `toy GPT-MoE` 配置，只用于理解 EP 的运行方式和参数约束。

## 常用覆盖方式

脚本中的大多数参数都可以通过环境变量覆盖，例如：

```bash
TRAIN_ITERS=2 DRY_RUN=1 bash examples/Qwen/run_qwen3_8b_tp2_2gpu.sh
```

```bash
SEQ_LENGTH=4096 GLOBAL_BATCH_SIZE=16 bash examples/Qwen/run_qwen3_8b_tp2_sp_2gpu.sh
```

常用变量包括：

- `TRAIN_ITERS`
- `SEQ_LENGTH`
- `MICRO_BATCH_SIZE`
- `GLOBAL_BATCH_SIZE`
- `OUT_DIR`
- `DRY_RUN=1`

`DRY_RUN=1` 只打印最终命令，不实际启动训练。
