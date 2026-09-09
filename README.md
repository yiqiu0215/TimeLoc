# Qwen3-VL 两阶段时序定位训练

当前版本采用 Transformers 官方 Qwen3-VL 模型及 AutoProcessor，使用标准 RGB 视频输入。自定义 RIT 模型、残差差分 token、残差 gate / LayerNorm / modality embedding、交错序列和连续时间编码均已移除。官方视觉编码、Merger、DeepStack、位置编码及生成流程保持原实现。

模型通过 AutoModelForImageTextToText 加载；Qwen3-VL 配置对应官方 Qwen3VLForConditionalGeneration。两阶段脚本关闭 LoRA 和 Liger 模型替换。接口依据 [Transformers 官方 Qwen3-VL 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)。

## 两阶段策略

1. Stage 1：从官方 Qwen3-VL-2B-Instruct 权重开始，在 GEB+ 上学习给定事件边界时间戳的前后状态描述。
2. Stage 2：加载 Stage 1 完整模型及 processor，在 TimeLens-100K 的 visual-only、按时长均衡采样子集上进行时序定位训练，目标样本数默认 20000。

保留原训练参数策略：冻结 ViT 和 Patch Embedding，训练官方 Merger、DeepStack Merger 和语言模型。两个阶段默认各 1 epoch，学习率均为 1e-5，seed=42。数据过滤、监督答案格式及评测指标未改变。实际保留样本数以数据过滤结果为准。

## 运行

沿用原脚本文件名以保持调用路径兼容；文件名中的 rit 不再表示使用自定义模型。以下命令需在已配置的服务器训练环境中执行：

```bash
bash train_scripts/run_two_stage_rit_qwen3_2b.sh \
  --model_path /path/to/Qwen3-VL-2B-Instruct \
  --gebplus_annotation_path /path/to/GEB+/train.json \
  --gebplus_video_root /path/to/GEB+/videos \
  --timelens_data_root /path/to/TimeLens-100K \
  --num_devices 8 \
  --output_root output/Qwen3VL-2B-TwoStage
```

输出包括 stage1-gebplus 和 stage2-timelens-20k。Stage 1 成功保存后才启动 Stage 2。保留各 epoch checkpoint，默认最终目录保存完整模型和 processor。

单独启动 Stage 2：

```bash
bash train_scripts/run_stage2_rit_qwen3_2b.sh \
  --stage1_model_path /path/to/stage1-gebplus \
  --timelens_data_root /path/to/TimeLens-100K
```

## 视频预处理与评测

默认 FPS=1、min_tokens=64、total_tokens=14336。帧数上限按纯 RGB 块计算：2*floor(total_tokens/min_tokens)，也可通过 fps_max_frames 指定。帧采样与缩放由 qwen_vl_utils 完成，官方 processor 生成视频 token、grid 和时间戳。

每个阶段的最终输出目录额外保存 video_preprocessing.json，记录 min_tokens、total_tokens、fps、fps_max_frames。该文件独立于官方模型 config.json。独立 Stage 2 从中读取预处理设置；评测入口默认读取该文件，显式 CLI 参数优先。

```bash
model_path=/path/to/stage2-timelens-20k \
pred_path=./eval_res \
bash scripts/eval_timelens_bench.sh
```

## 兼容性与验证

旧 RIT checkpoint 会被训练和评测入口拒绝，不能直接当作官方模型继续训练。请从官方权重重新进行 Stage 1，再接续 Stage 2。新版模型权重可由 Transformers 官方模型类加载，不需要项目中的自定义模型源码。

模型结构和视觉输入已改变，旧实验结果不能直接代表当前方案。这里只进行静态及局部逻辑检查，不启动训练、推理或正式评测。环境依赖沿用 requirements_train.txt / requirements.txt，不自动安装或修改环境。

ideas_docs 下的残差设计文档仅作为历史讨论记录。
