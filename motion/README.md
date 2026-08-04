# Charades 运动向量分布提取

该脚本从 Charades 原始视频中确定性选取固定百分比的视频，重新编码为 H.264，读取编码块运动向量，并绘制 1 fps 的运动特征分布图。原始视频和标注文件不会被修改。

## 输出

默认只生成两类文件：

```text
/workspace/s/lzw/datasets/TimeLens-Bench-H264/charades/
├── 3MSZA.mp4
└── 3MSZA.motion.png
```

其中 PNG 横轴为秒级时间戳，纵轴为残差运动特征。标注中的每个事件 span 使用一对同色竖虚线表示。原子状态区间使用状态专属的半透明背景色和区间顶部标签表示，不进行趋势拟合。

## 特征定义

视频保持原始帧率编码，并禁用 B 帧、限制为单参考帧。脚本对每个包含运动向量的 P 帧执行：

1. 根据 `motion_scale` 和帧间时间差，将运动向量转换为像素/秒。
2. 使用编码块面积加权中位数估计全局平移。
3. 计算去除全局平移后的残差向量模长。
4. 将残差模长截断至面积加权的第 95 百分位。
5. 计算面积加权 RMS，并通过 `log1p` 压缩动态范围。
6. 对每个一秒窗口内的帧级特征取第 90 百分位，得到 1 fps 曲线。

I 帧没有运动向量，因此不会作为零运动参与计算。整秒缺失时使用相邻有效秒线性插值。

## 阈值原子状态拆分

拆分直接使用当前视频的 1 fps 原始运动特征，不进行归一化、趋势拟合或数据集级统计。设相邻两秒的特征差为：

```text
delta[t] = motion[t + 1] - motion[t]
```

逐项暴力遍历并赋予状态：

```text
delta[t] > threshold   -> RISING
delta[t] < -threshold  -> FALLING
其他                    -> STABLE
```

连续相同的状态合并为一个时间区间，状态变化的位置直接作为新区间起点，因此所有区间连续、互斥且完整覆盖视频。

在生成最终区间前，`STABLE` 作为中性状态按相邻方向归并：

```text
RISING  -> STABLE... -> RISING   => RISING
FALLING -> STABLE... -> FALLING  => FALLING
```

视频开头或结尾只与一种方向相邻的 `STABLE` 也并入该方向。若 `STABLE` 两侧方向相反，例如 `RISING -> STABLE -> FALLING`，则保留 `STABLE` 作为独立区间，避免把两个相反趋势错误合并。

### 原子状态

脚本只输出和绘制以下三种互斥原子状态：

| 状态 | 含义 |
| --- | --- |
| `RISING` | 相邻秒运动特征上升幅度超过阈值 |
| `FALLING` | 相邻秒运动特征下降幅度超过阈值 |
| `STABLE` | 相邻秒运动特征变化幅度不超过阈值 |

阈值通过 `--change-threshold` 指定，默认是 `0.1`，单位与分布图纵轴一致。增大阈值会得到更多 `STABLE` 区间，减小阈值会得到更多 `RISING` 和 `FALLING` 区间。

## 环境

服务器需要可用的 `ffmpeg`，且包含 `libx264` 编码器。Python 依赖列在当前目录的 `requirements.txt` 中。请在项目指定的服务器环境中按需安装，避免使用默认 base 环境。

可先检查：

```bash
ffmpeg -hide_banner -encoders | grep libx264
python -c "import av, matplotlib, numpy; print(av.__version__)"
```

## 运行

默认选取 20%，`seed=42`，使用单进程：

```bash
python motion/extract_motion_features.py
```

建议先用较小比例检查环境和图像效果：

```bash
python motion/extract_motion_features.py \
  --percentage 0.1 \
  --workers 1
```

确认后再执行默认 20%：

```bash
python motion/extract_motion_features.py \
  --percentage 20 \
  --seed 42 \
  --workers 8 \
  --ffmpeg-threads 4 \
  --change-threshold 0.1
```

相同候选集合、`--percentage` 和 `--seed` 会选中相同的视频 ID。候选集合仅包含原始目录中唯一存在且在标注 JSON 中有记录的视频。

默认不会覆盖已有 MP4 或 PNG：编码视频存在时会复用；两种输出都存在时会跳过该视频。只有显式传入 `--overwrite` 才会替换已有结果。

如果编码视频已经存在，只需要重新生成带阈值原子状态分区的分布图，可执行：

```bash
python motion/extract_motion_features.py \
  --percentage 20 \
  --seed 42 \
  --workers 8 \
  --change-threshold 0.1 \
  --overwrite-plots
```

`--overwrite-plots` 会复用已有 H.264 视频，不会重新编码；`--overwrite` 才会同时替换视频和图像。

如需将分布图放在单独目录：

```bash
python motion/extract_motion_features.py \
  --plot-dir /workspace/s/lzw/datasets/TimeLens-Bench-H264/charades-motion-plots
```

完整参数可通过以下命令查看：

```bash
python motion/extract_motion_features.py --help
```
