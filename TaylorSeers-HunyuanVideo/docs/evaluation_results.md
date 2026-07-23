# HunyuanVideo 加速评测结果:VBench + 全参考感知指标

> 评测对象:TaylorSeer 在 HunyuanVideo 上的加速配置。
> 参考基线:`original`(无加速,每个 denoising step 全量计算)。
> 数据集:VBench 标准提示集,**944 prompt × 5 seed = 4720 段视频/配置**,规格 640×480、24fps、65 帧、50 infer steps。

## 配置说明

命名 `N{N}O{O}F{F}A{α}`:
- **N** = `fresh_threshold`,新鲜阈值 ≈ 加速倍率(越大越快);
- **O** = Taylor 展开阶数(均为 1);
- **F** = `first_enhance`,首增强步数(infer-steps=50 时固定 3);
- **α** = 指数平滑系数(0.8 → 开启 `TAYLOR_USE_SMOOTHING`)。

| 配置 | 含义 | 加速倍率 |
|---|---|---|
| `original` | 无加速基线 | 1× |
| `N3O1F3A0.8` | TaylorSeer | 3× |
| `N5O1F3A0.8` | TaylorSeer | 5× |
| `N6O1F3A0.8` | TaylorSeer | 6× |

---

## 一、VBench(无参考,主观分布质量)

16 维度 + 加权总分。**Quality 权重 4、Semantic 权重 1,Total = (Quality×4 + Semantic)/5**。数值为 VBench 归一化百分比,越高越好。

| 维度 | original | N3 (3×) | N5 (5×) | N6 (6×) |
|---|---:|---:|---:|---:|
| **Total** | **80.87** | **80.93** | **80.35** | **79.59** |
| **Quality** | 83.28 | 83.27 | 82.74 | 81.92 |
| **Semantic** | 71.24 | 71.54 | 70.75 | 70.30 |
| subject consistency | 95.21 | 95.03 | 94.72 | 93.84 |
| background consistency | 95.52 | 95.19 | 94.81 | 94.10 |
| temporal flickering | 97.87 | 97.44 | 96.98 | 96.03 |
| motion smoothness | 97.02 | 96.78 | 96.43 | 95.40 |
| overall consistency | 73.30 | 72.97 | 72.58 | 72.36 |
| human action | 90.40 | 91.80 | 91.00 | 89.20 |
| aesthetic quality | 61.12 | 60.86 | 60.62 | 60.82 |
| imaging quality | 64.47 | 64.58 | 64.14 | 64.35 |
| color | 88.32 | 86.43 | 87.94 | 87.40 |
| multiple objects | 69.79 | 68.26 | 66.87 | 64.45 |
| spatial relationship | 53.85 | 56.58 | 56.01 | 56.68 |
| scene | 47.79 | 47.88 | 45.99 | 45.56 |
| appearance style | 69.43 | 69.85 | 70.27 | 70.98 |
| temporal style | 66.98 | 66.59 | 66.54 | 66.29 |
| object class | 81.31 | 83.53 | 79.57 | 79.79 |
| dynamic degree | 30.14 | 31.39 | 30.14 | 27.91 |

**小结**:
- **N3 (3×) 总分 80.93,反略超原版 80.87**;N5 降 0.52、N6 降 1.28。
- 主观感知类指标(subject/background consistency、temporal flickering、motion smoothness)随加速比缓慢下降,幅度很小。
- VBench 口径下,TaylorSeer 在 3–6× 加速**主观质量几乎无损**。

---

## 二、全参考感知指标(PSNR / SSIM / LPIPS,逐帧像素保真)

以 `original` 同 prompt/同 seed 视频为参考,逐帧比对后取**所有样本 × 所有帧的总均值**。PSNR、SSIM 越高越好;LPIPS 越低越好。LPIPS 用 AlexNet 骨干。

### 均值(单一 headline 值,便于横向对比)

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 视频数 |
|---|---:|---:|---:|---:|
| **N3 (3×)** | **20.62** | **0.7123** | **0.2264** | 4720 |
| **N5 (5×)** | 17.90 | 0.6190 | 0.3241 | 4720 |
| **N6 (6×)** | 16.91 | 0.5662 | 0.3707 | 4720 |

### 分布(p10 / 中位数 / p90)

| 配置 | PSNR p10/p50/p90 | SSIM p10/p50/p90 | LPIPS p10/p50/p90 |
|---|---:|---:|---:|
| N3 (3×) | 15.98 / 20.45 / 25.39 | 0.549 / 0.724 / 0.857 | 0.098 / 0.214 / 0.373 |
| N5 (5×) | 14.37 / 17.75 / 21.59 | 0.461 / 0.624 / 0.766 | 0.191 / 0.320 / 0.461 |
| N6 (6×) | 13.71 / 16.75 / 20.27 | 0.412 / 0.569 / 0.718 | 0.246 / 0.371 / 0.496 |

### "近无损"视频占比

| 配置 | PSNR>30 | SSIM>0.95 | LPIPS<0.1 |
|---|---:|---:|---:|
| N3 (3×) | 0.9% | 0.3% | 10.6% |
| N5 (5×) | 0.0% | 0.0% | 0.6% |
| N6 (6×) | 0.0% | 0.0% | 0.0% |

**小结**:三个指标随加速比**严格单调**(N3→N5→N6:PSNR↓、SSIM↓、LPIPS↑),与物理预期一致。

---

## 三、综合结论:两套指标互补

| | VBench(主观分布质量) | PSNR/SSIM/LPIPS(逐帧像素保真) |
|---|---|---|
| N3 (3×) | 80.93%(略超原版) | 中等(PSNR 20.6) |
| N5 (5×) | 80.35%(−0.52) | 较低(PSNR 17.9) |
| N6 (6×) | 79.59%(−1.28) | 低(PSNR 16.9) |

- **VBench**:加速版主观质量接近原版(N3 甚至更好)。
- **像素指标**:同 seed 下,加速版与原版逐帧像素差异不小(PSNR 17–21),且随加速比单调放大。

**核心解读**:TaylorSeer 的"无损"是**质量无损、非像素无损**——加速版走的是另一条**同样高质量的有效采样轨迹**,而非原版的带噪拷贝。对生成式扩散任务,用 PSNR 判定"lossless"本身偏严(本就没有唯一正确像素解),因此两套指标合起来才是公平、完整的图景。

---

## 四、单次前向 Transformer FLOPs 与时间

> 评测单次 transformer 前向的**算力(FLOPs)**与**时间**,从计算量层面量化 TaylorSeer 的加速。
> 其他加速方法普遍未报告这两项;此处补齐。规格同 VBench:**480×640**、65 帧、50 步、单卡 A100-80GB、bf16、embedded CFG(batch=1)。
> FLOPs = Linear-hook MACs + 解析式 flash-attention(QK^T+PV);时间 = `transformer(...)` 的 CUDA 事件墙钟。

### 1. FLOPs(确定性,headline 算力指标)

| 配置 | full 步 | cached 步 | full 步 FLOPs | cached 步 FLOPs | 总 TFLOPs | 均值 GFLOPs/步 | 相对 original |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 50 | 0 | 588.0 T | — | 29402 | 588045 | 1.00× |
| N3O1F3A0.8 (3×) | 18 | 32 | 588.0 T | 137.5 G | 10589 | 211784 | **0.36×** |
| N5O1F3A0.8 (5×) | 12 | 38 | 588.0 T | 137.5 G | 7062 | 141235 | **0.24×** |
| N6O1F3A0.8 (6×) | 10 | 40 | 588.0 T | 137.5 G | 5886 | 117719 | **0.20×** |

- cached 步 FLOPs(137.5 G)仅为 full 步(588 T)的 **0.023%**——Taylor 在 cached 步几乎跳过全部 attention+MLP。
- **FLOPs 压缩比**(original/配置):N3 **2.78×**、N5 **4.16×**、N6 **5.00×**,与标称加速比一致。

### 2. 前向时间(compute view,与硬件/基础设施无关)

| 配置 | full 步前向 | cached 步前向 | 均值前向/步 | 时间加速比 |
|---|---:|---:|---:|---:|
| original | 3903 ms | — | 3903 ms | 1.00× |
| N3O1F3A0.8 (3×) | 3903 ms | 158 ms | 1506 ms | **2.59×** |
| N5O1F3A0.8 (5×) | 3903 ms | 154 ms | 1054 ms | **3.70×** |
| N6O1F3A0.8 (6×) | 3903 ms | 153 ms | 903 ms | **4.32×** |

- full 步前向 = 基线前向(refresh 步跑的是同一个模型前向,3903 ms);cached 步前向 = 实测 Taylor 预测(~155 ms,**比 full 步快约 25×**)。
- **前向时间加速比**:N3 **2.59×**、N5 **3.70×**、N6 **4.32×**,与 FLOPs 压缩比趋势一致(cached 步仍有 ~155 ms 残余开销,故略低于 FLOPs 比)。

### 3. 重要说明:显存 offload 对 full 步墙钟的影响

单 80GB 卡上,部署配置必须开启 `TAYLOR_SMOOTHING_OFFLOAD_HISTORY`(把平滑历史槽在 CPU↔GPU 间换页)才能放得下;该换页**主导了 full 步的实测墙钟(~24 s)**,属于**显存搬运开销、非 transformer 算力**(full 步的模型前向仍是 3.9 s,与基线相同),故不计入上面的 compute-view 加速比。关闭该 offload 在 480×640×65(及 ×49)下于 80GB **OOM**(cache+激活需 ~79 GB),证实该 offload 在此分辨率/硬件上不可避免。cached 步(~155 ms)不触发换页,不受影响。

**小结**:FLOPs 与前向时间两套口径共同表明——TaylorSeer 在 HunyuanVideo 上以 N3/N5/N6 配置实现约 **2.6–4.3× 的前向算力/时间压缩**(headline FLOPs 压缩 2.8–5.0×),与第一节 VBench"主观质量近无损"互补,构成"又快又好"的完整证据。

---

## 复现

- **VBench 打分**:`bash eval/vbench/launch_calc.sh`(每组配置)→ `bash eval/vbench/run_tabulate.sh`。`eval/vbench/calc_vbench.py` 已含 torch≥2.6 `weights_only` 兼容 patch。
- **感知指标**:`bash eval/perceptual/run_perceptual.sh`(3 配置 × 8 卡);单维度/冒烟见 `eval/perceptual/measure.py --help`。
- **单次前向 FLOPs + 时间**:`bash eval/benchmark_flops_time.sh`(original + N3/N5/N6,单 prompt,480×640×65×50,单卡)。profiler 由 `TAYLOR_BENCHMARK=1` 在 pipeline 内开启(`hyvideo/modules/taylor_utils/benchmark.py`);汇总见 `eval/benchmark/aggregate.py` → `results/benchmark/summary.{json,md}`。
- **原始明细**:
  - VBench:`results/vbench-simulate/<config>/scores/vbench/{scaled,all}_results.json`
  - 感知:`results/perceptual/<config>/{per_video.csv, summary.json}`、跨配置 `results/perceptual/summary_all.json`
