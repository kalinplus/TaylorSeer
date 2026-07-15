# TaylorSeer-HunyuanVideo 显存优化记录

> 记录在 VBench 评测（480×640 / 65 帧 / 50 步，单卡 80GB A100，`--use-cpu-offload`）下排查并解决 OOM 的全过程与最终落地的两项核心优化。
> 相关背景算法见 [TaylorSeer_Smooth_Algorithm.md](./TaylorSeer_Smooth_Algorithm.md)。

## 1. 问题现象

- 开启 `--use-cpu-offload` 后仍然 OOM，且**与 smoothing 开关无关**：`TAYLOR_USE_SMOOTHING=True` / `False` 都会爆。
- 失败位置最初在 DiT 去噪步：`hyvideo/modules/models.py:517`（single-stream block 的 `linear2(cat(...))`）。
- 80GB 卡上单进程（脚本每卡一个独立进程，按 prompt 做数据并行）即爆，说明是真实的单卡显存压力。

## 2. 根因分析（靠 `[MEM]` 探针定位，而非猜测）

在 pipeline 去噪循环里埋了 `torch.cuda.memory_allocated/reserved/max_memory_allocated` 打印（见第 5 节），单 prompt 跑一遍后从日志读到：

| 时机 | alloc | peak |
|---|---|---|
| `after model load` | **38.55 GiB** | — |
| `denoise-start` | 38.56 GiB | — |
| `step 0 post-fwd` | 52.75 GiB | 55.82 GiB |
| `step 1 post-fwd` | 66.93 GiB | 70.12 GiB |
| `step 2+ post-fwd`（稳定） | 66.93 GiB | 70.24 GiB |

关键结论：

1. **DiT 阶段其实能放下**（稳定峰值 70.24 GiB < 79.25 GiB）。
2. **`after model load` 就有 38.55 GiB**：即使开了 `enable_sequential_cpu_offload()`，权重并没有被真正搬到 CPU——offload 对本 pipeline 基本失效（见第 7 节遗留问题）。
3. **Taylor 特征缓存是大头**：`cache_dic['cache']` 给每个 stream/layer/module 都存了 full-size 的激活张量（feature + 各阶导数）。`cache[-2]`（smoothing 用的历史副本）开启 smoothing 时会再存一整份，把常驻显存从 ~15GB 翻到 ~30GB。

> 结构参考 `hyvideo/modules/cache_functions/cache_init.py`：`for hist in [-1, -2]:` 给 20 个 double-stream + 40 个 single-stream 层各建了因子存储。

## 3. 优化一：smoothing 历史 `cache[-2]` 顺序 offload 到 CPU

**原理**：`cache[-2]` 只在 full-compute step 的 `derivative_approximation_*_smoothing` 里被读取（用来做有限差分）。而 transformer 的 forward 是**按层循环**的，任一时刻只有一层的 smoothing 在算。所以没必要让所有层的历史都常驻 GPU——可以整体放 CPU，只在算到某一层时把**那一层**的一个张量临时搬上来，算完即释放。

这和 `enable_sequential_cpu_offload` 对权重的处理是同一个套路：一次只搬一层的量，所以**峰值不是「全部历史都在 GPU」**，而是「一层历史（~几百 MB）短暂在 GPU」。

**数值完全不变**（可用于和 baseline 严格对比）：
- `shift_cache_history` 把历史**深拷贝**到 CPU（值相同，只是存储位置不同）；
- 读取时 `f_prev2.to(feature.device)`——若已在目标设备则是 no-op。

**改动**（`hyvideo/modules/taylor_utils/__init__.py`）：
- `shift_cache_history`：开启 offload 时改为 `{k: v.cpu() for k,v in prev.items()}`，否则保留旧的 GPU 引用行为。
- `derivative_approximation_with_smoothing` / `derivative_approximation_hybrid_smoothing`：读取 `cache[-2]` 后 `.to(feature.device)` 临时取回。

**开关**（`hyvideo/modules/cache_functions/cache_init.py`）：
- 新增环境变量 `TAYLOR_SMOOTHING_OFFLOAD_HISTORY`，默认 `True`。
- 设为 `False` 可精确复现「历史在 GPU」的旧行为，用于 A/B 验证输出逐位一致。

**效果**：把 smoothing 路径下常驻的 ~15GB 第二份缓存挪到 CPU，DiT 阶段从「step 1 OOM」变为「稳定 66.9GB alloc / 70.2GB peak，跑通」。

## 4. 优化二：去噪循环结束后、VAE 解码前释放特征缓存

优化一让 DiT 跑通后，OOM **转移到了 VAE 解码**：

```
hyvideo/vae/unet_causal_3d_blocks.py:73  F.pad(...) → Tried to allocate 4.25 GiB ... 2.07 GiB free
```

原因：去噪循环结束后，`cache_dic['cache']`（~15–25GB）仍然常驻 GPU，但 **VAE 解码根本不读它**——它是纯死内存。VAE 想再申请就爆了。

**做法**（`hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py`，去噪循环之后、`if not output_type == "latent":` 之前）：

```python
cache_dic['cache'] = None   # 丢弃几十 GiB 的因子张量；保留 cache_dic 本体与标量标志
torch.cuda.empty_cache()    # 把释放的显存还给 driver
```

- **没有用 `del cache_dic`**：因为后面第 1153 行的 FLOPs 汇报还会读 `cache_dic['test_FLOPs']`，只清掉占显存的 `['cache']` 即可。
- 相比「把 `cache[-1]` 也按层流式 offload」，这里**零速度代价、零数学改动**——缓存本来在去噪后就死了，直接释放即可。（流式 `cache[-1]` 会在每个 cached step、每层都触发 H2D，抵消 Taylor 的加速，仅在需要进一步压低 DiT 阶段显存、例如上更高分辨率时才值得做。）

**效果**：`[MEM] after cache freed` 显示显存从 ~67GB 掉到权重基线（~38–50GB），VAE 解码顺利通过，整条 pipeline 跑通。

## 5. 显存诊断探针（临时，建议定位后清理）

为定位加的打印，标记为 `[MEM-DBG]` / `[MEM]`，均通过 `os.write(2, ...)` 写到 OS 级 stderr，**绕过 `sample_video_vbench.py` 里 `suppress_output()` 对 stdout/stderr 的重定向**，确保能落到 `results/HunyuanVideo_vbench_logs/device_*_*.log`：

- `sample_video_vbench.py`：`[MEM] after model load`（模型加载后，看权重是否被 offload）。
- `pipeline_hunyuan_video.py`：
  - `[MEM] denoise-start`（去噪前基线，含 `reset_peak_memory_stats`）
  - `[MEM] step i pre-fwd` / `[MEM] step i post-fwd ... peak=...`（每步前后 + 高水位）
  - `[MEM] after cache freed`（优化二释放后）

> 这些是诊断用的，问题彻底定位、稳定后可按 `grep -rn "MEM-DBG\|\[MEM\]"` 删除。

## 6. 修改文件清单

| 文件 | 改动 |
|---|---|
| `hyvideo/modules/cache_functions/cache_init.py` | 新增 `TAYLOR_SMOOTHING_OFFLOAD_HISTORY` 开关并写入 `cache_dic`；打印里输出该值 |
| `hyvideo/modules/taylor_utils/__init__.py` | `shift_cache_history` 历史深拷贝到 CPU；两个 smoothing 导数函数读取历史时 `.to(device)` |
| `hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py` | `import os`；`[MEM]` 探针；去噪后 `cache_dic['cache']=None` + `empty_cache()` |
| `sample_video_vbench.py` | `import torch`；`[MEM] after model load` 探针 |

> `eval/sample_vbench.sh` 的改动（新增 `--use-cpu-offload`、`FIRST_ENHANCE` 1→3）由使用者维护，不在本次代码优化范围内。

## 7. 效果对比

| 阶段 | 之前 | 之后 |
|---|---|---|
| DiT 去噪（smoothing 开） | step 1 即 OOM（`models.py:517`） | 稳定 66.9GB alloc / 70.2GB peak，跑通 |
| VAE 解码 | 循环跑完后 OOM（`F.pad` 4.25GB） | 缓存释放后显存回到 ~权重基线，跑通 |

## 8. 遗留问题（不阻塞，供后续处理）

- **`enable_sequential_cpu_offload()` 实际未把权重搬到 CPU**：`after model load` 仍有 38.55 GiB。当前靠「释放缓存」腾出空间已够用；若后续要上更高分辨率/更长视频，需要真正让 offload 生效（或用 fp8 压权重，但 fp8 会掉点、破坏与 baseline 的可比性，需谨慎）。
- `shift_cache_history` 原先以**引用**赋值 `cache[-2] = cache[-1]`，导致 smoothing 读取时 `f_prev2` 与 `f_prev1` 指向同一份（历史退化为 `[F_{n-1}, F_{n-1}, F_n]`）。本次 offload 改为深拷贝后**数值与旧行为逐位一致**（因为拷贝的是同一份 D_{n-1}），未改变该行为；若希望得到真正的「两步前」历史 `F_{n-2}`，需另行调整历史管理逻辑（注意这会改变输出，影响与现有 baseline 的对比）。
