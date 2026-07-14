# V-Bench 正式运行手册 (Runbook)

> 仓库:`/mnt/cpfs/hkl/TaylorSeers/TaylorSeers-HunyuanVideo` ｜ 8× A100 80GB
> 前置环境(权重 / text_encoder / infer+eval 两个 conda env / vbench+11 打分模型)**均已就绪**。
> 踩坑与细节见 `vbench_local_setup_notes.md`;本手册只讲怎么跑。

## 环境已内嵌进脚本(无需手动 export)

| 变量 | 值 | 所在脚本 | 作用 |
|---|---|---|---|
| `MODEL_BASE` | `/mnt/cpfs/hkl/models/tencent/HunyuanVideo` | `eval/sample_vbench.sh` | 生成端拼 VAE/text_encoder/tokenizer 路径 |
| `HOME` | `/mnt/workspace/hkl` | `eval/vbench/launch_calc.sh`、`run_tabulate.sh` | vbench 按 `$HOME/.cache/vbench` 找 11 个打分模型 |
| `VBENCH_CACHE_DIR` | `/mnt/workspace/hkl/.cache/vbench` | 同上 | 显式指定打分模型缓存(双保险) |

三个脚本都已 **pin 死对应 env 的 python**:`sample_vbench.sh` 用 `infer`,`launch_calc.sh`/`run_tabulate.sh` 用 `eval`。因此**在任意 conda env 下直接 `bash` 即可**,不用手动 activate。

---

## Step 1 — 生成视频(`infer` env)

```bash
cd /mnt/cpfs/hkl/TaylorSeers/TaylorSeers-HunyuanVideo
bash eval/sample_vbench.sh
```

运行前按需改 `eval/sample_vbench.sh` 顶部这几个参数:

| 参数 | 当前值 | 说明 |
|---|---|---|
| `Num_Devices` | `1` | 并行卡数;改 **8** 即 8 卡并行(946 prompt 自动均分)。须 ≤ `GPU_LIST` 长度 |
| `Num_Videos_per_Sample` | `5` | 每 prompt 生成几段;V-Bench 标准 **5** |
| `GPU_LIST` | `(0 1 2 3 4 5 6 7)` | 第 d 个任务用 `GPU_LIST[d]`;按空闲卡调整 |
| `configs` | 见下 | 要跑的配置列表,逐个遍历,每组存到各自目录 |

`configs` 数组(每组一条,逐个跑、各自存盘):
```bash
configs=(
    "original"          # 原版无加速基线 → results/vbench-simulate/original/
    "3  1  1  0.8"      # → .../N3O1F1A0.8/
    "5  1  1  0.8"      # → .../N5O1F1A0.8/
    "6  1  1  0.8"      # → .../N6O1F1A0.8/
)
```
- **`original`**:**无加速基线**(每个 denoising step 全量计算、不缓存),由 `TAYLOR_MODE=original` 触发(代码 `cache_init.py` 读取该 env,默认 `Taylor`)。感知类指标(subject/background consistency、temporal flicker 等)需要它作参考,**必跑**。
- 其余 `N O F alpha`:TaylorSeer 加速配置 —— `N`=新鲜阈值(≈加速倍率,越大越快)、`O`=Taylor 阶数、`F`=首增强步数、`α`=平滑系数(`α>0` 自动开指数平滑)。

固定生成参数(V-Bench 标准):`--video-size 480 640 --video-length 65 --infer-steps 50 --flow-reverse --use-cpu-offload`。

- **输出**:每组配置各自一个目录 `results/vbench-simulate/<config>/`(如 `original/`、`N6O1F1A0.8/`),命名 `<prompt>-<seed>.mp4`。
- **断点续跑**:已存在的 mp4 自动跳过,中断后重跑即可续上。
- **量级**:946 prompt × 5 = **4730 段**;480×640×65帧×50步 单卡约 3–5 分钟/段 → 强烈建议 8 卡。

---

## Step 2 — 16 维度打分(`eval` env)

```bash
cd /mnt/cpfs/hkl/TaylorSeers/TaylorSeers-HunyuanVideo
bash eval/vbench/launch_calc.sh
```

- 8 卡并行,16 维度切成 8 组(`START/END_INDEX_LIST`)同时算,脚本末尾 `wait` 等全部完成。
- `VIDEO_DIR` 默认指向 Step 1 输出 `results/vbench-simulate/N6O1F1A0.8`;分数写到 `VIDEO_DIR/scores`。
- **评测某组配置(含 `original` 基线)**:改 `VIDEO_DIR` 指向对应目录(如 `results/vbench-simulate/original`)。每组加速配置 + original 基线都要各自跑一遍 `launch_calc`,得到各自的 16 维度分;再用 Step 3 汇总,才能对比"加速版 vs 原版"的感知差异。
- 每维度日志:`VIDEO_DIR/scores/calc_vbench_{a..h}.log`。
- **换加速配置后**:同步改 `launch_calc.sh` 里的 `VIDEO_DIR` 目录名(如 `N3O1F1A0.8`)。

单维度调试(不经脚本):
```bash
export HOME=/mnt/workspace/hkl
/mnt/workspace/hkl/miniconda3/envs/eval/bin/python eval/vbench/calc_vbench.py \
    results/vbench-simulate/N6O1F1A0.8 results/vbench_scores --start 12 --end 13   # 例:只算 temporal_flickering
```
输出:`<save_path>/vbench/<dimension>_eval_results.json`。

---

## Step 3 — 汇总总分(`eval` env)

```bash
bash eval/vbench/run_tabulate.sh
```

读取所有 `*_eval_results.json`,按 VBench 预设 min/max 归一化,算:
- **Quality**(权重 4):subject/background consistency、temporal flickering、motion smoothness、dynamic degree、aesthetic、imaging quality
- **Semantic**(权重 1):object class、multiple objects、human action、color、spatial relationship、scene、appearance/temporal style、overall consistency
- **Total = (Quality×4 + Semantic×1) / 5**

输出:`all_results.json`(原始分)、`scaled_results.json`(归一化百分比)。

> ⚠️ `tabulate_vbench_scores.py` 会 **assert 全部 16 个维度都已算完**;缺任何一个都会报 `AssertionError: {...} not calculated yet`。必须等 Step 2 的 16 维度全部跑完再跑 Step 3。小规模验证时只算 1 个维度是正常的,届时 tabulate 会按设计拒绝运行(不算报错)。

---

## 易错点速查
- 打分前 `HOME` 必须是 `/mnt/workspace/hkl`(脚本已 export;手跑单维度记得自己 export)。见 setup 笔记**坑 10**。
- 生成用 **infer** env,打分用 **eval** env —— 脚本已 pin,别混。
- 想换加速配置,改 `sample_vbench.sh` 的 `configs`,并同步 `launch_calc.sh`/`run_tabulate.sh` 里的 `N6O1F1A0.8` 目录名。
- `Num_Devices` 必须 ≤ `GPU_LIST` 长度;否则越界。
