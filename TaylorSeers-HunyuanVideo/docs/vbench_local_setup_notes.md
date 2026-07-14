# V-Bench 本地运行环境踩坑与配置记录

> 机器:8× NVIDIA A100-SXM4-80GB ｜ 更新日期:2026-07-14
> 仓库:`/mnt/cpfs/hkl/TaylorSeers/TaylorSeers-HunyuanVideo`(注意目录名是 `TaylorSeers`,带 s)

本文档记录在本机把 TaylorSeer-HunyuanVideo 跑上 V-Bench 过程中踩到的坑与对应修复,供后续复跑时避坑。

---

## 一、关键路径

| 用途 | 路径 |
|---|---|
| 仓库根目录 | `/mnt/cpfs/hkl/TaylorSeers/TaylorSeers-HunyuanVideo` |
| HunyuanVideo 权重根 (MODEL_BASE) | `/mnt/cpfs/hkl/models/tencent/HunyuanVideo` |
| DiT 权重 | `$MODEL_BASE/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt` |
| VAE | `$MODEL_BASE/hunyuan-video-t2v-720p/vae/pytorch_model.pt` |
| CLIP (text_encoder_2) | `$MODEL_BASE/text_encoder_2/` |
| MLLM 原始权重 | `$MODEL_BASE/llava-llama-3-8b-v1_1-transformers/` |
| MLLM 预处理后 (text_encoder) | `$MODEL_BASE/text_encoder/` ← 需预处理生成 |
| VBench 打分模型缓存 | `/mnt/workspace/hkl/.cache/vbench` (= `$HOME/.cache/vbench`) |
| HOME | `/mnt/workspace/hkl` |

---

## 二、Conda 环境(miniconca 在 `/mnt/workspace/hkl/miniconda3`)

| env | 用途 | 关键包 |
|---|---|---|
| `infer` | **生成端**(sample_video_vbench.py) | torch 2.6.0+cu124, diffusers 0.36.0, transformers 4.57.6, flash-attn 2.8.3 |
| `eval` | **评测端**(vbench 打分) | torch 2.10.0+cu128, diffusers 0.37.0 + `vbench`(隔离安装,不污染 infer) |

显式调用:`/mnt/workspace/hkl/miniconda3/envs/<env>/bin/python`

> 设计原则:**vbench 装到独立的 `eval` env**,避免它拖来的重依赖(pyiqa / detectron2 / grit …)污染生成用的 `infer` env。生成与评测用不同 env。
>
> 补充(实际安装过程):`pip install vbench`(0.1.5,PyPI)装完后,还需 setuptools 降到 80.10.2、detectron2 0.6 从源码编译(详见坑 8);vbench 顺带把 transformers pin 到 4.33.2。`ram` / `mmcv` 实测不需要。

---

## 三、踩坑记录(按严重性排序)

### 坑 1(高危):`MODEL_BASE` 环境变量 ≠ `--model-base` CLI 参数
- **现象**:直接跑 `python sample_video_vbench.py --model-base <真路径> ...`,VAE/text_encoder/tokenizer 找的是不存在的 `./ckpts/...`,报路径不存在。
- **根因**:`hyvideo/constants.py:67` 在 **import 时**用 `MODEL_BASE = os.getenv("MODEL_BASE", "./ckpts")` 拼 VAE / TEXT_ENCODER_PATH / TOKENIZER_PATH。而 `--model-base` 这个 CLI 参数**只喂给 DiT 权重加载器**,根本不覆盖上面这些派生路径。
- **修复**:运行前 `export MODEL_BASE=/mnt/cpfs/hkl/models/tencent/HunyuanVideo`。
- **注意**:`eval/sample_vbench.sh` 里已经有 `export MODEL_BASE=...`(已改对),所以**用脚本跑正式生成时不会踩**;只有裸调 `sample_video_vbench.py` 时必须自己 export。

### 坑 2(高危):eval/ 下脚本写死了旧机器路径
- **现象**:`eval/sample_vbench.sh`、`eval/vbench/launch_calc.sh`、`eval/vbench/run_tabulate.sh` 里写死 `/mnt/data0/tencent/HunyuanVideo`、`/home/hkl/TaylorSeer/TaylorSeer-HunyuanVideo`,在本机均不存在。
- **修复**:已统一改为本机路径(`/mnt/cpfs/hkl/models/tencent/HunyuanVideo`、`/mnt/cpfs/hkl/TaylorSeers/TaylorSeers-HunyuanVideo`),并把 eval 脚本里的配置目录 `N3O1F1A0.8` 对齐到当前激活的生成配置 `N6O1F1A0.8`。
- **遗留**:`run_tabulate.sh` 第 8 行有一处文档性注释仍写 `N5O1F1A0.8`,不影响运行,未改。
- **复跑前自查**:`grep -rn '/mnt/data0\|/home/hkl' eval/` 应为 0 匹配。

### 坑 3(中危):`text_encoder/` 缺失,需预处理
- **现象**:权重目录只有原始的 `llava-llama-3-8b-v1_1-transformers/`,但代码要的是提取过的 `text_encoder/`(只含 LM 部分,省显存)。
- **修复**:跑一次预处理(见第四节命令),产出 ~15GB 的 4 个 safetensors 分片 + config/tokenizer。

### 坑 4(中危):`infer` env 缺 `imageio` / `imageio-ffmpeg`
- **现象**:生成时 `ModuleNotFoundError: No module named 'imageio'`(保存 mp4 用)。
- **修复**:已按 requirements.txt 版本补装 `imageio==2.34.0`、`imageio-ffmpeg==0.5.1`。

### 坑 5(低危,仅观察):依赖版本漂移
- `infer` env 实际版本比 requirements.txt 新不少(diffusers 0.36 vs 0.31、transformers 4.57 vs 4.46、torch 2.6 vs 2.4、numpy 2.2 vs 1.24)。生成端验证可正常跑,无功能性问题。若日后出诡异 bug,优先怀疑版本差异。

### 坑 6(低危):GPU 0 被其他进程占用
- 冒烟测试时 GPU 0 被别的进程占了 ~41GB。多卡跑 V-Bench 生成时建议从 GPU 1 起、或先 `nvidia-smi` 确认空闲卡再分配 `CUDA_VISIBLE_DEVICES`。

### 坑 7(低危):输出文件名含空格
- `sample_video_vbench.py` 用 `{prompt_text}-{seed_offset}.mp4` 命名,prompt 文本含空格/逗号。下游 shell 命令处理这些 mp4 时务必加引号。

### 坑 8(中危,评测端):eval env 装 vbench 需额外修依赖
- 直接 `pip install vbench`(PyPI,版本 0.1.5)装得上,但有几个重依赖要单独处理:
  - `pyiqa` 因 env 里 **setuptools 82 删掉了 `pkg_resources`** 而坏 → setuptools 降到 **80.10.2**。
  - `detectron2`(GRiT 需要,负责 object_class / multiple_objects / color / spatial_relationship 四个维度)PyPI 上没有 → 从源码编译:`pip install --no-build-isolation "detectron2 @ git+https://ghfast.top/https://github.com/facebookresearch/detectron2.git"`(必须关 build isolation,否则 setup.py 看不见已装的 torch 2.10)。
  - vbench 顺带把 **transformers pin 到 4.33.2**(仅 eval env 内,不影响 infer)。
- **不需要的依赖**:`recognize_anything`/`ram`(scene 用 vbench 自带的 Tag2Text)、`mmcv`/`mmdet`(grit_src 用 detectron2 + 自带 centernet2)——不用纠结这些的版本冲突。

### 坑 9(中危):`scripts/download_vbench_models.sh` 原有 3 个 bug(已修)
原脚本第一次跑直接 exit 1、一个模型都没下成。已修:
1. `download()` / `clone_repo()` 往子目录(`clip_model/`、`dino_model/` …)里写,但只 `mkdir` 了顶层 `$CACHE_DIR` → 每个 `wget -O …/foo.tmp` 都 "No such file or directory"。已加 `mkdir -p "$(dirname "$dest")"`。
2. `set -euo pipefail` + 失败路径 `return 1` → 第一个模型失败就整体退出、永远跑不到末尾自检。已把失败路径改 `return 0`(缺失仍由末尾自检报出)。
3. LAION aesthetic 的 hf-mirror URL 返回 **404** → 换成 github mirror `${GITHUB_MIRROR}/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth`(200)。

### 坑 10(中危,打分时必看):vbench 模型路径依赖 `$HOME`
- `vbench/utils.py` 里 `CACHE_DIR = $HOME/.cache/vbench`(可用环境变量 `VBENCH_CACHE_DIR` 覆盖)。本机 `HOME=/mnt/workspace/hkl`,正好对得上下载目录。
- **打分(`calc_vbench.py`)时 `HOME` 必须是 `/mnt/workspace/hkl`**。若在别的 `$HOME` 下跑,先 `export HOME=/mnt/workspace/hkl`,或 `export VBENCH_CACHE_DIR=/mnt/workspace/hkl/.cache/vbench`,否则 11 个模型全找不到。

### 坑 11(低危):ViCLIP bpe vocab 需预取
- ViCLIP 运行时从 `raw.githubusercontent.com` 拉 `bpe_simple_vocab_16e6.txt.gz`,该域名在本环境 TLS 证书坏 → 已预取进缓存(`$CACHE/ViCLIP/bpe_simple_vocab_16e6.txt.gz`)。若 `temporal_style` / `overall_consistency` 报 vocab 相关错误,先查这个文件在不在。

---

## 四、已验证可用的命令

### 4.1 预处理 text_encoder(一次性)
```bash
cd /mnt/cpfs/hkl/TaylorSeers/TaylorSeers-HunyuanVideo
/mnt/workspace/hkl/miniconda3/envs/infer/bin/python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py \
    --input_dir  /mnt/cpfs/hkl/models/tencent/HunyuanVideo/llava-llama-3-8b-v1_1-transformers \
    --output_dir /mnt/cpfs/hkl/models/tencent/HunyuanVideo/text_encoder
```

### 4.2 生成单条视频(直接 CLI,冒烟测试用)
```bash
cd /mnt/cpfs/hkl/TaylorSeers/TaylorSeers-HunyuanVideo
export MODEL_BASE=/mnt/cpfs/hkl/models/tencent/HunyuanVideo   # 必须!见坑1
CUDA_VISIBLE_DEVICES=1 /mnt/workspace/hkl/miniconda3/envs/infer/bin/python sample_video_vbench.py \
    --model-base  $MODEL_BASE \
    --dit-weight  $MODEL_BASE/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --vbench-json-path ./eval/VBench_full_info.json \
    --index-start 0 --index-end 0 \
    --seed 42 --num-videos-per-prompt 1 \
    --video-size 480 640 --video-length 65 \
    --infer-steps 50 --flow-reverse --use-cpu-offload \
    --save-path ./results/vbench
```
> 实测 480×640 / 65帧 / 20步 + cpu-offload ≈ 124 秒/条(A100)。正式 V-Bench 标准是 50 步、每条 prompt 5 个视频。

### 4.3 正式生成(多卡切分,推荐用脚本)
```bash
cd /mnt/cpfs/hkl/TaylorSeers/TaylorSeers-HunyuanVideo
./eval/sample_vbench.sh     # 脚本内已 export MODEL_BASE、改好路径
```

### 4.4 V-Bench 打分(Step 2-3,用 eval env)
```bash
export HOME=/mnt/workspace/hkl   # 必须!vbench 按 $HOME/.cache/vbench 找模型,见坑 10
# Step2: 16 维度打分
/mnt/workspace/hkl/miniconda3/envs/eval/bin/python eval/vbench/calc_vbench.py \
    ./results/vbench-simulate/N6O1F1A0.8 ./results/vbench_scores --start 0 --end -1
# Step3: 汇总总分
/mnt/workspace/hkl/miniconda3/envs/eval/bin/python eval/vbench/tabulate_vbench_scores.py \
    --score_dir ./results/vbench_scores/vbench
```

---

## 五、当前就绪状态

| 环节 | 状态 |
|---|---|
| 主权重 (DiT/VAE/CLIP/MLLM) | ✅ 已有 |
| text_encoder 预处理 | ✅ 已完成 |
| 生成端环境 (infer env) | ✅ 就绪(已补 imageio) |
| 生成冒烟测试 | ✅ 通过(有效 mp4) |
| 脚本路径修正 | ✅ 已完成 |
| vbench 库安装 (eval env) | ✅ vbench 0.1.5(+detectron2/pyiqa,见坑 8) |
| VBench 11 个打分模型下载 | ✅ 完成(8.5GB,自检通过) |
