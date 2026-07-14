#!/bin/bash
set -euo pipefail

# === 必需环境(与 launch_calc.sh 一致) ===
export HOME=/mnt/workspace/hkl
export VBENCH_CACHE_DIR=/mnt/workspace/hkl/.cache/vbench
EVAL_PY=/mnt/workspace/hkl/miniconda3/envs/eval/bin/python

# 汇总 VBench 评估结果为最终评分
#
# Usage: bash scripts/run_tabulate.sh [score_dir]
#   score_dir: 包含 vbench/ 子目录的路径（即 launch_calc.sh 的 CKPT_DIR）
#              默认: results/vbench-simulate/N5O1F1A0.8/scores

IMAGE_DIR="/mnt/cpfs/hkl/TaylorSeers/TaylorSeers-HunyuanVideo/results/vbench-simulate/N6O1F1A0.8"
SCORE_DIR="$IMAGE_DIR/scores/vbench"

${EVAL_PY} eval/vbench/tabulate_vbench_scores.py \
    --score_dir "$SCORE_DIR"
