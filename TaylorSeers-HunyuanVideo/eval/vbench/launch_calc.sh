#!/bin/bash

# 根据生成视频，并行计算 VBench 的 16 个评估维度

# === 必需环境: vbench 按 $HOME/.cache/vbench 查找 11 个打分模型 ===
export HOME=/mnt/workspace/hkl
export VBENCH_CACHE_DIR=/mnt/workspace/hkl/.cache/vbench
# vbench 装在独立的 eval env(infer env 里没有 vbench)
EVAL_PY=/mnt/workspace/hkl/miniconda3/envs/eval/bin/python

VIDEO_DIR="/mnt/cpfs/hkl/TaylorSeers/TaylorSeers-HunyuanVideo/results/vbench-simulate/N6O1F1A0.8"
# CKPT_DIR is actually the save_path arg
CKPT_DIR="$VIDEO_DIR/scores"
LOG_BASE=$CKPT_DIR
mkdir -p $LOG_BASE
echo "Logging to $LOG_BASE"

GPUS=(0 1 2 3 4 5 6 7)
START_INDEX_LIST=(0 2 6 7 8 9 10 13)
END_INDEX_LIST=(2 6 7 8 9 10 13 16)
TASK_ID_LIST=(calc_vbench_a calc_vbench_b calc_vbench_c calc_vbench_d calc_vbench_e calc_vbench_f calc_vbench_g calc_vbench_h) # for log records only

for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[i]} ${EVAL_PY} eval/vbench/calc_vbench.py $VIDEO_DIR $CKPT_DIR \
        --start ${START_INDEX_LIST[i]} \
        --end ${END_INDEX_LIST[i]} > ${LOG_BASE}/${TASK_ID_LIST[i]}.log 2>&1 &
done

wait
echo "All 16 dimensions evaluated. Check logs in $LOG_BASE"
