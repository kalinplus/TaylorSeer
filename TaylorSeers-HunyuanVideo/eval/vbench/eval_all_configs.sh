#!/bin/bash
# 对多个已完成生成的配置,依次跑 VBench 16 维度打分 + 总分汇总。
# 每个 config: 8 卡并行切 16 维度(launch_calc 逻辑) → wait → tabulate。
set -uo pipefail

export HOME=/mnt/workspace/hkl
export VBENCH_CACHE_DIR=/mnt/workspace/hkl/.cache/vbench
EVAL_PY=/mnt/workspace/hkl/miniconda3/envs/eval/bin/python

REPO=/mnt/cpfs/hkl/TaylorSeers/TaylorSeers-HunyuanVideo
cd "$REPO"

# 已完成生成的配置(各 ~4720 段)
CONFIGS=(
    "original"
    "N3O1F3A0.8"
    "N5O1F3A0.8"
    "N6O1F3A0.8"
)

GPUS=(0 1 2 3 4 5 6 7)
START_INDEX_LIST=(0 2 6 7 8 9 10 13)
END_INDEX_LIST=(2 6 7 8 9 10 13 16)
TASK_ID_LIST=(calc_vbench_a calc_vbench_b calc_vbench_c calc_vbench_d calc_vbench_e calc_vbench_f calc_vbench_g calc_vbench_h)

OVERALL_LOG="$REPO/results/vbench_eval_all.log"
echo "===== VBench eval-all start =====" | tee "$OVERALL_LOG"

for cfg in "${CONFIGS[@]}"; do
    VIDEO_DIR="$REPO/results/vbench-simulate/${cfg}"
    CKPT_DIR="$VIDEO_DIR/scores"
    LOG_BASE="$CKPT_DIR"
    mkdir -p "$LOG_BASE"

    echo "" | tee -a "$OVERALL_LOG"
    echo "########## CONFIG: ${cfg} ##########" | tee -a "$OVERALL_LOG"
    echo "VIDEO_DIR=$VIDEO_DIR" | tee -a "$OVERALL_LOG"
    nvid=$(ls "$VIDEO_DIR"/*.mp4 2>/dev/null | wc -l)
    echo "video count: $nvid" | tee -a "$OVERALL_LOG"

    # 若该 config 已汇总过(scaled_results.json 存在)则跳过
    if [ -f "$CKPT_DIR/vbench/scaled_results.json" ]; then
        echo "[SKIP] ${cfg} already has scaled_results.json, skip." | tee -a "$OVERALL_LOG"
        continue
    fi

    # --- Step 2: 8 卡并行算 16 维度 ---
    for i in "${!GPUS[@]}"; do
        CUDA_VISIBLE_DEVICES=${GPUS[i]} ${EVAL_PY} eval/vbench/calc_vbench.py "$VIDEO_DIR" "$CKPT_DIR" \
            --start ${START_INDEX_LIST[i]} \
            --end ${END_INDEX_LIST[i]} > ${LOG_BASE}/${TASK_ID_LIST[i]}.log 2>&1 &
    done
    wait
    echo "[${cfg}] 16-dim calc finished." | tee -a "$OVERALL_LOG"

    # --- Step 3: 汇总(失败不中断后续 config;缺维度会 assert,届时该 config 无 scaled_results) ---
    if ${EVAL_PY} eval/vbench/tabulate_vbench_scores.py \
        --score_dir "$CKPT_DIR/vbench" >> "$OVERALL_LOG" 2>&1; then
        echo "[${cfg}] tabulate finished -> $CKPT_DIR/vbench/{all,scaled}_results.json" | tee -a "$OVERALL_LOG"
    else
        echo "[${cfg}] tabulate FAILED (likely a dimension errored). Check $CKPT_DIR/vbench." | tee -a "$OVERALL_LOG"
    fi
done

echo "" | tee -a "$OVERALL_LOG"
echo "===== VBench eval-all ALL DONE =====" | tee -a "$OVERALL_LOG"
