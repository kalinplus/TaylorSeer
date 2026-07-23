#!/bin/bash
# 全参考感知指标(PSNR/SSIM/LPIPS)全量驱动:8 卡分 shard × 3 配置 + 合并。
# 参考=results/vbench-simulate/original;目标=N3/N5/N6-O1F3A0.8。
set -uo pipefail

export HOME=/mnt/workspace/hkl
export VBENCH_CACHE_DIR=/mnt/workspace/hkl/.cache/vbench
export PYTHONUNBUFFERED=1
EVAL_PY=/mnt/workspace/hkl/miniconda3/envs/eval/bin/python

REPO=/mnt/cpfs/hkl/TaylorSeers/TaylorSeers-HunyuanVideo
cd "$REPO"

REF="$REPO/results/vbench-simulate/original"
TOP="$REPO/results/perceptual"
CONFIGS=(N3O1F3A0.8 N5O1F3A0.8 N6O1F3A0.8)
GPUS=(0 1 2 3 4 5 6 7)
NUM_SHARDS=${#GPUS[@]}

LOG="$TOP/run.log"
mkdir -p "$TOP"
echo "===== perceptual eval-all start =====" | tee "$LOG"

for cfg in "${CONFIGS[@]}"; do
    TGT="$REPO/results/vbench-simulate/${cfg}"
    OUT="$TOP/${cfg}"
    mkdir -p "$OUT/shards"

    echo "" | tee -a "$LOG"
    echo "########## CONFIG: ${cfg} ##########" | tee -a "$LOG"

    if [ -f "$OUT/summary.json" ]; then
        echo "[SKIP] ${cfg} already has summary.json" | tee -a "$LOG"
        continue
    fi

    for i in "${!GPUS[@]}"; do
        CUDA_VISIBLE_DEVICES=${GPUS[i]} ${EVAL_PY} eval/perceptual/measure.py \
            --ref-dir "$REF" --tgt-dir "$TGT" --out-dir "$OUT" \
            --shard-idx $i --num-shards $NUM_SHARDS --gpu 0 \
            > "$OUT/shards/shard_${i}.log" 2>&1 &
    done
    wait
    echo "[${cfg}] all shards finished." | tee -a "$LOG"

    ${EVAL_PY} eval/perceptual/measure.py --merge --out-dir "$OUT" >> "$LOG" 2>&1
    echo "[${cfg}] merged -> $OUT/summary.json" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
${EVAL_PY} eval/perceptual/measure.py --merge --cross --cross-top "$TOP" >> "$LOG" 2>&1

echo "" | tee -a "$LOG"
echo "===== perceptual eval-all ALL DONE =====" | tee -a "$LOG"
