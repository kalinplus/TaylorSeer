#!/bin/bash
# Benchmark single-forward Transformer FLOPs + wall-clock time per TaylorSeer config.
#
# Runs ONE prompt (video discarded) through the real denoising loop with the in-pipeline
# profiler (TAYLOR_BENCHMARK=1), for: original (1x baseline) + N3/N5/N6 (O=1,F=3,alpha=0.8).
# Specs match the VBench eval (480x640, 65 frames, 50 steps, single GPU, NO cpu-offload so
# timing reflects real GPU compute). Then aggregates FLOPs/time + speedup vs original.
#
# Output: results/benchmark/<config>.json (+ .log), results/benchmark/summary.{json,md}

set -euo pipefail

export HOME=/mnt/workspace/hkl
export MODEL_BASE=/mnt/cpfs/hkl/models/tencent/HunyuanVideo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INFER_PY=/mnt/workspace/hkl/miniconda3/envs/infer/bin/python

REPO=/mnt/cpfs/hkl/TaylorSeers/TaylorSeers-HunyuanVideo
cd "$REPO"

DIT_WEIGHT=/mnt/cpfs/hkl/models/tencent/HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt
OUT_DIR="$REPO/results/benchmark"
mkdir -p "$OUT_DIR"

PROMPT="A cat walks on the grass, realistic style."
GPU=0

# configs: original | "N O F alpha"
CONFIGS=(
    "original"
    "3 1 3 0.8"
    "5 1 3 0.8"
    "6 1 3 0.8"
)

echo "===== benchmark_flops_time start (GPU=${GPU}) ====="

for cfg in "${CONFIGS[@]}"; do
    if [ "$cfg" = "original" ]; then
        export TAYLOR_MODE=original
        Config_Tag="original"
        echo "============================================"
        echo "Config: ORIGINAL (no acceleration, full every step)  [1x baseline]"
        echo "============================================"
    else
        export TAYLOR_MODE=Taylor
        read -r N O F alpha <<< "$cfg"
        Config_Tag="N${N}O${O}F${F}A${alpha}"
        if [ "$alpha" = "0" ]; then
            TAYLOR_USE_SMOOTHING="False"
        else
            TAYLOR_USE_SMOOTHING="True"
        fi
        export TAYLOR_FRESH_THRESHOLD="$N"
        export TAYLOR_MAX_ORDER="$O"
        export TAYLOR_FIRST_ENHANCE="$F"
        export TAYLOR_USE_SMOOTHING="$TAYLOR_USE_SMOOTHING"
        export TAYLOR_USE_HYBRID_SMOOTHING="False"
        export TAYLOR_SMOOTHING_METHOD="exponential"
        export TAYLOR_SMOOTHING_ALPHA="$alpha"
        echo "============================================"
        echo "Config: ${Config_Tag} (N=${N} O=${O} F=${F} alpha=${alpha} smoothing=${TAYLOR_USE_SMOOTHING})"
        echo "============================================"
    fi

    OUT_JSON="$OUT_DIR/${Config_Tag}.json"
    OUT_LOG="$OUT_DIR/${Config_Tag}.log"

    CUDA_VISIBLE_DEVICES=${GPU} \
    TAYLOR_BENCHMARK=1 \
    TAYLOR_BENCHMARK_TAG="$Config_Tag" \
    TAYLOR_BENCHMARK_OUT="$OUT_JSON" \
    ${INFER_PY} sample_video.py \
        --model-base "$MODEL_BASE" \
        --dit-weight "$DIT_WEIGHT" \
        --video-size 480 640 \
        --video-length 65 \
        --infer-steps 50 \
        --prompt "$PROMPT" \
        --seed 42 \
        --flow-reverse \
        --save-path /tmp/hyv_bench 2>&1 | tee "$OUT_LOG"

    echo "[done] ${Config_Tag} -> ${OUT_JSON}"
done

unset TAYLOR_MODE TAYLOR_FRESH_THRESHOLD TAYLOR_MAX_ORDER TAYLOR_FIRST_ENHANCE \
      TAYLOR_USE_SMOOTHING TAYLOR_USE_HYBRID_SMOOTHING TAYLOR_SMOOTHING_METHOD \
      TAYLOR_SMOOTHING_ALPHA TAYLOR_BENCHMARK TAYLOR_BENCHMARK_TAG TAYLOR_BENCHMARK_OUT

# ---------- aggregate ----------
# FLOPs are deterministic; the measured full-step wall-clock is dominated by the
# mandatory smoothing-history CPU offload (memory fit), so the aggregator reports a
# compute-view time (full step = baseline forward, cached step = measured) for the
# speedup, plus the raw measured times for transparency. See eval/benchmark/aggregate.py.
${INFER_PY} eval/benchmark/aggregate.py "$OUT_DIR"

echo "===== benchmark_flops_time ALL DONE ====="
