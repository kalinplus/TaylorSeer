#!/usr/bin/env python3
"""全参考感知指标测量:PSNR / SSIM / LPIPS。

以无加速基线(original)视频为参考,对加速版同 prompt/同 seed 的视频逐帧比对。
- 每视频:65 帧各指标取均值。
- 每配置:对所有视频统计 mean/median/std/p10/p90 + "近无损"占比。

用法见 run_perceptual.sh。依赖仅 eval env 现有:lpips / skimage / (decord|imageio) / torch。
"""

import argparse
import csv
import glob
import json
import os
import statistics
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 视频读帧:decord 优先(快、帧准确),失败回退 imageio
# ---------------------------------------------------------------------------


def read_frames(path):
    """读取全部帧,返回 uint8 ndarray [N, H, W, 3](RGB)。"""
    # decord
    try:
        import decord

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path, num_threads=1)
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()  # [N,H,W,3] uint8 RGB
        if frames.ndim == 4 and frames.shape[-1] == 3:
            return np.ascontiguousarray(frames)
    except Exception as e:  # noqa: BLE001
        pass  # 回退
    # imageio
    import imageio.v2 as imageio

    reader = imageio.get_reader(path, "ffmpeg")
    frames = np.stack([np.asarray(f) for f in reader])
    reader.close()
    if frames.ndim == 3:  # 单帧
        frames = frames[None]
    return np.ascontiguousarray(frames)


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # noqa: E402


def _ssim(a, b):
    # skimage 版本兼容:新 channel_axis,旧 multichannel
    try:
        return float(
            structural_similarity(a, b, channel_axis=-1, data_range=255)
        )
    except TypeError:
        return float(structural_similarity(a, b, multichannel=True, data_range=255))


def frame_psnr_ssim(ref, tgt):
    """ref/tgt: [N,H,W,3] uint8。返回 (mean_psnr, mean_ssim)。"""
    psnrs, ssims = [], []
    for i in range(ref.shape[0]):
        psnrs.append(peak_signal_noise_ratio(ref[i], tgt[i], data_range=255))
        ssims.append(_ssim(ref[i], tgt[i]))
    return float(np.mean(psnrs)), float(np.mean(ssims))


def make_lpips(net="alex", device="cuda"):
    import lpips

    model = lpips.LPIPS(net=net, verbose=False).to(device)
    model.eval()
    return model


@torch.no_grad()
def frame_lpips(model, ref, tgt, device="cuda", batch=16):
    """ref/tgt: [N,H,W,3] uint8 → 逐批 LPIPS,返回均值。"""
    n = ref.shape[0]
    vals = []
    for s in range(0, n, batch):
        a = torch.from_numpy(ref[s : s + batch].copy()).float().div(127.5).sub(1.0)
        b = torch.from_numpy(tgt[s : s + batch].copy()).float().div(127.5).sub(1.0)
        a = a.permute(0, 3, 1, 2).to(device)
        b = b.permute(0, 3, 1, 2).to(device)
        d = model(a, b).flatten()
        vals.extend(d.detach().cpu().tolist())
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# 测量主流程
# ---------------------------------------------------------------------------


def list_mp4(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(".mp4"))


def measure_pairs(ref_dir, tgt_dir, out_dir, shard_idx, num_shards, device,
                  backbone, limit, self_test):
    ref_set = set(list_mp4(ref_dir))
    tgt_set = set(list_mp4(tgt_dir))
    common_set = ref_set & tgt_set
    common = sorted(common_set)
    if self_test:
        # ref vs ref:取 ref 自身做配对,验证管线
        pairs = [(f, f) for f in common]
        tag = "selftest"
    else:
        missing = (ref_set | tgt_set) - common_set
        if missing:
            print(f"[warn] {len(missing)} unpaired files ignored, e.g. {sorted(missing)[:2]}")
        pairs = [(f, f) for f in common]
        tag = f"shard_{shard_idx}"

    # 切片
    pairs = pairs[shard_idx::num_shards]
    if limit:
        pairs = pairs[:limit]

    os.makedirs(os.path.join(out_dir, "shards"), exist_ok=True)
    out_file = os.path.join(out_dir, "shards", f"{tag}.json")

    lpips_model = make_lpips(net=backbone, device=device) if not self_test else make_lpips(net=backbone, device=device)

    rows = []
    t0 = time.time()
    for i, (fn, _) in enumerate(pairs):
        ref = read_frames(os.path.join(ref_dir, fn))
        if self_test:
            tgt = ref
        else:
            tgt = read_frames(os.path.join(tgt_dir, fn))
        nf = min(ref.shape[0], tgt.shape[0])
        if nf == 0:
            print(f"[skip] {fn}: 0 frames")
            continue
        ref, tgt = ref[:nf], tgt[:nf]
        psnr, ssim = frame_psnr_ssim(ref, tgt)
        lp = frame_lpips(lpips_model, ref, tgt, device=device)
        rows.append({"filename": fn, "psnr": psnr, "ssim": ssim, "lpips": lp})
        if (i + 1) % 20 == 0 or (i + 1) == len(pairs):
            dt = time.time() - t0
            print(f"  [{tag}] {i+1}/{len(pairs)}  last psnr={psnr:.2f} ssim={ssim:.4f} lpips={lp:.4f}  ({dt:.0f}s)")

    with open(out_file, "w") as f:
        json.dump(rows, f)
    print(f"[done] {tag}: {len(rows)} videos -> {out_file}")

    if self_test:
        r = rows[0]
        assert r["ssim"] > 0.999, f"self-test SSIM too low: {r['ssim']}"
        assert r["lpips"] < 0.01, f"self-test LPIPS too high: {r['lpips']}"
        assert r["psnr"] > 100 or np.isinf(r["psnr"]), f"self-test PSNR too low: {r['psnr']}"
        print(f"[self-test PASS] ssim={r['ssim']:.6f} lpips={r['lpips']:.6f} psnr={r['psnr']}")


# ---------------------------------------------------------------------------
# 合并 + 统计
# ---------------------------------------------------------------------------

THRESH = {  # "近无损"占比阈值
    "psnr_gt30": lambda r: r["psnr"] > 30,
    "ssim_gt095": lambda r: r["ssim"] > 0.95,
    "lpips_lt01": lambda r: r["lpips"] < 0.1,
}


def stat_rows(rows):
    out = {}
    for m in ("psnr", "ssim", "lpips"):
        # PSNR 对完全相同帧为 inf,统计时封顶 100 避免炸均值(仅影响 self-test/极端)
        vals = [min(r[m], 100.0) if m == "psnr" else r[m] for r in rows]
        arr = np.asarray(vals, dtype=float)
        out[m] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        }
    out["near_lossless_frac"] = {k: float(sum(1 for r in rows if fn(r)) / len(rows))
                                 for k, fn in THRESH.items()}
    out["num_videos"] = len(rows)
    return out


def merge_dir(out_dir):
    shard_files = sorted(glob.glob(os.path.join(out_dir, "shards", "*.json")))
    rows = []
    for sf in shard_files:
        # 跳过 selftest 的 shard(不参与正式统计)
        if os.path.basename(sf).startswith("selftest"):
            continue
        with open(sf) as f:
            rows.extend(json.load(f))
    if not rows:
        print(f"[merge] {out_dir}: no rows")
        return None
    # 写 per_video.csv
    with open(os.path.join(out_dir, "per_video.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "psnr", "ssim", "lpips"])
        for r in rows:
            w.writerow([r["filename"], f"{r['psnr']:.4f}", f"{r['ssim']:.6f}", f"{r['lpips']:.6f}"])
    summary = stat_rows(rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[merge] {out_dir}: {len(rows)} videos -> per_video.csv, summary.json")
    print(json.dumps(summary, indent=2))
    return summary


def merge_cross(top_dir, configs):
    table = {}
    for c in configs:
        sj = os.path.join(top_dir, c, "summary.json")
        if os.path.exists(sj):
            with open(sj) as f:
                table[c] = json.load(f)
        else:
            print(f"[cross] missing {sj}")
    out = os.path.join(top_dir, "summary_all.json")
    with open(out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"[cross] -> {out}")
    # 简洁表
    print(f"\n{'config':<14}{'PSNR':>8}{'SSIM':>9}{'LPIPS':>8}  {'SSIM>0.95':>9}{'LPIPS<0.1':>10}")
    for c, s in table.items():
        print(f"{c:<14}{s['psnr']['mean']:>8.2f}{s['ssim']['mean']:>9.4f}{s['lpips']['mean']:>8.4f}"
              f"  {s['near_lossless_frac']['ssim_gt095']*100:>8.1f}%{s['near_lossless_frac']['lpips_lt01']*100:>9.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default="results/vbench-simulate/original")
    ap.add_argument("--tgt-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--backbone", default="alex", choices=["alex", "vgg"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--cross", action="store_true", help="合并多配置 summary.json → summary_all.json")
    ap.add_argument("--cross-top", default="results/perceptual")
    ap.add_argument("--cross-configs", nargs="+", default=["N3O1F3A0.8", "N5O1F3A0.8", "N6O1F3A0.8"])
    args = ap.parse_args()

    if args.merge:
        if args.cross:
            merge_cross(args.cross_top, args.cross_configs)
        else:
            assert args.out_dir, "--out-dir required for --merge"
            merge_dir(args.out_dir)
        return

    assert args.tgt_dir and args.out_dir, "--tgt-dir and --out-dir required"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(args.gpu)
    print(f"[info] device={device} (cuda idx {args.gpu}), backbone={args.backbone}, "
          f"shard {args.shard_idx}/{args.num_shards}, limit={args.limit}, self_test={args.self_test}")
    measure_pairs(
        args.ref_dir, args.tgt_dir, args.out_dir,
        args.shard_idx, args.num_shards, device,
        args.backbone, args.limit, args.self_test,
    )


if __name__ == "__main__":
    main()
