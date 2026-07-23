"""Self-contained per-step FLOPs + wall-clock-time profiler for the HunyuanVideo DiT.

Activated by the pipeline when ``TAYLOR_BENCHMARK`` is set. It measures the *real*
transformer forward (not a calflops proxy) so the numbers reflect the actual compute,
including the cost reduction on Taylor-cached steps.

It counts two complementary contributions to a transformer forward:

1. **Linear FLOPs** — forward hooks on every ``nn.Linear`` under the transformer
   (QKV / output projections, FFN, embeddings, final layers). On cached steps the
   cached layers skip their Linears, so the hook total drops automatically.
   ``flops = 2 * in_features * out_features * num_output_positions``.

2. **Attention FLOPs** — wrapping the single attention entry point
   :func:`hyvideo.modules.attenion.attention` (and ``parallel_attention``) with the
   exact self-attention formula ``4 * heads * head_dim * sum(seg_len**2)`` (``QKᵀ`` +
   ``PV``). For varlen flash attention the segment lengths come from ``cu_seqlens_q``;
   the img/txt streams are separate segments in one cu_seqlens, so they are counted
   independently within a single call. Attention is only invoked when the layer
   actually computes it, so cached steps are excluded automatically.

3. **Time** — ``torch.cuda.Event`` around the transformer call (sync'd).

The attention counter accumulates GPU scalar tensors and reads them out once per step
(``.item()``) to avoid per-call syncs that would perturb the timing we measure.
"""

import torch
import torch.nn as nn


def _attention_pairs(q, k, cu_seqlens_q):
    """Total query-key token-pairs for one attention call.

    For varlen self-attention: ``sum(seg_len**2)`` over the segments in ``cu_seqlens_q``
    (img and txt are separate segments within a single call). Otherwise fall back to
    ``b * s_q * s_k`` from the q/k shapes.
    """
    if cu_seqlens_q is not None:
        seg = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).long()
        return seg.pow(2).sum()  # GPU scalar tensor (deferred sync)
    b = q.shape[0]
    s_q = q.shape[1]
    s_k = k.shape[1]
    return torch.tensor(float(b * s_q * s_k), device=q.device)


class FlopsTimeProfiler:
    """Attach to a transformer, then call :meth:`step_begin` / :meth:`step_end` per step."""

    def __init__(self, transformer):
        self.transformer = transformer
        self._handles = []
        self._linear_flops = 0          # python int, accumulated by Linear hooks
        self._attn_flops_t = None       # GPU scalar tensor, accumulated by attention wrap
        self._start_event = None
        self._end_event = None

        self._models_mod = None
        self._orig_attention = None
        self._orig_parallel_attention = None

        for m in transformer.modules():
            if isinstance(m, nn.Linear):
                self._handles.append(m.register_forward_hook(self._linear_hook))

    # ---------------- Linear hooks ----------------
    def _linear_hook(self, module, inputs, output):
        if output is None:
            return
        out_features = module.out_features
        # number of output positions = product of all dims except the last
        try:
            num_positions = output.shape[:-1].numel()
        except Exception:
            num_positions = output.numel() // out_features
        # MACs = in * out * positions; FLOPs = 2 * MACs
        self._linear_flops += 2 * module.in_features * out_features * num_positions

    # ---------------- Attention wrapping ----------------
    def attach_attention(self):
        """Monkeypatch ``attention``/``parallel_attention`` in the models module."""
        import hyvideo.modules.models as models_mod

        self._models_mod = models_mod
        self._orig_attention = models_mod.attention
        self._orig_parallel_attention = getattr(models_mod, "parallel_attention", None)

        profiler = self

        def wrapped_attention(q, k, v, *args, **kwargs):
            cu_seqlens_q = kwargs.get("cu_seqlens_q", None)
            pairs = _attention_pairs(q, k, cu_seqlens_q)
            heads = q.shape[-2]
            head_dim = q.shape[-1]
            flops = 4 * heads * head_dim * pairs  # GPU scalar tensor
            profiler._attn_flops_t = (
                flops if profiler._attn_flops_t is None else profiler._attn_flops_t + flops
            )
            return profiler._orig_attention(q, k, v, *args, **kwargs)

        models_mod.attention = wrapped_attention

        if self._orig_parallel_attention is not None:
            def wrapped_parallel_attention(*args, **kwargs):
                # args = (hybrid_seq_parallel_attn, q, k, v, ...)
                q = args[1]
                k = args[2]
                cu_seqlens_q = kwargs.get("cu_seqlens_q", None)
                pairs = _attention_pairs(q, k, cu_seqlens_q)
                heads = q.shape[-2]
                head_dim = q.shape[-1]
                flops = 4 * heads * head_dim * pairs
                profiler._attn_flops_t = (
                    flops if profiler._attn_flops_t is None else profiler._attn_flops_t + flops
                )
                return profiler._orig_parallel_attention(*args, **kwargs)

            models_mod.parallel_attention = wrapped_parallel_attention

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._models_mod is not None:
            if self._orig_attention is not None:
                self._models_mod.attention = self._orig_attention
            if self._orig_parallel_attention is not None:
                self._models_mod.parallel_attention = self._orig_parallel_attention
        self._models_mod = None

    # ---------------- Per-step API ----------------
    def step_begin(self):
        """Call immediately before the transformer forward."""
        self._linear_flops = 0
        self._attn_flops_t = None
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._start_event.record()

    def step_end(self):
        """Call immediately after the transformer forward. Returns per-step stats."""
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._end_event.record()
        torch.cuda.synchronize()
        ms = self._start_event.elapsed_time(self._end_event)
        attn_flops = float(self._attn_flops_t.item()) if self._attn_flops_t is not None else 0.0
        linear_flops = self._linear_flops
        return {
            "linear_flops": linear_flops,
            "attn_flops": attn_flops,
            "total_flops": linear_flops + attn_flops,
            "ms": ms,
        }
