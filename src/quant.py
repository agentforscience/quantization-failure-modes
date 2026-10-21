"""Weight-only round-to-nearest (RTN) quantization shared by the vLLM generation and HF KL scripts.

We simulate b-bit integer weight storage ("fake quantization"): each row of a linear
weight [out, in] is split into groups of `group` input channels; every group gets its own
asymmetric min-max scale/zero-point, the weights are rounded to the 2^b integer grid and
de-quantized back to bf16. Numerically this is exactly what an INT-b g128 kernel computes
(up to accumulation order), and it lets us sweep arbitrary bit widths with one code path.
Embeddings and the LM head stay in bf16 (standard practice in GPTQ/AWQ evaluations).
"""
import re

import torch

# Transformer-block linear layers in HF (q_proj, ..., down_proj) and vLLM (qkv_proj, gate_up_proj, ...)
LINEAR_RE = re.compile(r"layers\.\d+\..*(proj|fc\d?|dense|qkv|query_key_value)\.weight$")


@torch.no_grad()
def rtn_dequant(w: torch.Tensor, bits: int, group: int = 128) -> torch.Tensor:
    """Return the RTN-quantized-then-dequantized copy of a 2-D weight (same dtype/device)."""
    if bits >= 16:
        return w
    out, inp = w.shape
    g = group if (group > 0 and inp % group == 0) else inp  # group<=0 -> per-output-channel
    qmax = 2 ** bits - 1
    res = torch.empty_like(w)
    rows = max(1, (1 << 26) // inp)  # chunk rows to bound fp32 temp memory (~256 MB)
    for s in range(0, out, rows):
        x = w[s:s + rows].float().reshape(-1, g)
        mn, mx = x.amin(1, keepdim=True), x.amax(1, keepdim=True)
        scale = (mx - mn).clamp(min=1e-8) / qmax
        zero = torch.round(-mn / scale)
        q = torch.clamp(torch.round(x / scale) + zero, 0, qmax)
        res[s:s + rows] = ((q - zero) * scale).reshape(-1, inp).to(w.dtype)
    return res


@torch.no_grad()
def rtn_into_(dst: torch.Tensor, src: torch.Tensor, bits: int, group: int = 128, chunk_elems: int = 1 << 22):
    """Write RTN(src) into dst row-chunk by row-chunk (src may live on CPU); bounded GPU temp memory."""
    rows = max(1, chunk_elems // src.shape[1])
    for s in range(0, src.shape[0], rows):
        x = src[s:s + rows].to(dst.device)
        dst[s:s + rows].copy_(rtn_dequant(x, bits, group) if bits < 16 else x)


def quantizable(name: str, p: torch.Tensor) -> bool:
    return p.dim() == 2 and LINEAR_RE.search(name) is not None
