"""Activation-Hessian-tuned ARVQ fit for one (layer, projection).

Codebooks (c0[256,8], c1[256,8], FP4-constrained) are SHARED across all cold
experts of a (layer, projection), so this is a joint per-layer fit:

  1. per expert: raw-basis Hessian H_e, escalating-damp Cholesky hinv_e,
     per-(row,128block) init scale s_e, global scalar (fixed).
  2. codebook EM on a Hessian-importance-weighted subsample of normalized
     8-dim weight groups (k-means warm start -> alternating refine, FP4 project
     after every M-step).  <- "fit codebooks to data" (a)
  3. per expert final assignment via a GPTQ/LDLQ error-feedback column sweep
     with the fixed codebooks (plain-L2 inner assignment; cross-group feedback
     carries the off-diagonal Hessian), then LS refit of the E4M3 scales.
     <- "Hessian-aware code selection + error feedback" (b)

No incoherence rotation: Hessians and targets are in the raw weight basis
(the SM120 kernel feeds raw activations). Weighting is in the EM importance +
error feedback + scale LS, never a sqrt(diag(H)) whitening (the FP4 grid on the
codebook forbids rescaling the target space).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from arvqprep.pack import project_to_fp4


# --------------------------------------------------------------- Hessians
def hessian_fc1(x: torch.Tensor) -> torch.Tensor:
    """gate/up share E[x x^T] over the H axis. x: [T, H] (routed tokens)."""
    x = x.float()
    return (x.t() @ x) / max(x.shape[0], 1)


def hessian_fc2(x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor) -> torch.Tensor:
    """down E[m m^T], m = silu(x Wg^T) * (x Wu^T) over the I axis. x:[T,H]."""
    x = x.float()
    m = F.silu(x @ wg.float().t()) * (x @ wu.float().t())    # [T, I]
    return (m.t() @ m) / max(x.shape[0], 1)


def cholesky_inv_upper(hess: torch.Tensor, damp_start: float = 1e-3) -> torch.Tensor:
    """Escalating-damp upper Cholesky of H^-1 (same recipe as btxprep/encoder)."""
    k = hess.shape[0]
    mean_diag = hess.diagonal().mean().clamp(min=1e-12)
    damp = damp_start
    eye = torch.eye(k, device=hess.device, dtype=hess.dtype)
    for _ in range(12):
        try:
            return torch.linalg.cholesky(
                torch.linalg.inv(hess + damp * mean_diag * eye), upper=True)
        except Exception:  # noqa: BLE001
            damp *= 4.0
    raise RuntimeError("ARVQ Cholesky failed at maximum damp")


# ---------------------------------------------------------- codebook EM
def _nearest(x, c):
    """argmin_k ||x - c_k||^2 (plain L2). x:[M,8], c:[K,8] -> [M]."""
    return (c.square().sum(1)[None, :] - 2 * x @ c.t()).argmin(1)


def _wmeans(x, w, ids, count, prev):
    """Weighted per-cluster mean of x (scalar sample weights w); keep prev if empty."""
    sums = torch.zeros_like(prev)
    sums.index_add_(0, ids, x * w[:, None])
    wsum = torch.zeros(count, device=x.device, dtype=x.dtype)
    wsum.index_add_(0, ids, w)
    return torch.where(wsum[:, None] > 0, sums / wsum.clamp_min(1e-20)[:, None], prev)


def fit_codebooks(x, w, *, iters=8, seed=0):
    """Weighted FP4 additive-RVQ codebook fit on a subsample.
    x:[M,8] normalized group vectors, w:[M] importance weights.
    Returns c0[256,8], c1[256,8] (FP4-on-grid), best weighted relative-L2."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g, device=x.device)
    c0 = project_to_fp4(x[perm[:256]].clone())
    i = _nearest(x, c0)
    c1 = project_to_fp4(_kmeans(x - c0[i], 256, g))
    j = _nearest(x - c0[i], c1)
    wsum = (w * x.square().sum(1)).sum().clamp_min(1e-20)
    best = None
    for _ in range(iters):
        i = _nearest(x - c1[j], c0)
        c0 = project_to_fp4(_wmeans(x - c1[j], w, i, 256, c0))
        j = _nearest(x - c0[i], c1)
        c1 = project_to_fp4(_wmeans(x - c0[i], w, j, 256, c1))
        err = ((w * (x - c0[i] - c1[j]).square().sum(1)).sum() / wsum).item()
        if best is None or err < best[0]:
            best = (err, c0.clone(), c1.clone())
    err, c0, c1 = best
    return c0, c1, err ** 0.5


def _kmeans(x, count, gen, iters=12):
    c = x[torch.randperm(x.shape[0], generator=gen, device=x.device)[:count]].clone()
    for _ in range(iters):
        ids = _nearest(x, c)
        sums = torch.zeros_like(c)
        sums.index_add_(0, ids, x)
        cnt = torch.bincount(ids, minlength=count).float().clamp_min(1)[:, None]
        c = torch.where(torch.bincount(ids, minlength=count)[:, None] > 0, sums / cnt, c)
    return c


# ----------------------------------------------------- per-expert sweep
def _assign(tn, c0, c0n, c1, c1n, refine):
    """Plain-L2 additive assignment. tn:[N,8]. Returns a,b (long)."""
    a = (c0n[None, :] - 2 * tn @ c0.t()).argmin(1)
    r = tn - c0[a]
    b = (c1n[None, :] - 2 * r @ c1.t()).argmin(1)
    for _ in range(refine):
        r = tn - c1[b]
        a = (c0n[None, :] - 2 * r @ c0.t()).argmin(1)
        r = tn - c0[a]
        b = (c1n[None, :] - 2 * r @ c1.t()).argmin(1)
    return a, b


def sweep_expert(W, hinv, c0, c1, s, glob, *, col_block, refine):
    """LDLQ error-feedback assignment for one expert.

    W:[N,K] fp32 target; hinv:[K,K] upper fp32; s:[N,K/128] fp32 (E4M3-valued)
    scale; glob: scalar. Returns a,b uint8 [N,K/8] and reconstruction q [N,K]."""
    N, K = W.shape
    w = W.clone()
    q = torch.empty_like(w)
    a_full = torch.empty(N, K // 8, dtype=torch.uint8, device=W.device)
    b_full = torch.empty(N, K // 8, dtype=torch.uint8, device=W.device)
    c0n = c0.square().sum(1)
    c1n = c1.square().sum(1)
    for j0 in range(0, K, col_block):
        j1 = min(j0 + col_block, K)
        blk = w[:, j0:j1]
        qhat = torch.empty_like(blk)
        for gg in range((j1 - j0) // 8):
            c = j0 + gg * 8
            sden = (glob * s[:, c // 128]).clamp_min(1e-20)[:, None]      # [N,1]
            tn = blk[:, gg * 8:gg * 8 + 8] / sden
            a, b = _assign(tn, c0, c0n, c1, c1n, refine)
            qhat[:, gg * 8:gg * 8 + 8] = sden * (c0[a] + c1[b])
            a_full[:, c // 8] = a.to(torch.uint8)
            b_full[:, c // 8] = b.to(torch.uint8)
        q[:, j0:j1] = qhat
        err = torch.linalg.solve_triangular(
            hinv[j0:j1, j0:j1], blk - qhat, upper=True, left=False)
        if j1 < K:
            w[:, j1:] -= err @ hinv[j0:j1, j1:]
    return a_full, b_full, q


def refit_scales(W, a, b, c0, c1, glob, hdiag, K):
    """LS per-(row,128block) scale given codes; E4M3-quantized. Returns [N,K/128]."""
    N = W.shape[0]
    u = (c0[a.long()] + c1[b.long()]).reshape(N, K // 8, 8) * glob         # unit recon
    Wg = W.reshape(N, K // 8, 8)
    hd = hdiag.reshape(K // 8, 8)[None]                                    # [1,K/8,8]
    num = (hd * Wg * u).reshape(N, K // 128, 128).sum(-1)
    den = (hd * u * u).reshape(N, K // 128, 128).sum(-1).clamp_min(1e-20)
    s = (num / den)
    s = s.clamp_min(0).to(torch.float8_e4m3fn).float()
    return s


@dataclass
class EncodedLayerProj:
    c0: torch.Tensor            # [256,8] fp32 (FP4 grid)
    c1: torch.Tensor            # [256,8] fp32; [E,256,8] for expert scope
    glob: float
    a: torch.Tensor            # [E,N,K/8] uint8
    b: torch.Tensor            # [E,N,K/8] uint8
    s: torch.Tensor            # [E,N,K/128] fp32 (E4M3-valued)
    N: int
    K: int
    cb_rel_l2: float
    recon_rel_fro: float       # weight-space, Hessian-weighted, over experts
    scale_dtype: str = "fp8_e4m3"
    codebook_dtype: str = "fp4_grid"   # 'fp4_grid' (v2/v3) or 'fp16' (free atoms)


def _e4m3(x):
    return x.to(torch.float8_e4m3fn).float()


def fit_layer_projection(W, H, *, col_block=128, cb_iters=8, sweep_passes=2,
                         refine=1, subsample_per_expert=20000, seed=0,
                         device="cuda", verbose=False, global_scale=None,
                         codebook_scope="layer"):
    """W: list of E weight targets [N,K] (fp16/fp32, CPU or GPU).
    H: list of E Hessians [K,K] (fp32). Returns EncodedLayerProj."""
    dev = torch.device(device)
    E = len(W)
    N, K = W[0].shape
    gen = torch.Generator(device=dev).manual_seed(seed)
    if codebook_scope not in ('layer','expert'):
        raise ValueError('Unknown codebook scope')
    if codebook_scope == 'expert':
        # Retain one global scale per projection so the v2 index/scale layout
        # and arithmetic stay unchanged. Only the dictionaries gain an axis.
        if global_scale is None:
            pool=[]
            for weight in W:
                rms=weight.to(dev).float().reshape(N,K//128,128).square().mean(-1).sqrt().flatten()
                pick=torch.randperm(len(rms),generator=gen,device=dev)[:4096]
                pool.append(rms[pick])
            global_scale=float(torch.cat(pool).median().clamp_min(1e-8))
            del pool,rms
        results=[]
        for e in range(E):
            result=fit_layer_projection([W[e]],[H[e]],col_block=col_block,
                cb_iters=cb_iters,sweep_passes=sweep_passes,refine=refine,
                subsample_per_expert=subsample_per_expert,seed=seed+e,
                device=device,verbose=verbose,global_scale=global_scale)
            results.append(result)
        return EncodedLayerProj(torch.stack([r.c0 for r in results]),
            torch.stack([r.c1 for r in results]),float(global_scale),
            torch.cat([r.a for r in results]),torch.cat([r.b for r in results]),
            torch.cat([r.s for r in results]),N,K,
            sum(r.cb_rel_l2 for r in results)/E,
            sum(r.recon_rel_fro for r in results)/E)

    # ---- Phase A: hinv, hdiag, init scale, global
    hinv, hdiag, s = [], [], []
    rms_pool = []
    for e in range(E):
        We = W[e].to(dev).float()
        hd = H[e].to(dev).float().diagonal().clamp_min(1e-12)
        hinv.append(cholesky_inv_upper(H[e].to(dev).float()))
        hdiag.append(hd)
        blk_rms = We.reshape(N, K // 128, 128).square().mean(-1).sqrt()    # [N,K/128]
        s.append(blk_rms)                                                  # tmp (pre-global)
        rms_pool.append(blk_rms.flatten()[torch.randperm(N * (K // 128),
                        generator=gen, device=dev)[:4096]])
        del We
    glob = (float(torch.cat(rms_pool).median().clamp_min(1e-8).item())
            if global_scale is None else float(global_scale))
    if not __import__('math').isfinite(glob) or glob <= 0:
        raise ValueError('Global scale must be finite and positive')
    for e in range(E):
        s[e] = _e4m3(s[e] / glob)                                          # E4M3 scale

    # ---- Phase B: importance-weighted subsample of normalized group vectors
    xs, ws = [], []
    for e in range(E):
        We = W[e].to(dev).float().reshape(N, K // 8, 8)
        sden = (glob * s[e]).clamp_min(1e-20)                              # [N,K/128]
        sden = sden.repeat_interleave(16, dim=1)                          # [N,K/8]
        tn = (We / sden[:, :, None]).reshape(-1, 8)                       # [N*K/8,8]
        imp = (hdiag[e].float().reshape(K // 8, 8).mean(1)[None, :]
               * (s[e].repeat_interleave(16, dim=1) ** 2)).reshape(-1)   # [N*K/8]
        m = tn.shape[0]
        idx = torch.randperm(m, generator=gen, device=dev)[:subsample_per_expert]
        xs.append(tn[idx])
        ws.append(imp[idx])
        del We, tn, imp
    xs = torch.cat(xs)
    ws = torch.cat(ws).clamp_min(1e-12)
    c0, c1, cb_rel = fit_codebooks(xs, ws, iters=cb_iters, seed=seed)
    if verbose:
        print(f"  codebook subsample rel-L2 {cb_rel:.4f} (M={xs.shape[0]})",
              flush=True)
    del xs, ws

    # ---- Phase D: per-expert error-feedback sweep + scale refit
    a_all = torch.empty(E, N, K // 8, dtype=torch.uint8)
    b_all = torch.empty(E, N, K // 8, dtype=torch.uint8)
    s_all = torch.empty(E, N, K // 128, dtype=torch.float32)
    num_w = den_w = 0.0
    for e in range(E):
        We = W[e].to(dev).float()
        hv = hinv[e].float()
        hd = hdiag[e].float()
        se = s[e]
        a = b = q = None
        for p in range(sweep_passes):
            a, b, q = sweep_expert(We, hv, c0, c1, se, glob,
                                   col_block=col_block, refine=refine)
            if p + 1 < sweep_passes:
                se = refit_scales(We, a, b, c0, c1, glob, hd, K)
        se = refit_scales(We, a, b, c0, c1, glob, hd, K)
        a, b, q = sweep_expert(We, hv, c0, c1, se, glob,
                               col_block=col_block, refine=refine)
        a_all[e], b_all[e], s_all[e] = a.cpu(), b.cpu(), se.cpu()
        d = (We - q)
        # Half-stored Hessians can lose positive semidefiniteness. Evaluate
        # against the stabilized positive metric used by the sweep instead:
        # H_eff^-1 = U.T @ U, hence d H_eff d.T = ||d U^-1||^2.
        num_w += positive_quadratic(d, hv)
        den_w += positive_quadratic(We, hv)
        del We, hv, q, d
        hinv[e] = None
    recon = (num_w / max(den_w, 1e-20)) ** 0.5
    return EncodedLayerProj(c0.cpu(), c1.cpu(), glob, a_all, b_all, s_all,
                            N, K, cb_rel, recon)


def positive_quadratic(weight, inverse_cholesky):
    """Nonnegative squared norm under the regularized fitting Hessian."""
    z = torch.linalg.solve_triangular(inverse_cholesky.T, weight.T,
                                      upper=False)
    value = float(z.double().square().sum())
    if not __import__('math').isfinite(value):
        raise ValueError('Nonfinite regularized Hessian diagnostic')
    return value


def reconstruct(enc: EncodedLayerProj, e: int, device="cpu") -> torch.Tensor:
    """Decode one expert's weight [N,K] from stored codes/scale/codebooks."""
    dev = torch.device(device)
    c0, c1 = enc.c0.to(dev), enc.c1.to(dev)
    if c0.ndim == 3:
        c0, c1 = c0[e], c1[e]
    a = enc.a[e].to(dev).long()
    b = enc.b[e].to(dev).long()
    s = enc.s[e].to(dev).repeat_interleave(16, dim=1)                     # [N,K/8]
    cb = (c0[a] + c1[b])                                                  # [N,K/8,8]
    W = (enc.glob * s[:, :, None] * cb).reshape(enc.N, enc.K)
    return W
