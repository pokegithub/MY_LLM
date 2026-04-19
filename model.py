"""Model architecture components for attention, routing, and generation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from typing import Optional, Tuple, List, NamedTuple
from core.ops import apply_qk_norm, run_attention
from core.logging import get_logger
from config import ModelConfig

LOGGER = get_logger("model")


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

class ModelOutput(NamedTuple):
    logits: torch.Tensor
    kv_cache: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]
    unc_scores: Optional[torch.Tensor]
    load_balance_loss: torch.Tensor
    z_loss: torch.Tensor


class LossOutput(NamedTuple):
    total_loss: torch.Tensor
    ce_loss: torch.Tensor
    load_balance_loss: torch.Tensor
    z_loss: torch.Tensor
    aux_loss: torch.Tensor
    unc_loss: torch.Tensor


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    __constants__ = ["eps"]

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f32 = x.float()
        norm = x_f32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (self.scale * (x_f32 * norm)).to(dtype)


# ---------------------------------------------------------------------------
# YaRN Rotary Embeddings
# ---------------------------------------------------------------------------

def _yarn_correction_dim(
    beta: float, dim: int, max_seq_len: int, theta: float
) -> float:
    return (dim * math.log(max_seq_len / (beta * 2.0 * math.pi))) / (
        2.0 * math.log(theta)
    )


class YaRNRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 500000.0,
        scale: float = 4.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        super().__init__()
        half = head_dim // 2
        ext_len = int(max_seq_len * scale)

        freq_orig = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        freq_interp = freq_orig / scale

        low = max(
            math.floor(
                _yarn_correction_dim(beta_fast, head_dim, max_seq_len, theta)
            ),
            0,
        )
        high = min(
            math.ceil(
                _yarn_correction_dim(beta_slow, head_dim, max_seq_len, theta)
            ),
            half - 1,
        )

        ramp = (
            torch.ones(half)
            if low == high
            else (
                (torch.arange(half).float() - low) / (high - low)
            ).clamp(0, 1)
        )

        freqs = freq_interp * (1.0 - ramp) + freq_orig * ramp
        mscale = (0.1 * math.log(scale) + 1.0) if scale > 1.0 else 1.0

        pos = torch.arange(ext_len).float()
        angles = torch.outer(pos, freqs)
        cos_cache = torch.cat([angles, angles], dim=-1).cos() * mscale
        sin_cache = torch.cat([angles, angles], dim=-1).sin() * mscale
        self.register_buffer("cos_c", cos_cache)
        self.register_buffer("sin_c", sin_cache)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = q.shape[2]
        c = self.cos_c[offset: offset + T].unsqueeze(0).unsqueeze(0)
        s = self.sin_c[offset: offset + T].unsqueeze(0).unsqueeze(0)
        return (
            q * c + self._rotate_half(q) * s,
            k * c + self._rotate_half(k) * s,
        )


# ---------------------------------------------------------------------------
# Grouped-Query Attention
# ---------------------------------------------------------------------------

class GQAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.nh = cfg.n_heads
        self.nkv = cfg.n_kv_heads
        self.hd = cfg.head_dim
        self.latent_hd = cfg.latent_head_dim
        self.ng = cfg.n_heads // cfg.n_kv_heads
        self.local = layer_idx < cfg.n_layers // 2
        self.window = cfg.sliding_window
        self.backend = cfg.attention_backend
        self.use_flash = bool(cfg.use_flashattention)
        self.use_mla = bool(cfg.use_mla)
        self.qk_norm_eps = float(cfg.qk_norm_eps)

        self.q = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)

        if self.use_mla:
            # MLA-style latent compression uses a configurable d' ratio.
            self.k_down = nn.Linear(
                cfg.dim,
                cfg.n_kv_heads * self.latent_hd,
                bias=False,
            )
            self.v_down = nn.Linear(
                cfg.dim,
                cfg.n_kv_heads * self.latent_hd,
                bias=False,
            )
            self.k_up = nn.Linear(self.latent_hd, self.hd, bias=False)
            self.v_up = nn.Linear(self.latent_hd, self.hd, bias=False)
            self.k = None
            self.v = None
        else:
            self.k = nn.Linear(
                cfg.dim,
                cfg.n_kv_heads * cfg.head_dim,
                bias=False,
            )
            self.v = nn.Linear(
                cfg.dim,
                cfg.n_kv_heads * cfg.head_dim,
                bias=False,
            )
            self.k_down = None
            self.v_down = None
            self.k_up = None
            self.v_up = None

        self.o = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)
        self.rope = YaRNRotaryEmbedding(
            cfg.head_dim, cfg.max_seq_len, cfg.rope_theta, cfg.yarn_scale
        )

    def _causal_mask(
        self, T: int, S: int, device: torch.device
    ) -> torch.Tensor:
        q_pos = torch.arange(T, device=device)
        k_pos = torch.arange(S, device=device)
        causal = k_pos.unsqueeze(0) <= (S - T + q_pos).unsqueeze(1)
        if self.local:
            abs_q = S - T + q_pos
            dist = abs_q.unsqueeze(1) - k_pos.unsqueeze(0)
            causal = causal & (dist >= 0) & (dist <= self.window)
        return torch.where(causal, 0.0, float("-inf"))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)

        if self.use_mla:
            k_lat = self.k_down(x).view(B, T, self.nkv, self.latent_hd).transpose(1, 2)
            v_lat = self.v_down(x).view(B, T, self.nkv, self.latent_hd).transpose(1, 2)
            k = self.k_up(k_lat)
            v = self.v_up(v_lat)
        else:
            k = self.k(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
            v = self.v(x).view(B, T, self.nkv, self.hd).transpose(1, 2)

        q, k = apply_qk_norm(q, k, eps=self.qk_norm_eps)
        q, k = self.rope(q, k, offset)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv = (k, v)

        ke = k.repeat_interleave(self.ng, dim=1)
        ve = v.repeat_interleave(self.ng, dim=1)
        S = ke.shape[2]

        mask: Optional[torch.Tensor] = None
        if self.local:
            mask = self._causal_mask(T, S, x.device).unsqueeze(0).unsqueeze(0)

        out = run_attention(
            q,
            ke,
            ve,
            backend=self.backend,
            use_flashattention=self.use_flash,
            local_attention=self.local,
            head_dim=self.hd,
            attn_mask=mask,
            decode_single_token=(T == 1),
        )

        return (
            self.o(out.transpose(1, 2).contiguous().view(B, T, -1)),
            new_kv,
        )


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.g = nn.Linear(dim, ffn_dim, bias=False)
        self.u = nn.Linear(dim, ffn_dim, bias=False)
        self.d = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.d(F.silu(self.g(x)) * self.u(x))


# ---------------------------------------------------------------------------
# Mixture of Experts
# ---------------------------------------------------------------------------

class MoE(nn.Module):
    """
    Memory-efficient MoE using grouped token routing.

    Instead of gathering weight matrices per token (O(N * nea * D * edim) memory),
    we group tokens by expert assignment and run one matmul per expert.
    This is O(n_experts) kernel launches but each processes many tokens
    in parallel via standard batched matmul.

    Grouped routing keeps per-step memory bounded by expert batches.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ne = cfg.n_experts
        self.nea = cfg.n_experts_active
        self.dim = cfg.dim
        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.edim = max(cfg.ffn_dim // cfg.n_experts_active, 256)

        # Per-expert SwiGLU layers
        self.experts_gate = nn.ModuleList([
            nn.Linear(cfg.dim, self.edim, bias=False)
            for _ in range(cfg.n_experts)
        ])
        self.experts_up = nn.ModuleList([
            nn.Linear(cfg.dim, self.edim, bias=False)
            for _ in range(cfg.n_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(self.edim, cfg.dim, bias=False)
            for _ in range(cfg.n_experts)
        ])

        self.load_balance_loss: torch.Tensor = torch.tensor(0.0)
        self.z_loss: torch.Tensor = torch.tensor(0.0)
        self.router_prob_cv: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = B * T
        xf = x.reshape(N, D)

        logits = self.router(xf)
        logits_f32 = logits.float()
        probs = F.softmax(logits_f32, dim=-1)
        # z-loss uses log(Z)^2 in fp32 to reduce bf16 drift on router logits.
        self.z_loss = torch.logsumexp(logits_f32, dim=-1).pow(2).mean()

        topw, topi = torch.topk(probs, self.nea, dim=-1)
        topw = topw / topw.sum(-1, keepdim=True).clamp_min(1e-9)

        # Switch-Transformer style load-balance loss:
        #   n_experts * sum_i (f_i * p_i)
        # where f_i is token fraction routed to expert i and p_i is
        # average router probability for expert i.
        dispatch = F.one_hot(topi, self.ne).float()  # [N, top_k, ne]
        tokens_per_expert = dispatch.sum(dim=(0, 1))
        load_fraction = tokens_per_expert / max(float(N * self.nea), 1.0)
        prob_mean = probs.mean(dim=0)
        self.router_prob_cv = (
            prob_mean.std(unbiased=False)
            / prob_mean.mean().clamp_min(1e-9)
        )
        switch_load_balance = self.ne * torch.sum(load_fraction * prob_mean)

        self.load_balance_loss = switch_load_balance

        # --- Grouped token routing ---
        out = torch.zeros_like(xf)

        # Flatten top-k assignments and group by expert in one sort pass.
        flat_expert = topi.reshape(-1)                          # [N*nea]
        flat_weight = topw.to(dtype=xf.dtype).reshape(-1)       # [N*nea]
        flat_token = (
            torch.arange(N, device=xf.device)
            .unsqueeze(1)
            .expand(N, self.nea)
            .reshape(-1)
        )

        order = torch.argsort(flat_expert)
        exp_sorted = flat_expert[order]
        tok_sorted = flat_token[order]
        w_sorted = flat_weight[order]

        unique_exp, counts = torch.unique_consecutive(
            exp_sorted, return_counts=True
        )
        cursor = 0
        for exp_id, count in zip(unique_exp.tolist(), counts.tolist()):
            span = slice(cursor, cursor + count)
            cursor += count

            token_indices = tok_sorted[span]
            weights = w_sorted[span]
            x_expert = xf[token_indices]

            h = F.silu(self.experts_gate[exp_id](x_expert)) * \
                self.experts_up[exp_id](x_expert)
            h = self.experts_down[exp_id](h)

            out.scatter_add_(
                0,
                token_indices.unsqueeze(-1).expand(-1, D),
                h * weights.unsqueeze(-1),
            )

        return out.reshape(B, T, D)


# ---------------------------------------------------------------------------
# MoD Router
# ---------------------------------------------------------------------------

class MoDRouter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.gate.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(x))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """
    Unified transformer block.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = GQAttention(cfg, layer_idx)
        self.n2 = RMSNorm(cfg.dim, cfg.norm_eps)
        self.is_moe = cfg.use_moe and (layer_idx % cfg.moe_freq == 0)
        self.ffn = MoE(cfg) if self.is_moe else SwiGLU(cfg.dim, cfg.ffn_dim)
        self.use_mod = 2 <= layer_idx < cfg.n_layers - 2
        if self.use_mod:
            self.mod_router = MoDRouter(cfg.dim)

        if cfg.use_residual_gates:
            self.alpha_attn = nn.Parameter(torch.ones(1))
            self.alpha_ffn = nn.Parameter(torch.ones(1))
        else:
            self.alpha_attn = None
            self.alpha_ffn = None

    def _gate(
        self, alpha: Optional[nn.Parameter], val: torch.Tensor
    ) -> torch.Tensor:
        return alpha * val if alpha is not None else val

    def _training_body(self, x: torch.Tensor) -> torch.Tensor:
        """Training-only path: no KV cache needed."""
        a, _ = self.attn(self.n1(x))
        x = x + self._gate(self.alpha_attn, a)
        f = self.ffn(self.n2(x))
        return x + self._gate(self.alpha_ffn, f)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        # --- Training paths: no KV cache produced ---
        if self.training:
            if self.use_mod:
                w = self.mod_router(x)
                computed = self._training_body(x)
                return x + w * (computed - x), None
            return self._training_body(x), None

        # --- Inference paths: ALWAYS produce KV cache ---
        residual = x
        w = self.mod_router(residual) if self.use_mod else None
        a, kv = self.attn(self.n1(x), kv_cache, offset)
        x = x + self._gate(self.alpha_attn, a)
        f = self.ffn(self.n2(x))
        x = x + self._gate(self.alpha_ffn, f)
        if w is not None:
            x = residual + w * (x - residual)
        return x, kv


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

class LLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.mem_len = cfg.n_memory_tokens
        self.n_shallow = cfg.n_layers // 2
        self.n_loops = cfg.n_loops
        self.adapt_thr = cfg.adaptive_loop_threshold
        self.max_refine = cfg.max_refinement_loops

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx=0)
        self.memory = nn.Parameter(
            torch.randn(1, cfg.n_memory_tokens, cfg.dim) * 0.02
        )
        self.layers = nn.ModuleList(
            [Block(cfg, i) for i in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        self.conf_head = (
            nn.Sequential(
                nn.Linear(cfg.dim, 256, bias=False),
                nn.SiLU(),
                nn.Linear(256, cfg.confidence_dim, bias=False),
            )
            if cfg.use_confidence_head
            else None
        )

        self._n_moe_layers = sum(1 for b in self.layers if b.is_moe)
        self.apply(self._init_weights)
        total = sum(p.numel() for p in self.parameters())
        LOGGER.info("LLM initialized: %s parameters", f"{total:,}")

    def _init_weights(self, m: nn.Module):
        std = 0.02
        if isinstance(m, nn.Linear):
            # Output projections get scaled init
            nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, std)

    # -- layer runner -------------------------------------------------------

    def _run_layers(
        self,
        x: torch.Tensor,
        kv_cache: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]],
        offset: int,
        loops: int,
    ) -> Tuple[
        torch.Tensor,
        List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Run all layers. Shallow layers run once. Deep layers run `loops` times.
        Router gate and z-losses are accumulated and normalized.
        """
        dev = x.device
        gate = torch.tensor(0.0, device=dev)
        zl = torch.tensor(0.0, device=dev)
        new_kv: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None
        ] * len(self.layers)
        moe_fwd_count = 0

        # Shallow layers: always run once
        for i in range(self.n_shallow):
            ci = kv_cache[i] if kv_cache else None
            x, kv = self.layers[i](x, ci, offset)
            new_kv[i] = kv
            if self.layers[i].is_moe:
                gate = gate + self.layers[i].ffn.load_balance_loss
                zl = zl + self.layers[i].ffn.z_loss
                moe_fwd_count += 1

        # Deep layers: run `loops` times
        for loop in range(loops):
            for i in range(self.n_shallow, len(self.layers)):
                # Only use external cache on first loop
                ci = kv_cache[i] if (kv_cache and loop == 0) else None

                if self.training:
                    layer = self.layers[i]

                    def _run_deep_layer(
                        hidden: torch.Tensor,
                        layer_ref: Block = layer,
                        cache_ref=ci,
                        offset_ref: int = offset,
                    ) -> torch.Tensor:
                        out, _ = layer_ref(hidden, cache_ref, offset_ref)
                        return out

                    # Checkpoint deep recursive iterations on train path to
                    # cap activation memory on small VRAM devices (e.g., T4).
                    x = grad_ckpt(
                        _run_deep_layer,
                        x,
                        use_reentrant=False,
                    )
                    kv = None
                else:
                    x, kv = self.layers[i](x, ci, offset)

                # Always store the LAST loop's KV
                new_kv[i] = kv
                if self.layers[i].is_moe:
                    gate = gate + self.layers[i].ffn.load_balance_loss
                    zl = zl + self.layers[i].ffn.z_loss
                    moe_fwd_count += 1

        if moe_fwd_count > 0:
            gate = gate / moe_fwd_count
            zl = zl / moe_fwd_count

        return x, new_kv, gate, zl

    # -- refinement with synced cache ---------------------------------------

    def _run_refinement_with_cache(
        self, ids: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        """
        Complete re-run producing BOTH refined hidden states AND a
        consistent KV cache. Position encodings are correct (offset=0).
        """
        B = ids.shape[0]
        x = self.embed(ids)
        mem = self.memory.expand(B, -1, -1)
        x = torch.cat([mem, x], dim=1)

        total_deep_loops = self.n_loops + self.max_refine
        new_kv: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None
        ] * len(self.layers)

        # Shallow layers
        for i in range(self.n_shallow):
            x, kv = self.layers[i](x, kv_cache=None, offset=0)
            new_kv[i] = kv

        # Deep layers: multiple loops, only last loop stores KV
        for loop_idx in range(total_deep_loops):
            is_last = loop_idx == total_deep_loops - 1
            for i in range(self.n_shallow, len(self.layers)):
                x, kv = self.layers[i](x, kv_cache=None, offset=0)
                if is_last:
                    new_kv[i] = kv

        return x[:, self.mem_len:, :], new_kv

    # -- adaptive refinement for cached decode -------------------------------

    def _run_refinement_cached(
        self,
        x: torch.Tensor,
        kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
        offset: int,
    ) -> Tuple[
        torch.Tensor,
        List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        """
        Refinement for single-token cached decode.
        Re-runs deep layers one additional time using the same KV cache.
        Does NOT create new KV entries — operates on the existing state.
        """
        new_kv = list(kv_cache)  # shallow copy
        for i in range(self.n_shallow, len(self.layers)):
            layer = self.layers[i]
            residual = x
            mod_w = layer.mod_router(residual) if layer.use_mod else None

            # Re-run with the same cache (don't append to cache)
            x_normed = layer.n1(x)
            a, _ = layer.attn(x_normed, kv_cache[i], offset)
            x = x + layer._gate(layer.alpha_attn, a)
            f = layer.ffn(layer.n2(x))
            x = x + layer._gate(layer.alpha_ffn, f)
            if mod_w is not None:
                x = residual + mod_w * (x - residual)
        return x, new_kv

    # -- main forward -------------------------------------------------------

    def forward(
        self,
        ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[
            List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
        ] = None,
        offset: int = 0,
        n_loops: Optional[int] = None,
        loss_coeffs: Optional[dict[str, float]] = None,
    ):
        B, T = ids.shape
        x = self.embed(ids)
        first_call = kv_cache is None

        if first_call:
            x = torch.cat([self.memory.expand(B, -1, -1), x], dim=1)

        loops = (
            n_loops
            if n_loops is not None
            else (self.n_loops if self.training else 1)
        )

        x, new_kv, gate_loss, z_loss = self._run_layers(
            x, kv_cache, offset, loops
        )

        if first_call:
            x = x[:, self.mem_len:, :]

        h = self.norm(x)
        logits = self.head(h)

        # -- confidence & adaptive refinement -------------------------------
        unc_logits = None
        unc_scores = None

        if self.conf_head is not None:
            unc_logits = self.conf_head(h)

            if not self.training and first_call:
                unc_scores = torch.sigmoid(unc_logits)
                avg_conf = unc_scores[..., 0].mean().item()

                if avg_conf < self.adapt_thr:
                    h2_raw, refined_kv = self._run_refinement_with_cache(ids)
                    h2 = self.norm(h2_raw)
                    logits = self.head(h2)
                    unc_logits = self.conf_head(h2)
                    unc_scores = torch.sigmoid(unc_logits)
                    new_kv = refined_kv

            elif not self.training and not first_call:
                # Cached decode: adaptive refinement via deep re-run
                unc_scores = torch.sigmoid(unc_logits)
                avg_conf = unc_scores[..., 0].mean().item()

                if avg_conf < self.adapt_thr:
                    h2, refined_kv = self._run_refinement_cached(
                        x, new_kv, offset
                    )
                    h2 = self.norm(h2)
                    logits = self.head(h2)
                    unc_logits = self.conf_head(h2)
                    unc_scores = torch.sigmoid(unc_logits)
                    # Keep original KV (refinement didn't modify cache)

            elif not self.training:
                unc_scores = torch.sigmoid(unc_logits)

        # -- return inference -----------------------------------------------
        if targets is None:
            return ModelOutput(
                logits=logits,
                kv_cache=new_kv,
                unc_scores=unc_scores,
                load_balance_loss=gate_loss,
                z_loss=z_loss,
            )

        # -- compute loss ---------------------------------------------------
        ce = F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size),
            targets.reshape(-1),
            ignore_index=self.cfg.pad_token_id,
        )

        unc_loss = torch.tensor(0.0, device=ids.device)
        aux_terms = [
            layer.ffn.router_prob_cv
            for layer in self.layers
            if layer.is_moe
        ]
        aux_loss = (
            torch.stack(aux_terms).mean()
            if aux_terms
            else torch.tensor(0.0, device=ids.device)
        )
        if self.conf_head is not None and unc_logits is not None:
            with torch.no_grad():
                pred = logits.argmax(-1)
                correct = (pred == targets).float()
                # Entropy-based uncertainty target
                lp = F.log_softmax(logits, dim=-1)
                entropy = -(lp.exp() * lp).sum(-1)
                max_entropy = math.log(self.cfg.vocab_size)
                normalized_entropy = entropy / max_entropy
                # Target: [correctness, 1-entropy, entropy]
                unc_target = torch.stack([
                    correct,
                    1.0 - normalized_entropy,
                    normalized_entropy,
                ], dim=-1)

            unc_loss = F.binary_cross_entropy_with_logits(
                unc_logits,
                unc_target,
                reduction="mean",
            )

        ce_coeff = 1.0
        gate_coeff = 1.0
        z_coeff = 1.0
        aux_coeff = 1.0
        unc_coeff = 1.0
        if loss_coeffs:
            ce_coeff = float(loss_coeffs.get("ce", 1.0))
            gate_coeff = float(loss_coeffs.get("gate", 1.0))
            z_coeff = float(loss_coeffs.get("z", 1.0))
            aux_coeff = float(loss_coeffs.get("aux", 1.0))
            unc_coeff = float(loss_coeffs.get("unc", 1.0))

        total_loss = (
            ce_coeff * ce
            + gate_coeff * gate_loss
            + z_coeff * z_loss
            + aux_coeff * aux_loss
            + unc_coeff * unc_loss
        )

        return LossOutput(
            total_loss=total_loss,
            ce_loss=ce,
            load_balance_loss=gate_loss,
            z_loss=z_loss,
            aux_loss=aux_loss,
            unc_loss=unc_loss,
        )

    # -- generation helper --------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = 2,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Autoregressive generation with KV caching.
        Returns (generated_ids, uncertainty_scores).
        """
        self.eval()
        B = ids.shape[0]
        generated = ids.clone()
        all_unc = []

        # Prefill
        out = self(ids)
        logits = out.logits[:, -1:, :]
        kv = out.kv_cache
        unc = out.unc_scores
        if unc is not None:
            all_unc.append(unc[:, -1:, :])

        offset = ids.shape[1] + self.mem_len

        for _ in range(max_new_tokens):
            # Sample
            if temperature > 0:
                scaled = logits[:, -1, :] / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled[scaled < v[:, -1:]] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(
                        scaled, descending=True
                    )
                    cumprobs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    mask = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[mask] = float("-inf")
                    scaled = sorted_logits.scatter(1, sorted_idx, sorted_logits)
                probs = F.softmax(scaled, dim=-1)
                next_id = torch.multinomial(probs, 1)
            else:
                next_id = logits[:, -1, :].argmax(-1, keepdim=True)

            generated = torch.cat([generated, next_id], dim=1)

            if (next_id == eos_token_id).all():
                break

            # Decode step
            out = self(next_id, kv_cache=kv, offset=offset)
            logits = out.logits
            kv = out.kv_cache
            if out.unc_scores is not None:
                all_unc.append(out.unc_scores)
            offset += 1

        unc_tensor = (
            torch.cat(all_unc, dim=1) if all_unc else None
        )
        return generated, unc_tensor