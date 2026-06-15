"""Render a vectorized PDF that explains the MDLM (joint-sequence) architecture.

Sources (kept faithful to the code):
  - jsm/models/transformer.py            (SpeciesSpecificJointSequenceTransformer)
  - jsm/diffusion/modules.py             (JointSequenceDiffusion)
  - jsm/diffusion/noise_schedule.py      (LogLinearNoise)
  - jsm/diffusion/core.py                (q_xt, _subs_parameterization)
  - jsm/data/species_specific.py         (joint-sequence layout)
  - diffusion_configs/model/transformer_300M.yaml
"""

import os
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Patch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "pdf.fonttype": 42,          # embed real glyphs (vector text, selectable)
    "ps.fonttype":  42,
    "font.family":  "DejaVu Sans",
    "font.size":    9.5,
    "axes.linewidth": 0.6,
})

# Color palette: muted, paper-friendly
COL = {
    "input":   "#dde7f2",
    "input2":  "#cfe0c3",
    "input3":  "#f4d8b0",
    "input4":  "#ead4f0",
    "noise":   "#f5c0c0",
    "core":    "#bcd2e8",
    "core_dk": "#7397c4",
    "head":    "#c8e2c8",
    "loss":    "#f3d3a4",
    "ghost":   "#f0f0f0",
    "edge":    "#222222",
}


def box(ax, xy, w, h, text, fc, fontsize=9.5, ec="#222", lw=0.8,
        weight="normal", style="round,pad=0.05,rounding_size=0.08"):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle=style,
                       linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, weight=weight,
            wrap=True)
    return (x, y, w, h)


def arrow(ax, p0, p1, text=None, color="#222", lw=0.9, rad=0.0,
          style="-|>", fontsize=8.5, text_offset=(0.0, 0.05)):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=10,
                        connectionstyle=f"arc3,rad={rad}",
                        color=color, linewidth=lw)
    ax.add_patch(a)
    if text:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx + text_offset[0], my + text_offset[1], text,
                ha="center", va="center", fontsize=fontsize,
                color=color)


def title(ax, txt, y=0.965):
    ax.text(0.5, y, txt, transform=ax.transAxes,
            ha="center", va="top", fontsize=14, weight="bold")


def subtitle(ax, txt, y=0.925):
    ax.text(0.5, y, txt, transform=ax.transAxes,
            ha="center", va="top", fontsize=10, style="italic", color="#444")


def new_page(pdf, figsize=(11, 8.5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------------------
# Page 1 — End-to-end overview
# ---------------------------------------------------------------------------
def page_overview(pdf):
    fig, ax = new_page(pdf)
    title(ax, "MDLM Joint-Sequence Diffusion — End-to-end overview")
    subtitle(ax, "Absorbing-state masked discrete diffusion over mRNA "
                 "(5'UTR + CDS + 3'UTR) conditioned on protein, species, modality")

    # --- Inputs column (left) ---
    box(ax, (0.3, 6.0), 2.4, 0.55, "5'UTR  (nt tokens)",      COL["input"])
    box(ax, (0.3, 5.3), 2.4, 0.55, "CDS  (codon tokens)",     COL["input"])
    box(ax, (0.3, 4.6), 2.4, 0.55, "3'UTR  (nt tokens)",      COL["input"])
    box(ax, (0.3, 3.7), 2.4, 0.55, "Protein seq → ESM-style\nfrozen encoder", COL["input2"])
    box(ax, (0.3, 2.8), 2.4, 0.55, "Species ID (≤256)",       COL["input3"])
    box(ax, (0.3, 2.1), 2.4, 0.55, "Modality tag per token\n(UTR5 / CDS / UTR3 / special)", COL["input4"])

    # --- Joint sequence assembly ---
    box(ax, (3.4, 3.7), 2.6, 1.6,
        "Joint sequence builder\n"
        "[CLS] species [SEP] protein_emb [SEP]\n"
        "5'UTR [SEP] CDS [SEP] 3'UTR [EOS]\n"
        "+ row-wise column permutation\n"
        "+ attention_mask, special_token_mask",
        COL["core"], fontsize=9)

    # --- Forward noise ---
    box(ax, (6.5, 5.6), 3.2, 1.0,
        "Forward noising  q(x_t | x_0)\n"
        "MASK with prob = 1 − e^{−σ(t)}\n"
        "(only RNA tokens; specials pinned)",
        COL["noise"])

    # --- Backbone ---
    box(ax, (6.5, 3.5), 3.2, 1.6,
        "Transformer backbone\n"
        "24 × {pre-LN  →  RoPE QKV  →  FlashAttn (varlen, bidirectional)\n"
        "→ residual  →  pre-LN  →  GELU MLP (×4)  →  residual}\n"
        "hidden = 1024,  heads = 8,  ≈300 M params",
        COL["core_dk"], fontsize=8.8)

    # --- Heads ---
    box(ax, (6.5, 2.3), 1.45, 0.85,
        "RNA LM head\n(Linear → vocab logits)", COL["head"], fontsize=8.5)
    box(ax, (8.25, 2.3), 1.45, 0.85,
        "SUBS parameterization\n(mask logit → −∞,\nunmasked → identity)",
        COL["head"], fontsize=8.5)

    # --- Loss ---
    box(ax, (6.5, 0.9), 3.2, 1.05,
        "NLL  =  −log pθ(x0 | xt) · dσ/expm1(σ)\n"
        "Split per region (UTR5 / CDS / UTR3)\n"
        "mean over the three regions  → final loss",
        COL["loss"], fontsize=8.8)

    # --- Arrows from inputs into builder ---
    for y in (6.27, 5.57, 4.87, 3.97):
        arrow(ax, (2.7, y), (3.4, 4.5), lw=0.8)
    arrow(ax, (2.7, 3.07), (3.4, 4.3), lw=0.8)
    arrow(ax, (2.7, 2.37), (3.4, 4.1), lw=0.8)

    # builder -> noising
    arrow(ax, (6.0, 5.0), (6.5, 6.1), text="x0", lw=1.0)
    # noising -> backbone
    arrow(ax, (8.1, 5.6), (8.1, 5.1), text="xt", lw=1.0,
          text_offset=(0.2, 0.0))
    # backbone -> head
    arrow(ax, (8.1, 3.5), (8.1, 3.15), lw=1.0)
    # head -> SUBS
    arrow(ax, (7.95, 2.72), (8.25, 2.72), lw=0.9)
    # SUBS -> loss
    arrow(ax, (8.97, 2.3), (8.97, 1.95), lw=1.0)

    # legend
    legend_handles = [
        Patch(facecolor=COL["input"],  edgecolor="#222", label="RNA tokens"),
        Patch(facecolor=COL["input2"], edgecolor="#222", label="Protein (frozen)"),
        Patch(facecolor=COL["input3"], edgecolor="#222", label="Species"),
        Patch(facecolor=COL["input4"], edgecolor="#222", label="Modality tag"),
        Patch(facecolor=COL["noise"],  edgecolor="#222", label="Forward noise (absorbing MASK)"),
        Patch(facecolor=COL["core_dk"], edgecolor="#222", label="Transformer backbone"),
        Patch(facecolor=COL["head"],   edgecolor="#222", label="Output head / parameterization"),
        Patch(facecolor=COL["loss"],   edgecolor="#222", label="Training loss"),
    ]
    ax.legend(handles=legend_handles, loc="lower left",
              bbox_to_anchor=(0.005, 0.005), ncol=4, fontsize=7.8,
              frameon=False, handlelength=1.4, columnspacing=1.0)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 2 — Joint sequence layout + permutation
# ---------------------------------------------------------------------------
def page_layout(pdf):
    fig, ax = new_page(pdf, figsize=(11, 6))
    ax.set_ylim(0, 5)
    title(ax, "Joint sequence layout & per-token conditioning")
    subtitle(ax,
             "Special tokens are pinned (never masked). "
             "Inputs are concatenated, then a per-row permutation is applied "
             "so unpadded tokens are contiguous for FlashAttention varlen.")

    # token strip
    tokens = [
        ("[CLS]",   "S", COL["input3"]),
        ("species", "S", COL["input3"]),
        ("[SEP]",   "S", COL["ghost"]),
        ("p_1",     "P", COL["input2"]),
        ("p_2",     "P", COL["input2"]),
        ("...",     "P", COL["input2"]),
        ("p_Lp",    "P", COL["input2"]),
        ("[SEP]",   "S", COL["ghost"]),
        ("u5_1",    "U", COL["input"]),
        ("u5_2",    "U", COL["input"]),
        ("[SEP]",   "S", COL["ghost"]),
        ("c_1",     "C", COL["input"]),
        ("c_2",     "C", COL["input"]),
        ("c_3",     "C", COL["input"]),
        ("[SEP]",   "S", COL["ghost"]),
        ("u3_1",    "U", COL["input"]),
        ("u3_2",    "U", COL["input"]),
        ("[EOS]",   "S", COL["ghost"]),
    ]
    x0, y0, w, h = 0.4, 3.4, 0.5, 0.55
    for i, (tok, role, c) in enumerate(tokens):
        box(ax, (x0 + i * w, y0), w, h, tok, c, fontsize=7.5, lw=0.5,
            style="round,pad=0.02,rounding_size=0.05")
        ax.text(x0 + i * w + w / 2, y0 - 0.18, role,
                ha="center", va="top", fontsize=7, color="#555")

    ax.annotate("modality_type_ids", xy=(x0, y0 - 0.4),
                xytext=(x0, y0 - 0.4), fontsize=8.5, color="#444")
    ax.annotate(
        "S = special (pinned, special_token_mask=1)   "
        "P = protein-embedding slot   U = UTR nt   C = codon",
        xy=(x0, 2.55), fontsize=8.2, color="#444"
    )

    # arrows showing permutation step
    arrow(ax, (5.0, 3.25), (5.0, 2.30), text="row-wise column permutation\n→ unpad → FlashAttention varlen",
          lw=0.9, fontsize=8.5, text_offset=(2.0, -0.15))

    # second strip: post-perm contiguous
    y1 = 1.55
    for i, (tok, role, c) in enumerate(tokens):
        box(ax, (x0 + i * w, y1), w, h, tok, c, fontsize=7.5, lw=0.5,
            style="round,pad=0.02,rounding_size=0.05")

    # callouts on the right
    ax.text(0.4, 0.95, "Embedding sum at each position:",
            fontsize=9.5, weight="bold")
    ax.text(0.4, 0.65,
            "h_i  =  rna_embed(x_i)                 (RNA positions)\n"
            "     +  modality_embed(m_i) · 1[i ∈ RNA]\n"
            "     ⊕  species_embed(s)               (species slots)\n"
            "     ⊕  proj(LayerNorm(protein_emb))   (protein slots)",
            fontsize=9, family="DejaVu Sans Mono")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 3 — Forward noising (absorbing-state MDLM)
# ---------------------------------------------------------------------------
def page_noising(pdf):
    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0],
                          width_ratios=[1.0, 1.0])
    ax_diag = fig.add_subplot(gs[0, :])
    ax_diag.set_xlim(0, 10); ax_diag.set_ylim(0, 5); ax_diag.set_aspect("equal")
    ax_diag.axis("off")
    title(ax_diag, "Forward process — absorbing-state masked diffusion")
    subtitle(ax_diag,
             "Each RNA token independently jumps to [MASK] with probability "
             "1 − e^{−σ(t)}. Special tokens are pinned.")

    # diagram: x0 -> q(xt|x0) -> xt
    box(ax_diag, (0.4, 1.7), 2.6, 1.4,
        "x0\n[CLS] species ... 5'UTR ... CDS ... 3'UTR ... [EOS]\n(clean tokens)",
        COL["input"], fontsize=9)
    box(ax_diag, (3.7, 1.7), 2.6, 1.4,
        "q(x_t | x_0)\nBernoulli mask per RNA position\np_mask = 1 − e^{−σ(t)}",
        COL["noise"], fontsize=9)
    box(ax_diag, (7.0, 1.7), 2.6, 1.4,
        "x_t\nMixture of original tokens and [MASK]\nfed to the network",
        COL["input"], fontsize=9)
    arrow(ax_diag, (3.0, 2.4), (3.7, 2.4), lw=1.0)
    arrow(ax_diag, (6.3, 2.4), (7.0, 2.4), lw=1.0)

    # Eq box
    ax_diag.text(0.4, 0.9,
                 "t ~ U(ε, 1)            (antithetic sampling enabled)\n"
                 "σ(t)  =  −log(1 − (1−ε) t)            (log-linear)\n"
                 "dσ/dt  =  (1−ε) / (1 − (1−ε) t)\n"
                 "move_chance(t)  =  1 − e^{−σ(t)}            "
                 "(0 on special-token positions)",
                 fontsize=9.5, family="DejaVu Sans Mono")

    # σ(t) plot
    ax_sigma = fig.add_subplot(gs[1, 0])
    t = np.linspace(0, 1, 400)
    eps = 1e-3
    sigma = -np.log1p(-(1 - eps) * t)
    pmask = 1 - np.exp(-sigma)
    ax_sigma.plot(t, sigma, color="#7397c4", lw=1.5, label=r"$\sigma(t)$")
    ax_sigma.plot(t, pmask, color="#c25b5b", lw=1.5, ls="--",
                  label=r"$p_{\mathrm{mask}}(t)=1-e^{-\sigma(t)}$")
    ax_sigma.set_xlabel("t")
    ax_sigma.set_ylabel("value")
    ax_sigma.set_title("Log-linear noise schedule", fontsize=10)
    ax_sigma.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax_sigma.grid(True, alpha=0.3, lw=0.4)
    for s in ("top", "right"):
        ax_sigma.spines[s].set_visible(False)

    # Mask example as a strip evolving with t
    ax_strip = fig.add_subplot(gs[1, 1])
    ax_strip.set_title("Example: same x0 at three t values", fontsize=10)
    ax_strip.set_xlim(0, 20); ax_strip.set_ylim(0, 3.6)
    ax_strip.axis("off")
    rng = np.random.default_rng(0)
    x0 = np.array(list("AUGCUAGCUUACGGAUCCAU"))
    for j, tval in enumerate([0.1, 0.4, 0.85]):
        sig = -np.log1p(-(1 - eps) * tval)
        p = 1 - np.exp(-sig)
        keep = rng.random(len(x0)) >= p
        y = 2.7 - j * 1.05
        ax_strip.text(-0.3, y + 0.25, f"t={tval:>4.2f}", fontsize=8.5)
        for i, ch in enumerate(x0):
            shown = ch if keep[i] else "M"
            fc = COL["input"] if keep[i] else COL["noise"]
            ax_strip.add_patch(Rectangle((i * 0.95, y), 0.85, 0.55,
                                         facecolor=fc, edgecolor="#222", lw=0.4))
            ax_strip.text(i * 0.95 + 0.425, y + 0.27, shown,
                          ha="center", va="center", fontsize=7.5,
                          family="DejaVu Sans Mono")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 4 — Backbone block (Transformer with RoPE + FlashAttention)
# ---------------------------------------------------------------------------
def page_backbone(pdf):
    fig, ax = new_page(pdf)
    title(ax, "Transformer backbone — 24 × pre-LN blocks (~300 M params)")
    subtitle(ax,
             "Bidirectional self-attention (no causal mask). "
             "Varlen FlashAttention on unpadded sequences; "
             "RoPE applied to Q and K only.")

    # ----- left: full stack -----
    sx, sy = 0.5, 0.6
    bw, bh = 1.9, 0.45
    spacing = 0.1
    blocks = [
        ("Input tokens / latents",         COL["ghost"]),
        ("RNA embed + modality embed",     COL["input"]),
        ("Species embed (slot)",           COL["input3"]),
        ("Protein proj (slot, frozen src)", COL["input2"]),
        ("Concatenate + permute + unpad",  COL["core"]),
    ]
    y = 6.5
    last_pt = None
    for name, c in blocks:
        x, yy, w, h = box(ax, (sx, y), bw, bh, name, c, fontsize=8.2)
        if last_pt is not None:
            arrow(ax, last_pt, (x + w / 2, y + h), lw=0.7)
        last_pt = (x + w / 2, y)
        y -= bh + spacing

    # 24 stacked blocks (drawn as a striped column with × N tag)
    stack_x, stack_y, stack_w, stack_h = sx, 4.3, bw, 1.4
    box(ax, (stack_x, stack_y), stack_w, stack_h,
        "Transformer block × 24\n"
        "(see right panel)",
        COL["core_dk"], fontsize=9, weight="bold")
    arrow(ax, last_pt, (stack_x + stack_w / 2, stack_y + stack_h), lw=0.7)

    box(ax, (sx, 3.6), bw, 0.45, "Final LayerNorm", COL["core"], fontsize=8.5)
    arrow(ax, (stack_x + stack_w / 2, stack_y),
          (sx + bw / 2, 3.6 + 0.45), lw=0.7)

    box(ax, (sx, 2.9), bw, 0.45, "Pad → re-permute (inverse)\n→ slice RNA region",
        COL["core"], fontsize=8.0)
    arrow(ax, (sx + bw / 2, 3.6), (sx + bw / 2, 2.9 + 0.45), lw=0.7)

    box(ax, (sx, 2.2), bw, 0.45, "RNA LM head  (Linear → V)", COL["head"], fontsize=8.5)
    arrow(ax, (sx + bw / 2, 2.9), (sx + bw / 2, 2.2 + 0.45), lw=0.7)

    box(ax, (sx, 1.5), bw, 0.45, "SUBS parameterization", COL["head"], fontsize=8.5)
    arrow(ax, (sx + bw / 2, 2.2), (sx + bw / 2, 1.5 + 0.45), lw=0.7)

    box(ax, (sx, 0.8), bw, 0.45, "log p_θ(x0 | xt)", COL["loss"], fontsize=8.5)
    arrow(ax, (sx + bw / 2, 1.5), (sx + bw / 2, 0.8 + 0.45), lw=0.7)

    # ----- right: zoom of one block -----
    ax.add_patch(Rectangle((3.3, 0.5), 6.5, 6.4,
                           facecolor="#fafafa", edgecolor="#bbb", lw=0.5))
    ax.text(6.55, 6.78, "Zoom: one Transformer block", ha="center",
            fontsize=11, weight="bold")

    bx, by = 3.6, 0.9
    bw2, bh2 = 5.9, 0.5
    items = [
        ("residual in (fp32)",                        COL["ghost"]),
        ("LayerNorm (fused, pre-norm)",               COL["core"]),
        ("Linear  →  QKV  (no bias)",                  COL["core"]),
        ("Rotary embed (cos,sin) on Q and K  (V skipped)", COL["core"]),
        ("FlashAttention  varlen,  bidirectional,  dropout=0", COL["core_dk"]),
        ("Linear  →  out  (no bias)",                  COL["core"]),
        ("+ residual",                                COL["ghost"]),
        ("LayerNorm (fused, pre-norm)",               COL["core"]),
        ("Linear → 4·d  →  GELU(tanh)  →  Linear → d", COL["core"]),
        ("+ residual",                                COL["ghost"]),
    ]
    y = by + (len(items) - 1) * (bh2 + 0.05)
    prev = None
    for name, c in items:
        x, yy, w, h = box(ax, (bx, y), bw2, bh2, name, c, fontsize=8.6)
        if prev is not None:
            arrow(ax, prev, (x + w / 2, y + h), lw=0.7)
        prev = (x + w / 2, y)
        y -= bh2 + 0.05

    # config table
    ax.text(3.6, 6.55,
            "hidden = 1024     n_heads = 8     head_dim = 128     "
            "n_blocks = 24     mlp_ratio = 4     vocab pad multiple = 8",
            fontsize=8.5, color="#333")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 5 — SUBS head + loss + reverse sampling
# ---------------------------------------------------------------------------
def page_loss_and_sampling(pdf):
    fig, ax = new_page(pdf)
    title(ax, "SUBS head, loss decomposition, and reverse sampling")
    subtitle(ax,
             "SUBS forces the predictor onto valid clean tokens; "
             "unmasked positions are pinned to identity. "
             "The DDPM-cache sampler iteratively denoises from a fully-masked prior.")

    # ---- SUBS head box ----
    box(ax, (0.4, 5.0), 4.6, 1.8,
        "SUBS parameterization (per token)\n\n"
        "1.  logits[mask_idx] ← −∞                 (never predict MASK)\n"
        "2.  logits ← logits − logsumexp(logits)   (log-softmax)\n"
        "3.  if x_t[i] ≠ MASK:\n"
        "        logits[i, :] ← −∞;  logits[i, x_t[i]] ← 0\n"
        "    (already-revealed tokens are pinned to themselves)",
        COL["head"], fontsize=8.6)

    # ---- Loss box ----
    box(ax, (5.3, 5.0), 4.3, 1.8,
        "Per-token NLL (continuous-time MDLM ELBO)\n\n"
        "ℓ_i  =  − log p_θ(x0,i | x_t) · (dσ/dt) / (e^{σ(t)} − 1)\n\n"
        "Aggregated separately on UTR5 / CDS / UTR3 masks;\n"
        "final loss is the mean of the three region-NLLs.\n"
        "Validation also tracks per-region perplexity\n"
        "and codon→amino-acid translation error.",
        COL["loss"], fontsize=8.6)

    # ---- Reverse sampling diagram ----
    ax.text(0.4, 4.55, "Reverse process — DDPM-cache sampler (128 steps)",
            fontsize=11, weight="bold")

    # strip showing denoising over time
    n_panels = 5
    strip_x, strip_y = 0.4, 2.7
    panel_w, panel_h = 1.7, 1.2
    gap = 0.2
    rng = np.random.default_rng(7)
    # build a target sequence and progressively reveal
    final = np.array(list("AUGGCCUAACGUAUGCCUAA"))
    t_vals = [1.0, 0.75, 0.5, 0.25, 0.0]
    revealed_mask = np.zeros(len(final), dtype=bool)
    # decide reveal order
    order = rng.permutation(len(final))
    for j, tval in enumerate(t_vals):
        keep_count = int(round((1 - tval) * len(final)))
        revealed_mask = np.zeros(len(final), dtype=bool)
        revealed_mask[order[:keep_count]] = True

        x = strip_x + j * (panel_w + gap)
        ax.add_patch(Rectangle((x, strip_y), panel_w, panel_h,
                               facecolor="#fafafa", edgecolor="#bbb", lw=0.5))
        ax.text(x + panel_w / 2, strip_y + panel_h - 0.15,
                f"t = {tval:.2f}", ha="center", fontsize=8.5, weight="bold")
        # draw token boxes (two rows of 10)
        for k in range(len(final)):
            r, c = divmod(k, 10)
            tx = x + 0.07 + c * 0.16
            ty = strip_y + panel_h - 0.55 - r * 0.32
            shown = final[k] if revealed_mask[k] else "M"
            fc = COL["input"] if revealed_mask[k] else COL["noise"]
            ax.add_patch(Rectangle((tx, ty), 0.15, 0.27,
                                   facecolor=fc, edgecolor="#222", lw=0.3))
            ax.text(tx + 0.075, ty + 0.135, shown, ha="center", va="center",
                    fontsize=6.5, family="DejaVu Sans Mono")

        if j < n_panels - 1:
            arrow(ax,
                  (x + panel_w + 0.02, strip_y + panel_h / 2),
                  (x + panel_w + gap - 0.02, strip_y + panel_h / 2),
                  lw=0.8)

    # algorithm pseudocode
    ax.text(0.4, 2.3, "Algorithm (sampling.predictor = ddpm_cache)",
            fontsize=10, weight="bold")
    ax.text(0.4, 0.4,
            "x ← all-MASK prior over RNA positions (specials fixed)\n"
            "for k = K..1:\n"
            "    t = k / K,    s = (k-1) / K\n"
            "    p_θ(x0 | x)  ←  SUBS_head(backbone(x, batch))   # cached if unchanged\n"
            "    sample x_s  ~  q(x_s | x_t, x0=p_θ)  using  σ(t), σ(s)\n"
            "    x ← x_s\n"
            "return x  (apply final noise_removal step if configured)",
            fontsize=9, family="DejaVu Sans Mono",
            verticalalignment="bottom")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    out = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       "..", "claude-notes",
                                       "mdlm_architecture.pdf"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with PdfPages(out) as pdf:
        page_overview(pdf)
        page_layout(pdf)
        page_noising(pdf)
        page_backbone(pdf)
        page_loss_and_sampling(pdf)
        d = pdf.infodict()
        d["Title"]    = "MDLM joint-sequence diffusion — architecture"
        d["Author"]   = "Auto-generated from jsm/* source"
        d["Subject"]  = "Absorbing-state MDLM with species / protein / modality conditioning"
        d["Keywords"] = "MDLM, masked diffusion, RNA, joint sequence modeling"
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
