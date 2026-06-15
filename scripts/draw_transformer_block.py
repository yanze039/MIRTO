"""Render a publication-style vector PDF of the Transformer block used in the
MDLM joint-sequence backbone.

Faithful to jsm/models/transformer.py :: TransformerBlock.forward:
    - pre-norm (fused layer_norm_fn), residual kept in fp32
    - QKV projection (bias-free), packed (ts, 3, h, d_h)
    - RoPE applied to Q and K only (V skipped)
    - varlen bidirectional FlashAttention, dropout = 0
    - out-projection (bias-free)
    - second pre-norm
    - MLP: Linear(d → 4d) → GELU(tanh) → Linear(4d → d), bias=True

Output: claude-notes/transformer_block.pdf  (single page, landscape, vector)
"""

import os
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from types import SimpleNamespace

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
    "font.family":  "DejaVu Sans",
    "font.size":    9.0,
    "axes.linewidth": 0.5,
    "mathtext.fontset": "cm",
})

C = {
    "norm":    "#e9eef5",
    "lin":     "#dde7f2",
    "rope":    "#f5e2c8",
    "attn_dk": "#7e9fc4",
    "mlp":     "#dbe9d4",
    "act":     "#cde0c0",
    "add":     "#f3d9d4",
    "edge":    "#1f1f1f",
    "resid":   "#9b2d20",
    "spec":    "#666",
    "frame":   "#cccccc",
}


def draw_box(ax, x, y, w, h, text, fc, fs=8.5, ec=C["edge"], lw=0.6,
             weight="normal"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.03,rounding_size=0.06",
                       linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight)
    return SimpleNamespace(x=x, y=y, w=w, h=h,
                           cx=x + w / 2, top=y + h, bot=y,
                           left=x, right=x + w)


def arrow(ax, p0, p1, lw=0.9, color=C["edge"], ms=10):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                                 mutation_scale=ms, color=color,
                                 linewidth=lw))


def line(ax, p0, p1, lw=0.9, color=C["edge"]):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-",
                                 color=color, linewidth=lw))


def circle_op(ax, xy, txt, r=0.16, fc=C["add"]):
    cx, cy = xy
    ax.add_patch(plt.Circle((cx, cy), r, facecolor=fc,
                            edgecolor=C["edge"], linewidth=0.7))
    ax.text(cx, cy, txt, ha="center", va="center", fontsize=11, weight="bold")
    return SimpleNamespace(cx=cx, cy=cy, r=r,
                           top=cy + r, bot=cy - r,
                           left=cx - r, right=cx + r)


def shape_lbl(ax, x, y, txt):
    ax.text(x, y, txt, fontsize=7.5, color=C["spec"], ha="right", va="center",
            family="DejaVu Sans Mono")


# ---------------------------------------------------------------------------
def draw_block(ax, col_x=3.2, top=8.55, bw=2.9):
    """Left panel — vertical block diagram. Stores each shape's coords so
    arrows are always anchored to actual box edges."""

    bh = 0.42

    # residual stream x (right of column)
    rx = col_x + bw / 2 + 1.40

    # We'll lay out boxes top→bottom with explicit gaps.
    gaps = {
        "after_input": 0.55,
        "default":     0.40,
        "small":       0.30,
    }

    # ---- input ----
    cur_y = top
    b_in = draw_box(ax, col_x - bw/2, cur_y, bw, bh,
                    "input  from previous block", C["norm"], fs=8.0)
    shape_lbl(ax, col_x - bw/2 - 0.05, cur_y + bh/2, "(T, d)")

    # branch residual to the right at the input
    ry_top = b_in.bot + bh/2
    line(ax,  (b_in.right, ry_top), (rx, ry_top), color=C["resid"])
    ax.text(rx + 0.10, ry_top, "residual r  (fp32)",
            color=C["resid"], fontsize=7.5, va="center")

    def step(prev_box, gap, label, fc, fs=8.5, weight="normal",
             extra_lbl=None, h=None):
        h = h or bh
        y = prev_box.bot - gap - h
        b = draw_box(ax, col_x - bw/2, y, bw, h, label, fc, fs=fs,
                     weight=weight)
        arrow(ax, (col_x, prev_box.bot), (col_x, b.top))
        if extra_lbl is not None:
            shape_lbl(ax, col_x - bw/2 - 0.05, y + h/2, extra_lbl)
        return b

    # ---- LN 1 ----
    b_ln1 = step(b_in,  gaps["after_input"],
                 "LayerNorm  (fused, pre-norm)", C["norm"],
                 extra_lbl="(T, d)")

    # ---- QKV proj ----
    b_qkv = step(b_ln1, gaps["default"],
                 "Linear  →  QKV     (bias = False)", C["lin"],
                 extra_lbl="(T, 3·d)")

    # ---- rearrange ----
    b_rea = step(b_qkv, gaps["small"],
                 "rearrange  →  (T, 3, h, d_h)", C["lin"],
                 extra_lbl="")

    # ---- RoPE ----
    b_rope = step(b_rea, gaps["small"],
                  "RoPE on  Q  and  K   only      (V skipped)",
                  C["rope"], extra_lbl="Q,K,V : (T, h, d_h)")

    # ---- FlashAttention (taller) ----
    b_attn = step(b_rope, gaps["default"],
                  "FlashAttention  varlen, bidirectional\n"
                  "(cu_seqlens, max_seqlen,  dropout = 0)",
                  C["attn_dk"], fs=8.4, weight="bold",
                  extra_lbl="(T, h, d_h)", h=0.62)

    # ---- out proj ----
    b_out = step(b_attn, gaps["default"],
                 "Linear  →  out     (bias = False)", C["lin"],
                 extra_lbl="(T, d)")

    # ---- first residual add  +  ----
    add1_y = b_out.bot - 0.45
    add1 = circle_op(ax, (col_x, add1_y), "+")
    arrow(ax, (col_x, b_out.bot), (col_x, add1.top))

    # residual merges from the right
    line(ax,  (rx, ry_top), (rx, add1.cy), color=C["resid"])
    arrow(ax, (rx, add1.cy), (add1.right, add1.cy), color=C["resid"])

    # branch new residual back out to the right
    ry_mid = add1.bot - 0.10
    line(ax, (col_x, add1.bot), (col_x, ry_mid - 0.05))
    line(ax, (col_x, ry_mid), (rx, ry_mid), color=C["resid"])
    ax.text(rx + 0.10, ry_mid, r"r  +=  AttnOut",
            color=C["resid"], fontsize=7.3, va="center",
            family="DejaVu Sans Mono")

    # placeholder box just below the +
    class P:  # tiny shim so step() can keep its protocol
        pass
    p = P(); p.bot = ry_mid

    # ---- LN 2 ----
    b_ln2 = step(p, gaps["small"],
                 "LayerNorm  (fused, pre-norm)", C["norm"],
                 extra_lbl="(T, d)")

    # ---- MLP up ----
    b_mlp1 = step(b_ln2, gaps["default"],
                  "Linear  →  4·d         (bias = True)", C["mlp"],
                  extra_lbl="(T, 4·d)")

    # ---- GELU ----
    b_act = step(b_mlp1, gaps["small"],
                 "GELU  (approximate = tanh)", C["act"], extra_lbl="")

    # ---- MLP down ----
    b_mlp2 = step(b_act, gaps["small"],
                  "Linear  →  d            (bias = True)", C["mlp"],
                  extra_lbl="(T, d)")

    # ---- second residual add  +  ----
    add2_y = b_mlp2.bot - 0.45
    add2 = circle_op(ax, (col_x, add2_y), "+")
    arrow(ax, (col_x, b_mlp2.bot), (col_x, add2.top))

    # residual line continues down on the right then in to +
    line(ax,  (rx, ry_mid), (rx, add2.cy), color=C["resid"])
    arrow(ax, (rx, add2.cy), (add2.right, add2.cy), color=C["resid"])

    # ---- output ----
    out_y = add2.bot - 0.45
    line(ax, (col_x, add2.bot), (col_x, out_y + bh))
    b_outbox = draw_box(ax, col_x - bw/2, out_y, bw, bh,
                        "output  to next block",
                        C["norm"], fs=8.0)
    shape_lbl(ax, col_x - bw/2 - 0.05, out_y + bh/2, "(T, d)")

    ax.text(rx + 0.10, add2.cy - 0.30, r"r  +=  MLP(LN(r))",
            color=C["resid"], fontsize=7.3, va="center",
            family="DejaVu Sans Mono")

    # block-section header (centred above the column, clear of header text)
    ax.text(col_x, top + 0.80,
            "Block diagram   ( ×N = 24 )",
            ha="center", fontsize=11.5, weight="bold")

    return b_outbox.bot   # bottom y of the column


# ---------------------------------------------------------------------------
def draw_details(ax, x0=7.5, x_right=13.6, top=8.95, bot=0.50):
    """Right panel — equations, shape table, config, notes."""
    ax.add_patch(Rectangle((x0 - 0.10, bot), x_right - x0 + 0.20, top - bot,
                           facecolor="#fbfbfb", edgecolor=C["frame"],
                           linewidth=0.5))

    y = top - 0.20
    ax.text(x0 + 0.10, y, "Equations  (single block)",
            fontsize=11, weight="bold")
    y -= 0.32

    eqs = [
        r"$\tilde{x}_1 \;=\; \mathrm{LN}_1(\,r + x\,),\;\;\;\;\;r \leftarrow r + x$",
        r"$[Q,\,K,\,V] \;=\; W_{qkv}\,\tilde{x}_1\;\;\;(W_{qkv}\!\in\!\mathbb{R}^{3d\times d},\;\text{no bias})$",
        r"$Q,K \leftarrow \mathrm{RoPE}(Q,K)\;\;\;(V\;\text{unrotated})$",
        r"$A \;=\; \mathrm{FlashAttn}_{\mathrm{varlen}}(Q,K,V),\;\;\mathrm{causal}=\mathrm{False}$",
        r"$u \;=\; W_o\,A\;\;\;(W_o\!\in\!\mathbb{R}^{d\times d},\;\text{no bias})$",
        r"$\tilde{x}_2 \;=\; \mathrm{LN}_2(\,r + u\,),\;\;\;\;\;r \leftarrow r + u$",
        r"$m \;=\; W_2\,\mathrm{GELU}(W_1\,\tilde{x}_2 + b_1) + b_2$",
        r"$x_{\mathrm{out}} = m;\;\;\;r\leftarrow r + m\;\;(\text{added in next block's }\mathrm{LN})$",
    ]
    for e in eqs:
        ax.text(x0 + 0.25, y, e, fontsize=9.0, va="top")
        y -= 0.34

    y -= 0.05
    ax.text(x0 + 0.10, y, "Tensor shapes  (varlen-packed, T = Σ valid tokens)",
            fontsize=10.5, weight="bold")
    y -= 0.30
    shapes_tbl = [
        ("input  /  residual",   "(T, d)"),
        ("QKV projection",       "(T, 3·d)  →  (T, 3, h, d_h)"),
        ("Q,  K,  V",            "(T, h, d_h)     h = 8,  d_h = d/h"),
        ("attention output",     "(T, h, d_h)  →  (T, d)"),
        ("MLP hidden",           "(T, 4·d)"),
        ("block output",         "(T, d)"),
    ]
    for k, v in shapes_tbl:
        ax.text(x0 + 0.25, y, k, fontsize=8.4, va="top")
        ax.text(x0 + 2.55, y, v, fontsize=8.2, va="top",
                family="DejaVu Sans Mono", color="#333")
        y -= 0.26

    y -= 0.05
    ax.text(x0 + 0.10, y, "Configuration  (transformer_300M.yaml)",
            fontsize=10.5, weight="bold")
    y -= 0.30
    cfg = [
        ("hidden size  d",       "1024"),
        ("# heads  h",           "8"),
        ("head dim  d_h",        "128"),
        ("# blocks  N",          "24"),
        ("MLP ratio",            "4   (hidden = 4096)"),
        ("attention dropout",    "0.0"),
        ("residual precision",   "fp32"),
        ("training precision",   "bf16-mixed"),
        ("parameters",           "≈ 300 M"),
    ]
    for k, v in cfg:
        ax.text(x0 + 0.25, y, k, fontsize=8.4, va="top")
        ax.text(x0 + 2.55, y, v, fontsize=8.2, va="top",
                family="DejaVu Sans Mono", color="#333")
        y -= 0.24

    y -= 0.05
    ax.text(x0 + 0.10, y, "Implementation notes",
            fontsize=10.5, weight="bold")
    y -= 0.28
    notes = [
        ("Pre-LN with fused LayerNorm + residual add",
         "(flash-attn  layer_norm_fn,  prenorm = True)."),
        ("Residual stream kept in fp32 even when activations",
         "run in bf16 — stabilises 24-deep sums."),
        ("RoPE (base = 10 000, max_len = 65 536) on Q, K only;",
         "V is left unrotated."),
        ("Variable-length FlashAttention kernel on tokens un-",
         "padded by get_unpad_data — no padding inside T."),
        ("Attention is bidirectional (causal = False), as",
         "required by the masked-diffusion training objective."),
        ("Init:  Xavier-uniform on  W_qkv,  W_1;",
         "N(0, σ²)  with  σ = 0.02 / √(2N)  on  W_o,  W_2."),
    ]
    for line1, line2 in notes:
        ax.text(x0 + 0.25, y,         "•  " + line1, fontsize=8.2, va="top")
        ax.text(x0 + 0.45, y - 0.22,  line2,          fontsize=8.2, va="top")
        y -= 0.52


# ---------------------------------------------------------------------------
def main():
    out = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       "..", "claude-notes",
                                       "transformer_block.pdf"))
    os.makedirs(os.path.dirname(out), exist_ok=True)

    with PdfPages(out) as pdf:
        fig, ax = plt.subplots(figsize=(13.5, 10.0))   # landscape, generous
        ax.set_xlim(0, 13.8)
        ax.set_ylim(0, 10.3)
        ax.set_aspect("equal")
        ax.axis("off")

        # Page header — kept clear of the block title (y = 9.10)
        ax.text(6.9, 10.05,
                "Transformer block of the MDLM joint-sequence backbone",
                ha="center", fontsize=13.5, weight="bold")
        ax.text(6.9, 9.78,
                "pre-LN, RoPE on Q/K, varlen bidirectional FlashAttention,  "
                "GELU MLP (×4 widening),  fp32 residual stream",
                ha="center", fontsize=9.5, color="#444", style="italic")

        draw_block(ax)
        draw_details(ax)

        ax.text(0.02, 0.012,
                "Source: jsm/models/transformer.py  ::  "
                "TransformerBlock,  TransformerBackbone",
                transform=ax.transAxes, fontsize=7.5, color="#888")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        d = pdf.infodict()
        d["Title"]    = "MDLM transformer block"
        d["Author"]   = "Auto-generated from jsm/models/transformer.py"
        d["Subject"]  = "Pre-norm Transformer block with RoPE and varlen FlashAttention"
        d["Keywords"] = "Transformer, MDLM, RoPE, FlashAttention, pre-norm"

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
