"""
================================================================================
  LINEAR ALGEBRA AND ITS APPLICATIONS (LAA) — IMAGE PROCESSING PROJECT
================================================================================

Author  : LAA Project
Purpose : Demonstrate core linear algebra concepts applied to image processing
          covering SVD compression, denoising, matrix transformations,
          eigenanalysis, and PCA.

Mathematical Foundations Covered
─────────────────────────────────
  • Matrix representation of images
  • Singular Value Decomposition (SVD)  →  A = U Σ Vᵀ
  • Low-rank approximation             →  Aₖ = Uₖ Σₖ Vₖᵀ
  • Noise modelling and SVD-based denoising
  • Affine transformation matrices (rotation, scaling, shear)
  • Eigendecomposition                 →  Av = λv
  • Principal Component Analysis (PCA) via covariance eigendecomposition

Libraries : NumPy, Pillow (PIL), OpenCV (cv2), Matplotlib, SciPy, Time
================================================================================
"""

# ─── Standard / Third-party imports ──────────────────────────────────────────
import os
import sys
import time
import argparse
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")                    # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches

from PIL import Image, ImageDraw
import cv2
from scipy.ndimage import gaussian_filter

# ─── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#e6edf3",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.6,
    "axes.titlecolor":  "#f0f6fc",
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
    "figure.titlesize": 13,
    "figure.titleweight": "bold",
    "font.family":      "DejaVu Sans",
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"#0d1117",
})

ACCENT   = "#58a6ff"   # blue highlight
SUCCESS  = "#3fb950"   # green
WARNING  = "#d29922"   # amber
ERROR    = "#f85149"   # red
MUTED    = "#8b949e"   # grey
PURPLE   = "#bc8cff"

OUTPUT_DIR = "laa_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 0 — SYNTHETIC IMAGE GENERATION
#  (Provides self-contained demo images so the script runs with zero
#   external files.)
# ══════════════════════════════════════════════════════════════════════════════

def generate_demo_images(size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesise a grayscale and an RGB demo image using NumPy geometry.

    Mathematical idea
    ─────────────────
    Each pixel value is computed from a closed-form expression over a
    coordinate grid — equivalent to sampling a continuous 2-D function on a
    regular lattice.  The result is a matrix (2-D array for gray, 3-D tensor
    for RGB) whose entries lie in [0, 255].

    Returns
    -------
    gray_img : ndarray, shape (size, size),        dtype uint8
    rgb_img  : ndarray, shape (size, size, 3),     dtype uint8
    """
    print("[INFO] Generating synthetic demo images …")

    x = np.linspace(-3, 3, size)          # 1-D sample grid
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)              # 2-D coordinate matrices

    # ── Grayscale: superposition of sinusoids + radial Gaussian
    #    f(x,y) = sin(πx)·cos(πy) + exp(-(x²+y²)/4)
    #    This creates a smooth pattern rich in spatial frequency content —
    #    ideal for demonstrating SVD rank truncation.
    gray_raw = (
        0.5 * np.sin(np.pi * X) * np.cos(np.pi * Y)
        + 0.3 * np.exp(-(X**2 + Y**2) / 4)
        + 0.2 * np.sin(2 * np.pi * (X + Y) / 3)
    )
    gray_img = _normalise_to_uint8(gray_raw)

    # ── RGB: three independent channels with different frequency content
    r_raw = 0.6 * np.sin(np.pi * X) * np.cos(0.5 * np.pi * Y) + \
            0.4 * np.exp(-((X - 1)**2 + (Y - 1)**2) / 2)
    g_raw = 0.5 * np.cos(np.pi * Y) + \
            0.5 * np.exp(-((X + 1)**2 + (Y + 1)**2) / 3)
    b_raw = 0.7 * np.sin(2 * np.pi * X / 3) * np.sin(2 * np.pi * Y / 3) + \
            0.3 * np.exp(-(X**2 + Y**2) / 5)

    rgb_img = np.stack([
        _normalise_to_uint8(r_raw),
        _normalise_to_uint8(g_raw),
        _normalise_to_uint8(b_raw),
    ], axis=-1)

    return gray_img, rgb_img


def _normalise_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Linear normalisation  →  [min, max]  ⟹  [0, 255].

    Mathematically:  out = 255 · (arr − min) / (max − min)
    This is a linear (affine) transformation of the value range.
    """
    lo, hi = arr.min(), arr.max()
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — IMAGE REPRESENTATION
# ══════════════════════════════════════════════════════════════════════════════

def load_and_represent_image(
    gray_img: np.ndarray,
    rgb_img:  np.ndarray,
) -> None:
    """
    Illustrate how a digital image IS a matrix (or tensor).

    ┌──────────────────────────────────────────────────────────────────┐
    │  MATHEMATICAL CONCEPT — Image as a Matrix                        │
    │                                                                  │
    │  Grayscale image  →  A ∈ ℝ^(m×n)                                │
    │     A[i,j] = intensity of pixel at row i, column j              │
    │     Values: 0 (black) … 255 (white)                              │
    │                                                                  │
    │  RGB image        →  T ∈ ℝ^(m×n×3)                              │
    │     T[:,:,0] = Red channel matrix                                │
    │     T[:,:,1] = Green channel matrix                              │
    │     T[:,:,2] = Blue channel matrix                               │
    │                                                                  │
    │  Every image processing operation is a linear (or affine)        │
    │  transformation on these matrices.                               │
    └──────────────────────────────────────────────────────────────────┘
    """
    print("\n" + "═" * 60)
    print("  SECTION 1 — IMAGE REPRESENTATION")
    print("═" * 60)

    m, n = gray_img.shape
    print(f"  Grayscale matrix shape : {m} × {n}  ({m*n:,} pixels)")
    print(f"  RGB tensor shape       : {rgb_img.shape}  "
          f"({m*n*3:,} values)")
    print(f"  Grayscale rank (NumPy) : {np.linalg.matrix_rank(gray_img.astype(float))}")
    print(f"  Memory (grayscale)     : {gray_img.nbytes / 1024:.1f} KB")
    print(f"  Memory (RGB)           : {rgb_img.nbytes / 1024:.1f} KB")

    # ── Visualise a small pixel-value patch as numbers ──────────────────────
    patch_size = 8
    patch = gray_img[:patch_size, :patch_size]

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("Section 1 — Image Representation: Images as Matrices",
                 color="#f0f6fc", y=1.01)

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel A — grayscale image
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(gray_img, cmap="gray", vmin=0, vmax=255)
    ax0.set_title(f"Grayscale  A ∈ ℝ^({m}×{n})")
    ax0.axis("off")
    _add_subtitle(ax0, "Each pixel = one matrix entry ∈ [0, 255]")

    # Panel B — RGB image
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(rgb_img)
    ax1.set_title(f"RGB Tensor  T ∈ ℝ^({m}×{n}×3)")
    ax1.axis("off")
    _add_subtitle(ax1, "3 channel matrices stacked along depth axis")

    # Panel C — numeric pixel patch
    ax2 = fig.add_subplot(gs[2])
    ax2.axis("off")
    ax2.set_title(f"Pixel Matrix (top-left {patch_size}×{patch_size} patch)")
    _add_subtitle(ax2, "A[i,j] shown as text; colour = intensity")

    cell_w = 1.0 / patch_size
    cell_h = 1.0 / patch_size
    for i in range(patch_size):
        for j in range(patch_size):
            val = patch[i, j]
            norm = val / 255.0
            bg = (norm * 0.6, norm * 0.6, norm * 0.6 + 0.2 * norm)
            fc = "white" if norm < 0.5 else "black"
            rect = mpatches.FancyBboxPatch(
                (j * cell_w, 1 - (i + 1) * cell_h),
                cell_w, cell_h,
                boxstyle="square,pad=0",
                facecolor=bg,
                edgecolor="#30363d",
                lw=0.4,
                transform=ax2.transAxes,
                clip_on=True,
            )
            ax2.add_patch(rect)
            ax2.text(
                (j + 0.5) * cell_w, 1 - (i + 0.5) * cell_h,
                str(val),
                ha="center", va="center",
                fontsize=5.5, color=fc,
                transform=ax2.transAxes,
            )

    _save(fig, "01_image_representation.png")
    print(f"  [✓] Saved  →  {OUTPUT_DIR}/01_image_representation.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — SVD COMPRESSION
# ══════════════════════════════════════════════════════════════════════════════

def svd_compress(
    gray_img:    np.ndarray,
    k_values:    list[int] | None = None,
    show_energy: bool = True,
) -> dict:
    """
    Compress a grayscale image via SVD low-rank approximation.

    ┌──────────────────────────────────────────────────────────────────┐
    │  MATHEMATICAL CONCEPT — Singular Value Decomposition (SVD)       │
    │                                                                  │
    │  Every real matrix A ∈ ℝ^(m×n) admits the decomposition:        │
    │                                                                  │
    │         A  =  U  Σ  Vᵀ                                          │
    │                                                                  │
    │  where                                                           │
    │    U ∈ ℝ^(m×m)  — left singular vectors  (orthonormal cols)     │
    │    Σ ∈ ℝ^(m×n)  — diagonal matrix, σ₁ ≥ σ₂ ≥ … ≥ σᵣ ≥ 0      │
    │    V ∈ ℝ^(n×n)  — right singular vectors (orthonormal cols)     │
    │                                                                  │
    │  Rank-k approximation (best possible in Frobenius norm):         │
    │                                                                  │
    │         Aₖ  =  Σᵢ₌₁ᵏ  σᵢ · uᵢ · vᵢᵀ                          │
    │                                                                  │
    │  ▸ Each term σᵢ · uᵢ · vᵢᵀ is a rank-1 matrix (outer product)  │
    │  ▸ Eckart–Young theorem: Aₖ minimises ‖A − B‖_F over all        │
    │    rank-k matrices B.                                            │
    │  ▸ Compression ratio ≈  k(m + n + 1) / (mn)                     │
    └──────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    gray_img    : uint8 grayscale image matrix
    k_values    : list of k (number of singular values to retain)
    show_energy : plot singular value spectrum

    Returns
    -------
    dict mapping k → reconstructed image (uint8 ndarray)
    """
    print("\n" + "═" * 60)
    print("  SECTION 2 — SVD IMAGE COMPRESSION")
    print("═" * 60)

    if k_values is None:
        k_values = [1, 5, 15, 30, 60, 100]

    A = gray_img.astype(np.float64)       # work in float64 for accuracy
    m, n = A.shape

    t0 = time.perf_counter()
    # numpy's SVD — economy (thin) decomposition for efficiency
    # full_matrices=False gives U:(m×r), Σ:(r,), Vt:(r×n) where r=min(m,n)
    U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
    t_svd = time.perf_counter() - t0

    print(f"  Image shape          : {m} × {n}")
    print(f"  Number of sing. vals : {len(sigma)}")
    print(f"  σ₁ (largest)         : {sigma[0]:.2f}")
    print(f"  σ_r (smallest)       : {sigma[-1]:.6f}")
    print(f"  SVD wall-clock time  : {t_svd*1000:.1f} ms")

    # ── Cumulative energy captured by top-k singular values ────────────────
    #   Energy fraction = (Σᵢ₌₁ᵏ σᵢ²) / (Σᵢ σᵢ²)
    #   This is the fraction of total Frobenius norm² preserved.
    energy_total = np.sum(sigma**2)
    cum_energy   = np.cumsum(sigma**2) / energy_total

    # Find k for 90%, 95%, 99% energy thresholds
    for pct in (0.90, 0.95, 0.99):
        k_thresh = int(np.searchsorted(cum_energy, pct)) + 1
        ratio = k_thresh * (m + n + 1) / (m * n) * 100
        print(f"  k for {pct*100:.0f}% energy      : {k_thresh:4d}  "
              f"(compression ratio ≈ {ratio:.1f}%)")

    # ── Reconstruct images at each k ────────────────────────────────────────
    compressed = {}
    metrics    = {}

    for k in k_values:
        k = min(k, len(sigma))              # clamp to available rank

        # Rank-k reconstruction:  Aₖ = U[:, :k] · diag(σ[:k]) · Vt[:k, :]
        # This is equivalent to summing k outer products: Σᵢ σᵢ uᵢ vᵢᵀ
        Ak = (U[:, :k] * sigma[:k]) @ Vt[:k, :]   # shape (m, n)

        # Clip to valid pixel range then cast back to uint8
        Ak_uint8 = np.clip(Ak, 0, 255).astype(np.uint8)
        compressed[k] = Ak_uint8

        # ── Quality metrics ──────────────────────────────────────────────────
        # PSNR = 10 log₁₀(MAX² / MSE)  — higher is better
        # SSIM — structural similarity (simplified)
        mse  = np.mean((A - Ak) ** 2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")
        comp_ratio = k * (m + n + 1) / (m * n) * 100
        metrics[k] = {
            "mse":        mse,
            "psnr":       psnr,
            "energy_pct": cum_energy[k - 1] * 100,
            "comp_ratio": comp_ratio,
        }
        print(f"  k={k:4d} | PSNR={psnr:6.2f} dB | "
              f"energy={cum_energy[k-1]*100:6.2f}% | "
              f"ratio={comp_ratio:.1f}%")

    # ──────────────────────────────────────────────────────────────────────
    # PLOT A — Singular Value Spectrum
    # ──────────────────────────────────────────────────────────────────────
    if show_energy:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle("Section 2a — Singular Value Spectrum  (A = UΣVᵀ)")

        ax = axes[0]
        k_plot = min(120, len(sigma))
        ax.bar(range(1, k_plot + 1), sigma[:k_plot],
               color=ACCENT, width=0.9, alpha=0.75)
        ax.set_xlabel("Singular value index  i")
        ax.set_ylabel("σᵢ")
        ax.set_title("Singular Values  σ₁ ≥ σ₂ ≥ … ≥ σᵣ ≥ 0")
        _add_subtitle(ax,
            "Large early values encode global structure; "
            "small later values encode fine detail/noise")
        ax.grid(axis="y", alpha=0.4)

        ax = axes[1]
        ax.plot(range(1, len(sigma) + 1), cum_energy * 100,
                color=ACCENT, lw=2)
        for pct in (0.90, 0.95, 0.99):
            k_t = int(np.searchsorted(cum_energy, pct)) + 1
            ax.axhline(pct * 100, color=WARNING, lw=0.8, ls="--", alpha=0.7)
            ax.axvline(k_t, color=SUCCESS, lw=0.8, ls=":", alpha=0.7)
            ax.text(k_t + 1, pct * 100 - 1.5, f"k={k_t}",
                    color=SUCCESS, fontsize=7)
        ax.set_xlabel("k (number of singular values retained)")
        ax.set_ylabel("Cumulative energy captured (%)")
        ax.set_title("Energy Fraction  =  (Σᵢ₌₁ᵏ σᵢ²) / (Σᵢ σᵢ²)")
        _add_subtitle(ax, "Eckart–Young: rank-k truncation is OPTIMAL "
                          "in Frobenius / spectral norm")
        ax.grid(alpha=0.3)

        _save(fig, "02a_svd_spectrum.png")
        print(f"  [✓] Saved  →  {OUTPUT_DIR}/02a_svd_spectrum.png")

    # ──────────────────────────────────────────────────────────────────────
    # PLOT B — Compressed images side-by-side
    # ──────────────────────────────────────────────────────────────────────
    show_k = [k for k in k_values if k <= min(100, len(sigma))][:6]
    ncols  = len(show_k) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3.8))
    fig.suptitle("Section 2b — SVD Low-Rank Approximations  "
                 "Aₖ = Uₖ Σₖ Vₖᵀ")

    axes[0].imshow(gray_img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original\n(full rank)", color=SUCCESS)
    axes[0].axis("off")

    for ax, k in zip(axes[1:], show_k):
        ax.imshow(compressed[k], cmap="gray", vmin=0, vmax=255)
        m_ = metrics[k]
        ax.set_title(f"k = {k}\nPSNR={m_['psnr']:.1f} dB")
        ax.axis("off")
        _add_subtitle(ax, f"energy={m_['energy_pct']:.1f}%  "
                          f"ratio={m_['comp_ratio']:.1f}%")

    _save(fig, "02b_svd_compression.png")
    print(f"  [✓] Saved  →  {OUTPUT_DIR}/02b_svd_compression.png")

    # ──────────────────────────────────────────────────────────────────────
    # PLOT C — PSNR vs k
    # ──────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ks    = sorted(metrics.keys())
    psnrs = [metrics[k]["psnr"] for k in ks]
    ax.plot(ks, psnrs, color=ACCENT, lw=2, marker="o", ms=5)
    ax.set_xlabel("k (singular values retained)")
    ax.set_ylabel("PSNR (dB)  ↑ better")
    ax.set_title("Compression Quality vs. Rank k\n"
                 "PSNR = 10 log₁₀(255² / MSE)")
    ax.grid(alpha=0.3)
    _save(fig, "02c_psnr_vs_k.png")
    print(f"  [✓] Saved  →  {OUTPUT_DIR}/02c_psnr_vs_k.png")

    return compressed


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — IMAGE DENOISING
# ══════════════════════════════════════════════════════════════════════════════

def add_gaussian_noise(
    img: np.ndarray,
    sigma: float = 25.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Add additive white Gaussian noise (AWGN).

    Model:  Ã = A + N,   N ~ 𝒩(0, σ²)

    Gaussian noise is the most commonly modelled disturbance in signal
    processing because of the Central Limit Theorem — many independent
    small disturbances sum to a Gaussian distribution.
    """
    rng   = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=img.shape)
    noisy = img.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(
    img: np.ndarray,
    prob: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """
    Add salt-and-pepper (impulse) noise.

    Each pixel independently becomes 0 (pepper) or 255 (salt) with
    probability `prob`.  This models stuck / dead sensor pixels.
    """
    rng    = np.random.default_rng(seed)
    noisy  = img.copy()
    mask   = rng.random(img.shape)
    noisy[mask < prob / 2]       = 0      # pepper
    noisy[mask > 1 - prob / 2]   = 255    # salt
    return noisy


def svd_denoise(
    noisy_img:  np.ndarray,
    k:          int  = 30,
    return_all: bool = False,
) -> np.ndarray | tuple:
    """
    SVD-based denoising via rank-k truncation.

    ┌──────────────────────────────────────────────────────────────────┐
    │  MATHEMATICAL INTUITION                                          │
    │                                                                  │
    │  Noise in an image tends to be spread across ALL singular        │
    │  values uniformly.  The signal (true image content) is           │
    │  concentrated in the FIRST FEW large singular values.            │
    │                                                                  │
    │  By retaining only top-k singular values we project the noisy   │
    │  image onto the subspace spanned by the first k left/right       │
    │  singular vectors — effectively a LOW-PASS FILTER in the         │
    │  spectral domain of the image.                                   │
    │                                                                  │
    │  Ã ≈ A + N  →  SVD  →  keep top k  →  Aₖ ≈ A                  │
    └──────────────────────────────────────────────────────────────────┘
    """
    A = noisy_img.astype(np.float64)
    U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(k, len(sigma))
    Ak = (U[:, :k] * sigma[:k]) @ Vt[:k, :]
    denoised = np.clip(Ak, 0, 255).astype(np.uint8)
    if return_all:
        return denoised, U, sigma, Vt
    return denoised


def gaussian_filter_denoise(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Gaussian blur as a convolution-based (linear) denoising baseline.

    Convolution with a Gaussian kernel G is a linear operator:
        A_smooth = G * A
    In the frequency domain this multiplies the DFT of A by the DFT of G,
    which suppresses high-frequency noise components.
    """
    blurred = gaussian_filter(img.astype(np.float64), sigma=sigma)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def image_denoise_demo(gray_img: np.ndarray) -> None:
    """
    End-to-end denoising demonstration covering:
      1. Gaussian noise  → SVD denoising + Gaussian blur comparison
      2. S&P noise       → SVD denoising + median filter comparison
    """
    print("\n" + "═" * 60)
    print("  SECTION 3 — IMAGE DENOISING")
    print("═" * 60)

    # ── (a) Gaussian noise ───────────────────────────────────────────────────
    gauss_noisy  = add_gaussian_noise(gray_img, sigma=30)
    gauss_svd    = svd_denoise(gauss_noisy, k=25)
    gauss_blur   = gaussian_filter_denoise(gauss_noisy, sigma=1.8)

    # ── (b) Salt & pepper noise ──────────────────────────────────────────────
    sp_noisy     = add_salt_pepper_noise(gray_img, prob=0.07)
    sp_svd       = svd_denoise(sp_noisy, k=20)
    sp_median    = cv2.medianBlur(sp_noisy, ksize=3)

    _print_denoise_metrics("Gaussian → SVD", gray_img, gauss_noisy, gauss_svd)
    _print_denoise_metrics("Gaussian → Blur", gray_img, gauss_noisy, gauss_blur)
    _print_denoise_metrics("S&P → SVD", gray_img, sp_noisy, sp_svd)
    _print_denoise_metrics("S&P → Median", gray_img, sp_noisy, sp_median)

    # ── Plot ─────────────────────────────────────────────────────────────────
    rows   = [
        ("Gaussian Noise", gray_img, gauss_noisy, gauss_svd,   gauss_blur,
         "SVD k=25",                "Gaussian Blur σ=1.8"),
        ("Salt & Pepper",  gray_img, sp_noisy,    sp_svd,      sp_median,
         "SVD k=20",                "Median Filter 3×3"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    fig.suptitle("Section 3 — Image Denoising  (Signal ≈ Low-rank; "
                 "Noise ≈ Spread across all singular values)")

    col_titles = ["Original", "Noisy Input", "", ""]

    for r, (label, orig, noisy, svd_d, alt_d, svd_lbl, alt_lbl) in \
            enumerate(rows):
        for c, (im, title) in enumerate([
            (orig,  "Original  A"),
            (noisy, f"Noisy  Ã = A + N\n{label}"),
            (svd_d, f"SVD Denoised\n{svd_lbl}"),
            (alt_d, f"Linear Filter\n{alt_lbl}"),
        ]):
            ax = axes[r][c]
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            color = SUCCESS if c == 0 else (WARNING if c == 1 else ACCENT)
            ax.set_title(title, color=color)

    plt.tight_layout()
    _save(fig, "03_denoising.png")
    print(f"  [✓] Saved  →  {OUTPUT_DIR}/03_denoising.png")

    # ── Singular value comparison: clean vs noisy ────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    _, sg_clean, _ = np.linalg.svd(gray_img.astype(float), full_matrices=False)
    _, sg_noisy, _ = np.linalg.svd(gauss_noisy.astype(float), full_matrices=False)

    k_show = 80
    idx = np.arange(1, k_show + 1)
    ax.semilogy(idx, sg_clean[:k_show], color=SUCCESS, lw=2, label="Clean image")
    ax.semilogy(idx, sg_noisy[:k_show], color=WARNING, lw=2,
                ls="--", label="Noisy image")
    ax.set_xlabel("Singular value index i")
    ax.set_ylabel("σᵢ  (log scale)")
    ax.set_title("SVD Spectrum: Clean vs Noisy\n"
                 "Noise raises the 'noise floor' of small singular values")
    ax.legend(facecolor="#161b22", edgecolor="#30363d")
    ax.grid(alpha=0.3)
    _save(fig, "03b_noise_spectrum.png")
    print(f"  [✓] Saved  →  {OUTPUT_DIR}/03b_noise_spectrum.png")


def _print_denoise_metrics(
    label:    str,
    original: np.ndarray,
    noisy:    np.ndarray,
    denoised: np.ndarray,
) -> None:
    A  = original.astype(float)
    N  = noisy.astype(float)
    D  = denoised.astype(float)
    mse_in  = np.mean((A - N) ** 2)
    mse_out = np.mean((A - D) ** 2)
    psnr_in  = 10 * np.log10(255**2 / mse_in)  if mse_in  > 0 else 99
    psnr_out = 10 * np.log10(255**2 / mse_out) if mse_out > 0 else 99
    print(f"  {label:22s} | input PSNR={psnr_in:5.1f} dB → "
          f"output PSNR={psnr_out:5.1f} dB  "
          f"(Δ = {psnr_out - psnr_in:+.1f} dB)")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — MATRIX TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════════════════════

def build_rotation_matrix(theta_deg: float) -> np.ndarray:
    """
    2-D rotation matrix for angle θ (in degrees).

    ┌──────────────────────────────────────────────────────────────────┐
    │  R(θ) = ⎡ cos θ  −sin θ ⎤                                       │
    │          ⎣ sin θ   cos θ ⎦                                       │
    │                                                                  │
    │  Properties:                                                     │
    │    det(R) = 1     →  area-preserving                             │
    │    R⁻¹ = Rᵀ       →  R is orthogonal (rotation is reversible)   │
    │    eigenvalues: e^{±iθ} (complex, unit modulus)                  │
    └──────────────────────────────────────────────────────────────────┘
    """
    theta = np.radians(theta_deg)
    c, s  = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


def build_scale_matrix(sx: float, sy: float) -> np.ndarray:
    """
    2-D scaling matrix.

    S = ⎡ sₓ  0 ⎤
        ⎣  0  sy ⎦

    Eigenvalues: sₓ, sy  (real)
    det(S) = sₓ·sy  →  area scales by this factor
    """
    return np.array([[sx, 0.0],
                     [0.0, sy]])


def build_shear_matrix(kx: float = 0.3, ky: float = 0.0) -> np.ndarray:
    """
    2-D shear matrix.

    H = ⎡ 1   kx ⎤
        ⎣ ky   1 ⎦

    Shear preserves area (det = 1 when kx·ky=0).
    Eigenvalues for x-shear: both = 1 (defective if k≠0).
    """
    return np.array([[1.0, kx],
                     [ky,  1.0]])


def apply_affine_transform(
    img:    np.ndarray,
    M:      np.ndarray,
    flags:  int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Apply a 2×2 linear transformation matrix M to an image via cv2.warpAffine.

    OpenCV expects a 2×3 affine matrix [M | t].  We embed M with zero
    translation and centre the transformation on the image centre.

    p' = M (p − c) + c   where c = image centre
    """
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2

    # Build the 2×3 matrix that centres the transform
    M23 = np.zeros((2, 3), dtype=np.float64)
    M23[:2, :2] = M
    M23[0, 2]   = cx - M[0, 0] * cx - M[0, 1] * cy
    M23[1, 2]   = cy - M[1, 0] * cx - M[1, 1] * cy

    return cv2.warpAffine(img, M23, (w, h), flags=flags,
                          borderMode=cv2.BORDER_REFLECT)


def matrix_transformations_demo(gray_img: np.ndarray) -> None:
    """
    Demonstrate rotation, scaling, shear, and their eigenvalue analysis.
    """
    print("\n" + "═" * 60)
    print("  SECTION 4 — MATRIX TRANSFORMATIONS")
    print("═" * 60)

    transforms = [
        ("Rotation 30°",        build_rotation_matrix(30)),
        ("Rotation 60°",        build_rotation_matrix(60)),
        ("Scale (0.7×, 1.3×)",  build_scale_matrix(0.7, 1.3)),
        ("Scale (1.5×, 0.6×)",  build_scale_matrix(1.5, 0.6)),
        ("Shear x (k=0.4)",     build_shear_matrix(kx=0.4)),
        ("Shear x (k=0.8)",     build_shear_matrix(kx=0.8)),
    ]

    ncols = len(transforms) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 4))
    fig.suptitle("Section 4 — Linear (Affine) Image Transformations  "
                 "p′ = M(p − c) + c")

    axes[0].imshow(gray_img, cmap="gray")
    axes[0].set_title("Original", color=SUCCESS)
    axes[0].axis("off")

    for ax, (name, M) in zip(axes[1:], transforms):
        transformed = apply_affine_transform(gray_img, M)
        ax.imshow(transformed, cmap="gray")
        eigvals = np.linalg.eigvals(M)
        eig_str = "  ".join([f"{v:.2f}" for v in eigvals])
        ax.set_title(f"{name}\ndet={np.linalg.det(M):.2f}")
        _add_subtitle(ax, f"λ = [{eig_str}]")
        ax.axis("off")

        print(f"  {name:20s} | det={np.linalg.det(M):.3f} | "
              f"eigenvalues={eigvals}")

    _save(fig, "04_transformations.png")
    print(f"  [✓] Saved  →  {OUTPUT_DIR}/04_transformations.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — EIGENVALUES & EIGENVECTORS
# ══════════════════════════════════════════════════════════════════════════════

def eigenanalysis_demo(gray_img: np.ndarray) -> None:
    """
    Perform and visualise eigendecomposition of the image's covariance matrix.

    ┌──────────────────────────────────────────────────────────────────┐
    │  MATHEMATICAL CONCEPT — Eigendecomposition                       │
    │                                                                  │
    │  For a square matrix A:    A v = λ v                             │
    │    v = eigenvector (direction unchanged by A)                    │
    │    λ = eigenvalue  (scaling factor along v)                      │
    │                                                                  │
    │  Covariance matrix C of image patches:                           │
    │    C = (1/n) Xᵀ X   (positive semi-definite, symmetric)         │
    │    All eigenvalues λ ≥ 0                                         │
    │    Eigenvectors = principal directions of variance               │
    │    This is the mathematical foundation of PCA.                   │
    └──────────────────────────────────────────────────────────────────┘
    """
    print("\n" + "═" * 60)
    print("  SECTION 5 — EIGENVALUE / EIGENVECTOR ANALYSIS")
    print("═" * 60)

    A = gray_img.astype(np.float64)

    # ── Column-wise covariance of the image ──────────────────────────────────
    # Treat each COLUMN of the image as one data point in ℝ^m
    # Mean-centre:  X = A − mean(A, axis=1, keepdims=True)
    X   = A - A.mean(axis=1, keepdims=True)
    C   = (X.T @ X) / (X.shape[0] - 1)    # covariance matrix ∈ ℝ^(n×n)

    print(f"  Covariance matrix shape : {C.shape}")
    print(f"  Symmetric?              : {np.allclose(C, C.T)}")

    # ── Eigendecomposition of symmetric positive semi-definite matrix ────────
    # np.linalg.eigh is numerically stable for symmetric matrices
    # Returns eigenvalues in ascending order; we flip for descending.
    t0   = time.perf_counter()
    eigvals, eigvecs = np.linalg.eigh(C)
    t_eig = time.perf_counter() - t0

    eigvals = eigvals[::-1]            # descending order
    eigvecs = eigvecs[:, ::-1]

    print(f"  Eigendecomposition time : {t_eig*1000:.1f} ms")
    print(f"  Top-5 eigenvalues       : "
          + "  ".join([f"{v:.1f}" for v in eigvals[:5]]))
    print(f"  % variance explained:")
    total_var = eigvals.sum()
    for i in [1, 5, 10, 20]:
        pct = eigvals[:i].sum() / total_var * 100
        print(f"    Top {i:3d} eigenvectors : {pct:.2f}%")

    # ── Reconstruct image using top-k eigenvectors (eigen-compression) ───────
    # Project X onto top-k eigenvectors, then reconstruct:
    #   Z   = X Vₖ         (score matrix in reduced space)
    #   X̂  = Z Vₖᵀ        (reconstruction)
    k_vals = [5, 20, 50, 100]
    recons = {}
    for k in k_vals:
        Vk    = eigvecs[:, :k]         # top-k eigenvectors
        Z     = X @ Vk                 # projection (data in reduced space)
        Xrec  = Z @ Vk.T              # reconstruction
        Arec  = np.clip(Xrec + A.mean(axis=1, keepdims=True), 0, 255)
        recons[k] = Arec.astype(np.uint8)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Section 5 — Eigendecomposition of Image Covariance Matrix\n"
                 "A·v = λ·v  (eigenvectors = principal directions of variance)")

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.3)

    # Eigenvalue spectrum
    ax0 = fig.add_subplot(gs[0, :2])
    k_show = min(60, len(eigvals))
    ax0.bar(range(1, k_show + 1), eigvals[:k_show], color=PURPLE, alpha=0.8)
    ax0.set_xlabel("Index i")
    ax0.set_ylabel("Eigenvalue λᵢ")
    ax0.set_title("Eigenvalue Spectrum of Covariance Matrix C")
    _add_subtitle(ax0, "Large λᵢ = directions of high image variance")
    ax0.grid(axis="y", alpha=0.35)

    # Cumulative variance
    ax1 = fig.add_subplot(gs[0, 2:])
    cum_var = np.cumsum(eigvals) / total_var * 100
    ax1.plot(range(1, len(eigvals) + 1), cum_var, color=PURPLE, lw=2)
    ax1.axhline(95, color=WARNING, lw=0.9, ls="--", alpha=0.7,
                label="95% threshold")
    ax1.set_xlabel("Number of eigenvectors retained")
    ax1.set_ylabel("Cumulative variance explained (%)")
    ax1.set_title("Cumulative Explained Variance")
    ax1.legend(facecolor="#161b22", edgecolor="#30363d")
    ax1.grid(alpha=0.3)

    # Reconstructed images
    for idx, k in enumerate(k_vals):
        ax = fig.add_subplot(gs[1, idx])
        ax.imshow(recons[k], cmap="gray", vmin=0, vmax=255)
        pct = eigvals[:k].sum() / total_var * 100
        ax.set_title(f"k = {k} eigenvectors\n{pct:.1f}% var")
        ax.axis("off")

    _save(fig, "05_eigenanalysis.png")
    print(f"  [✓] Saved  →  {OUTPUT_DIR}/05_eigenanalysis.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — PCA FOR IMAGE DATA
# ══════════════════════════════════════════════════════════════════════════════

def pca_demo(gray_img: np.ndarray, patch_size: int = 8) -> None:
    """
    Apply PCA to a dataset of image patches.

    ┌──────────────────────────────────────────────────────────────────┐
    │  MATHEMATICAL CONCEPT — PCA                                      │
    │                                                                  │
    │  Given n data points {xᵢ} ⊂ ℝᵈ  (each patch is a vector):      │
    │                                                                  │
    │  1. Centre: X̄ = (1/n) Σ xᵢ                                     │
    │  2. Covariance: C = (1/n) Xᵀ X  where X = {xᵢ − X̄}           │
    │  3. Eigendecompose: C = V Λ Vᵀ                                  │
    │  4. Project: z = Vₖᵀ (x − X̄)   (reduce to k dimensions)        │
    │  5. Reconstruct: x̂ = Vₖ z + X̄   (approx. original patch)      │
    │                                                                  │
    │  PCA is equivalent to applying SVD to the centred data matrix X  │
    │  (the eigenvectors of C ARE the right singular vectors of X).    │
    └──────────────────────────────────────────────────────────────────┘
    """
    print("\n" + "═" * 60)
    print("  SECTION 6 — PCA ON IMAGE PATCHES")
    print("═" * 60)

    A = gray_img.astype(np.float64)
    m, n = A.shape
    p    = patch_size
    d    = p * p              # dimensionality of each patch

    # ── Extract non-overlapping patches ──────────────────────────────────────
    patches = []
    for i in range(0, m - p + 1, p):
        for j in range(0, n - p + 1, p):
            patch = A[i:i+p, j:j+p].flatten()   # vectorise: ℝ^(p²)
            patches.append(patch)

    X = np.array(patches)            # data matrix: (n_patches × d)
    n_patches = X.shape[0]
    print(f"  Patch size     : {p}×{p} = {d} dims")
    print(f"  Total patches  : {n_patches}")
    print(f"  Data matrix X  : {X.shape}")

    # ── Perform PCA via SVD of centred data matrix ───────────────────────────
    mu   = X.mean(axis=0)            # mean patch (centroid in ℝᵈ)
    Xc   = X - mu                    # centred data matrix

    # SVD of centred data ≡ PCA
    # The right singular vectors V are the principal components (eigenvectors of C)
    # Singular values σ relate to eigenvalues: λᵢ = σᵢ² / (n-1)
    U, sigma, Vt = np.linalg.svd(Xc, full_matrices=False)

    eigvals   = sigma**2 / (n_patches - 1)
    cum_var   = np.cumsum(eigvals) / eigvals.sum()

    print(f"  Top-5 eigenvalues (PCA): "
          + "  ".join([f"{v:.2f}" for v in eigvals[:5]]))

    # ── Reconstruct patches at different k ───────────────────────────────────
    k_values = [1, 4, 8, 16, 32]
    errors   = []
    for k in k_values:
        Vk    = Vt[:k, :].T          # principal component matrix (d × k)
        Z     = Xc @ Vk              # scores: (n_patches × k)
        Xrec  = Z @ Vk.T + mu       # reconstruction: (n_patches × d)
        err   = np.mean((Xc - (Xrec - mu))**2)
        errors.append(err)
        print(f"  k={k:3d}  reconstruction MSE = {err:.4f}")

    # ── Visualise principal components (eigenpatches) ────────────────────────
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Section 6 — PCA on Image Patches  "
                 "(PCA ≡ SVD on centred data matrix)")

    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.5, wspace=0.3)

    # Top row: first 10 principal components (eigenpatches)
    n_show = min(10, d)
    for i in range(5):
        ax = fig.add_subplot(gs[0, i])
        pc = Vt[i, :].reshape(p, p)
        ax.imshow(pc, cmap="RdBu_r")
        ax.set_title(f"PC{i+1}\nλ={eigvals[i]:.2f}")
        ax.axis("off")
        if i == 0:
            _add_subtitle(ax, "Most variance")

    # Bottom row: explained variance + reconstruction error
    ax_ev = fig.add_subplot(gs[1, :3])
    k_plot = min(d, 40)
    ax_ev.bar(range(1, k_plot + 1), eigvals[:k_plot], color=PURPLE, alpha=0.8)
    ax_ev2 = ax_ev.twinx()
    ax_ev2.plot(range(1, k_plot + 1), cum_var[:k_plot] * 100,
                color=ACCENT, lw=2)
    ax_ev2.set_ylabel("Cumulative variance (%)", color=ACCENT)
    ax_ev.set_xlabel("Principal Component index")
    ax_ev.set_ylabel("Eigenvalue λᵢ", color=PURPLE)
    ax_ev.set_title("Eigenvalue Spectrum & Cumulative Variance")
    ax_ev.grid(alpha=0.25)

    ax_err = fig.add_subplot(gs[1, 3:])
    ax_err.plot(k_values, errors, color=ERROR, lw=2, marker="s", ms=6)
    ax_err.set_xlabel("k (components retained)")
    ax_err.set_ylabel("Reconstruction MSE")
    ax_err.set_title("PCA Reconstruction Error vs k")
    ax_err.grid(alpha=0.3)

    _save(fig, "06_pca.png")
    print(f"  [✓] Saved  →  {OUTPUT_DIR}/06_pca.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — RGB SVD COMPRESSION
# ══════════════════════════════════════════════════════════════════════════════

def rgb_svd_compress(rgb_img: np.ndarray, k_values: list[int]) -> None:
    """
    Apply SVD compression independently to each RGB channel.

    Because the RGB image is a 3-D tensor T ∈ ℝ^(m×n×3), we decompose each
    channel matrix separately:

        Rₖ = Uᴿₖ Σᴿₖ (Vᴿₖ)ᵀ
        Gₖ = Uᴳₖ Σᴳₖ (Vᴳₖ)ᵀ
        Bₖ = Uᴮₖ Σᴮₖ (Vᴮₖ)ᵀ

    Then recombine: Tₖ = stack(Rₖ, Gₖ, Bₖ)
    """
    print("\n" + "═" * 60)
    print("  SECTION 7 — RGB SVD COMPRESSION")
    print("═" * 60)

    channel_names = ["Red", "Green", "Blue"]
    channel_colors = [ERROR, SUCCESS, ACCENT]

    fig, axes = plt.subplots(
        len(k_values) + 1, 4,
        figsize=(12, 3.2 * (len(k_values) + 1))
    )
    fig.suptitle("Section 7 — RGB Image SVD Compression\n"
                 "T = (Rₖ ‖ Gₖ ‖ Bₖ)  each channel compressed independently")

    # Plot original
    axes[0][0].imshow(rgb_img)
    axes[0][0].set_title("Original RGB", color=SUCCESS)
    axes[0][0].axis("off")
    for c, (ch_name, col) in enumerate(zip(channel_names, channel_colors)):
        axes[0][c + 1].imshow(rgb_img[:, :, c], cmap="gray")
        axes[0][c + 1].set_title(f"{ch_name} channel", color=col)
        axes[0][c + 1].axis("off")

    for row, k in enumerate(k_values, start=1):
        channels_k = []
        for c in range(3):
            ch    = rgb_img[:, :, c].astype(np.float64)
            U, sg, Vt = np.linalg.svd(ch, full_matrices=False)
            k_c   = min(k, len(sg))
            ch_k  = (U[:, :k_c] * sg[:k_c]) @ Vt[:k_c, :]
            channels_k.append(np.clip(ch_k, 0, 255).astype(np.uint8))

        rgb_k = np.stack(channels_k, axis=-1)
        mse   = np.mean((rgb_img.astype(float) - rgb_k.astype(float))**2)
        psnr  = 10 * np.log10(255**2 / mse)

        axes[row][0].imshow(rgb_k)
        axes[row][0].set_title(f"k={k}  PSNR={psnr:.1f} dB")
        axes[row][0].axis("off")
        print(f"  k={k:4d} | PSNR = {psnr:.2f} dB")

        for c, col in enumerate(channel_colors):
            axes[row][c + 1].imshow(channels_k[c], cmap="gray")
            axes[row][c + 1].axis("off")

    plt.tight_layout()
    _save(fig, "07_rgb_compression.png")
    print(f"  [✓] Saved  →  {OUTPUT_DIR}/07_rgb_compression.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — PERFORMANCE BENCHMARKING
# ══════════════════════════════════════════════════════════════════════════════

def performance_benchmark(gray_img: np.ndarray) -> None:
    """
    Time and space trade-off analysis for SVD compression.

    ┌──────────────────────────────────────────────────────────────────┐
    │  STORAGE COST OF RANK-k APPROXIMATION                           │
    │                                                                  │
    │  Original  :  m × n  float64 values  =  8mn  bytes              │
    │  Rank-k    :  k(m + 1 + n)           =  8k(m+n+1) bytes         │
    │                                                                  │
    │  Compression ratio  ρ = k(m+n+1) / (mn)                         │
    │  Break-even point   k* = mn / (m+n+1)   (always > min(m,n)/2)   │
    └──────────────────────────────────────────────────────────────────┘
    """
    print("\n" + "═" * 60)
    print("  SECTION 8 — PERFORMANCE BENCHMARKING")
    print("═" * 60)

    m, n = gray_img.shape
    A    = gray_img.astype(np.float64)

    k_range = list(range(5, min(130, min(m, n)), 10))
    times, psnrs, sizes = [], [], []

    for k in k_range:
        t0 = time.perf_counter()
        U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
        Ak = (U[:, :k] * sigma[:k]) @ Vt[:k, :]
        t_svd = time.perf_counter() - t0

        mse  = np.mean((A - Ak)**2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 99
        size = k * (m + n + 1) / (m * n) * 100   # % of original

        times.append(t_svd * 1000)
        psnrs.append(psnr)
        sizes.append(size)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Section 8 — Time & Space Trade-offs for SVD Compression")

    ax1.plot(k_range, times, color=WARNING, lw=2, marker="o", ms=4)
    ax1.set_xlabel("k")
    ax1.set_ylabel("Wall-clock time (ms)")
    ax1.set_title("Computation Time vs k\n(dominated by SVD, O(mn·min(m,n)))")
    ax1.grid(alpha=0.3)

    ax2.plot(k_range, sizes, color=PURPLE, lw=2, marker="s", ms=4)
    ax2.axhline(100, color=MUTED, lw=0.8, ls="--", label="100% = original size")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Storage (% of original)")
    ax2.set_title("Storage Cost vs k\nρ = k(m+n+1)/(mn)")
    ax2.legend(facecolor="#161b22", edgecolor="#30363d")
    ax2.grid(alpha=0.3)

    ax3.scatter(sizes, psnrs, c=k_range, cmap="plasma", s=50, zorder=3)
    ax3.set_xlabel("Storage (%)")
    ax3.set_ylabel("PSNR (dB)  ↑ better")
    ax3.set_title("Quality–Storage Pareto Curve\n"
                  "(choose k on the 'knee' for best trade-off)")
    ax3.grid(alpha=0.3)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(cmap="plasma",
                              norm=plt.Normalize(min(k_range), max(k_range))),
        ax=ax3
    )
    cb.set_label("k", color="#e6edf3")

    _save(fig, "08_benchmarks.png")
    print(f"  [✓] Saved  →  {OUTPUT_DIR}/08_benchmarks.png")

    # Print summary table
    print(f"\n  {'k':>5}  {'Time (ms)':>10}  {'Size (%)':>10}  {'PSNR (dB)':>10}")
    print("  " + "─" * 40)
    for k, t, s, p in zip(k_range, times, sizes, psnrs):
        print(f"  {k:>5}  {t:>10.1f}  {s:>10.1f}  {p:>10.2f}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — SUMMARY DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def summary_dashboard(
    gray_img:   np.ndarray,
    rgb_img:    np.ndarray,
    compressed: dict,
) -> None:
    """
    One-page academic summary figure combining key results.
    """
    print("\n" + "═" * 60)
    print("  SECTION 9 — SUMMARY DASHBOARD")
    print("═" * 60)

    k_best = min(30, max(compressed.keys()))
    comp   = compressed.get(k_best, list(compressed.values())[-1])
    noisy  = add_gaussian_noise(gray_img, sigma=25)
    den    = svd_denoise(noisy, k=25)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Linear Algebra & Image Processing — Project Summary Dashboard\n"
        "A = UΣVᵀ  (SVD)  ·  Av = λv  (Eigen)  ·  p′ = Mp  (Transform)",
        fontsize=14, y=1.01,
    )

    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.55, wspace=0.35)

    # Row 0
    _dash_img(fig, gs[0, 0], gray_img, "Grayscale\nA ∈ ℝ^(m×n)", "gray")
    _dash_img(fig, gs[0, 1], rgb_img,  "RGB Tensor\nT ∈ ℝ^(m×n×3)")
    _dash_img(fig, gs[0, 2], comp,     f"SVD Compressed\nk={k_best}", "gray")
    _dash_img(fig, gs[0, 3], noisy,    "Noisy (Gaussian)", "gray")
    _dash_img(fig, gs[0, 4], den,      "SVD Denoised\nk=25", "gray")

    # Row 1 — transformation
    rot   = apply_affine_transform(gray_img, build_rotation_matrix(45))
    scale = apply_affine_transform(gray_img, build_scale_matrix(1.3, 0.7))
    shear = apply_affine_transform(gray_img, build_shear_matrix(kx=0.5))
    _dash_img(fig, gs[1, 0], rot,   "Rotation 45°\ndet=1", "gray")
    _dash_img(fig, gs[1, 1], scale, "Scaling (1.3, 0.7)\ndet=0.91", "gray")
    _dash_img(fig, gs[1, 2], shear, "Shear kₓ=0.5\ndet=1", "gray")

    # Singular value plot
    ax_sv = fig.add_subplot(gs[1, 3:])
    A     = gray_img.astype(float)
    _, sg, _ = np.linalg.svd(A, full_matrices=False)
    ax_sv.semilogy(sg[:80], color=ACCENT, lw=2)
    ax_sv.set_xlabel("Index i")
    ax_sv.set_ylabel("σᵢ")
    ax_sv.set_title("Singular Value Decay")
    ax_sv.grid(alpha=0.3)

    # Row 2 — PCA components
    A    = gray_img.astype(np.float64)
    X    = A - A.mean(axis=1, keepdims=True)
    C    = (X.T @ X) / (X.shape[0] - 1)
    _, evecs = np.linalg.eigh(C)
    evecs = evecs[:, ::-1]

    for i in range(4):
        ax = fig.add_subplot(gs[2, i])
        pc = evecs[:, i].reshape(-1, 1) * np.ones((1, 10))
        ax.imshow(pc, cmap="RdBu_r", aspect="auto")
        ax.set_title(f"Eigenvector {i+1}", fontsize=9)
        ax.axis("off")

    ax_cv = fig.add_subplot(gs[2, 4])
    _, sg2, _ = np.linalg.svd(X, full_matrices=False)
    ev2 = sg2**2 / (X.shape[0] - 1)
    cum = np.cumsum(ev2) / ev2.sum() * 100
    ax_cv.plot(cum[:60], color=PURPLE, lw=2)
    ax_cv.axhline(95, color=WARNING, lw=0.8, ls="--", alpha=0.7)
    ax_cv.set_title("PCA Cum. Variance")
    ax_cv.set_xlabel("Components")
    ax_cv.set_ylabel("Variance (%)")
    ax_cv.grid(alpha=0.3)

    _save(fig, "09_summary_dashboard.png")
    print(f"  [✓] Saved  →  {OUTPUT_DIR}/09_summary_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — USER IMAGE UPLOAD + PROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_uploaded_image(image_path: str) -> None:
    """
    End-to-end processing pipeline for a user-supplied image.

    Steps
    -----
    1. Load  → convert to RGB and Grayscale (a linear projection)
    2. Denoise directly on the real image:
         a) SVD rank-k truncation  →  keeps top-k singular values
         b) Gaussian filter        →  linear convolution G * A
    3. Scale  up 1.5×  and  down 0.5×  via cv2.resize
    4. Rotate 45° and 90°  using existing rotation-matrix helpers
    5. Visualise in a 4-row grid and save all outputs
    """
    print("\n" + "═" * 60)
    print("  SECTION 10 — USER IMAGE UPLOAD PROCESSING PIPELINE")
    print("═" * 60)

    # ── Validate path ────────────────────────────────────────────────────────
    if not os.path.isfile(image_path):
        print(f"  [ERROR] File not found: {image_path}")
        sys.exit(1)

    # ── Output directory ─────────────────────────────────────────────────────
    upload_dir = os.path.join(OUTPUT_DIR, "upload_processing")
    os.makedirs(upload_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 1 — LOAD IMAGE
    #  PIL handles all formats (JPEG, PNG, CMYK, RGBA …) cleanly.
    #  Grayscale:  y = 0.299R + 0.587G + 0.114B  (linear projection ℝ³→ℝ)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n  ── Step 1: Load image ──")
    print(f"  [INFO] Loading: {image_path}")
    pil_img  = Image.open(image_path)
    rgb_img  = np.array(pil_img.convert("RGB"))   # shape (H, W, 3)
    gray_img = np.array(pil_img.convert("L"))     # shape (H, W)

    h, w = gray_img.shape
    print(f"  Image size       : {w} × {h}")
    print(f"  Grayscale matrix : {gray_img.shape}")
    print(f"  RGB tensor       : {rgb_img.shape}")

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 2 — DENOISING  (applied directly to the real image)
    #
    #  SVD denoising:  decompose A = U Σ Vᵀ, keep top-k singular values.
    #  → Signal is in the large singular values; noise spreads uniformly.
    #  Gaussian filter: linear low-pass convolution A_smooth = G * A.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  ── Step 2: Denoising ──")

    # Apply both methods directly on the original grayscale image (no synthetic noise)
    svd_denoised   = svd_denoise(gray_img, k=30)             # reuse existing fn
    gauss_denoised = gaussian_filter_denoise(gray_img, sigma=1.5)  # reuse existing fn

    print("  SVD denoising  (k=30)       : done")
    print("  Gaussian filter (sigma=1.5) : done")

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 3 — SCALING
    #  Scaling matrix  S = diag(sₓ, sᵧ).  cv2.resize resamples the image
    #  matrix on a finer / coarser lattice using bilinear interpolation.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  ── Step 3: Scaling ──")

    # 1.5× scale-up  (INTER_LINEAR = bilinear interpolation)
    scaled_up = cv2.resize(gray_img, (int(w * 1.5), int(h * 1.5)),
                           interpolation=cv2.INTER_LINEAR)
    print(f"  Scaled UP  (1.5×) : {gray_img.shape} → {scaled_up.shape}")

    # 0.5× scale-down  (INTER_AREA = better for shrinking)
    scaled_dn = cv2.resize(gray_img, (max(1, int(w * 0.5)), max(1, int(h * 0.5))),
                           interpolation=cv2.INTER_AREA)
    print(f"  Scaled DOWN (0.5×): {gray_img.shape} → {scaled_dn.shape}")

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 4 — ROTATION
    #  R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]  — orthogonal matrix:
    #  det(R) = 1  (area-preserving),  R⁻¹ = Rᵀ.
    #  Reuses build_rotation_matrix + apply_affine_transform from Section 4.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  ── Step 4: Rotation ──")

    R45 = build_rotation_matrix(45)                           # reuse existing fn
    R90 = build_rotation_matrix(90)                           # reuse existing fn
    rotated_45 = apply_affine_transform(gray_img, R45)        # reuse existing fn
    rotated_90 = apply_affine_transform(gray_img, R90)        # reuse existing fn

    print(f"  Rotated 45°  det(R) = {np.linalg.det(R45):.3f}")
    print(f"  Rotated 90°  det(R) = {np.linalg.det(R90):.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 5 — VISUALISATION  (4-row grid)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  ── Step 5: Visualising results ──")

    fig, axes = plt.subplots(4, 3, figsize=(14, 17))
    fig.suptitle(
        "LAA Image Processing Pipeline\n"
        "Original  ·  Denoising  ·  Scaling  ·  Rotation",
        fontsize=13,
    )

    # Row 1: Original | Grayscale | RGB
    axes[0][0].imshow(gray_img, cmap="gray", vmin=0, vmax=255)
    axes[0][0].set_title("Original", color=SUCCESS)
    axes[0][0].axis("off")

    axes[0][1].imshow(gray_img, cmap="gray", vmin=0, vmax=255)
    axes[0][1].set_title("Grayscale\ny = 0.299R + 0.587G + 0.114B", color=ACCENT)
    axes[0][1].axis("off")

    axes[0][2].imshow(rgb_img)
    axes[0][2].set_title("RGB  T ∈ ℝ^(m×n×3)", color=ACCENT)
    axes[0][2].axis("off")

    # Row 2: SVD Denoised | Gaussian Denoised | original (reference)
    axes[1][0].imshow(svd_denoised, cmap="gray", vmin=0, vmax=255)
    axes[1][0].set_title("SVD Denoised (k=30)\nAₖ = Uₖ Σₖ Vₖᵀ", color=ACCENT)
    axes[1][0].axis("off")

    axes[1][1].imshow(gauss_denoised, cmap="gray", vmin=0, vmax=255)
    axes[1][1].set_title("Gaussian Denoised\nG * A  (linear convolution)", color=ACCENT)
    axes[1][1].axis("off")

    # Third panel in row 2: show original for easy comparison
    axes[1][2].imshow(gray_img, cmap="gray", vmin=0, vmax=255)
    axes[1][2].set_title("Original (reference)", color=MUTED)
    axes[1][2].axis("off")

    # Row 3: Scaled Up | Scaled Down | Rotated 45°
    axes[2][0].imshow(scaled_up, cmap="gray", vmin=0, vmax=255)
    axes[2][0].set_title(f"Scaled UP 1.5×\n{scaled_up.shape[1]}×{scaled_up.shape[0]}",
                         color=PURPLE)
    axes[2][0].axis("off")

    axes[2][1].imshow(scaled_dn, cmap="gray", vmin=0, vmax=255)
    axes[2][1].set_title(f"Scaled DOWN 0.5×\n{scaled_dn.shape[1]}×{scaled_dn.shape[0]}",
                         color=PURPLE)
    axes[2][1].axis("off")

    axes[2][2].imshow(rotated_45, cmap="gray", vmin=0, vmax=255)
    axes[2][2].set_title("Rotated 45°\nR(θ), det = 1", color=PURPLE)
    axes[2][2].axis("off")

    # Row 4: Rotated 90° | (blank) | (blank)
    axes[3][0].imshow(rotated_90, cmap="gray", vmin=0, vmax=255)
    axes[3][0].set_title("Rotated 90°\nR(θ), det = 1", color=PURPLE)
    axes[3][0].axis("off")

    # Hide unused panels in row 4
    axes[3][1].axis("off")
    axes[3][2].axis("off")

    plt.tight_layout()
    grid_path = os.path.join(upload_dir, "processing_grid.png")
    fig.savefig(grid_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  [✓] Grid saved  → {grid_path}")

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 5 (cont.) — SAVE INDIVIDUAL IMAGES
    #  Required output filenames as specified in the task brief.
    # ─────────────────────────────────────────────────────────────────────────
    saves = [
        ("original.png",          gray_img,        "L"),
        ("grayscale.png",         gray_img,        "L"),
        ("svd_denoised.png",      svd_denoised,    "L"),
        ("gaussian_denoised.png", gauss_denoised,  "L"),
        ("scaled_up.png",         scaled_up,       "L"),
        ("scaled_down.png",       scaled_dn,       "L"),
        ("rotated_45.png",        rotated_45,      "L"),
        ("rotated_90.png",        rotated_90,      "L"),
    ]

    for fname, arr, mode in saves:
        out = os.path.join(upload_dir, fname)
        Image.fromarray(arr, mode=mode).save(out)
        print(f"  [✓] Saved  → {out}")

    # Also save the colour original
    rgb_out = os.path.join(upload_dir, "original_rgb.png")
    Image.fromarray(rgb_img, mode="RGB").save(rgb_out)
    print(f"  [✓] Saved  → {rgb_out}")

    print(f"\n  ✓  Pipeline complete. All outputs in: {upload_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _add_subtitle(ax: plt.Axes, text: str, y: float = -0.08) -> None:
    ax.text(0.5, y, text, ha="center", va="top",
            fontsize=6.5, color=MUTED, transform=ax.transAxes,
            wrap=True)


def _save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close(fig)


def _dash_img(fig, gs_cell, img, title, cmap=None):
    ax = fig.add_subplot(gs_cell)
    if cmap:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
    else:
        ax.imshow(img)
    ax.set_title(title, fontsize=8.5)
    ax.axis("off")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

SECTIONS = {
    "1": ("Image Representation",     "load_and_represent_image"),
    "2": ("SVD Compression",          "svd_compress"),
    "3": ("Image Denoising",          "image_denoise_demo"),
    "4": ("Matrix Transformations",   "matrix_transformations_demo"),
    "5": ("Eigenvalue Analysis",      "eigenanalysis_demo"),
    "6": ("PCA",                      "pca_demo"),
    "7": ("RGB SVD Compression",      "rgb_svd_compress"),
    "8": ("Performance Benchmarks",   "performance_benchmark"),
    "9": ("Summary Dashboard",        "summary_dashboard"),
}


def print_banner() -> None:
    banner = textwrap.dedent("""
    ╔══════════════════════════════════════════════════════════════╗
    ║   LINEAR ALGEBRA AND ITS APPLICATIONS — IMAGE PROCESSING    ║
    ║   A = UΣVᵀ  ·  Av = λv  ·  p′ = Mp                         ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    print(banner)
    print("  Output directory :", OUTPUT_DIR)
    print()
    print("  Available sections:")
    for num, (name, _) in SECTIONS.items():
        print(f"    [{num}] {name}")
    print("   [all] Run all sections")
    print()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LAA Image Processing Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python laa_image_processing.py                  # run all sections
          python laa_image_processing.py -s 2 3           # sections 2 and 3 only
          python laa_image_processing.py -i photo.jpg -s 2
          python laa_image_processing.py --upload photo.jpg  # upload pipeline only
        """),
    )
    p.add_argument(
        "-i", "--image",
        metavar="PATH",
        help="Path to an input image (optional; synthetic image used by default)",
    )
    p.add_argument(
        "--upload",
        metavar="IMAGE_PATH",
        help="Run ONLY the upload processing pipeline on the given image",
    )
    p.add_argument(
        "-s", "--sections",
        nargs="+",
        default=["all"],
        choices=list(SECTIONS.keys()) + ["all"],
        help="Section(s) to run (default: all)",
    )
    p.add_argument(
        "-k", "--kvals",
        nargs="+",
        type=int,
        default=[1, 5, 15, 30, 60, 100],
        help="Singular value counts for SVD compression (default: 1 5 15 30 60 100)",
    )
    p.add_argument(
        "--size",
        type=int,
        default=256,
        help="Size of synthetic demo image in pixels (default: 256)",
    )
    return p


def main() -> None:
    print_banner()
    parser = build_parser()
    args   = parser.parse_args()

    # ── If --upload is given, run ONLY the upload pipeline then exit ──────────
    if args.upload:
        process_uploaded_image(args.upload)
        return

    # ── No --upload provided: require the user to supply an image ─────────────
    print("[ERROR] Please provide an image using --upload <image_path>")
    print("  Example:")
    print("    python laa_image_processing.py --upload photo.jpg")
    sys.exit(1)


if __name__ == "__main__":
    main()
