"""
================================================================================
  LINEAR ALGEBRA AND ITS APPLICATIONS (LAA) — IMAGE PROCESSING PROJECT
  ENHANCED VERSION WITH RGB DENOISING & MENU-DRIVEN INTERFACE
================================================================================

Author  : LAA Project (Enhanced)
Purpose : Demonstrate core linear algebra concepts applied to image processing
          with added RGB channel denoising and interactive menu system

New Features:
  • RGB channel-wise SVD and Gaussian denoising
  • Interactive menu-driven CLI interface
  • Processing pipeline selector
  • Enhanced visualization for multi-channel results

Mathematical Foundations Covered
─────────────────────────────────
  • Matrix representation of images
  • Singular Value Decomposition (SVD)  →  A = U Σ Vᵀ
  • Low-rank approximation             →  Aₖ = Uₖ Σₖ Vₖᵀ
  • Noise modelling and SVD-based denoising (Grayscale & RGB)
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
from typing import Optional, Tuple, Dict

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
# ══════════════════════════════════════════════════════════════════════════════

def generate_demo_images(size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesise a grayscale and an RGB demo image using NumPy geometry.
    """
    print("[INFO] Generating synthetic demo images …")

    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)

    gray_raw = (
        0.5 * np.sin(np.pi * X) * np.cos(np.pi * Y)
        + 0.3 * np.exp(-(X**2 + Y**2) / 4)
        + 0.2 * np.sin(2 * np.pi * (X + Y) / 3)
    )
    gray_img = _normalise_to_uint8(gray_raw)

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
    """Linear normalisation  →  [min, max]  ⟹  [0, 255]."""
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

    patch_size = 8
    patch = gray_img[:patch_size, :patch_size]

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("Section 1 — Image Representation: Images as Matrices",
                 color="#f0f6fc", y=1.01)

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(gray_img, cmap="gray", vmin=0, vmax=255)
    ax0.set_title(f"Grayscale  A ∈ ℝ^({m}×{n})")
    ax0.axis("off")
    _add_subtitle(ax0, "Each pixel = one matrix entry ∈ [0, 255]")

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(rgb_img)
    ax1.set_title(f"RGB Tensor  T ∈ ℝ^({m}×{n}×3)")
    ax1.axis("off")
    _add_subtitle(ax1, "3 channel matrices stacked along depth axis")

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
    k_values:    Optional[list] = None,
    show_energy: bool = True,
) -> Dict:
    """
    Compress a grayscale image via SVD low-rank approximation.
    """
    print("\n" + "═" * 60)
    print("  SECTION 2 — SVD IMAGE COMPRESSION")
    print("═" * 60)

    if k_values is None:
        k_values = [1, 5, 15, 30, 60, 100]

    A = gray_img.astype(np.float64)
    m, n = A.shape

    t0 = time.perf_counter()
    U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
    t_svd = time.perf_counter() - t0

    print(f"  Image shape          : {m} × {n}")
    print(f"  Number of sing. vals : {len(sigma)}")
    print(f"  σ₁ (largest)         : {sigma[0]:.2f}")
    print(f"  σ_r (smallest)       : {sigma[-1]:.6f}")
    print(f"  SVD wall-clock time  : {t_svd*1000:.1f} ms")

    energy_total = np.sum(sigma**2)
    cum_energy   = np.cumsum(sigma**2) / energy_total

    for pct in (0.90, 0.95, 0.99):
        k_thresh = int(np.searchsorted(cum_energy, pct)) + 1
        ratio = k_thresh * (m + n + 1) / (m * n) * 100
        print(f"  k for {pct*100:.0f}% energy      : {k_thresh:4d}  "
              f"(compression ratio ≈ {ratio:.1f}%)")

    compressed = {}
    metrics    = {}

    for k in k_values:
        k = min(k, len(sigma))
        Ak = (U[:, :k] * sigma[:k]) @ Vt[:k, :]
        Ak_uint8 = np.clip(Ak, 0, 255).astype(np.uint8)
        compressed[k] = Ak_uint8

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
        ax.grid(alpha=0.3)

        _save(fig, "02a_svd_spectrum.png")
        print(f"  [✓] Saved  →  {OUTPUT_DIR}/02a_svd_spectrum.png")

    show_k = [k for k in k_values if k <= min(100, len(sigma))][:6]
    ncols  = len(show_k) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3.8))
    fig.suptitle("Section 2b — SVD Low-Rank Approximations  Aₖ = Uₖ Σₖ Vₖᵀ")

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

    fig, ax = plt.subplots(figsize=(7, 4))
    ks    = sorted(metrics.keys())
    psnrs = [metrics[k]["psnr"] for k in ks]
    ax.plot(ks, psnrs, color=ACCENT, lw=2, marker="o", ms=5)
    ax.set_xlabel("k (singular values retained)")
    ax.set_ylabel("PSNR (dB)  ↑ better")
    ax.set_title("Compression Quality vs. Rank k\nPSNR = 10 log₁₀(255² / MSE)")
    ax.grid(alpha=0.3)
    _save(fig, "02c_psnr_vs_k.png")
    print(f"  [✓] Saved  →  {OUTPUT_DIR}/02c_psnr_vs_k.png")

    return compressed


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — IMAGE DENOISING (GRAYSCALE & RGB)
# ══════════════════════════════════════════════════════════════════════════════

def add_gaussian_noise(
    img: np.ndarray,
    sigma: float = 25.0,
    seed: int = 42,
) -> np.ndarray:
    """Add additive white Gaussian noise (AWGN)."""
    rng   = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=img.shape)
    noisy = img.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(
    img: np.ndarray,
    prob: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Add salt-and-pepper (impulse) noise."""
    rng    = np.random.default_rng(seed)
    noisy  = img.copy()
    mask   = rng.random(img.shape)
    noisy[mask < prob / 2]       = 0
    noisy[mask > 1 - prob / 2]   = 255
    return noisy


def svd_denoise(
    noisy_img:  np.ndarray,
    k:          int  = 30,
    return_all: bool = False,
):
    """SVD-based denoising via rank-k truncation."""
    A = noisy_img.astype(np.float64)
    U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(k, len(sigma))
    Ak = (U[:, :k] * sigma[:k]) @ Vt[:k, :]
    denoised = np.clip(Ak, 0, 255).astype(np.uint8)
    if return_all:
        return denoised, U, sigma, Vt
    return denoised


def svd_denoise_rgb(
    noisy_rgb: np.ndarray,
    k: int = 30,
) -> np.ndarray:
    """
    SVD-based denoising for RGB image (channel-wise).
    
    Each RGB channel is treated as an independent 2D matrix and
    denoised separately using SVD rank-k truncation.
    """
    denoised_channels = []
    channel_names = ["Red", "Green", "Blue"]
    
    print(f"  [RGB Denoising] Processing {k} singular values per channel...")
    
    for c in range(3):
        channel = noisy_rgb[:, :, c].astype(np.float64)
        U, sigma, Vt = np.linalg.svd(channel, full_matrices=False)
        k_c = min(k, len(sigma))
        channel_denoised = (U[:, :k_c] * sigma[:k_c]) @ Vt[:k_c, :]
        channel_uint8 = np.clip(channel_denoised, 0, 255).astype(np.uint8)
        denoised_channels.append(channel_uint8)
        print(f"    {channel_names[c]:6s} channel: σ₁={sigma[0]:.2f}, "
              f"σₖ={sigma[k_c-1]:.2f}, MSE={np.mean((channel - channel_denoised)**2):.4f}")
    
    return np.stack(denoised_channels, axis=-1)


def gaussian_filter_denoise(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian blur as a convolution-based (linear) denoising baseline."""
    blurred = gaussian_filter(img.astype(np.float64), sigma=sigma)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def gaussian_filter_denoise_rgb(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian blur denoising for RGB image (channel-wise)."""
    denoised_channels = []
    for c in range(3):
        blurred = gaussian_filter(img[:, :, c].astype(np.float64), sigma=sigma)
        denoised_channels.append(np.clip(blurred, 0, 255).astype(np.uint8))
    return np.stack(denoised_channels, axis=-1)


def image_denoise_demo(gray_img: np.ndarray, rgb_img: np.ndarray) -> None:
    """
    End-to-end denoising demonstration covering:
      1. Grayscale: Gaussian noise → SVD denoising + Gaussian blur comparison
      2. RGB: Gaussian noise → SVD denoising + Gaussian blur comparison
      3. S&P noise comparison
    """
    print("\n" + "═" * 60)
    print("  SECTION 3 — IMAGE DENOISING (GRAYSCALE & RGB)")
    print("═" * 60)

    # ── (a) Grayscale Gaussian noise ─────────────────────────────────────────
    print("\n  ── Grayscale Channel ──")
    gauss_noisy  = add_gaussian_noise(gray_img, sigma=30)
    gauss_svd    = svd_denoise(gauss_noisy, k=25)
    gauss_blur   = gaussian_filter_denoise(gauss_noisy, sigma=1.8)

    # ── (b) Salt & pepper noise ──────────────────────────────────────────────
    sp_noisy     = add_salt_pepper_noise(gray_img, prob=0.07)
    sp_svd       = svd_denoise(sp_noisy, k=20)
    sp_median    = cv2.medianBlur(sp_noisy, ksize=3)

    _print_denoise_metrics("Grayscale: Gaussian → SVD", gray_img, gauss_noisy, gauss_svd)
    _print_denoise_metrics("Grayscale: Gaussian → Blur", gray_img, gauss_noisy, gauss_blur)
    _print_denoise_metrics("Grayscale: S&P → SVD", gray_img, sp_noisy, sp_svd)
    _print_denoise_metrics("Grayscale: S&P → Median", gray_img, sp_noisy, sp_median)

    # ── (c) RGB Gaussian noise ───────────────────────────────────────────────
    print("\n  ── RGB Channels ──")
    rgb_gauss_noisy = add_gaussian_noise(rgb_img, sigma=30)
    rgb_gauss_svd   = svd_denoise_rgb(rgb_gauss_noisy, k=25)
    rgb_gauss_blur  = gaussian_filter_denoise_rgb(rgb_gauss_noisy, sigma=1.8)

    _print_denoise_metrics_rgb("RGB: Gaussian → SVD", rgb_img, rgb_gauss_noisy, rgb_gauss_svd)
    _print_denoise_metrics_rgb("RGB: Gaussian → Blur", rgb_img, rgb_gauss_noisy, rgb_gauss_blur)

    # ── Plot Grayscale ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    fig.suptitle("Section 3a — Grayscale Image Denoising")

    rows = [
        ("Gaussian Noise", gray_img, gauss_noisy, gauss_svd, gauss_blur,
         "SVD k=25", "Gaussian Blur σ=1.8"),
        ("Salt & Pepper", gray_img, sp_noisy, sp_svd, sp_median,
         "SVD k=20", "Median Filter 3×3"),
    ]

    for r, (label, orig, noisy, svd_d, alt_d, svd_lbl, alt_lbl) in enumerate(rows):
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
    _save(fig, "03a_grayscale_denoising.png")
    print(f"  [✓] Saved  →  {OUTPUT_DIR}/03a_grayscale_denoising.png")

    # ── Plot RGB ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    fig.suptitle("Section 3b — RGB Image Denoising (Channel-wise SVD & Gaussian)")

    rows_rgb = [
        ("Gaussian Noise", rgb_img, rgb_gauss_noisy, rgb_gauss_svd, rgb_gauss_blur,
         "SVD k=25", "Gaussian Blur σ=1.8"),
    ]

    for r, (label, orig, noisy, svd_d, alt_d, svd_lbl, alt_lbl) in enumerate(rows_rgb):
        for c, (im, title) in enumerate([
            (orig,   "Original RGB"),
            (noisy,  f"Noisy RGB\n{label}"),
            (svd_d,  f"SVD Denoised\n{svd_lbl}"),
            (alt_d,  f"Gaussian Blur\n{alt_lbl}"),
        ]):
            ax = axes[0][c]
            ax.imshow(im)
            ax.axis("off")
            color = SUCCESS if c == 0 else (WARNING if c == 1 else ACCENT)
            ax.set_title(title, color=color, fontsize=10)

    # Hide second row
    for c in range(4):
        axes[1][c].axis("off")

    plt.tight_layout()
    _save(fig, "03b_rgb_denoising.png")
    print(f"  [✓] Saved  →  {OUTPUT_DIR}/03b_rgb_denoising.png")

    # ── Singular value comparison: clean vs noisy ────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Section 3c — Noise Impact on Singular Value Spectrum")
    
    _, sg_clean, _ = np.linalg.svd(gray_img.astype(float), full_matrices=False)
    _, sg_noisy, _ = np.linalg.svd(gauss_noisy.astype(float), full_matrices=False)

    k_show = 80
    idx = np.arange(1, k_show + 1)
    ax1.semilogy(idx, sg_clean[:k_show], color=SUCCESS, lw=2, label="Clean image")
    ax1.semilogy(idx, sg_noisy[:k_show], color=WARNING, lw=2,
                ls="--", label="Noisy image")
    ax1.set_xlabel("Singular value index i")
    ax1.set_ylabel("σᵢ  (log scale)")
    ax1.set_title("Grayscale: SVD Spectrum: Clean vs Noisy")
    ax1.legend(facecolor="#161b22", edgecolor="#30363d")
    ax1.grid(alpha=0.3)

    # RGB channel comparison
    _, sg_clean_r, _ = np.linalg.svd(rgb_img[:, :, 0].astype(float), full_matrices=False)
    _, sg_noisy_r, _ = np.linalg.svd(rgb_gauss_noisy[:, :, 0].astype(float), full_matrices=False)

    ax2.semilogy(idx, sg_clean_r[:k_show], color=ERROR, lw=2, label="Clean (Red)")
    ax2.semilogy(idx, sg_noisy_r[:k_show], color=ACCENT, lw=2, ls="--", label="Noisy (Red)")
    ax2.set_xlabel("Singular value index i")
    ax2.set_ylabel("σᵢ  (log scale)")
    ax2.set_title("RGB Red Channel: SVD Spectrum: Clean vs Noisy")
    ax2.legend(facecolor="#161b22", edgecolor="#30363d")
    ax2.grid(alpha=0.3)

    _save(fig, "03c_noise_spectrum.png")
    print(f"  [✓] Saved  →  {OUTPUT_DIR}/03c_noise_spectrum.png")


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
    print(f"  {label:30s} | input PSNR={psnr_in:5.1f} dB → "
          f"output PSNR={psnr_out:5.1f} dB  "
          f"(Δ = {psnr_out - psnr_in:+.1f} dB)")


def _print_denoise_metrics_rgb(
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
    print(f"  {label:30s} | input PSNR={psnr_in:5.1f} dB → "
          f"output PSNR={psnr_out:5.1f} dB  "
          f"(Δ = {psnr_out - psnr_in:+.1f} dB)")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — MATRIX TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════════════════════

def build_rotation_matrix(theta_deg: float) -> np.ndarray:
    """2-D rotation matrix for angle θ (in degrees)."""
    theta = np.radians(theta_deg)
    c, s  = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


def build_scale_matrix(sx: float, sy: float) -> np.ndarray:
    """2-D scaling matrix."""
    return np.array([[sx, 0.0],
                     [0.0, sy]])


def build_shear_matrix(kx: float = 0.3, ky: float = 0.0) -> np.ndarray:
    """2-D shear matrix."""
    return np.array([[1.0, kx],
                     [ky,  1.0]])


def apply_affine_transform(
    img:    np.ndarray,
    M:      np.ndarray,
    flags:  int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Apply a 2×2 linear transformation matrix M to an image."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2

    M23 = np.zeros((2, 3), dtype=np.float64)
    M23[:2, :2] = M
    M23[0, 2]   = cx - M[0, 0] * cx - M[0, 1] * cy
    M23[1, 2]   = cy - M[1, 0] * cx - M[1, 1] * cy

    return cv2.warpAffine(img, M23, (w, h), flags=flags,
                          borderMode=cv2.BORDER_REFLECT)


def matrix_transformations_demo(gray_img: np.ndarray) -> None:
    """Demonstrate rotation, scaling, shear, and their eigenvalue analysis."""
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
    fig.suptitle("Section 4 — Linear (Affine) Image Transformations  p′ = M(p − c) + c")

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
    """Perform and visualise eigendecomposition of the image's covariance matrix."""
    print("\n" + "═" * 60)
    print("  SECTION 5 — EIGENVALUE / EIGENVECTOR ANALYSIS")
    print("═" * 60)

    A = gray_img.astype(np.float64)
    X   = A - A.mean(axis=1, keepdims=True)
    C   = (X.T @ X) / (X.shape[0] - 1)

    print(f"  Covariance matrix shape : {C.shape}")
    print(f"  Symmetric?              : {np.allclose(C, C.T)}")

    t0   = time.perf_counter()
    eigvals, eigvecs = np.linalg.eigh(C)
    t_eig = time.perf_counter() - t0

    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    print(f"  Eigendecomposition time : {t_eig*1000:.1f} ms")
    print(f"  Top-5 eigenvalues       : "
          + "  ".join([f"{v:.1f}" for v in eigvals[:5]]))
    print(f"  % variance explained:")
    total_var = eigvals.sum()
    for i in [1, 5, 10, 20]:
        pct = eigvals[:i].sum() / total_var * 100
        print(f"    Top {i:3d} eigenvectors : {pct:.2f}%")

    k_vals = [5, 20, 50, 100]
    recons = {}
    for k in k_vals:
        Vk    = eigvecs[:, :k]
        Z     = X @ Vk
        Xrec  = Z @ Vk.T
        Arec  = np.clip(Xrec + A.mean(axis=1, keepdims=True), 0, 255)
        recons[k] = Arec.astype(np.uint8)

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Section 5 — Eigendecomposition of Image Covariance Matrix\n"
                 "A·v = λ·v  (eigenvectors = principal directions of variance)")

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.3)

    ax0 = fig.add_subplot(gs[0, :2])
    k_show = min(60, len(eigvals))
    ax0.bar(range(1, k_show + 1), eigvals[:k_show], color=PURPLE, alpha=0.8)
    ax0.set_xlabel("Index i")
    ax0.set_ylabel("Eigenvalue λᵢ")
    ax0.set_title("Eigenvalue Spectrum of Covariance Matrix C")
    _add_subtitle(ax0, "Large λᵢ = directions of high image variance")
    ax0.grid(axis="y", alpha=0.35)

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
    """Apply PCA to a dataset of image patches."""
    print("\n" + "═" * 60)
    print("  SECTION 6 — PCA ON IMAGE PATCHES")
    print("═" * 60)

    A = gray_img.astype(np.float64)
    m, n = A.shape
    p    = patch_size
    d    = p * p

    patches = []
    for i in range(0, m - p + 1, p):
        for j in range(0, n - p + 1, p):
            patch = A[i:i+p, j:j+p].flatten()
            patches.append(patch)

    X = np.array(patches)
    n_patches = X.shape[0]
    print(f"  Patch size     : {p}×{p} = {d} dims")
    print(f"  Total patches  : {n_patches}")
    print(f"  Data matrix X  : {X.shape}")

    mu   = X.mean(axis=0)
    Xc   = X - mu

    U, sigma, Vt = np.linalg.svd(Xc, full_matrices=False)

    eigvals   = sigma**2 / (n_patches - 1)
    cum_var   = np.cumsum(eigvals) / eigvals.sum()

    print(f"  Top-5 eigenvalues (PCA): "
          + "  ".join([f"{v:.2f}" for v in eigvals[:5]]))

    k_values = [1, 4, 8, 16, 32]
    errors   = []
    for k in k_values:
        Vk    = Vt[:k, :].T
        Z     = Xc @ Vk
        Xrec  = Z @ Vk.T + mu
        err   = np.mean((Xc - (Xrec - mu))**2)
        errors.append(err)
        print(f"  k={k:3d}  reconstruction MSE = {err:.4f}")

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Section 6 — PCA on Image Patches  (PCA ≡ SVD on centred data matrix)")

    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.5, wspace=0.3)

    n_show = min(10, d)
    for i in range(5):
        ax = fig.add_subplot(gs[0, i])
        pc = Vt[i, :].reshape(p, p)
        ax.imshow(pc, cmap="RdBu_r")
        ax.set_title(f"PC{i+1}\nλ={eigvals[i]:.2f}")
        ax.axis("off")
        if i == 0:
            _add_subtitle(ax, "Most variance")

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

def rgb_svd_compress(rgb_img: np.ndarray, k_values: list) -> None:
    """Apply SVD compression independently to each RGB channel."""
    print("\n" + "═" * 60)
    print("  SECTION 7 — RGB SVD COMPRESSION")
    print("═" * 60)

    channel_names = ["Red", "Green", "Blue"]
    channel_colors = [ERROR, SUCCESS, ACCENT]

    fig, axes = plt.subplots(
        len(k_values) + 1, 4,
        figsize=(12, 3.2 * (len(k_values) + 1))
    )
    fig.suptitle("Section 7 — RGB Image SVD Compression\nT = (Rₖ ‖ Gₖ ‖ Bₖ)")

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
    """Time and space trade-off analysis for SVD compression."""
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
        size = k * (m + n + 1) / (m * n) * 100

        times.append(t_svd * 1000)
        psnrs.append(psnr)
        sizes.append(size)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Section 8 — Time & Space Trade-offs for SVD Compression")

    ax1.plot(k_range, times, color=WARNING, lw=2, marker="o", ms=4)
    ax1.set_xlabel("k")
    ax1.set_ylabel("Wall-clock time (ms)")
    ax1.set_title("Computation Time vs k")
    ax1.grid(alpha=0.3)

    ax2.plot(k_range, sizes, color=PURPLE, lw=2, marker="s", ms=4)
    ax2.axhline(100, color=MUTED, lw=0.8, ls="--", label="100% = original size")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Storage (% of original)")
    ax2.set_title("Storage Cost vs k")
    ax2.legend(facecolor="#161b22", edgecolor="#30363d")
    ax2.grid(alpha=0.3)

    ax3.scatter(sizes, psnrs, c=k_range, cmap="plasma", s=50, zorder=3)
    ax3.set_xlabel("Storage (%)")
    ax3.set_ylabel("PSNR (dB)  ↑ better")
    ax3.set_title("Quality–Storage Pareto Curve")
    ax3.grid(alpha=0.3)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(cmap="plasma",
                              norm=plt.Normalize(min(k_range), max(k_range))),
        ax=ax3
    )
    cb.set_label("k", color="#e6edf3")

    _save(fig, "08_benchmarks.png")
    print(f"  [✓] Saved  →  {OUTPUT_DIR}/08_benchmarks.png")

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
    """One-page academic summary figure combining key results."""
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

    _dash_img(fig, gs[0, 0], gray_img, "Grayscale\nA ∈ ℝ^(m×n)", "gray")
    _dash_img(fig, gs[0, 1], rgb_img,  "RGB Tensor\nT ∈ ℝ^(m×n×3)")
    _dash_img(fig, gs[0, 2], comp,     f"SVD Compressed\nk={k_best}", "gray")
    _dash_img(fig, gs[0, 3], noisy,    "Noisy (Gaussian)", "gray")
    _dash_img(fig, gs[0, 4], den,      "SVD Denoised\nk=25", "gray")

    rot   = apply_affine_transform(gray_img, build_rotation_matrix(45))
    scale = apply_affine_transform(gray_img, build_scale_matrix(1.3, 0.7))
    shear = apply_affine_transform(gray_img, build_shear_matrix(kx=0.5))
    _dash_img(fig, gs[1, 0], rot,   "Rotation 45°\ndet=1", "gray")
    _dash_img(fig, gs[1, 1], scale, "Scaling (1.3, 0.7)\ndet=0.91", "gray")
    _dash_img(fig, gs[1, 2], shear, "Shear kₓ=0.5\ndet=1", "gray")

    ax_sv = fig.add_subplot(gs[1, 3:])
    A     = gray_img.astype(float)
    _, sg, _ = np.linalg.svd(A, full_matrices=False)
    ax_sv.semilogy(sg[:80], color=ACCENT, lw=2)
    ax_sv.set_xlabel("Index i")
    ax_sv.set_ylabel("σᵢ")
    ax_sv.set_title("Singular Value Decay")
    ax_sv.grid(alpha=0.3)

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
    
    Steps:
    1. Load → convert to RGB and Grayscale
    2. Denoise (both grayscale and RGB)
    3. Scale up 1.5× and down 0.5×
    4. Rotate 45° and 90°
    5. Visualise and save all outputs
    """
    print("\n" + "═" * 60)
    print("  SECTION 10 — USER IMAGE UPLOAD PROCESSING PIPELINE")
    print("═" * 60)

    if not os.path.isfile(image_path):
        print(f"  [ERROR] File not found: {image_path}")
        sys.exit(1)

    upload_dir = os.path.join(OUTPUT_DIR, "upload_processing")
    os.makedirs(upload_dir, exist_ok=True)

    print(f"\n  ── Step 1: Load image ──")
    print(f"  [INFO] Loading: {image_path}")
    pil_img  = Image.open(image_path)
    rgb_img  = np.array(pil_img.convert("RGB"))
    gray_img = np.array(pil_img.convert("L"))

    h, w = gray_img.shape
    print(f"  Image size       : {w} × {h}")
    print(f"  Grayscale matrix : {gray_img.shape}")
    print(f"  RGB tensor       : {rgb_img.shape}")

    print("\n  ── Step 2: Denoising ──")

    svd_denoised_gray   = svd_denoise(gray_img, k=30)
    gauss_denoised_gray = gaussian_filter_denoise(gray_img, sigma=1.5)
    
    # NEW: RGB denoising
    svd_denoised_rgb    = svd_denoise_rgb(rgb_img, k=30)
    gauss_denoised_rgb  = gaussian_filter_denoise_rgb(rgb_img, sigma=1.5)

    print("  Grayscale SVD denoising  (k=30)       : done")
    print("  Grayscale Gaussian filter (sigma=1.5) : done")
    print("  RGB SVD denoising        (k=30)       : done")
    print("  RGB Gaussian filter      (sigma=1.5)  : done")

    print("\n  ── Step 3: Scaling ──")

    scaled_up   = cv2.resize(gray_img, (int(w * 1.5), int(h * 1.5)),
                             interpolation=cv2.INTER_LINEAR)
    scaled_dn   = cv2.resize(gray_img, (max(1, int(w * 0.5)), max(1, int(h * 0.5))),
                             interpolation=cv2.INTER_AREA)
    print(f"  Scaled UP  (1.5×) : {gray_img.shape} → {scaled_up.shape}")
    print(f"  Scaled DOWN (0.5×): {gray_img.shape} → {scaled_dn.shape}")

    print("\n  ── Step 4: Rotation ──")

    R45 = build_rotation_matrix(45)
    R90 = build_rotation_matrix(90)
    rotated_45 = apply_affine_transform(gray_img, R45)
    rotated_90 = apply_affine_transform(gray_img, R90)

    print(f"  Rotated 45°  det(R) = {np.linalg.det(R45):.3f}")
    print(f"  Rotated 90°  det(R) = {np.linalg.det(R90):.3f}")

    print("\n  ── Step 5: Visualisation ──")

    # Grayscale processing grid
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle("LAA Image Processing Pipeline - Grayscale", fontsize=13)

    axes[0][0].imshow(gray_img, cmap="gray", vmin=0, vmax=255)
    axes[0][0].set_title("Original", color=SUCCESS)
    axes[0][0].axis("off")

    axes[0][1].imshow(svd_denoised_gray, cmap="gray", vmin=0, vmax=255)
    axes[0][1].set_title("SVD Denoised (k=30)", color=ACCENT)
    axes[0][1].axis("off")

    axes[0][2].imshow(gauss_denoised_gray, cmap="gray", vmin=0, vmax=255)
    axes[0][2].set_title("Gaussian Denoised", color=ACCENT)
    axes[0][2].axis("off")

    axes[0][3].imshow(scaled_up, cmap="gray", vmin=0, vmax=255)
    axes[0][3].set_title(f"Scaled UP 1.5× ({scaled_up.shape[1]}×{scaled_up.shape[0]})", color=PURPLE)
    axes[0][3].axis("off")

    axes[1][0].imshow(scaled_dn, cmap="gray", vmin=0, vmax=255)
    axes[1][0].set_title(f"Scaled DOWN 0.5× ({scaled_dn.shape[1]}×{scaled_dn.shape[0]})", color=PURPLE)
    axes[1][0].axis("off")

    axes[1][1].imshow(rotated_45, cmap="gray", vmin=0, vmax=255)
    axes[1][1].set_title("Rotated 45°", color=PURPLE)
    axes[1][1].axis("off")

    axes[1][2].imshow(rotated_90, cmap="gray", vmin=0, vmax=255)
    axes[1][2].set_title("Rotated 90°", color=PURPLE)
    axes[1][2].axis("off")

    axes[1][3].axis("off")

    plt.tight_layout()
    gray_grid = os.path.join(upload_dir, "grayscale_processing_grid.png")
    fig.savefig(gray_grid, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  [✓] Grayscale grid saved → {gray_grid}")

    # RGB processing grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("LAA Image Processing Pipeline - RGB", fontsize=13)

    axes[0][0].imshow(rgb_img)
    axes[0][0].set_title("Original RGB", color=SUCCESS)
    axes[0][0].axis("off")

    axes[0][1].imshow(svd_denoised_rgb)
    axes[0][1].set_title("SVD Denoised RGB (k=30)", color=ACCENT)
    axes[0][1].axis("off")

    axes[0][2].imshow(gauss_denoised_rgb)
    axes[0][2].set_title("Gaussian Denoised RGB", color=ACCENT)
    axes[0][2].axis("off")

    # Hide bottom row
    for c in range(3):
        axes[1][c].axis("off")

    plt.tight_layout()
    rgb_grid = os.path.join(upload_dir, "rgb_processing_grid.png")
    fig.savefig(rgb_grid, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  [✓] RGB grid saved → {rgb_grid}")

    # Save individual images
    print("\n  ── Step 6: Saving outputs ──")
    
    saves = [
        ("original_gray.png",         gray_img,             "L"),
        ("grayscale.png",             gray_img,             "L"),
        ("svd_denoised_gray.png",     svd_denoised_gray,    "L"),
        ("gaussian_denoised_gray.png",gauss_denoised_gray,  "L"),
        ("scaled_up.png",             scaled_up,            "L"),
        ("scaled_down.png",           scaled_dn,            "L"),
        ("rotated_45.png",            rotated_45,           "L"),
        ("rotated_90.png",            rotated_90,           "L"),
        ("original_rgb.png",          rgb_img,              "RGB"),
        ("svd_denoised_rgb.png",      svd_denoised_rgb,     "RGB"),
        ("gaussian_denoised_rgb.png", gauss_denoised_rgb,   "RGB"),
    ]

    for fname, arr, mode in saves:
        out = os.path.join(upload_dir, fname)
        Image.fromarray(arr, mode=mode).save(out)
        print(f"  [✓] Saved → {fname}")

    print(f"\n  ✓ Pipeline complete. Outputs in: {upload_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _add_subtitle(ax: plt.Axes, text: str, y: float = -0.08) -> None:
    ax.text(0.5, y, text, ha="center", va="top",
            fontsize=6.5, color=MUTED, transform=ax.transAxes, wrap=True)


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
#  MENU-DRIVEN INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def print_main_menu():
    """Display main menu."""
    print("\n" + "═" * 70)
    print("  LINEAR ALGEBRA & IMAGE PROCESSING — INTERACTIVE MENU")
    print("═" * 70)
    print("\n  Main Menu Options:\n")
    print("    [1] Run All Sections (Full Project)")
    print("    [2] Run Custom Section Selection")
    print("    [3] Process Uploaded Image (Full Pipeline)")
    print("    [4] Image Denoising Demo (Grayscale & RGB)")
    print("    [5] SVD Compression Demo")
    print("    [6] Matrix Transformations Demo")
    print("    [7] Eigenanalysis & PCA Demo")
    print("    [8] Generate & Save Summary Dashboard")
    print("    [0] Exit\n")


def print_section_menu():
    """Display section selection menu."""
    print("\n  Available Sections:\n")
    print("    [1] Image Representation")
    print("    [2] SVD Compression")
    print("    [3] Image Denoising (Grayscale & RGB)")
    print("    [4] Matrix Transformations")
    print("    [5] Eigenvalue Analysis")
    print("    [6] PCA on Image Patches")
    print("    [7] RGB SVD Compression")
    print("    [8] Performance Benchmarking")
    print("    [9] Summary Dashboard\n")


def run_all_sections():
    """Run all available sections."""
    print("\n[INFO] Running all sections...\n")
    
    gray_img, rgb_img = generate_demo_images(size=256)
    
    load_and_represent_image(gray_img, rgb_img)
    compressed = svd_compress(gray_img, k_values=[1, 5, 15, 30, 60, 100])
    image_denoise_demo(gray_img, rgb_img)
    matrix_transformations_demo(gray_img)
    eigenanalysis_demo(gray_img)
    pca_demo(gray_img, patch_size=8)
    rgb_svd_compress(rgb_img, k_values=[5, 20, 50, 100])
    performance_benchmark(gray_img)
    summary_dashboard(gray_img, rgb_img, compressed)
    
    print("\n" + "═" * 70)
    print("  ✓ All sections completed!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("═" * 70)


def run_custom_sections():
    """Run user-selected sections."""
    print_section_menu()
    
    selected = []
    while True:
        choice = input("  Enter section number (or 'done' to finish): ").strip()
        if choice.lower() == 'done':
            break
        if choice in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            selected.append(int(choice))
        else:
            print("  [ERROR] Invalid choice. Try again.")
    
    if not selected:
        print("  [INFO] No sections selected.")
        return
    
    gray_img, rgb_img = generate_demo_images(size=256)
    compressed = None
    
    section_funcs = {
        1: lambda: load_and_represent_image(gray_img, rgb_img),
        2: lambda: svd_compress(gray_img),
        3: lambda: image_denoise_demo(gray_img, rgb_img),
        4: lambda: matrix_transformations_demo(gray_img),
        5: lambda: eigenanalysis_demo(gray_img),
        6: lambda: pca_demo(gray_img),
        7: lambda: rgb_svd_compress(rgb_img, k_values=[5, 20, 50, 100]),
        8: lambda: performance_benchmark(gray_img),
        9: lambda: summary_dashboard(gray_img, rgb_img, 
                                     svd_compress(gray_img, show_energy=False))
    }
    
    for sec in sorted(selected):
        try:
            section_funcs[sec]()
        except Exception as e:
            print(f"  [ERROR] Section {sec} failed: {e}")
    
    print("\n[✓] Selected sections completed!")


def process_user_image():
    """Process user-uploaded image."""
    image_path = input("\n  Enter image path: ").strip()
    
    if not os.path.isfile(image_path):
        print(f"  [ERROR] File not found: {image_path}")
        return
    
    process_uploaded_image(image_path)


def run_denoising_demo():
    """Run just denoising demo."""
    print("\n[INFO] Running denoising demo (Grayscale & RGB)...\n")
    gray_img, rgb_img = generate_demo_images(size=256)
    image_denoise_demo(gray_img, rgb_img)
    print("\n[✓] Denoising demo completed!")


def run_svd_compression_demo():
    """Run just SVD compression demo."""
    print("\n[INFO] Running SVD compression demo...\n")
    gray_img, rgb_img = generate_demo_images(size=256)
    svd_compress(gray_img, k_values=[1, 5, 15, 30, 60, 100])
    print("\n[✓] SVD compression demo completed!")


def run_transformations_demo():
    """Run just transformations demo."""
    print("\n[INFO] Running transformations demo...\n")
    gray_img, rgb_img = generate_demo_images(size=256)
    matrix_transformations_demo(gray_img)
    print("\n[✓] Transformations demo completed!")


def run_eigen_pca_demo():
    """Run eigenanalysis and PCA demo."""
    print("\n[INFO] Running eigenanalysis & PCA demo...\n")
    gray_img, rgb_img = generate_demo_images(size=256)
    eigenanalysis_demo(gray_img)
    pca_demo(gray_img, patch_size=8)
    print("\n[✓] Eigenanalysis & PCA demo completed!")


def run_dashboard():
    """Generate summary dashboard."""
    print("\n[INFO] Generating summary dashboard...\n")
    gray_img, rgb_img = generate_demo_images(size=256)
    compressed = svd_compress(gray_img, show_energy=False)
    summary_dashboard(gray_img, rgb_img, compressed)
    print("\n[✓] Dashboard generated!")


def interactive_menu():
    """Main interactive menu loop."""
    while True:
        print_main_menu()
        choice = input("  Enter your choice: ").strip()
        
        if choice == '1':
            run_all_sections()
        elif choice == '2':
            run_custom_sections()
        elif choice == '3':
            process_user_image()
        elif choice == '4':
            run_denoising_demo()
        elif choice == '5':
            run_svd_compression_demo()
        elif choice == '6':
            run_transformations_demo()
        elif choice == '7':
            run_eigen_pca_demo()
        elif choice == '8':
            run_dashboard()
        elif choice == '0':
            print("\n  [INFO] Exiting. Goodbye!\n")
            sys.exit(0)
        else:
            print("  [ERROR] Invalid choice. Please try again.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    interactive_menu()
