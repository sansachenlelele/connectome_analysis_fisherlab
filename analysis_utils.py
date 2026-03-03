"""analysis_utils.py

Reusable helper functions extracted from analysis_get-main-value-25summer.ipynb.

Design goals
- No hidden notebook globals: configs (images/masks/z-ranges) are passed in as args.
- Works with both .czi and .tif/.tiff inputs (best-effort shape normalization to CZYX).
- Plot helpers return (fig, ax) so the notebook decides save/show.

Author: extracted with ChatGPT
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tifffile as tf
from skimage.measure import regionprops_table

try:
    from aicsimageio import AICSImage
except Exception:  # pragma: no cover
    AICSImage = None


# -----------------------------
# Config/path helpers
# -----------------------------

def get_image_path(fly: str, images: Dict[str, Union[str, Path]], require: bool = True) -> Optional[str]:
    """Return image path for a fly key from `images` dict."""
    if fly in images:
        return str(images[fly])
    if require:
        raise KeyError(f"image path for {fly} is missing.")
    return None


def get_mask_paths(
    fly: str,
    label_masks: Dict[str, Union[str, Path]],
    label_masks_checking: Dict[str, Union[str, Path]],
    require: bool = True,
) -> Dict[str, Optional[str]]:
    """Return dict with keys {'mask','bgmask'} for a fly."""
    mask = label_masks.get(fly)
    bgmask = label_masks_checking.get(fly)
    if require and (mask is None or bgmask is None):
        missing = [name for name, val in [("mask", mask), ("bgmask", bgmask)] if val is None]
        raise KeyError(f"{', '.join(missing)} path(s) for {fly} are missing.")
    return {"mask": None if mask is None else str(mask), "bgmask": None if bgmask is None else str(bgmask)}


def _normalize_ranges(ranges: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Ensure ints and (start <= end)."""
    norm: List[Tuple[int, int]] = []
    for a, b in ranges:
        a, b = int(a), int(b)
        if a > b:
            a, b = b, a
        norm.append((a, b))
    return norm


def get_z_ranges(
    fly: str,
    z_ranges_by_fly: Dict[str, Iterable[Tuple[int, int]]],
    Z: Optional[int] = None,
    clip: bool = False,
) -> List[Tuple[int, int]]:
    """Get (start,end) z-ranges for `fly`.

    If Z is provided:
      - clip=True: clips to [0, Z-1]
      - clip=False: raises if any range is out of bounds
    """
    rngs = _normalize_ranges(z_ranges_by_fly[fly])  # KeyError if missing

    if Z is not None:
        if clip:
            rngs = [(max(0, min(a, Z - 1)), max(0, min(b, Z - 1))) for a, b in rngs]
        else:
            bad = [(i, (a, b)) for i, (a, b) in enumerate(rngs) if not (0 <= a <= b < Z)]
            if bad:
                preview = bad[:5]
                raise ValueError(
                    f"z_ranges for {fly} exceed Z={Z}: bad entries {preview}{'...' if len(bad) > 5 else ''}"
                )
    return rngs


# -----------------------------
# Image/mask loading
# -----------------------------

def _as_czyx_from_tif(arr: np.ndarray) -> np.ndarray:
    """Best-effort conversion of tifffile array to CZYX."""
    a = np.asarray(arr)
    if a.ndim == 2:
        # YX -> 1x1xYxX
        return a[None, None, :, :]
    if a.ndim == 3:
        # Usually ZYX
        return a[None, :, :, :]
    if a.ndim == 4:
        # Could be CZYX or ZYXC. Heuristic: if last dim small, assume channels last.
        if a.shape[-1] <= 6 and a.shape[0] > 6:
            # ZYXC -> CZYX
            return np.moveaxis(a, -1, 0)
        # else assume CZYX already
        return a
    raise ValueError(f"Unsupported tif array ndim={a.ndim} shape={a.shape}")


def load_image_czyx(image_path: Union[str, Path]) -> np.ndarray:
    """Load image and return numpy array in CZYX order.

    Supports .czi via AICSImage (preferred) and .tif/.tiff via tifffile.

    Note: this mirrors your notebook's expectation: `data = img.get_image_data('CZYX', S=0, T=0)`
    """
    p = Path(image_path)
    suf = p.suffix.lower()

    if suf == ".czi":
        if AICSImage is None:
            raise ImportError("aicsimageio is required to read .czi files (AICSImage import failed).")
        img = AICSImage(str(p))
        data = img.get_image_data("CZYX", S=0, T=0)
        if data.ndim != 4:
            raise ValueError(f"Expected CZYX array from AICSImage; got shape {data.shape}")
        return np.asarray(data)

    if suf in {".tif", ".tiff"}:
        arr = tf.imread(str(p))
        return _as_czyx_from_tif(arr)

    raise ValueError(f"Unsupported image type: {p.name} (suffix={suf})")


def load_fly(
    fly: str,
    images: Dict[str, Union[str, Path]],
    red_idx: int = 1,
    green_idx: int = 2,
    blue_idx: int = 0,
) -> Dict[str, object]:
    """Load channels for a fly.

    Returns dict with keys: red_channel, green_channel, blue_channel, shape, paths
    where channels are ZYX arrays.
    """
    image_path = get_image_path(fly, images=images, require=True)
    data = load_image_czyx(image_path)

    C = data.shape[0]
    for ix, nm in [(red_idx, "red_idx"), (green_idx, "green_idx"), (blue_idx, "blue_idx")]:
        if ix >= C:
            raise ValueError(f"{nm}={ix} out of bounds for C={C}")

    return {
        "red_channel": data[red_idx],
        "green_channel": data[green_idx],
        "blue_channel": data[blue_idx],
        "shape": data.shape,
        "paths": {"image": image_path},
    }


def load_fly_mask(
    fly: str,
    label_masks: Dict[str, Union[str, Path]],
    label_masks_checking: Dict[str, Union[str, Path]],
    require_masks: bool = False,
) -> Dict[str, np.ndarray]:
    """Load label masks for a fly.

    If masks are missing and require_masks=False, returns {}.
    """
    paths = get_mask_paths(
        fly, label_masks=label_masks, label_masks_checking=label_masks_checking, require=require_masks
    )
    mask_path, bgmask_path = paths.get("mask"), paths.get("bgmask")

    if not mask_path or not bgmask_path:
        return {}

    label_mask = tf.imread(mask_path)
    label_mask_checking = tf.imread(bgmask_path)

    return {
        "label_mask": label_mask,
        "label_mask_checking": label_mask_checking,
    }


# -----------------------------
# Quant helpers (from notebook)
# -----------------------------

def qtile(x: np.ndarray, q: float) -> float:
    """Quantile helper used throughout the notebook."""
    x = np.asarray(x)
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def per_slice_percentiles_red_and_green_diff(
    red_channel: np.ndarray,
    green_channel: np.ndarray,
    label_mask_checking: np.ndarray,
    q: Union[float, Sequence[float]],
    roi_id: Optional[int] = None,
    exclude_zero: bool = False,
    round_pct: int = 1,
) -> pd.DataFrame:
    """Compute per-Z percentile thresholds in a 2D ROI footprint.

    q:
      - float -> same percentile for red and green
      - (q_red, q_green) -> different percentiles per channel
    """
    if isinstance(q, (tuple, list, np.ndarray)):
        if len(q) != 2:
            raise ValueError("If q is a sequence, it must have length 2: (q_red, q_green)")
        q_red, q_green = float(q[0]), float(q[1])
    else:
        q_red = q_green = float(q)

    labels = np.asarray(label_mask_checking)
    if labels.ndim != 2:
        raise ValueError(f"label_mask_checking must be 2D (YX). Got {labels.shape}")

    if roi_id is None:
        # pick largest positive label
        vals, counts = np.unique(labels[labels > 0], return_counts=True)
        if vals.size == 0:
            raise ValueError("No positive labels found in label_mask_checking.")
        roi_id = int(vals[np.argmax(counts)])

    roi2d = labels == roi_id
    if roi2d.sum() == 0:
        raise ValueError(f"ROI id {roi_id} not found (0 pixels).")

    red = np.asarray(red_channel)
    green = np.asarray(green_channel)
    if red.shape != green.shape:
        raise ValueError(f"red_channel and green_channel must have same shape. {red.shape} vs {green.shape}")
    if red.ndim != 3:
        raise ValueError(f"Channels must be ZYX. Got {red.shape}")

    Z = red.shape[0]
    rows = []
    for z in range(Z):
        r = red[z][roi2d]
        g = green[z][roi2d]
        if exclude_zero:
            keep = (r != 0) & (g != 0)
            r = r[keep]
            g = g[keep]

        rt = qtile(r, q_red)
        gt = qtile(g, q_green)

        # fraction <= threshold (as percent)
        r_frac = np.nan if r.size == 0 else np.mean(r <= rt) * 100.0
        g_frac = np.nan if g.size == 0 else np.mean(g <= gt) * 100.0

        rows.append(
            dict(
                z=z,
                roi_pixels=int(r.size),
                red_threshold=rt,
                red_frac_leq_pct=round(float(r_frac), round_pct) if np.isfinite(r_frac) else np.nan,
                percentile_q_red=q_red,
                green_threshold=gt,
                green_frac_leq_pct=round(float(g_frac), round_pct) if np.isfinite(g_frac) else np.nan,
                percentile_q_green=q_green,
                roi_id=int(roi_id),
            )
        )
    return pd.DataFrame(rows)


def per_slice_percentiles_and_mean_for_roi(
    channel: np.ndarray,
    label_mask: np.ndarray,
    q: float,
    roi_id: int,
    exclude_zero: bool = False,
) -> pd.DataFrame:
    """Per-Z percentile threshold and mean for a specific ROI id in label_mask."""
    labels = np.asarray(label_mask)
    if labels.ndim != 2:
        raise ValueError(f"label_mask must be 2D (YX). Got {labels.shape}")
    roi2d = labels == int(roi_id)
    if roi2d.sum() == 0:
        raise ValueError(f"ROI id {roi_id} not found (0 pixels).")

    ch = np.asarray(channel)
    if ch.ndim != 3:
        raise ValueError(f"channel must be ZYX. Got {ch.shape}")

    rows = []
    for z in range(ch.shape[0]):
        x = ch[z][roi2d]
        if exclude_zero:
            x = x[x != 0]
        thr = qtile(x, q)
        mean = float(np.mean(x)) if x.size else float("nan")
        rows.append(dict(z=z, roi_pixels=int(x.size), threshold=thr, mean=mean, q=float(q), roi_id=int(roi_id)))
    return pd.DataFrame(rows)


# -----------------------------
# Plot helpers
# -----------------------------

def make_hist(ax, data: np.ndarray, bins: int = 200, xlabel: str = "Intensity", ylabel: str = "Count", title: str = ""):
    data = np.asarray(data).ravel()
    data = data[np.isfinite(data)]
    ax.hist(data, bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def plot_intensity_histograms_with_percentile(
    red_channel: np.ndarray,
    green_channel: np.ndarray,
    label_mask_checking: np.ndarray,
    q: Union[float, Sequence[float]],
    roi_id: Optional[int] = None,
    z: Optional[int] = None,
    bins: int = 200,
):
    """Plot red and green histograms for one z-slice (or middle slice) within a background ROI."""
    df = per_slice_percentiles_red_and_green_diff(
        red_channel=red_channel,
        green_channel=green_channel,
        label_mask_checking=label_mask_checking,
        q=q,
        roi_id=roi_id,
    )

    if z is None:
        z = int(df["z"].median())

    roi_id = int(df.iloc[0]["roi_id"])
    roi2d = (np.asarray(label_mask_checking) == roi_id)

    r = np.asarray(red_channel)[z][roi2d]
    g = np.asarray(green_channel)[z][roi2d]
    rt = float(df.loc[df["z"] == z, "red_threshold"].values[0])
    gt = float(df.loc[df["z"] == z, "green_threshold"].values[0])

    fig, ax = plt.subplots()
    make_hist(ax, r, bins=bins, title=f"Red ROI z={z}")
    ax.axvline(rt)
    plt.close(fig)

    fig2, ax2 = plt.subplots()
    make_hist(ax2, g, bins=bins, title=f"Green ROI z={z}")
    ax2.axvline(gt)
    plt.close(fig2)

    return (fig, ax), (fig2, ax2), df


def plot_roi_histograms_per_slice(
    channel: np.ndarray,
    label_mask: np.ndarray,
    roi_id: int,
    z_list: Sequence[int],
    q: Optional[float] = None,
    bins: int = 200,
):
    """Plot histograms for selected z slices for a given ROI."""
    roi2d = (np.asarray(label_mask) == int(roi_id))
    figs = []
    for z in z_list:
        x = np.asarray(channel)[int(z)][roi2d]
        fig, ax = plt.subplots()
        make_hist(ax, x, bins=bins, title=f"ROI {roi_id} z={z}")
        if q is not None:
            ax.axvline(qtile(x, float(q)))
        plt.close(fig)
        figs.append((fig, ax))
    return figs


def two_group_hist(ax, a: np.ndarray, b: np.ndarray, bins: int = 200, label_a: str = "A", label_b: str = "B", title: str = ""):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    ax.hist(a, bins=bins, alpha=0.5, label=label_a)
    ax.hist(b, bins=bins, alpha=0.5, label=label_b)
    if title:
        ax.set_title(title)
    ax.legend()


def plot_pb_vs_background_histograms(pb_vals: np.ndarray, bg_vals: np.ndarray, bins: int = 200, title: str = "PB vs BG"):
    fig, ax = plt.subplots()
    two_group_hist(ax, pb_vals, bg_vals, bins=bins, label_a="PB", label_b="Background", title=title)
    plt.close(fig)
    return fig, ax


def plot_pb_histograms_only(pb_vals: np.ndarray, bins: int = 200, title: str = "PB"):
    fig, ax = plt.subplots()
    make_hist(ax, pb_vals, bins=bins, title=title)
    plt.close(fig)
    return fig, ax


def plot_glomeruli_vs_background_histograms(glom_vals: np.ndarray, bg_vals: np.ndarray, bins: int = 200, title: str = "Glomeruli vs BG"):
    fig, ax = plt.subplots()
    two_group_hist(ax, glom_vals, bg_vals, bins=bins, label_a="Glomeruli", label_b="Background", title=title)
    plt.close(fig)
    return fig, ax
