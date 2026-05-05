"""
Utility functions for extracting fluorescence signal from airyscan images.
"""

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from aicsimageio import AICSImage
from skimage import io, measure
from skimage.measure import regionprops_table
import tifffile as tf
import czifile
from czifile import CziFile
from PIL import Image
import imageio

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import re

from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit

from matplotlib.patches import Patch
from typing import Dict, Iterable


def get_image_path(fly, images, require=True):
    """
    Return the image path for one fly.

    Parameters
    ----------
    fly : str
        Fly/sample name, such as "fly1".
    images : dict
        Dictionary mapping fly names to image file paths.
    require : bool, default=True
        If True, raise an error when the fly is missing.

    Returns
    -------
    str or None
        Image path for the selected fly, or None if missing and require=False.
    """
    if fly in images:
        return images[fly]
    if require:
        raise KeyError(f"Image path for {fly} is missing.")
    return None


def get_mask_paths(fly, label_masks, label_masks_checking, require=True):
    """
    Return the ROI mask and background/checking mask paths for one fly.

    Parameters
    ----------
    fly : str
        Fly/sample name.
    label_masks : dict
        Dictionary mapping fly names to full ROI label mask paths.
    label_masks_checking : dict
        Dictionary mapping fly names to background/checking mask paths.
    require : bool, default=True
        If True, raise an error when either mask path is missing.

    Returns
    -------
    dict
        Dictionary with keys "mask" and "bgmask".
    """
    mask = label_masks.get(fly)
    bgmask = label_masks_checking.get(fly)
    if require and (mask is None or bgmask is None):
        missing = [
            name for name, val in [("mask", mask), ("bgmask", bgmask)]
            if val is None]
        raise KeyError(f"{', '.join(missing)} path(s) for {fly} are missing.")
    return {"mask": mask, "bgmask": bgmask}


def load_fly(fly, images, red_idx=1, green_idx=2, blue_idx=0):
    """
    Load one Airyscan image and return red, green, and blue channel arrays. The image is loaded as a CZYX array using AICSImage.

    Parameters
    ----------
    fly : str
        Fly/sample name.
    images : dict
        Dictionary mapping fly names to image file paths.
    red_idx : int, default=1
        Channel index for the red channel.
    green_idx : int, default=2
        Channel index for the green channel.
    blue_idx : int, default=0
        Channel index for the blue channel.

    Returns
    -------
    dict
        Dictionary containing:
        - "red_channel"
        - "green_channel"
        - "blue_channel"
        - "shape"
        - "paths"
    """
    image_path = get_image_path(fly, images, require=True)

    img = AICSImage(image_path)
    data = img.get_image_data("CZYX", S=0, T=0)

    if data.ndim != 4:
        raise ValueError(f"Expected CZYX array; got shape {data.shape}")
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


def load_fly_mask(fly, label_masks, label_masks_checking, require_masks=False):
    """
    Load the full ROI label mask and background/checking label mask for one fly.

    Parameters
    ----------
    fly : str
        Fly/sample name.
    label_masks : dict
        Dictionary mapping fly names to full ROI mask paths.
    label_masks_checking : dict
        Dictionary mapping fly names to background/checking mask paths.
    require_masks : bool, default=False
        If True, raise an error when masks are missing. If False, return an
        empty dictionary when masks are missing.

    Returns
    -------
    dict
        Dictionary containing "label_mask" and "label_mask_checking",
        or an empty dictionary if masks are missing and require_masks=False.
    """
    paths = get_mask_paths(
        fly,
        label_masks=label_masks,
        label_masks_checking=label_masks_checking,
        require=require_masks,)
    mask_path = paths.get("mask")
    bgmask_path = paths.get("bgmask")

    if not mask_path or not bgmask_path:
        return {}

    label_mask = tf.imread(mask_path)
    label_mask_checking = tf.imread(bgmask_path)
    return {
        "label_mask": label_mask,
        "label_mask_checking": label_mask_checking,
    }


def _normalize_ranges(ranges):
    """
    Convert z-range values to integers and ensure each range is ordered.

    Parameters
    ----------
    ranges : list of tuple
        List of z-ranges, where each range is given as (start, end).

    Returns
    -------
    list of tuple
        Normalized list of integer z-ranges.
    """
    norm = []
    for a, b in ranges:
        a, b = int(a), int(b)
        if a > b:
            a, b = b, a
        norm.append((a, b))
    return norm


def get_z_ranges(fly, z_ranges_by_fly, Z=None, clip=False):
    """
    Return normalized z-ranges for one fly.

    Parameters
    ----------
    fly : str
        Fly/sample name.
    z_ranges_by_fly : dict
        Dictionary mapping fly names to manually defined z-ranges.
    Z : int, optional
        Number of z-slices in the image. If provided, ranges are checked
        against this value.
    clip : bool, default=False
        If True, clip z-ranges to fit inside [0, Z-1]. If False, raise an
        error when ranges exceed the image z-depth.

    Returns
    -------
    list of tuple
        Normalized z-ranges for the selected fly.
    """
    rngs = _normalize_ranges(z_ranges_by_fly[fly])

    if Z is not None:
        if clip:
            rngs = [
                (max(0, min(a, Z - 1)), max(0, min(b, Z - 1)))
                for a, b in rngs]
        else:
            bad = [
                (i, (a, b))
                for i, (a, b) in enumerate(rngs)
                if not (0 <= a <= b < Z)]
            if bad:
                raise ValueError(
                    f"z_ranges for {fly} exceed Z={Z}: "
                    f"bad entries {bad[:5]}{'...' if len(bad) > 5 else ''}"
                )
    return rngs



def per_slice_percentiles_red_and_green_diff(
    red_channel,
    green_channel,
    label_mask_checking,
    q,                      # float (old behavior) OR (q_red, q_green) as tuple/list (see docstring)
    roi_id=1,               # assume i draw label=1 as the background ROI for red and green.
    exclude_zero=False,     # ignore zeros inside the ROI (both channels)
    round_pct=1             # round the reported percentages to this many decimals
):
    """
    A background ROI is drawn on one z-slice in `label_mask_checking`. 
    This function extracts that 2D ROI footprint and applies it to every z-slice of the red and green image channels.
    For each z-slice, it calculates the requested percentile threshold separately for red and green.

    Parameters
    ----------
    red_channel, green_channel : ndarray
            3D image arrays with shape (Z, Y, X).
    label_mask_checking : ndarray
            2D or 3D label mask containing the background ROI.
    q : float or tuple of float
            Percentile value(s). If float, the same percentile is used for both channels. If tuple/list, use (q_red, q_green).
    roi_id : int, optional
            Label ID of the background ROI. If None, the largest positive label is used.
    exclude_zero : bool, default=False
        Whether to exclude zero-valued pixels before calculating percentiles.
    round_pct : int, default=1
        Number of decimal places for reporting fraction-below-threshold values.

    Returns
    -------
    pandas.DataFrame
        One row per z-slice, with red and green threshold values.
    """
    # -------- parse q into per-channel percentiles --------
    if isinstance(q, (list, tuple)):
        if len(q) != 2:
            raise ValueError("If q is a list/tuple, it must be length 2: (q_red, q_green).")
        q_red, q_green = float(q[0]), float(q[1])
    else:
        q_red = q_green = float(q)

    # ---- normalize labels to (Z,Y,X) or (Y,X) ----
    lbl = label_mask_checking                                                                               # should always be 3, so no need to squeeze. i changed this line from "bl = np.asarray(label_mask_checking)" to the current line
    if lbl.ndim > 3:
        lbl = np.squeeze(lbl)

    # ---- figure out ROI id ----
    pos_labels = np.unique(lbl[lbl > 0])
    if pos_labels.size == 0:
        raise ValueError("No positive labels found in label_mask_checking.")

    roi_id = int(roi_id)
    if roi_id not in pos_labels:
        raise ValueError(
            f"roi_id={roi_id} not present in label_mask_checking "
            f"(found {pos_labels.tolist()}).")
    
    # ---- build a 2D ROI mask (footprint) ----
    if lbl.ndim == 3:
        z_hits = np.where(np.any(lbl == roi_id, axis=(1, 2)))[0]                                            # the z slice that i draw this huge roi on.
        if z_hits.size == 0:
            raise ValueError(f"Label {roi_id} has no painted pixels.")
        z_draw = int(z_hits[0])                                                                             # z_draw should = z_hits, because i only draw one roi on one z slice.
        roi2d = (lbl[z_draw] == roi_id)   # (Y,X)                                                           # create a 2D array (X, Y) with T/F as entries: only T for pixels within the roi.
    elif lbl.ndim == 2:
        roi2d = (lbl == roi_id)
        z_draw = None
    else:
        raise ValueError(f"label_mask_checking must be 2D or 3D; got shape {lbl.shape}")

    if not roi2d.any():
        raise ValueError("ROI footprint is empty.")

    # ---- sanity: channels must be 3D (Z,Y,X) and spatially match ROI ----
    if red_channel.ndim != 3 or green_channel.ndim != 3:
        raise ValueError("red_channel and green_channel must be 3D (Z,Y,X).")
    Z, Y, X = red_channel.shape
    if green_channel.shape != (Z, Y, X):
        raise ValueError(f"green_channel shape {green_channel.shape} != red_channel shape {(Z, Y, X)}.")
    if roi2d.shape != (Y, X):
        raise ValueError(f"ROI footprint shape {roi2d.shape} does not match channel XY {(Y, X)}.")

    # ---- precompute ROI coordinates (apply same XY mask at every Z) ----
    y_ix, x_ix = np.where(roi2d)                                                                            # roi2d is a 2D array: the same positioned entry in array 1 (y coordinate) and array 2 (x coordinate) makes up one pixel
    roi_pixels = int(roi2d.sum())                                                                           # roi_pixels = total number of pixels included in this roi.

    # ---- helpers ----
    def qtile(a, qf):
        a = np.asarray(a)                                                                                   # when qtile is called, a is the array of all the VALUES for all the selected pixels. 
        a = a[np.isfinite(a)]                                                                               # just to make sure that no value is a infinite number (which there shouldn't be any).
        if a.size == 0:
            return np.nan
        try:
            return float(np.quantile(a, qf, method="linear"))                                               # interpolation: estimating a value between known data points, useful when the 80% percentile cutoff is not one exact rank.
        except TypeError:
            return float(np.quantile(a, qf, interpolation="linear"))

    rows = []
    for z in range(Z):                                                                                      # Z = total number of z slices (acquired from red_channel.shape). So it goes to all the z slices.
        rvals = red_channel[z,  y_ix, x_ix]                                                                 # it works: red_channel[16, [544, 544], [1097, 1098]] gives the red siganl value at pixel [16, 544, 1097] and pixel [16, 544, 1098]. So output an array like [21, 17].
        gvals = green_channel[z, y_ix, x_ix]

        if exclude_zero:                                                                                    # i should NEVER set exclude_zero = True.
            rvals = rvals[rvals != 0]
            gvals = gvals[gvals != 0]

        # thresholds at per-channel q
        thr_r = qtile(rvals, q_red)                                                                         # q_red doesn't = q_green! --> the changed part!!!!!!!!!!
        thr_g = qtile(gvals, q_green)

        # actual fractions ≤ threshold (ties may make this > q)
        frac_r = float((rvals <= thr_r).mean()) if rvals.size else np.nan                                   # just to check: what is the actual fraction of pixels that is below the calculated threshold.
        frac_g = float((gvals <= thr_g).mean()) if gvals.size else np.nan

        rows.append({
            "z": z,
            "roi_pixels": roi_pixels,
            "red_threshold":   thr_r,
            "red_frac_leq_pct":   (np.round(frac_r*100.0, round_pct) if np.isfinite(frac_r) else np.nan),
            "green_threshold": thr_g,
            "green_frac_leq_pct": (np.round(frac_g*100.0, round_pct) if np.isfinite(frac_g) else np.nan),
        })

    df = pd.DataFrame(rows, columns=[
        "z", "roi_pixels",
        "red_threshold", "red_frac_leq_pct", 
        "green_threshold", "green_frac_leq_pct",
    ])
    return df


def plot_per_slice_thresholds(
    df_thresholds,
    red_col="red_threshold",
    green_col="green_threshold",
    z_col="z",
    q_col=None,
):
    """
    Plot red and green per-z-slice threshold values.
    I used the more generalized version of this function: plot_per_slice_thresholds_rgb

    Parameters
    ----------
    df_thresholds : pandas.DataFrame
        DataFrame containing one row per z-slice and threshold columns.
    red_col : str, default="red_threshold"
        Column containing red-channel thresholds.
    green_col : str, default="green_threshold"
        Column containing green-channel thresholds.
    z_col : str, default="z"
        Column containing z-slice indices.
    q_col : str, optional
        Optional column containing percentile q values. If provided and present,
        the q value is added to plot titles and y-axis labels.

    Returns
    -------
    None
        Displays two matplotlib figures: one for red thresholds and one for
        green thresholds.
    """
    _df = df_thresholds.copy()
    _df = _df.sort_values(z_col)

    if q_col is not None and q_col in _df.columns and _df[q_col].notna().any():
        q = float(_df[q_col].dropna().iloc[0])
        q_pct = f"{q * 100:.1f}%"
    else:
        q_pct = None

    x = _df[z_col].to_numpy()

    # Red thresholds
    y_red = _df[red_col].to_numpy()
    mask_red = np.isfinite(y_red)
    plt.figure()
    plt.plot(x[mask_red], y_red[mask_red], marker="o", linewidth=1)
    plt.xlabel("Z slice")
    ylabel = "Red threshold"
    if q_pct: ylabel += f" (q={q_pct})"
    plt.ylabel(ylabel)
    title = "Per-slice red thresholds"
    if q_pct: title += f" – q={q_pct}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Green thresholds
    y_green = _df[green_col].to_numpy()
    mask_green = np.isfinite(y_green)
    plt.figure()
    plt.plot(x[mask_green], y_green[mask_green], marker="o", linewidth=1)
    plt.xlabel("Z slice")
    ylabel = "Green threshold"
    if q_pct: ylabel += f" (q={q_pct})"
    plt.ylabel(ylabel)
    title = "Per-slice green thresholds"
    if q_pct: title += f" – q={q_pct}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()



def build_threshold_masks_per_roi(
    red_channel,
    green_channel,
    label_mask,
    z_ranges,
    df_thresholds,
    missing_threshold_value=-np.inf,
):
    """
    Build thresholded red and green masks for each ROI using per-z-slice thresholds.

    For each ROI label in `label_mask`, the 2D ROI footprint is taken from the
    slice where that ROI was drawn, then broadcast across that ROI's manually
    assigned z-range. Red pixels are kept if they pass the red threshold for
    that z-slice. Green pixels are kept only if they pass both the red threshold
    and green threshold.

    Parameters
    ----------
    red_channel, green_channel : ndarray
        3D arrays with shape (Z, Y, X).
    label_mask : ndarray
        3D integer label mask with ROI labels. Label 0 is background.
    z_ranges : list of tuple
        One (z0, z1) range per ROI label.
    df_thresholds : pandas.DataFrame
        DataFrame with columns "z", "red_threshold", and "green_threshold".
    missing_threshold_value : float, default=-np.inf
        Value used when a z-slice is missing from `df_thresholds`.

    Returns
    -------
    dict
        Dictionary containing:
        - "mask_red_labels": integer label volume where red-passing pixels are
          labeled by ROI ID
        - "mask_green_full": boolean volume where green-passing pixels are True
        - "thr_red_by_z": red threshold array aligned to z
        - "thr_green_by_z": green threshold array aligned to z
    """
    Z, Y, X = red_channel.shape

    if green_channel.shape != (Z, Y, X):
        raise ValueError(f"green_channel shape {green_channel.shape} != red_channel shape {(Z, Y, X)}.")

    if label_mask.shape != (Z, Y, X):
        raise ValueError(f"label_mask shape {label_mask.shape} != red_channel shape {(Z, Y, X)}.")

    n_rois = int(label_mask.max())

    if len(z_ranges) != n_rois:
        raise ValueError(
            f"Need one (z0, z1) per ROI; got {len(z_ranges)} z-ranges for {n_rois} ROIs."
        )

    required_cols = {"z", "red_threshold", "green_threshold"}
    missing_cols = required_cols - set(df_thresholds.columns)

    if missing_cols:
        raise ValueError(f"df_thresholds is missing required columns: {sorted(missing_cols)}")

    df_aligned = df_thresholds.set_index("z").reindex(np.arange(Z))

    thr_red_by_z = df_aligned["red_threshold"].to_numpy(dtype=float)
    thr_green_by_z = df_aligned["green_threshold"].to_numpy(dtype=float)

    thr_red_by_z = np.where(np.isfinite(thr_red_by_z), thr_red_by_z, missing_threshold_value)
    thr_green_by_z = np.where(np.isfinite(thr_green_by_z), thr_green_by_z, missing_threshold_value)

    mask_red_labels = np.zeros((Z, Y, X), dtype=np.int32)
    mask_green_full = np.zeros((Z, Y, X), dtype=bool)

    for roi_id in range(1, n_rois + 1):
        zs = np.where(label_mask == roi_id)[0]

        if zs.size == 0:
            continue

        z_draw = int(zs[0])
        roi2d = label_mask[z_draw] == roi_id

        z0, z1 = z_ranges[roi_id - 1]
        zlen = z1 - z0 + 1

        roi_sub = np.repeat(roi2d[None, :, :], zlen, axis=0)

        red_sub = red_channel[z0:z1 + 1]
        green_sub = green_channel[z0:z1 + 1]

        thr_red_sub = thr_red_by_z[z0:z1 + 1][:, None, None]
        thr_green_sub = thr_green_by_z[z0:z1 + 1][:, None, None]

        mask_red_sub = roi_sub & (red_sub > thr_red_sub)
        mask_green_sub = mask_red_sub & (green_sub > thr_green_sub)

        z_rel, y_idx, x_idx = np.where(mask_red_sub)
        mask_red_labels[z0 + z_rel, y_idx, x_idx] = roi_id

        mask_green_full[z0:z1 + 1] |= mask_green_sub

    return {
        "mask_red_labels": mask_red_labels,
        "mask_green_full": mask_green_full,
        "thr_red_by_z": thr_red_by_z,
        "thr_green_by_z": thr_green_by_z,
    }


def extract_roi_signal_with_per_z_thresholds(
    red_channel,
    green_channel,
    label_mask,
    z_ranges,
    df_thresholds,
    round_digits=1,
    missing_threshold_value=-np.inf,
):
    """
    Extract red and green signal from each ROI using per-z-slice thresholds.

    For each ROI, the 2D ROI footprint is taken from the z-slice where the ROI
    was drawn, then copied across that ROI's manually assigned z-range.

    The red threshold is applied first inside the ROI. Green signal is then
    measured only from the subset of red-positive pixels that also pass the
    green threshold.

    Parameters
    ----------
    red_channel, green_channel : ndarray
        3D image arrays with shape (Z, Y, X).
    label_mask : ndarray
        3D integer label mask with ROI IDs. Label 0 is background.
    z_ranges : list of tuple
        One (z0, z1) range per ROI.
    df_thresholds : pandas.DataFrame
        DataFrame with columns "z", "red_threshold", and "green_threshold".
    round_digits : int, default=1
        Number of decimal places used for output values.
    missing_threshold_value : float, default=-np.inf
        Value used when a z-slice is missing from df_thresholds.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by ROI ID, with columns:
        - mean_red
        - mean_green
        - total_red
        - total_green
    """
    required_cols = {"z", "red_threshold", "green_threshold"}
    if not required_cols.issubset(df_thresholds.columns):
        raise ValueError(
            f"df_thresholds must have columns: {sorted(required_cols)}"
        )

    if red_channel.ndim != 3 or green_channel.ndim != 3:
        raise ValueError("red_channel and green_channel must be 3D arrays with shape (Z, Y, X).")

    Z, Y, X = red_channel.shape

    if green_channel.shape != (Z, Y, X):
        raise ValueError(
            f"green_channel shape {green_channel.shape} does not match red_channel shape {(Z, Y, X)}."
        )

    if label_mask.shape != (Z, Y, X):
        raise ValueError(
            f"label_mask shape {label_mask.shape} does not match image shape {(Z, Y, X)}."
        )

    n_rois = int(label_mask.max())

    if len(z_ranges) != n_rois:
        raise ValueError(
            f"z_ranges has length {len(z_ranges)} but label_mask has {n_rois} labels."
        )

    df_aligned = df_thresholds.set_index("z").reindex(np.arange(Z))

    thr_red_by_z = df_aligned["red_threshold"].to_numpy(dtype=float)
    thr_green_by_z = df_aligned["green_threshold"].to_numpy(dtype=float)

    thr_red_by_z = np.where(np.isfinite(thr_red_by_z), thr_red_by_z, missing_threshold_value)
    thr_green_by_z = np.where(np.isfinite(thr_green_by_z), thr_green_by_z, missing_threshold_value)

    mean_red = np.zeros(n_rois, dtype=float)
    mean_green = np.zeros(n_rois, dtype=float)
    total_red = np.zeros(n_rois, dtype=float)
    total_green = np.zeros(n_rois, dtype=float)
    n_red_px = np.zeros(n_rois, dtype=int)
    n_green_px = np.zeros(n_rois, dtype=int)

    for roi_id in range(1, n_rois + 1):
        zs, ys, xs = np.where(label_mask == roi_id)

        if zs.size == 0:
            mean_red[roi_id - 1] = np.nan
            mean_green[roi_id - 1] = np.nan
            total_red[roi_id - 1] = np.nan
            total_green[roi_id - 1] = np.nan
            continue

        z_draw = int(zs[0])
        roi2d = label_mask[z_draw] == roi_id

        z0, z1 = z_ranges[roi_id - 1]

        red_sub = red_channel[z0:z1 + 1]
        green_sub = green_channel[z0:z1 + 1]

        roi_sub = np.repeat(
            roi2d[None, :, :],
            z1 - z0 + 1,
            axis=0,
        )

        thr_red_sub = thr_red_by_z[z0:z1 + 1][:, None, None]
        thr_green_sub = thr_green_by_z[z0:z1 + 1][:, None, None]

        mask_red = roi_sub & (red_sub > thr_red_sub)
        mask_green = mask_red & (green_sub > thr_green_sub)

        red_vals = red_sub[mask_red]
        green_vals = green_sub[mask_green]

        n_red_px[roi_id - 1] = red_vals.size
        n_green_px[roi_id - 1] = green_vals.size

        mean_red[roi_id - 1] = (
            np.round(red_vals.mean(), round_digits) if red_vals.size else np.nan
        )
        total_red[roi_id - 1] = (
            np.round(red_vals.sum(), round_digits) if red_vals.size else np.nan
        )
        mean_green[roi_id - 1] = (
            np.round(green_vals.mean(), round_digits) if green_vals.size else np.nan
        )
        total_green[roi_id - 1] = (
            np.round(green_vals.sum(), round_digits) if green_vals.size else np.nan
        )

    if not np.all(n_green_px <= n_red_px):
        print("Warning: some ROIs have more green pixels than red pixels — check thresholds and masks.")

    df_means = pd.DataFrame(
        {
            "mean_red": mean_red,
            "mean_green": mean_green,
            "total_red": total_red,
            "total_green": total_green,
        },
        index=np.arange(1, n_rois + 1),
    )

    #df_means.index.name = "roi_id"

    return df_means



def per_slice_percentiles_red_green_and_blue_diff(
    red_channel,
    green_channel,
    blue_channel,
    label_mask_checking,
    q,                      # float OR (q_red, q_green, q_blue)
    roi_id_rg=1,            # label id for red/green background ROI
    roi_id_blue=2,          # label id for blue background ROI
    exclude_zero=False,
    round_pct=1
):
    """
    Use one background ROI (label 1) for red/green and another (label 2) for blue.
    Compute per-slice percentile thresholds for all three channels.
    (This codes works just like the previous one (per_slice_percentiles_red_and_green_diff), 
    but now the background tiff contains label number =1 as the bg roi for red and green
    and label number=2 as the bg roi for blue.)

    (I didn't use it in the get_signal_from_airyscan.ipynb)

    q:
        - float -> same percentile for all 3 channels
        - (q_red, q_green, q_blue) -> different percentiles per channel

    Returns a DataFrame with columns:
        z,
        roi_pixels_rg, roi_pixels_blue,
        red_threshold, red_frac_leq_pct,
        green_threshold, green_frac_leq_pct,
        blue_threshold, blue_frac_leq_pct
    """

    # -------- parse q into per-channel percentiles --------
    if isinstance(q, (list, tuple)):
        if len(q) != 3:
            raise ValueError(
                "If q is a list/tuple, it must be length 3: (q_red, q_green, q_blue)."
            )
        q_red, q_green, q_blue = map(float, q)
    else:
        q_red = q_green = q_blue = float(q)

    # ---- normalize labels ----
    lbl = label_mask_checking
    if lbl.ndim > 3:
        lbl = np.squeeze(lbl)

    if lbl.ndim not in (2, 3):
        raise ValueError(f"label_mask_checking must be 2D or 3D; got shape {lbl.shape}")

    pos_labels = np.unique(lbl[lbl > 0])
    if pos_labels.size == 0:
        raise ValueError("No positive labels found in label_mask_checking.")

    if roi_id_rg not in pos_labels:
        raise ValueError(f"roi_id_rg={roi_id_rg} not present in label_mask_checking (found {pos_labels.tolist()}).")
    if roi_id_blue not in pos_labels:
        raise ValueError(f"roi_id_blue={roi_id_blue} not present in label_mask_checking (found {pos_labels.tolist()}).")

    # ---- helper: get one 2D ROI footprint from one label id ----
    def get_roi2d(lbl, roi_id):
        if lbl.ndim == 3:
            z_hits = np.where(np.any(lbl == roi_id, axis=(1, 2)))[0]
            if z_hits.size == 0:
                raise ValueError(f"Label {roi_id} has no painted pixels.")
            z_draw = int(z_hits[0])
            roi2d = (lbl[z_draw] == roi_id)
        else:  # 2D
            roi2d = (lbl == roi_id)

        if not roi2d.any():
            raise ValueError(f"ROI footprint for label {roi_id} is empty.")
        return roi2d

    roi2d_rg = get_roi2d(lbl, roi_id_rg)                                                    # rio_id_rg is set to be 1, which means label number=1, so it's the one i drew for red and green channel
    roi2d_blue = get_roi2d(lbl, roi_id_blue)                                                # rio_id_rg is set to be 2, which means label number=2, so it's the one i drew for blue channel

    # ---- sanity checks on channel shapes ----
    if red_channel.ndim != 3 or green_channel.ndim != 3 or blue_channel.ndim != 3:
        raise ValueError("red_channel, green_channel, and blue_channel must all be 3D (Z,Y,X).")

    Z, Y, X = red_channel.shape
    if green_channel.shape != (Z, Y, X):
        raise ValueError(f"green_channel shape {green_channel.shape} != red_channel shape {(Z, Y, X)}.")
    if blue_channel.shape != (Z, Y, X):
        raise ValueError(f"blue_channel shape {blue_channel.shape} != red_channel shape {(Z, Y, X)}.")

    if roi2d_rg.shape != (Y, X):
        raise ValueError(f"Red/green ROI footprint shape {roi2d_rg.shape} does not match channel XY {(Y, X)}.")
    if roi2d_blue.shape != (Y, X):
        raise ValueError(f"Blue ROI footprint shape {roi2d_blue.shape} does not match channel XY {(Y, X)}.")

    # ---- coordinates for the two background ROIs ----
    y_rg, x_rg = np.where(roi2d_rg)
    y_blue, x_blue = np.where(roi2d_blue)

    roi_pixels_rg = int(roi2d_rg.sum())
    roi_pixels_blue = int(roi2d_blue.sum())

    # ---- helper ----
    def qtile(a, qf):
        a = np.asarray(a)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return np.nan
        try:
            return float(np.quantile(a, qf, method="linear"))
        except TypeError:
            return float(np.quantile(a, qf, interpolation="linear"))

    rows = []
    for z in range(Z):
        rvals = red_channel[z, y_rg, x_rg]
        gvals = green_channel[z, y_rg, x_rg]
        bvals = blue_channel[z, y_blue, x_blue]

        if exclude_zero:
            rvals = rvals[rvals != 0]
            gvals = gvals[gvals != 0]
            bvals = bvals[bvals != 0]

        thr_r = qtile(rvals, q_red)
        thr_g = qtile(gvals, q_green)
        thr_b = qtile(bvals, q_blue)

        frac_r = float((rvals <= thr_r).mean()) if rvals.size else np.nan
        frac_g = float((gvals <= thr_g).mean()) if gvals.size else np.nan
        frac_b = float((bvals <= thr_b).mean()) if bvals.size else np.nan

        rows.append({
            "z": z,
            "roi_pixels_rg": roi_pixels_rg,
            "roi_pixels_blue": roi_pixels_blue,
            "red_threshold": thr_r,
            "red_frac_leq_pct": np.round(frac_r * 100.0, round_pct) if np.isfinite(frac_r) else np.nan,
            "green_threshold": thr_g,
            "green_frac_leq_pct": np.round(frac_g * 100.0, round_pct) if np.isfinite(frac_g) else np.nan,
            "blue_threshold": thr_b,
            "blue_frac_leq_pct": np.round(frac_b * 100.0, round_pct) if np.isfinite(frac_b) else np.nan,
        })

    df = pd.DataFrame(rows, columns=[
        "z",
        "roi_pixels_rg", "roi_pixels_blue",
        "red_threshold", "red_frac_leq_pct",
        "green_threshold", "green_frac_leq_pct",
        "blue_threshold", "blue_frac_leq_pct",
    ])
    return df



def plot_per_slice_thresholds_rgb(df_thresholds):
    """
    Plot per-slice threshold curves for red, green, and blue channels from the dataframe returned by
    per_slice_percentiles_red_green_and_blue_diff().

    Expected columns:
        z
        red_threshold
        green_threshold
        blue_threshold

    Optional columns:
        red_percentile_q
        green_percentile_q
        blue_percentile_q
        percentile_q   # old-style single percentile for all channels
    """

    # copy + sort
    _df = df_thresholds.copy().sort_values("z")

    x = _df["z"].to_numpy()

    # helper to get nice q label
    def get_q_label(df, channel_name):
        colname = f"{channel_name}_percentile_q"

        if colname in df.columns and df[colname].notna().any():
            q = float(df[colname].dropna().iloc[0])
            return f"{q*100:.1f}%"

        elif "percentile_q" in df.columns and df["percentile_q"].notna().any():
            q = float(df["percentile_q"].dropna().iloc[0])
            return f"{q*100:.1f}%"

        else:
            return None

    # helper to make one plot
    def plot_one_channel(x, y, channel_name):
        q_pct = get_q_label(_df, channel_name)
        mask = np.isfinite(y)

        plt.figure()
        plt.plot(x[mask], y[mask], marker="o", linewidth=1)
        plt.xlabel("Z slice")

        ylabel = f"{channel_name.capitalize()} threshold"
        if q_pct:
            ylabel += f" (q={q_pct})"
        plt.ylabel(ylabel)

        title = f"Per-slice {channel_name} thresholds"
        if q_pct:
            title += f" – q={q_pct}"
        plt.title(title)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    # red
    y_red = _df["red_threshold"].to_numpy()
    plot_one_channel(x, y_red, "red")

    # green
    y_green = _df["green_threshold"].to_numpy()
    plot_one_channel(x, y_green, "green")

    # blue
    y_blue = _df["blue_threshold"].to_numpy()
    plot_one_channel(x, y_blue, "blue")



def per_slice_percentiles_by_channel(
    channels,
    label_mask_checking,
    q_by_channel,
    roi_id_by_channel=None,
    exclude_zero=False,
    round_pct=1,
):
    """
    Compute per-z-slice percentile thresholds for multiple image channels.
    This is the most general version of the per_slice_percentile function,
    which can handle any number of channels and separate background ROIs for each channel.
    
    Parameters
    ----------
    channels : dict
        Dictionary mapping channel names to 3D arrays, e.g.
        {"red": red_channel, "green": green_channel, "blue": blue_channel}.

    label_mask_checking : ndarray
        2D or 3D label mask containing background ROI labels.

    q_by_channel : dict
        Dictionary mapping channel names to percentile values, e.g.
        {"red": 0.975, "green": 0.95, "blue": 0.95}.

    roi_id_by_channel : dict, optional
        Dictionary mapping channel names to background ROI label IDs.
        Default is {"red": 1, "green": 1, "blue": 2}.

    exclude_zero : bool, default=False
        Whether to exclude zero-valued pixels before calculating percentiles.

    round_pct : int, default=1
        Number of decimal places for reporting fraction-below-threshold values.

    Returns
    -------
    pandas.DataFrame
        One row per z-slice, with threshold and fraction columns for each channel.
    """
    if roi_id_by_channel is None:
        roi_id_by_channel = {
            "red": 1,
            "green": 1,
            "blue": 2,
        }

    # Check that every channel has q and ROI ID
    for ch in channels:
        if ch not in q_by_channel:
            raise ValueError(f"Missing q value for channel '{ch}'.")
        if ch not in roi_id_by_channel:
            raise ValueError(f"Missing ROI label ID for channel '{ch}'.")

    # Normalize label mask
    lbl = label_mask_checking
    if lbl.ndim > 3:
        lbl = np.squeeze(lbl)

    if lbl.ndim not in (2, 3):
        raise ValueError(f"label_mask_checking must be 2D or 3D; got shape {lbl.shape}")

    pos_labels = np.unique(lbl[lbl > 0])
    if pos_labels.size == 0:
        raise ValueError("No positive labels found in label_mask_checking.")

    # Check channels have same shape
    first_ch = next(iter(channels))
    first_shape = channels[first_ch].shape

    if len(first_shape) != 3:
        raise ValueError(f"Channel '{first_ch}' must be 3D (Z,Y,X); got shape {first_shape}")

    Z, Y, X = first_shape

    for ch, arr in channels.items():
        if arr.ndim != 3:
            raise ValueError(f"Channel '{ch}' must be 3D (Z,Y,X); got shape {arr.shape}")
        if arr.shape != (Z, Y, X):
            raise ValueError(
                f"Channel '{ch}' shape {arr.shape} does not match first channel shape {(Z, Y, X)}."
            )

    def get_roi2d(lbl, roi_id):
        roi_id = int(roi_id)
        if roi_id not in pos_labels:
            raise ValueError(
                f"roi_id={roi_id} not present in label_mask_checking "
                f"(found {pos_labels.tolist()})."
            )
        if lbl.ndim == 3:
            z_hits = np.where(np.any(lbl == roi_id, axis=(1, 2)))[0]
            if z_hits.size == 0:
                raise ValueError(f"Label {roi_id} has no painted pixels.")
            z_draw = int(z_hits[0])
            roi2d = lbl[z_draw] == roi_id
        else:
            roi2d = lbl == roi_id

        if not roi2d.any():
            raise ValueError(f"ROI footprint for label {roi_id} is empty.")

        if roi2d.shape != (Y, X):
            raise ValueError(
                f"ROI footprint shape {roi2d.shape} does not match channel XY {(Y, X)}."
            )

        return roi2d

    def qtile(a, qf):
        a = np.asarray(a)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return np.nan
        try:
            return float(np.quantile(a, qf, method="linear"))
        except TypeError:
            return float(np.quantile(a, qf, interpolation="linear"))

    # Precompute ROI coordinates for each channel, so the coordinate of the masked pixels are recorded to avoid doing it 3 times for 3 channels.
    roi_info = {}
    for ch in channels:
        roi_id = roi_id_by_channel[ch]
        roi2d = get_roi2d(lbl, roi_id)          # make a 2D boolean mask of a full plane (full Y * full X)
        y_ix, x_ix = np.where(roi2d)            # np.where returns the coordiantes of all True pixels.

        roi_info[ch] = {
            "roi_id": int(roi_id),
            "y_ix": y_ix,
            "x_ix": x_ix,
            "roi_pixels": int(roi2d.sum()),
        }

    rows = []
    for z in range(Z):
        row = {"z": z}

        for ch, arr in channels.items():
            y_ix = roi_info[ch]["y_ix"]
            x_ix = roi_info[ch]["x_ix"]
            vals = arr[z, y_ix, x_ix]           # a list of the values of all masked pixels in this z slice in this channel.

            if exclude_zero:
                vals = vals[vals != 0]

            q = float(q_by_channel[ch])
            thr = qtile(vals, q)
            frac = float((vals <= thr).mean()) if vals.size else np.nan

            row[f"{ch}_roi_id"] = roi_info[ch]["roi_id"]
            row[f"{ch}_roi_pixels"] = roi_info[ch]["roi_pixels"]
            row[f"{ch}_threshold"] = thr
            row[f"{ch}_frac_leq_pct"] = (
                np.round(frac * 100.0, round_pct) if np.isfinite(frac) else np.nan
            )
            row[f"{ch}_percentile_q"] = q

        rows.append(row)

    return pd.DataFrame(rows)




def extract_rg_and_blue_on_green_pixels_exact_old_logic(
    red_channel,
    green_channel,
    blue_channel,
    label_mask,
    df_thresholds,
    z_ranges,
):
    """
    Use the exact same ROI + z-range + threshold logic as the old code
    for red/green extraction, then measure blue values on the same mask_green pixels.

    Returns one row per ROI.
    """

    pd.set_option("display.float_format", lambda v: f"{v:.1f}")

    # -------------------- checks --------------------
    if not {'z', 'red_threshold', 'green_threshold', 'blue_threshold'}.issubset(df_thresholds.columns):
        raise ValueError("df_thresholds must have columns: 'z', 'red_threshold', 'green_threshold', 'blue_threshold'.")

    if red_channel.ndim != 3 or green_channel.ndim != 3 or blue_channel.ndim != 3:
        raise ValueError("red_channel, green_channel, and blue_channel must all be 3D (Z,Y,X).")

    Z_all, Y, X = red_channel.shape
    if green_channel.shape != (Z_all, Y, X):
        raise ValueError("green_channel shape does not match red_channel.")
    if blue_channel.shape != (Z_all, Y, X):
        raise ValueError("blue_channel shape does not match red_channel.")
    if label_mask.shape != (Z_all, Y, X):
        raise ValueError("label_mask shape must match channel shape.")

    # -------------------- thresholds by z --------------------
    _bg = df_thresholds.set_index('z').reindex(np.arange(Z_all))

    thr_red_by_z   = _bg['red_threshold'].to_numpy(dtype=float)
    thr_green_by_z = _bg['green_threshold'].to_numpy(dtype=float)
    thr_blue_by_z  = _bg['blue_threshold'].to_numpy(dtype=float)

    thr_red_by_z   = np.where(np.isfinite(thr_red_by_z),   thr_red_by_z,   -np.inf)
    thr_green_by_z = np.where(np.isfinite(thr_green_by_z), thr_green_by_z, -np.inf)
    thr_blue_by_z  = np.where(np.isfinite(thr_blue_by_z),  thr_blue_by_z,  -np.inf)

    # -------------------- outputs --------------------
    n_glom = int(label_mask.max())

    mean_red    = np.zeros(n_glom, dtype=float)
    mean_green  = np.zeros(n_glom, dtype=float)
    total_red   = np.zeros(n_glom, dtype=float)
    total_green = np.zeros(n_glom, dtype=float)

    n_red_px    = np.zeros(n_glom, dtype=int)
    n_green_px  = np.zeros(n_glom, dtype=int)

    mean_blue_on_green  = np.zeros(n_glom, dtype=float)
    total_blue_on_green = np.zeros(n_glom, dtype=float)
    n_blue_positive_on_green = np.zeros(n_glom, dtype=int)
    pct_blue_positive_on_green = np.zeros(n_glom, dtype=float)

    if len(z_ranges) != n_glom:
        raise ValueError(f"z_ranges has length {len(z_ranges)} but label_mask has {n_glom} labels.")

    # -------------------- main loop --------------------
    for i in range(1, n_glom + 1):
        # same as old code
        zs, ys, xs = np.where(label_mask == i)
        if zs.size == 0:
            mean_red[i - 1] = np.nan
            mean_green[i - 1] = np.nan
            total_red[i - 1] = np.nan
            total_green[i - 1] = np.nan
            mean_blue_on_green[i - 1] = np.nan
            total_blue_on_green[i - 1] = np.nan
            pct_blue_positive_on_green[i - 1] = np.nan
            continue

        z_draw = zs[0]
        roi2d = (label_mask[z_draw] == i)

        z0, z1 = z_ranges[i - 1]

        roi = np.zeros_like(label_mask, dtype=bool)
        roi[z0:z1+1, :, :] = roi2d[None, :, :]

        red_sub   = red_channel[z0:z1+1]
        green_sub = green_channel[z0:z1+1]
        blue_sub  = blue_channel[z0:z1+1]
        roi_sub   = roi[z0:z1+1]

        thr_red_sub   = thr_red_by_z[z0:z1+1][:, None, None]
        thr_green_sub = thr_green_by_z[z0:z1+1][:, None, None]
        thr_blue_sub  = thr_blue_by_z[z0:z1+1][:, None, None]

        # exact old red/green logic
        mask_roi   = roi_sub
        mask_red   = mask_roi & (red_sub > thr_red_sub)
        mask_green = mask_red & (green_sub > thr_green_sub)

        red_vals   = red_sub[mask_red]
        green_vals = green_sub[mask_green]

        # new blue logic: measure blue on the SAME green-selected pixels
        blue_vals_on_green = blue_sub[mask_green]
        blue_positive_mask = mask_green & (blue_sub > thr_blue_sub)
        n_blue_pos = int(blue_positive_mask.sum())

        # save old columns exactly in old style
        n_red_px[i - 1]   = red_vals.size
        n_green_px[i - 1] = green_vals.size

        mean_red[i - 1]    = np.round(red_vals.mean(), 1)    if red_vals.size else np.nan
        total_red[i - 1]   = np.round(red_vals.sum(), 1)     if red_vals.size else np.nan
        mean_green[i - 1]  = np.round(green_vals.mean(), 1)  if green_vals.size else np.nan
        total_green[i - 1] = np.round(green_vals.sum(), 1)   if green_vals.size else np.nan

        # new blue columns
        mean_blue_on_green[i - 1]  = np.round(blue_vals_on_green.mean(), 1) if blue_vals_on_green.size else np.nan
        total_blue_on_green[i - 1] = np.round(blue_vals_on_green.sum(), 1)  if blue_vals_on_green.size else np.nan
        n_blue_positive_on_green[i - 1] = n_blue_pos
        pct_blue_positive_on_green[i - 1] = (
            np.round(100 * n_blue_pos / green_vals.size, 1) if green_vals.size else np.nan
        )

    df_out = pd.DataFrame(
        {
            "n_red_pixels": n_red_px,
            "n_green_pixels": n_green_px,
            "mean_red": mean_red,
            "total_red": total_red,
            "mean_green": mean_green,
            "total_green": total_green,
            "mean_blue_on_green_pixels": mean_blue_on_green,
            "total_blue_on_green_pixels": total_blue_on_green,
            "n_blue_positive_on_green_pixels": n_blue_positive_on_green,
            "percent_blue_positive_on_green_pixels": pct_blue_positive_on_green,
        },
        index=np.arange(1, n_glom + 1)
    )

    return df_out



def plot_one_column_across_rois(
    df,
    column_name,
    *,
    x=None,
    xlabel="ROI",
    ylabel=None,
    title=None,
    marker="o",
    linewidth=1,
    ylim=None   # NEW
):
    """
    Plot a single column from a DataFrame across ROIs.
    This is to visualize the df obtained from extract_roi_signal_with_per_z_thresholds().

    Parameters
    ----------
    df : pandas.DataFrame
    column_name : str
        Column to plot.
    x : array-like, optional
        Custom x-axis values (default: df index).
    xlabel, ylabel, title : str
    marker : str
    linewidth : float
    ylim : tuple, optional
        (ymin, ymax)
    """

    if column_name not in df.columns:
        raise ValueError(f"'{column_name}' not found in dataframe columns.")

    # choose x
    if x is None:
        try:
            x_vals = df.index.to_numpy()
            if len(x_vals) != len(df):
                x_vals = np.arange(1, len(df) + 1)
        except:
            x_vals = np.arange(1, len(df) + 1)
    else:
        x_vals = np.asarray(x)
        if len(x_vals) != len(df):
            raise ValueError("Length of x does not match number of rows in df.")

    y_vals = df[column_name].to_numpy()
    mask = np.isfinite(y_vals)

    plt.figure()
    plt.plot(x_vals[mask], y_vals[mask], marker=marker, linewidth=linewidth)

    plt.xlabel(xlabel)
    plt.ylabel(column_name if ylabel is None else ylabel)
    plt.title(column_name if title is None else title)

    # force integer x ticks
    plt.xticks(x_vals)

    # set y-axis range if provided
    if ylim is not None:
        plt.ylim(ylim)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()



def build_rgb_threshold_masks_per_roi(
    red_channel,
    green_channel,
    blue_channel,
    label_mask,
    z_ranges,
    df_thresholds,
    missing_threshold_value=-np.inf,
):
    """
    Build red, green, and blue threshold masks for each ROI.

    Red is thresholded first within each ROI. Green is thresholded only among
    red-positive pixels. Blue is thresholded only among the same green-selected pixels.
    (This cell opens the image and the masks together, so that i can check whether
    the thresholds i set are reasonable, now including blue mask as image.)


    Parameters
    ----------
    red_channel, green_channel, blue_channel : ndarray
        3D arrays with shape (Z, Y, X).
    label_mask : ndarray
        3D ROI label mask with label 0 as background.
    z_ranges : list of tuple
        One (z0, z1) z-range per ROI.
    df_thresholds : pandas.DataFrame
        DataFrame with columns "z", "red_threshold", "green_threshold",
        and "blue_threshold".
    missing_threshold_value : float, default=-np.inf
        Value used for missing threshold values.

    Returns
    -------
    dict
        Contains:
        - mask_red_labels
        - mask_green_full
        - mask_blue_full
        - thr_red_by_z
        - thr_green_by_z
        - thr_blue_by_z
    """
    Z, Y, X = red_channel.shape

    if green_channel.shape != (Z, Y, X):
        raise ValueError("green_channel shape does not match red_channel.")
    if blue_channel.shape != (Z, Y, X):
        raise ValueError("blue_channel shape does not match red_channel.")
    if label_mask.shape != (Z, Y, X):
        raise ValueError("label_mask shape must match channel shape.")

    n_rois = int(label_mask.max())
    if len(z_ranges) != n_rois:
        raise ValueError(
            f"Need one (z0, z1) per ROI; got {len(z_ranges)} for {n_rois}."
        )

    required_cols = {"z", "red_threshold", "green_threshold", "blue_threshold"}
    if not required_cols.issubset(df_thresholds.columns):
        raise ValueError(
            f"df_thresholds must have columns: {sorted(required_cols)}"
        )

    df_aligned = df_thresholds.set_index("z").reindex(np.arange(Z))

    thr_red_by_z = df_aligned["red_threshold"].to_numpy(dtype=float)
    thr_green_by_z = df_aligned["green_threshold"].to_numpy(dtype=float)
    thr_blue_by_z = df_aligned["blue_threshold"].to_numpy(dtype=float)

    thr_red_by_z = np.where(np.isfinite(thr_red_by_z), thr_red_by_z, missing_threshold_value)
    thr_green_by_z = np.where(np.isfinite(thr_green_by_z), thr_green_by_z, missing_threshold_value)
    thr_blue_by_z = np.where(np.isfinite(thr_blue_by_z), thr_blue_by_z, missing_threshold_value)

    mask_red_labels = np.zeros((Z, Y, X), dtype=np.int32)
    mask_green_full = np.zeros((Z, Y, X), dtype=bool)
    mask_blue_full = np.zeros((Z, Y, X), dtype=bool)

    for roi_id in range(1, n_rois + 1):
        zs = np.where(label_mask == roi_id)[0]
        if zs.size == 0:
            continue

        z_draw = zs[0]
        roi2d = label_mask[z_draw] == roi_id

        z0, z1 = z_ranges[roi_id - 1]
        zlen = z1 - z0 + 1

        roi_sub = np.repeat(roi2d[None, :, :], zlen, axis=0)

        red_sub = red_channel[z0:z1 + 1]
        green_sub = green_channel[z0:z1 + 1]
        blue_sub = blue_channel[z0:z1 + 1]

        thr_red_sub = thr_red_by_z[z0:z1 + 1][:, None, None]
        thr_green_sub = thr_green_by_z[z0:z1 + 1][:, None, None]
        thr_blue_sub = thr_blue_by_z[z0:z1 + 1][:, None, None]

        mask_red_sub = roi_sub & (red_sub > thr_red_sub)
        mask_green_sub = mask_red_sub & (green_sub > thr_green_sub)
        mask_blue_sub = mask_green_sub & (blue_sub > thr_blue_sub)

        z_rel, y_idx, x_idx = np.where(mask_red_sub)
        mask_red_labels[z0 + z_rel, y_idx, x_idx] = roi_id

        mask_green_full[z0:z1 + 1] |= mask_green_sub
        mask_blue_full[z0:z1 + 1] |= mask_blue_sub

    return {
        "mask_red_labels": mask_red_labels,
        "mask_green_full": mask_green_full,
        "mask_blue_full": mask_blue_full,
        "thr_red_by_z": thr_red_by_z,
        "thr_green_by_z": thr_green_by_z,
        "thr_blue_by_z": thr_blue_by_z,
    }




#-----------------end of the section for getting signal from raw airyscan-----------------


















#-----------------below are for the "reorder" section-----------------
# much of below functions are modified from analysis_trend-25summer.ipynb,
# because here I am dealing with one df at a time.

def subtype_to_part_dict(subtype):
    """
    Convert a subtype string into part1/part2 fields.
    Handles normal 2-glomerulus subtypes and special 3-glomerulus subtypes.

    Examples
    --------
    "L3R6"    -> {"part1": "L3",   "part2": "R6"}
    "L1L9R8" -> {"part1": "L1+9", "part2": "R8"}
    "L8R1R9" -> {"part1": "L8",   "part2": "R1+9"}
    """
    parts = re.findall(r"[LR]\d+", subtype)

    if subtype == "L1L9R8":
        return {
            "part1": parts[0] + "+" + parts[1][-1],
            "part2": parts[2],}
    if subtype == "L8R1R9":
        return {
            "part1": parts[0],
            "part2": parts[1] + "+" + parts[2][-1],}

    return {
        "part1": parts[0],
        "part2": parts[1],
    }


def split_df_into_left_right_halves(df, n_per_half=9, reset_index=True):
    """
    Split one 18-row PB DataFrame into left and right halves.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with glomeruli ordered as L9-L1 followed by R1-R9.
    n_per_half : int, default=9
        Number of rows in each half.
    reset_index : bool, default=True
        Whether to reset index in each output DataFrame.

    Returns
    -------
    tuple of pandas.DataFrame
        (left_half, right_half)
    """
    left = df.iloc[:n_per_half].copy()
    right = df.iloc[n_per_half:].copy()
    if reset_index:
        left = left.reset_index(drop=True)
        right = right.reset_index(drop=True)

    return left, right



def combine_1_9(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine glomeruli 1 and 9 on each side into a single row, and then take the average of the "mean" columns.
    The name of the columns are fixed...

    Rules:
    - L1 + L9 → L1+9 (placed at L9 position)
    - R1 + R9 → R1+9 (placed at R1 position)

    """
    df = df.copy()

    for side in ['L', 'R']:
        g1, g9 = f'{side}1', f'{side}9'
        idx1 = df.index[df['glomerulus'] == g1]
        idx9 = df.index[df['glomerulus'] == g9]

        # Only combine if both rows exist
        if len(idx1) > 0 and len(idx9) > 0:
            i1, i9 = idx1[0], idx9[0]             # indices of the '1' and '9' rows
            rows = df.loc[[i1, i9]]               # two-row slice

            # Build the combined values
            new_glom        = f'{side}1+9'
            new_mean_red   = round(rows["mean_red"].sum() / 2.0, 1)
            new_mean_green = round(rows["mean_green"].sum() / 2.0, 1)
            new_total_red   = rows['total_red'].sum()
            new_total_green = rows['total_green'].sum()

            # Placement/drop rule:
            # - For 'R': put new row where '1' was, drop the '9' row
            # - For 'L': put new row where '9' was, drop the '1' row
            keep_idx = i1 if side == 'R' else i9
            drop_idx = i9 if side == 'R' else i1

            # Overwrite the kept row with combined values
            df.loc[keep_idx, 'glomerulus'] = new_glom
            df.loc[keep_idx, 'mean_red']   = new_mean_red
            df.loc[keep_idx, 'mean_green'] = new_mean_green
            df.loc[keep_idx, 'total_red']  = new_total_red
            df.loc[keep_idx, 'total_green']= new_total_green

            # Drop the other row
            df = df.drop(index=drop_idx)

    return df


def reorder_half_to_center(df, center, target_idx=4):
    """
    Reorder one PB half DataFrame so that the center glomerulus is moved to
    a target row index.

    Parameters
    ----------
    df : pandas.DataFrame
        One half-PB DataFrame containing a "glomerulus" column.
    center : str
        Glomerulus label to center, such as "L3", "R6", "L1+9", or "R1+9".
    target_idx : int, default=4
        Target row index for the center glomerulus.

    Returns
    -------
    pandas.DataFrame
        Reordered DataFrame with reset index.
    """
    if "glomerulus" not in df.columns:
        raise ValueError("df must contain a 'glomerulus' column.")

    idxs = df.index[df["glomerulus"] == center]

    if len(idxs) == 0:
        raise ValueError(f"Center glomerulus {center!r} was not found in df.")

    current_idx = int(idxs[0])
    shift = target_idx - current_idx
    order = np.roll(np.arange(len(df)), shift)

    return df.iloc[order].reset_index(drop=True)



def add_wrapped_row(
    df,
    insert_pos=8,
    source_pos=0,
    metric_cols=("mean_red", "mean_green", "total_red", "total_green"),
    glomerulus_col="glomerulus",
    glomerulus_value="added",
):
    """
    Insert one copied/wrapped row into a DataFrame.

    The new row is inserted at `insert_pos`. Metric values are copied from
    `source_pos`, while the glomerulus label is set to `glomerulus_value`.

    This is useful after combining glomeruli 1 and 9, when an 8-row PB half is
    treated as circular and one row is added back for plotting/curve fitting.

    Parameters
    ----------
    df : pandas.DataFrame
        Input half-PB DataFrame.
    insert_pos : int, default=8
        Position where the new row is inserted.
    source_pos : int, default=0
        Row position copied into the new row.
    metric_cols : iterable of str
        Columns whose values are copied from `source_pos`.
    glomerulus_col : str, default="glomerulus"
        Name of the glomerulus label column.
    glomerulus_value : str, default="added"
        Value assigned to the glomerulus column in the inserted row.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one inserted row.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df must be a non-empty pandas DataFrame.")

    base = df.reset_index(drop=True).copy()

    if source_pos >= len(base):
        raise ValueError(f"source_pos={source_pos} is out of range for df length {len(base)}.")

    new_row = pd.DataFrame(columns=base.columns)
    new_row.loc[0, glomerulus_col] = glomerulus_value

    for col in metric_cols:
        if col in base.columns:
            new_row.loc[0, col] = pd.to_numeric(base.loc[source_pos, col], errors="coerce")

    pos = max(0, min(insert_pos, len(base)))

    return pd.concat(
        [base.iloc[:pos], new_row, base.iloc[pos:]],
        ignore_index=True,
    )

# there are some functions that I didn't bring over from analysis_trend-25summer.ipynb, because they are not needed for the current analysis, but I can always add them back later if needed.
# those were for example functions for plotting, making a summary df of all flies, and making the graph for that.
#-----------------end of the "reorder" section-----------------