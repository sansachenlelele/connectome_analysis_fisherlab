import os
from pathlib import Path
import re
import math
import itertools

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from scipy.optimize import curve_fit, least_squares

from kylie_lib import syn_specs

from typing import Dict, Iterable, List, Optional, Sequence, Mapping
import matplotlib as mpl



#-------------------------PART ZERO-------------------------

def process_delta7_naming_csv(
    input_path,
    output_path=None,
    drop_notes=True,
    instance_prefix=r'^Delta7\(PB15\)_'
):
    """
    Load and process a Delta7 naming CSV.

    Steps:
    - Optionally drop 'notes' column
    - Clean 'instance' column by removing prefix
    - Sort by 'instance'
    - Reset index

    Parameters
    ----------
    input_path : str or Path
        Path to input CSV file

    output_path : str or Path or None
        If provided, save processed CSV to this path

    drop_notes : bool
        Whether to drop the 'notes' column

    instance_prefix : str (regex)
        Pattern to remove from the beginning of 'instance'

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(input_path)

    if drop_notes and "notes" in df.columns:
        df = df.drop(columns=["notes"])

    if "instance" in df.columns:
        df["instance"] = df["instance"].str.replace(instance_prefix, "", regex=True)

    df = df.sort_values("instance").reset_index(drop=True)

    if output_path is not None:
        df.to_csv(output_path, index=False)

    return df


def fetch_d7_upstream_by_glomerulus(
    d7_names_df,
    output_dir=None,
    conn_id="P1-9",
    rois_sides=("R", "L"),
    glom_range=range(1, 10),
):
    """
    Fetch upstream synapse counts onto Delta7 neurons from a specified neuron type,
    separated by PB glomerulus.
    This function uses the syn_specs class/function from Kylie's library (Kylie_lib)

    Parameters
    ----------
    d7_names_df : pd.DataFrame
        Must contain:
        - 'id' : neuron bodyId
        - 'instance' : neuron instance name

    output_dir : str or Path or None
        If provided, saves one CSV per neuron.

    conn_id : str
        Presynaptic neuron type (e.g. "P1-9")

    rois_sides : tuple
        ('R', 'L') typically

    glom_range : iterable
        Typically range(1, 10)

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    all_results = {}

    for _, row in d7_names_df.iterrows():
        neuron_id = row["id"]
        inst_name = row["instance"]
        result_name = f"delta7_{neuron_id}_{inst_name}_{conn_id}_upstream"

        summary = []

        for side in rois_sides:
            for i in glom_range:
                glom = f"PB({side}{i})"
                spec = syn_specs(
                    target_neuron=neuron_id,
                    scale="type",
                    conn_type="pre",
                    conn_id=conn_id,
                    rois=[glom],
                    primary_only=False,
                )

                df = spec.fetch_syn_conns()
                synapse_count = len(df) if isinstance(df, pd.DataFrame) else 0

                summary.append(
                    {
                        "Glomerulus": glom,
                        "SynapseCount": synapse_count,
                    }
                )

        all_results[result_name] = pd.DataFrame(summary)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, df in all_results.items():
            df.to_csv(output_dir / f"{name}.csv", index=False)

    return all_results


def fetch_d7_upstream_by_neuron_type(
    d7_names_df: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    rois="PB",
    include_nonprimary: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    For each Delta7 neuron, fetch all upstream inputs and summarize synapse count
    by presynaptic neuron type.

    Uses Kylie library function:
        fetch_connectivity(
            target_scale="neuron",
            conn_scale="all",
            conn_type="pre",
            target_id=<Delta7 bodyId>,
            rois=<rois>,
            include_nonprimary=<include_nonprimary>,
        )

    Returns
    -------
    dict[str, pd.DataFrame]
        One DataFrame per Delta7 neuron.
    """

    from kylie_lib import fetch_connectivity

    all_results = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in d7_names_df.iterrows():
        neuron_id = int(row["id"])
        inst_name = row["instance"]

        print(f"Fetching upstream types for Delta7 {neuron_id} ({inst_name})...")

        conn_df = fetch_connectivity(
            target_scale="neuron",
            conn_scale="all",
            conn_type="pre",
            target_id=neuron_id,
            conn_id=None,
            rois=rois,
            include_nonprimary=include_nonprimary,
        )

        if conn_df is None or len(conn_df) == 0:
            summary_df = pd.DataFrame(
                columns=[
                    "target_id",
                    "target_instance",
                    "upstream_type",
                    "synapse_count",
                    "n_upstream_cells",
                ]
            )
        else:
            summary_df = (
                conn_df
                .groupby("type_pre", dropna=False)
                .agg(
                    synapse_count=("weight", "sum"),
                    n_upstream_cells=("bodyId_pre", "nunique"),
                )
                .reset_index()
                .rename(columns={"type_pre": "upstream_type"})
                .sort_values("synapse_count", ascending=False)
                .reset_index(drop=True)
            )

            summary_df.insert(0, "target_instance", inst_name)
            summary_df.insert(0, "target_id", neuron_id)

        result_name = f"delta7_{neuron_id}_{inst_name}_upstream_by_type"
        all_results[result_name] = summary_df

        if output_dir is not None:
            summary_df.to_csv(output_dir / f"{result_name}.csv", index=False)

    return all_results





# -------------------------PART ONE-------------------------

def reorder_pb_glomeruli(
    dfs_dict,
    *,
    verbose=False
):
    """
    Reorder rows in each DataFrame so glomeruli are:

    PB(L9) ... PB(L1), PB(R1) ... PB(R9)

    Parameters
    ----------
    dfs_dict : dict[str, pd.DataFrame]
        Dictionary of DataFrames with a 'Glomerulus' column

    verbose : bool

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    desired = [f"PB(L{i})" for i in range(9, 0, -1)] + \
              [f"PB(R{i})" for i in range(1, 10)]

    out = {}
    skipped = []

    for name, df in dfs_dict.items():
        gcol = (
            "Glomerulus"
            if "Glomerulus" in df.columns
            else ("glomerulus" if "glomerulus" in df.columns else None)
        )

        if gcol is None:
            skipped.append(name)
            continue

        tmp = df.copy()
        tmp[gcol] = tmp[gcol].astype(str)

        tmp["__ord"] = pd.Categorical(
            tmp[gcol],
            categories=desired,
            ordered=True
        )

        tmp = (
            tmp.sort_values("__ord")
               .drop(columns="__ord")
               .reset_index(drop=True)
        )

        out[name] = tmp

    if verbose and skipped:
        print(f"[warn] skipped {len(skipped)} items: {skipped[:3]}")

    return out



def split_pb_left_right_halves(source_dict):
    """
    Split each PB glomerulus summary DataFrame into left and right PB halves.

    Assumes each DataFrame has already been reordered as:
    PB(L9) ... PB(L1), PB(R1) ... PB(R9)

    Parameters
    ----------
    source_dict : dict[str, pd.DataFrame]
        Dictionary of DataFrames to split. Each DataFrame should have 18 rows.

    Returns
    -------
    dict[str, pd.DataFrame]
        New dictionary where each input DataFrame becomes two entries:
        '<name>_L' for the first 9 rows and '<name>_R' for the last 9 rows.
    """
    halves = {}

    for name, df in source_dict.items():
        df_left = df.iloc[:9].reset_index(drop=True)
        df_right = df.iloc[9:].reset_index(drop=True)

        halves[f"{name}_L"] = df_left
        halves[f"{name}_R"] = df_right

    return halves



def combine_pb_1_and_9(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine PB(1) and PB(9) glomeruli into a single row per side.

    For each side (L and R):
    - Combine PB(side1) and PB(side9) into PB(side1+9)
    - Sum SynapseCount
    - Placement rule:
        * R side → replace position of PB(R1), drop PB(R9)
        * L side → replace position of PB(L9), drop PB(L1)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
        - 'Glomerulus'
        - 'SynapseCount'

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    for side in ['L', 'R']:
        g1, g9 = f'PB({side}1)', f'PB({side}9)'

        idx1 = df.index[df['Glomerulus'] == g1]
        idx9 = df.index[df['Glomerulus'] == g9]

        if len(idx1) > 0 and len(idx9) > 0:
            i1, i9 = idx1[0], idx9[0]
            rows = df.loc[[i1, i9]]

            new_glom = f'PB({side}1+9)'
            new_sc = rows['SynapseCount'].sum()

            keep_idx = i1 if side == 'R' else i9
            drop_idx = i9 if side == 'R' else i1

            df.loc[keep_idx, 'Glomerulus'] = new_glom
            df.loc[keep_idx, 'SynapseCount'] = new_sc

            df = df.drop(index=drop_idx)

    return df



def build_d7_subtype_parts(d7_names_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build Delta7 subtype-to-PB-glomerulus mapping columns.

    From a Delta7 naming table, this function:
    - keeps 'id' and 'instance'
    - removes the final side suffix from 'instance' to create 'subtype'
      e.g. 'L1L9R8_R' -> 'L1L9R8'
    - extracts PB glomerulus tokens such as 'L1', 'L9', 'R8'
    - combines edge glomeruli:
        * L1L9R8 -> PB(L1+9), PB(R8)
        * L8R1R9 -> PB(L8), PB(R1+9)

    Parameters
    ----------
    d7_names_df : pd.DataFrame
        Must contain columns:
        - 'id'
        - 'instance'

    Returns
    -------
    pd.DataFrame
        Columns:
        - 'id'
        - 'subtype'
        - 'part1'
        - 'part2'
    """
    df = d7_names_df.loc[:, ["id", "instance"]].copy()

    df["subtype"] = df["instance"].str.slice(0, -2)

    tokens = df["subtype"].str.findall(r"[LR]\d+")
    max_parts = tokens.map(len).max()
    col_names = [f"part{i+1}" for i in range(max_parts)]

    df[col_names] = pd.DataFrame(tokens.tolist(), index=df.index)

    m1 = df["subtype"].eq("L1L9R8")
    if "part3" in df.columns:
        df.loc[m1, "part1"] = (
            df.loc[m1, "part1"] + "+" + df.loc[m1, "part2"].str[-1]
        )
        df.loc[m1, "part2"] = df.loc[m1, "part3"]

    m2 = df["subtype"].eq("L8R1R9")
    if "part3" in df.columns:
        df.loc[m2, "part2"] = (
            df.loc[m2, "part2"] + "+" + df.loc[m2, "part3"].str[-1]
        )

    if "part3" in df.columns:
        df = df.drop(columns=["part3"])

    for col in ["part1", "part2"]:
        mask = df[col].notna()
        df.loc[mask, col] = "PB(" + df.loc[mask, col].astype(str) + ")"

    return df.loc[:, ["id", "subtype", "part1", "part2"]]


def center_halves_on_subtype_glomerulus(
    halves_dict: dict,
    neuron_subtype: pd.DataFrame,
    *,
    target_idx: int = 4,
    add_suffix: bool = False,
) -> dict:
    """
    Reorder each half-PB DataFrame so the Delta7 subtype-associated center
    glomerulus is moved to a target row index.

    For each DataFrame, the neuron bodyId and PB side are parsed from the dict key.
    The center glomerulus is looked up from `neuron_subtype`:
    - left half uses 'part1'
    - right half uses 'part2'

    The row matching that center glomerulus is shifted to `target_idx` using
    circular row rotation.

    Parameters
    ----------
    halves_dict : dict[str, pd.DataFrame]
        Dictionary of half-PB DataFrames. Keys are expected to contain:
        delta7_<bodyId>_<subtype>_<side>_upstream_<side>
        or a similar underscore-separated format.

    neuron_subtype : pd.DataFrame
        Must contain columns:
        - 'id'
        - 'part1'
        - 'part2'

    target_idx : int
        Desired index for the center glomerulus after reordering.
        Default is 4, the middle row of a 9-row half-PB DataFrame.

    add_suffix : bool
        If True, append '_reordered' to output keys.
        If False, keep original keys.

    Returns
    -------
    dict[str, pd.DataFrame]
        Reordered DataFrames.
    """
    parts_lookup = (
        neuron_subtype
        .set_index("id")[["part1", "part2"]]
        .to_dict("index")
    )

    reordered = {}

    for name, df in halves_dict.items():
        name_parts = name.split("_")
        neuron_id = int(name_parts[1])
        side_name = name_parts[-1]
        column_to_look = "part1" if "L" in side_name else "part2"

        parts = parts_lookup.get(neuron_id)
        if parts is None:
            continue

        center = parts.get(column_to_look)
        if pd.isna(center):
            continue

        idxs = df.index[df["Glomerulus"] == center]
        if len(idxs) == 0:
            continue

        current_idx = int(idxs[0])
        shift = target_idx - current_idx

        order = np.roll(np.arange(len(df)), shift)
        df_reordered = df.iloc[order].reset_index(drop=True)

        out_name = f"{name}_reordered" if add_suffix else name
        reordered[out_name] = df_reordered

    return reordered



def add_percentage_synapse_count(
    dfs: Dict[str, pd.DataFrame],
    *,
    count_col: str = "SynapseCount",
    new_col: str = "percentage_SynapseCount"
) -> Dict[str, pd.DataFrame]:
    """
    Add a normalized synapse-count column to each DataFrame in a dictionary.

    For each DataFrame, this computes:

        new_col = count_col / max(count_col)

    If the count column is missing, the DataFrame is empty, or the maximum value
    is missing/non-positive, the new column is filled with 0.0.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Dictionary of DataFrames.

    count_col : str
        Column containing raw synapse counts.

    new_col : str
        Name of the normalized output column.

    Returns
    -------
    dict[str, pd.DataFrame]
        New dictionary with the same keys and updated DataFrames.
    """
    out = {}

    for name, df in dfs.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            out[name] = df
            continue

        base = df.copy()

        if count_col not in base.columns:
            base[new_col] = 0.0
            out[name] = base
            continue

        s = pd.to_numeric(base[count_col], errors="coerce")
        m = s.max(skipna=True)

        if pd.isna(m) or m <= 0:
            base[new_col] = 0.0
        else:
            base[new_col] = (s / m).astype(float)

        out[name] = base

    return out



def make_add1row_versions(
    dfs: Dict[str, pd.DataFrame],
    *,
    glomerulus_col: str = "Glomerulus",
    metric_cols: Iterable[str] = ("SynapseCount", "percentage_SynapseCount"),
    suffix: str = "_add1row",
    add_suffix: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Add one copied bottom row to each DataFrame in a dictionary.

    For each DataFrame, append a final row where:
    - glomerulus_col = "added"
    - each column in metric_cols copies the value from the first row, index 0

    This is useful for circular PB plots where the first point needs to be
    repeated at the end to visually close the curve.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Dictionary of DataFrames.

    glomerulus_col : str
        Column containing glomerulus labels.

    metric_cols : iterable of str
        Columns whose values should be copied from the first row into the added row.

    suffix : str
        Suffix to append to keys if add_suffix=True.

    add_suffix : bool
        If True, output keys become '<old_key><suffix>'.
        If False, output keys stay unchanged.

    Returns
    -------
    dict[str, pd.DataFrame]
        New dictionary with added bottom rows.
    """
    out = {}

    for name, df in dfs.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        base = df.reset_index(drop=True).copy()

        if glomerulus_col not in base.columns:
            base[glomerulus_col] = pd.NA

        new_row = {c: pd.NA for c in base.columns}
        new_row[glomerulus_col] = "added"

        for c in metric_cols:
            if c in base.columns:
                new_row[c] = base.loc[0, c]

        df_new = pd.concat(
            [base, pd.DataFrame([new_row], columns=base.columns)],
            ignore_index=True,
        )

        out_name = f"{name}{suffix}" if add_suffix else name
        out[out_name] = df_new

    return out


# new function for "2 halves into one full PB"
def make_add1row_versions_for_combined_halves(
    dfs: Dict[str, pd.DataFrame],
    *,
    first_half_start_idx: int = 0,
    second_half_start_idx: int = 8,
    glomerulus_col: str = "Glomerulus",
    added_label: str = "added",
) -> Dict[str, pd.DataFrame]:
    """
    Add one duplicated row to the end of each half in a DataFrame that contains
    two PB halves stacked together.

    Assumes each input DataFrame is arranged as:
        old rows 0-7  = first half
        old rows 8-15 = second half

    This function:
    - duplicates old row 0 and inserts it after old row 7
    - duplicates old row 8 and appends it after old row 15

    So each DataFrame goes from 16 rows to 18 rows.
    """
    out = {}

    for name, df in dfs.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        base = df.reset_index(drop=True).copy()

        if len(base) <= second_half_start_idx:
            raise ValueError(
                f"DataFrame '{name}' has only {len(base)} rows, "
                f"but second_half_start_idx={second_half_start_idx}."
            )

        first_added = base.iloc[[first_half_start_idx]].copy()
        second_added = base.iloc[[second_half_start_idx]].copy()

        if glomerulus_col in base.columns:
            first_added.loc[:, glomerulus_col] = added_label
            second_added.loc[:, glomerulus_col] = added_label

        first_half = base.iloc[:second_half_start_idx].copy()
        second_half = base.iloc[second_half_start_idx:].copy()

        df_new = pd.concat(
            [
                first_half,
                first_added,
                second_half,
                second_added,
            ],
            ignore_index=True,
        )

        out[name] = df_new

    return out


def sum_and_mean_synapse_counts(
    dfs_dict: dict[str, pd.DataFrame],
    column: str = "SynapseCount",
    make_glomerulus_labels: bool = True,
) -> pd.DataFrame:
    """
    Sum and average row-wise synapse-count values across a dictionary of DataFrames.

    For each row position, this function computes:
    - sum of `column`
    - mean of `column`
    - sum of `percentage_<column>`
    - mean of `percentage_<column>`
    - normalized summed `column`, called 'percentage_SynapseCount_mean_NEW'

    Notes
    -----
    `percentage_<column>_mean` is the average of already-normalized values
    from the individual DataFrames.

    `percentage_SynapseCount_mean_NEW` is different: it is computed from the
    summed raw `column` in the output summary DataFrame, then divided by that
    output column's maximum value.

    Parameters
    ----------
    dfs_dict : dict[str, pd.DataFrame]
        Dictionary of DataFrames with matching row counts.

    column : str
        Raw count column to summarize. Default is 'SynapseCount'.

    make_glomerulus_labels : bool
        If True, create centered x-axis labels like -4..4.
        Requires an odd number of rows.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with row-wise summed and averaged metrics.
    """
    if not dfs_dict:
        raise ValueError("Input dictionary is empty.")

    perc_col = f"percentage_{column}"

    series_main = []
    series_perc = []
    nrows = None

    for name, df in dfs_dict.items():
        if column not in df.columns:
            raise ValueError(f"'{column}' not found in DataFrame '{name}'.")

        s_main = pd.to_numeric(df[column], errors="coerce").reset_index(drop=True)

        if nrows is None:
            nrows = len(s_main)
        elif len(s_main) != nrows:
            raise ValueError(
                f"Row count mismatch in '{name}': expected {nrows}, got {len(s_main)}"
            )

        series_main.append(s_main)

        if perc_col not in df.columns:
            raise ValueError(f"'{perc_col}' not found in DataFrame '{name}'.")

        s_perc = pd.to_numeric(df[perc_col], errors="coerce").reset_index(drop=True)

        if len(s_perc) != nrows:
            raise ValueError(
                f"Row count mismatch for '{perc_col}' in '{name}': "
                f"expected {nrows}, got {len(s_perc)}"
            )

        series_perc.append(s_perc)

    mat_main = pd.concat(series_main, axis=1)
    mat_perc = pd.concat(series_perc, axis=1)

    sum_main = mat_main.sum(axis=1, skipna=True)
    n_main = mat_main.notna().sum(axis=1)
    mean_main = sum_main.divide(n_main).where(n_main > 0)

    sum_perc = mat_perc.sum(axis=1, skipna=True)
    n_perc = mat_perc.notna().sum(axis=1)
    mean_perc = sum_perc.divide(n_perc).where(n_perc > 0)

    if make_glomerulus_labels:
        n = len(sum_main)
        if n % 2 != 1:
            raise ValueError(f"Expected an odd number of rows, got {n}.")
        half = n // 2
        glomerulus = list(range(-half, half + 1))
    else:
        glomerulus = list(range(len(sum_main)))

    out = pd.DataFrame(
        {
            "glomerulus": glomerulus,
            column: sum_main.values,
            f"{column}_mean": mean_main.values,
            perc_col: sum_perc.values,
            f"{perc_col}_mean": mean_perc.values,
        }
    )

    s = pd.to_numeric(out[column], errors="coerce")
    m = s.max(skipna=True)

    out["percentage_SynapseCount_mean_NEW"] = np.where(
        np.isfinite(m) & (m != 0),
        s / m,
        0.0,
    ).astype(float)

    return out


# new for "2 halves to one full PB"
def sum_and_mean_synapse_counts_combined_halves(
    dfs_dict: dict[str, pd.DataFrame],
    column: str = "SynapseCount",
) -> pd.DataFrame:
    """
    Sum and average row-wise synapse-count values across DataFrames that contain
    two 9-row circular PB halves stacked together.

    Assumes each DataFrame has 18 rows:
        rows 0-8   = first half, with duplicated closing row
        rows 9-17  = second half, with duplicated closing row
    """

    if not dfs_dict:
        raise ValueError("Input dictionary is empty.")

    perc_col = f"percentage_{column}"

    series_main = []
    series_perc = []
    nrows = None

    for name, df in dfs_dict.items():
        if column not in df.columns:
            raise ValueError(f"'{column}' not found in DataFrame '{name}'.")

        if perc_col not in df.columns:
            raise ValueError(f"'{perc_col}' not found in DataFrame '{name}'.")

        s_main = pd.to_numeric(df[column], errors="coerce").reset_index(drop=True)
        s_perc = pd.to_numeric(df[perc_col], errors="coerce").reset_index(drop=True)

        if nrows is None:
            nrows = len(s_main)
        elif len(s_main) != nrows:
            raise ValueError(
                f"Row count mismatch in '{name}': expected {nrows}, got {len(s_main)}"
            )

        series_main.append(s_main)
        series_perc.append(s_perc)

    mat_main = pd.concat(series_main, axis=1)
    mat_perc = pd.concat(series_perc, axis=1)

    sum_main = mat_main.sum(axis=1, skipna=True)
    mean_main = mat_main.mean(axis=1, skipna=True)

    sum_perc = mat_perc.sum(axis=1, skipna=True)
    mean_perc = mat_perc.mean(axis=1, skipna=True)

    if nrows == 18:
        glomerulus = list(range(0, 9)) + list(range(0, 9))
        half = ["half1"] * 9 + ["half2"] * 9
    else:
        glomerulus = list(range(nrows))
        half = ["combined"] * nrows

    out = pd.DataFrame(
        {
            "half": half,
            "glomerulus": glomerulus,
            column: sum_main.values,
            f"{column}_mean": mean_main.values,
            perc_col: sum_perc.values,
            f"{perc_col}_mean": mean_perc.values,
        }
    )

    s = pd.to_numeric(out[column], errors="coerce")
    m = s.max(skipna=True)

    out["percentage_SynapseCount_mean_NEW"] = np.where(
        np.isfinite(m) & (m != 0),
        s / m,
        0.0,
    ).astype(float)

    return out






#-------------------------PART TWO (and THREE)-------------------------

def add_pct_of_max_channels(
    dfs: Dict[str, pd.DataFrame],
    *,
    channels: Dict[str, str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Add normalized percentage columns for multiple channels.

    Each channel is normalized by its maximum value within each DataFrame.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]

    channels : dict[str, str]
        Mapping from raw column → normalized column name.
        Example:
        {
            "total_green": "percentage_total_green",
            "total_red": "percentage_total_red"
        }

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    if channels is None:
        channels = {
            "total_green": "percentage_total_green",
            "total_red": "percentage_total_red",
        }

    out = {}

    for name, df in dfs.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            out[name] = df
            continue

        tmp = df.copy()

        for value_col, new_col in channels.items():
            if value_col not in tmp.columns:
                continue

            s = pd.to_numeric(tmp[value_col], errors="coerce")
            m = s.max(skipna=True)

            if pd.notna(m) and m != 0:
                tmp[new_col] = (s / m).astype(float)
            else:
                tmp[new_col] = 0.0

        out[name] = tmp

    return out


# for "2 halves to one full PB"
def combine_rotated_halves_by_cell(
    dfs_dict: dict[str, pd.DataFrame],
    *,
    center_idx: int = 4,
) -> dict[str, pd.DataFrame]:
    """
    For half-dataframes named like:

        fly12_L
        fly12_R

    1. rotate each dataframe so center_idx becomes row 0
    2. combine L and R halves from the same fly
       (L first, then R)

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are fly names (e.g. fly12)
    """

    # rotate
    rotated = {}

    for name, df in dfs_dict.items():
        df_rot = (
            df.iloc[np.roll(np.arange(len(df)), -center_idx)]
              .reset_index(drop=True)
        )
        rotated[name] = df_rot

    # combine
    combined = {}

    fly_names = sorted({
        name.rsplit("_", 1)[0]
        for name in rotated
    })

    for fly_name in fly_names:

        left_name = f"{fly_name}_L"
        right_name = f"{fly_name}_R"

        if left_name not in rotated:
            raise ValueError(f"Missing {left_name}")

        if right_name not in rotated:
            raise ValueError(f"Missing {right_name}")

        combined[fly_name] = pd.concat(
            [
                rotated[left_name],
                rotated[right_name],
            ],
            ignore_index=True,
        )

    return combined


def add_comparison_columns_to_halves(
    dct: dict[str, pd.DataFrame],
    denom_d7_df: pd.DataFrame,
    *,
    value_col: str = "percentage_total_green",
    denom_glut_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Add columns to each IHC half-PB DataFrame by dividing a chosen value column
    by connectome population summary columns.

    For example, if value_col='percentage_total_green', adds:
    - percentage_total_green_divided_by_d7countall
    - percentage_total_green_divided_by_d7%all

    If denom_glut_df is provided, also adds:
    - percentage_total_green_divided_by_glutamatecountall
    - percentage_total_green_divided_by_glutamate%all

    Division is aligned by row index. Zero or invalid denominators produce NaN.
    """
    for needed in ("SynapseCount_mean", "percentage_SynapseCount_mean"):
        if needed not in denom_d7_df.columns:
            raise ValueError(f"denom_d7_df must have '{needed}' column.")

    d7_count = pd.to_numeric(
        denom_d7_df["SynapseCount_mean"], errors="coerce"
    ).reset_index(drop=True)

    d7_pct = pd.to_numeric(
        denom_d7_df["percentage_SynapseCount_mean"], errors="coerce"
    ).reset_index(drop=True)

    glut_count = glut_pct = None

    if denom_glut_df is not None:
        for needed in ("SynapseCount_mean", "percentage_SynapseCount_mean"):
            if needed not in denom_glut_df.columns:
                raise ValueError(f"denom_glut_df must have '{needed}' column.")

        glut_count = pd.to_numeric(
            denom_glut_df["SynapseCount_mean"], errors="coerce"
        ).reset_index(drop=True)

        glut_pct = pd.to_numeric(
            denom_glut_df["percentage_SynapseCount_mean"], errors="coerce"
        ).reset_index(drop=True)

    out = {}

    out_d7_count_col = f"{value_col}_divided_by_d7countall"
    out_d7_pct_col = f"{value_col}_divided_by_d7%all"
    out_glut_count_col = f"{value_col}_divided_by_glutamatecountall"
    out_glut_pct_col = f"{value_col}_divided_by_glutamate%all"

    for name, df in dct.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            out[name] = df
            continue

        tmp = df.copy()

        if value_col not in tmp.columns:
            out[name] = tmp
            continue

        num = pd.to_numeric(tmp[value_col], errors="coerce").reset_index(drop=True)

        n = min(len(num), len(d7_count))
        res_d7_count = np.divide(
            num.iloc[:n].to_numpy(),
            d7_count.iloc[:n].to_numpy(),
            out=np.full(n, np.nan, dtype=float),
            where=(d7_count.iloc[:n].to_numpy() != 0)
            & np.isfinite(d7_count.iloc[:n].to_numpy()),
        )
        tmp[out_d7_count_col] = np.nan
        tmp.loc[tmp.index[:n], out_d7_count_col] = res_d7_count

        m = min(len(num), len(d7_pct))
        res_d7_pct = np.divide(
            num.iloc[:m].to_numpy(),
            d7_pct.iloc[:m].to_numpy(),
            out=np.full(m, np.nan, dtype=float),
            where=(d7_pct.iloc[:m].to_numpy() != 0)
            & np.isfinite(d7_pct.iloc[:m].to_numpy()),
        )
        tmp[out_d7_pct_col] = np.nan
        tmp.loc[tmp.index[:m], out_d7_pct_col] = res_d7_pct

        if (glut_count is not None) and (glut_pct is not None):
            k = min(len(num), len(glut_count))
            res_glut_count = np.divide(
                num.iloc[:k].to_numpy(),
                glut_count.iloc[:k].to_numpy(),
                out=np.full(k, np.nan, dtype=float),
                where=(glut_count.iloc[:k].to_numpy() != 0)
                & np.isfinite(glut_count.iloc[:k].to_numpy()),
            )
            tmp[out_glut_count_col] = np.nan
            tmp.loc[tmp.index[:k], out_glut_count_col] = res_glut_count

            r = min(len(num), len(glut_pct))
            res_glut_pct = np.divide(
                num.iloc[:r].to_numpy(),
                glut_pct.iloc[:r].to_numpy(),
                out=np.full(r, np.nan, dtype=float),
                where=(glut_pct.iloc[:r].to_numpy() != 0)
                & np.isfinite(glut_pct.iloc[:r].to_numpy()),
            )
            tmp[out_glut_pct_col] = np.nan
            tmp.loc[tmp.index[:r], out_glut_pct_col] = res_glut_pct

        out[name] = tmp

    return out



def build_population_df(
    dfs: Dict[str, pd.DataFrame],
    *,
    channels: List[str] = ("green", "red"),
    include_normalized: bool = True,
    n_positions: int = 9,
) -> pd.DataFrame:
    """
    Build a population-level summary DataFrame from IHC half-PB DataFrames.

    For each channel, this function computes:
    - sum across all inputs
    - average across all inputs

    Optionally includes normalized and cross-normalized columns.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]

    channels : list[str]
        Channel names (e.g., ["green", "red"])

    include_normalized : bool
        Whether to include percentage and cross-normalized columns

    n_positions : int
        Number of PB positions (typically 9)

    Returns
    -------
    pd.DataFrame
    """
    cols = []
    avg_cols = []

    for ch in channels:
        cols += [
            f"mean_{ch}",
            f"total_{ch}",
        ]
        avg_cols += [
            f"total_{ch}",
        ]

        if include_normalized:
            cols += [
                f"percentage_total_{ch}",
                f"percentage_total_{ch}_divided_by_d7countall",
                f"percentage_total_{ch}_divided_by_d7%all",
            ]
            avg_cols += [
                f"percentage_total_{ch}",
                f"percentage_total_{ch}_divided_by_d7countall",
                f"percentage_total_{ch}_divided_by_d7%all",
            ]

    stack = []

    for _, df in dfs.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        tmp = (
            df.reindex(columns=cols)
              .apply(pd.to_numeric, errors="coerce")
              .reset_index(drop=True)
              .reindex(range(n_positions))
              .fillna(0)
        )

        stack.append(tmp)

    n_inputs = len(stack)

    if n_inputs == 0:
        pop = pd.DataFrame(0, index=range(n_positions), columns=cols)
    else:
        pop = (
            pd.concat(stack, keys=range(n_inputs))
              .groupby(level=1)
              .sum()
              .reindex(range(n_positions), fill_value=0)
        )

    if n_positions == 9:
        gloms = list(range(-4, 5))
    else:
        half = n_positions // 2
        gloms = list(range(-half, n_positions - half))

    pop.insert(0, "glomerulus", gloms)

    if n_inputs > 0:
        for c in avg_cols:
            if c in pop.columns:
                pop[f"{c}_avg"] = pop[c] / n_inputs
    else:
        for c in avg_cols:
            pop[f"{c}_avg"] = np.nan

    return pop




'''
OLD function, just in case...

def plot_individual_vs_population_ihc(
    indiv_dict,
    pop_df,
    metric="mean",
    channel="green",
    line_color=None,
    *,
    column=None,
    show_legend: bool = True,
    save_svg: str | Path | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    individual_color_mode="same",
    individual_color="gray",
    individual_alpha=0.45,
    individual_linewidth=1,
):
    """
    Plot individual IHC traces (thin lines) together with a population average
    (thick line), with flexible input modes and optional SVG export.

    This function supports two ways to specify which column to plot:

    1. **metric + channel mode** (default)
    The column name is constructed as:
        f"{metric}_{channel}"

    Examples:
        metric="total", channel="green"        → "total_green"
        metric="percentage", channel="total_green" → "percentage_total_green"

    2. **column mode (recommended for clarity)**
    Provide the full column name directly:
        column="percentage_total_green"

    When `column` is provided, it overrides `metric` and `channel`.

    The function also automatically looks for the population column:
        f"{signal}_avg"

    ---

    Parameters
    ----------
    indiv_dict : dict[str, pd.DataFrame]
        Dictionary of individual DataFrames (e.g., per neuron).
        Each DataFrame must contain the selected signal column.

    pop_df : pd.DataFrame
        Population summary DataFrame.
        Must contain the corresponding averaged column (signal + "_avg").

    metric : str, default "mean"
        Metric prefix used to construct column names (only used if `column` is None).

    channel : str, default "green"
        Channel name or partial column name (only used if `column` is None).

    line_color : str or None
        Color of the population (thick) line.
        If None, automatically inferred from the signal name.

    column : str or None
        Full column name to plot (overrides metric + channel).
        Recommended for explicit and unambiguous usage.

    show_legend : bool, default True
        Whether to display the legend.

    save_svg : str | Path | None
        If provided, saves the figure as an SVG at this path.
        If None, no file is saved.

    xlabel, ylabel : str or None
        Custom axis labels. Defaults are used if None.

    title : str or None
        Custom plot title. If None, a default title is used.

    individual_color_mode : {"same", "different"}
        - "same": all individual traces use the same color
        - "different": each individual trace uses a different color

    individual_color : str, default "gray"
        Color for individual traces when using "same" mode.

    individual_alpha : float, default 0.45
        Transparency of individual traces.

    individual_linewidth : float, default 1
        Line width of individual traces.

    ---

    Returns
    -------
    None
        Displays the plot and optionally saves it as an SVG.
    """
    if not indiv_dict:
        raise ValueError("indiv_dict is empty.")

    signal = column if column is not None else f"{metric}_{channel}"
    signal_avg = f"{signal}_avg"

    if line_color is None:
        if "green" in signal:
            line_color = "green"
        elif "red" in signal:
            line_color = "red"
        else:
            line_color = "tab:blue"

    if signal_avg not in pop_df.columns:
        raise ValueError(f"pop_df must contain '{signal_avg}'.")

    individual_color_mode = str(individual_color_mode).lower()
    if individual_color_mode not in {"same", "different"}:
        raise ValueError("individual_color_mode must be 'same' or 'different'.")

    lengths = [len(df) for df in indiv_dict.values()]
    lengths.append(len(pop_df))
    L = min(lengths)
    x = np.arange(L)

    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(7, 5))

        if individual_color_mode == "different":
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, (name, df) in enumerate(indiv_dict.items()):
            if signal not in df.columns:
                raise ValueError(f"'{signal}' not found in '{name}'.")

            y = pd.to_numeric(df[signal], errors="coerce").to_numpy()[:L]

            color = (
                individual_color
                if individual_color_mode == "same"
                else color_cycle[i % len(color_cycle)]
            )

            ax.plot(
                x,
                y,
                linewidth=individual_linewidth,
                alpha=individual_alpha,
                color=color,
            )

        if show_legend:
            legend_color = individual_color if individual_color_mode == "same" else "gray"
            ax.plot(
                [],
                [],
                linewidth=individual_linewidth,
                color=legend_color,
                alpha=individual_alpha,
                label=f"individuals ({signal})",
            )

        y_pop = pd.to_numeric(pop_df[signal_avg], errors="coerce").to_numpy()[:L]

        ax.plot(
            x,
            y_pop,
            linewidth=4,
            color=line_color,
            marker="o",
            label=(f"population ({signal_avg})" if show_legend else None),
        )

        if "glomerulus" in pop_df.columns:
            xtick_labels = pop_df["glomerulus"].astype(str).tolist()[:L]
        else:
            xtick_labels = [str(i) for i in range(L)]

        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels)

        ax.set_xlabel(xlabel if xlabel is not None else "Glomerulus")
        ax.set_ylabel(ylabel if ylabel is not None else signal.replace("_", " "))
        ax.set_title(
            title if title is not None else f"Individuals vs Population (n={len(indiv_dict)})",
            pad=20,
        )

        if show_legend:
            ax.legend(
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.07),
                ncol=2,
                borderaxespad=0.0,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.92])
        else:
            fig.tight_layout()

        if save_svg is not None:
            save_svg = Path(save_svg)
            save_svg.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_svg,
                format="svg",
                bbox_inches="tight",
                transparent=True,
                metadata={"Creator": "plot_individual_vs_population_ihc"},
            )

        plt.show()
'''


# NEWWWWWW
def plot_individual_vs_population_ihc(
    indiv_dict,
    pop_df,
    metric="mean",
    channel="green",
    line_color=None,
    *,
    column=None,
    show_legend: bool = True,
    number_of_legendcolumns: int = 2,
    save_svg: str | Path | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,

    # colors
    individual_color_mode="same",   # "same" or "different"
    individual_color="gray",
    mean_line_color=None,
    dot_color=None,

    # line / dot sizes
    individual_alpha=0.45,
    individual_linewidth=1.0,
    mean_linewidth=4.0,
    dot_size=8.0,

    # figure
    figsize=(7, 5),

    # font sizes
    title_fontsize=16,
    xlabel_fontsize=14,
    ylabel_fontsize=14,
    xtick_fontsize=12,
    ytick_fontsize=12,
    legend_fontsize=10,

    # axis label style
    ylabel_rotation=90,
    ylabel_labelpad=10,

    # axis/tick style
    axis_linewidth=1.5,
    tick_width=1.5,
    tick_length=5,
    axis_outward=6,

    # y axis
    y_max_pad_frac=0.08,
    y_tick_decimals=1,
    ymax_mode: str = "padded",   # "padded" or "exact"
    auto_y_scale: bool = True,

    # x axis
    x_margin=0.5,
):
    if not indiv_dict:
        raise ValueError("indiv_dict is empty.")

    signal = column if column is not None else f"{metric}_{channel}"
    signal_avg = f"{signal}_avg"

    if line_color is None:
        if "green" in signal:
            line_color = "green"
        elif "red" in signal:
            line_color = "red"
        else:
            line_color = "tab:blue"

    if mean_line_color is None:
        mean_line_color = line_color

    if dot_color is None:
        dot_color = line_color

    if signal_avg not in pop_df.columns:
        raise ValueError(f"pop_df must contain '{signal_avg}'.")

    individual_color_mode = individual_color_mode.lower()
    if individual_color_mode not in {"same", "different"}:
        raise ValueError("individual_color_mode must be 'same' or 'different'.")

    lengths = [len(df) for df in indiv_dict.values()]
    lengths.append(len(pop_df))
    L = min(lengths)
    x = np.arange(L)

    all_y = []

    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=figsize)

        # individual traces
        if individual_color_mode == "different":
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, (name, df) in enumerate(indiv_dict.items()):
            if signal not in df.columns:
                raise ValueError(f"'{signal}' not found in '{name}'.")

            y = pd.to_numeric(df[signal], errors="coerce").to_numpy()[:L]
            all_y.extend(y[np.isfinite(y)])

            color = (
                individual_color
                if individual_color_mode == "same"
                else color_cycle[i % len(color_cycle)]
            )

            ax.plot(
                x,
                y,
                linewidth=individual_linewidth,
                alpha=individual_alpha,
                color=color,
                zorder=1,
            )

        # legend handle for individuals
        if show_legend:
            ax.plot(
                [],
                [],
                linewidth=individual_linewidth,
                color=individual_color,
                alpha=individual_alpha,
                label=f"individuals ({signal})",
            )

        # population trace
        y_pop = pd.to_numeric(pop_df[signal_avg], errors="coerce").to_numpy()[:L]
        all_y.extend(y_pop[np.isfinite(y_pop)])

        ax.plot(
            x,
            y_pop,
            linewidth=mean_linewidth,
            color=mean_line_color,
            marker="o",
            markersize=dot_size,
            markerfacecolor=dot_color,
            markeredgecolor=dot_color,
            label=(f"population ({signal_avg})" if show_legend else None),
            zorder=3,
        )

        # x ticks
        if "glomerulus" in pop_df.columns:
            xtick_labels = pop_df["glomerulus"].astype(str).tolist()[:L]
        else:
            xtick_labels = [str(i) for i in range(L)]

        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels)

        # y-axis range
        if len(all_y) == 0:
            ymax_use = 1
        else:
            ymax_data = np.nanmax(all_y)

            if ymax_mode == "exact":
                ymax_use = ymax_data
            elif ymax_mode == "padded":
                ymax_use = ymax_data * (1 + y_max_pad_frac)
            else:
                raise ValueError("ymax_mode must be 'exact' or 'padded'.")

            if ymax_use == 0:
                ymax_use = 1

        ax.set_ylim(0, ymax_use)

        # y ticks: always 0, middle, max
        y_ticks = np.linspace(0, ymax_use, 3)
        ax.set_yticks(y_ticks)

        # automatic y-axis scaling for display only
        if auto_y_scale and ymax_use != 0:
            y_scale_exp = int(np.floor(np.log10(abs(ymax_use))))

            # do not scale ordinary values like 0.1–999
            if -1 <= y_scale_exp <= 2:
                y_scale_exp = 0
        else:
            y_scale_exp = 0

        y_scale = 10 ** y_scale_exp

        ax.set_yticklabels(
            [f"{v / y_scale:.{y_tick_decimals}f}" for v in y_ticks]
        )

        # show scale factor only once, near y max
        if y_scale_exp != 0:
            ax.text(
                -0.03,
                1.02,
                rf"$\times 10^{{{y_scale_exp}}}$",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=ytick_fontsize,
            )

        # apply tick font sizes robustly
        ax.tick_params(
            axis="x",
            labelsize=xtick_fontsize,
            direction="out",
            width=tick_width,
            length=tick_length,
        )

        ax.tick_params(
            axis="y",
            labelsize=ytick_fontsize,
            direction="out",
            width=tick_width,
            length=tick_length,
        )

        # labels
        ax.set_xlabel(
            xlabel if xlabel is not None else "Glomerulus",
            fontsize=xlabel_fontsize,
        )

        ax.set_ylabel(
            ylabel if ylabel is not None else signal.replace("_", " "),
            fontsize=ylabel_fontsize,
            rotation=ylabel_rotation,
            labelpad=ylabel_labelpad,
        )

        ax.set_title(
            title if title is not None else f"Individuals vs Population (n={len(indiv_dict)})",
            fontsize=title_fontsize,
            pad=20,
        )

        # remove box: keep only left and bottom spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_linewidth(axis_linewidth)
        ax.spines["bottom"].set_linewidth(axis_linewidth)

        # move axes outward so x/y axes do not touch
        ax.spines["left"].set_position(("outward", axis_outward))
        ax.spines["bottom"].set_position(("outward", axis_outward))

        # shorten axis lines
        ax.spines["left"].set_bounds(0, ymax_use)
        ax.spines["bottom"].set_bounds(x[0], x[-1])

        # keep x-axis with margin
        ax.set_xlim(x[0] - x_margin, x[-1] + x_margin)

        if show_legend:
            ax.legend(
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.07),
                ncol=number_of_legendcolumns,
                borderaxespad=0.0,
                fontsize=legend_fontsize,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.92])
        else:
            fig.tight_layout()

        if save_svg is not None:
            save_svg = Path(save_svg)
            save_svg.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_svg,
                format="svg",
                bbox_inches="tight",
                transparent=True,
                metadata={"Creator": "plot_individual_vs_population_ihc"},
            )

        plt.show()






'''
OLD function, just in case...

def plot_individual_vs_population_connectome(
    indiv_dict,
    pop_df,
    line_color="tab:blue",
    title=None,
    *,
    metric="SynapseCount",
    mean_version="old",
    show_legend: bool = True,
    save_svg: str | Path | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    individual_color_mode="same",
    individual_color="gray",
    individual_alpha=0.45,
    individual_linewidth=1,
):
    """
    Plot individual connectome traces and a population summary trace.

    Supports optional SVG saving, custom axis labels/title, legend control,
    and individual trace coloring.

    Parameters
    ----------
    indiv_dict : dict[str, pd.DataFrame]
        Individual DataFrames. Each must contain `metric`.

    pop_df : pd.DataFrame
        Population summary DataFrame.

    line_color : str
        Color for the population line.

    title : str or None
        Plot title.

    metric : {"SynapseCount", "percentage_SynapseCount"}
        Individual-trace column to plot.

    mean_version : {"old", "new"}
        Only used when metric="percentage_SynapseCount".
        - "old": plot population column "percentage_SynapseCount_mean"
        - "new": plot "percentage_SynapseCount_mean_NEW" if present,
          otherwise compute SynapseCount_mean / max(SynapseCount_mean).

    show_legend : bool
        Whether to show the legend.

    save_svg : str | Path | None
        If provided, save the plot as an SVG at this path.
        If None, no file is saved.

    xlabel, ylabel : str or None
        Custom axis labels.

    individual_color_mode : {"same", "different"}
        "same": all individual traces use `individual_color`.
        "different": individual traces use the matplotlib color cycle.

    individual_color : str
        Color for individual traces when individual_color_mode="same".

    individual_alpha : float
        Transparency of individual traces.

    individual_linewidth : float
        Line width of individual traces.

    Returns
    -------
    None
        Displays the plot and optionally saves it as SVG.
    """
    if not indiv_dict:
        raise ValueError("indiv_dict is empty.")

    metric = str(metric)
    if metric not in {"SynapseCount", "percentage_SynapseCount"}:
        raise ValueError("metric must be 'SynapseCount' or 'percentage_SynapseCount'.")

    if metric == "SynapseCount":
        pop_col = "SynapseCount_mean"
        if pop_col not in pop_df.columns:
            raise ValueError(f"pop_df must contain '{pop_col}'.")
        y_pop_raw = pd.to_numeric(pop_df[pop_col], errors="coerce")
        pop_label = "population (SynapseCount_mean)"

    else:
        mean_version = str(mean_version).lower()
        if mean_version not in {"old", "new"}:
            raise ValueError("mean_version must be 'old' or 'new'.")

        if mean_version == "old":
            pop_col = "percentage_SynapseCount_mean"
            if pop_col not in pop_df.columns:
                raise ValueError(f"pop_df must contain '{pop_col}'.")
            y_pop_raw = pd.to_numeric(pop_df[pop_col], errors="coerce")
            pop_label = "population (percentage_SynapseCount_mean)"

        else:
            if "percentage_SynapseCount_mean_NEW" in pop_df.columns:
                y_pop_raw = pd.to_numeric(
                    pop_df["percentage_SynapseCount_mean_NEW"],
                    errors="coerce",
                )
                pop_label = "population (percentage_SynapseCount_mean_NEW)"
            else:
                if "SynapseCount_mean" not in pop_df.columns:
                    raise ValueError(
                        "pop_df must contain 'SynapseCount_mean' to compute NEW version."
                    )

                scm = pd.to_numeric(pop_df["SynapseCount_mean"], errors="coerce")
                m = scm.max(skipna=True)
                y_pop_raw = (scm / m) if (np.isfinite(m) and m != 0) else scm * 0
                pop_label = "population (SynapseCount_mean / max)"

    individual_color_mode = str(individual_color_mode).lower()
    if individual_color_mode not in {"same", "different"}:
        raise ValueError("individual_color_mode must be 'same' or 'different'.")

    lengths = [len(df) for df in indiv_dict.values()]
    lengths.append(len(pop_df))
    L = int(min(lengths))
    x = np.arange(L)

    if "glomerulus" in pop_df.columns:
        xtick_labels = pop_df["glomerulus"].astype(str).tolist()[:L]
    elif "Glomerulus" in pop_df.columns:
        xtick_labels = pop_df["Glomerulus"].astype(str).tolist()[:L]
    else:
        xtick_labels = [str(i) for i in range(L)]

    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(7, 5))

        if individual_color_mode == "different":
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, (name, df) in enumerate(indiv_dict.items()):
            if metric not in df.columns:
                raise ValueError(f"'{metric}' not found in individual df '{name}'.")

            y_ind = pd.to_numeric(df[metric], errors="coerce").to_numpy()[:L]

            color = (
                individual_color
                if individual_color_mode == "same"
                else color_cycle[i % len(color_cycle)]
            )

            ax.plot(
                x,
                y_ind,
                linewidth=individual_linewidth,
                alpha=individual_alpha,
                color=color,
            )

        if show_legend:
            legend_color = individual_color if individual_color_mode == "same" else "gray"
            ax.plot(
                [],
                [],
                linewidth=individual_linewidth,
                color=legend_color,
                alpha=individual_alpha,
                label=f"individuals ({metric})",
            )

        y_pop = y_pop_raw.to_numpy()[:L]

        ax.plot(
            x,
            y_pop,
            linewidth=4,
            color=line_color,
            marker="o",
            label=(pop_label if show_legend else None),
        )

        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels)

        ax.set_xlabel(xlabel if xlabel is not None else "Glomerulus")
        ax.set_ylabel(ylabel if ylabel is not None else metric.replace("_", " "))
        ax.set_title(
            title if title is not None else f"Individuals vs Population (n={len(indiv_dict)})",
            pad=20,
        )

        if show_legend:
            ax.legend(frameon=False)
            fig.tight_layout()
        else:
            fig.tight_layout()

        if save_svg is not None:
            save_svg = Path(save_svg)
            save_svg.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_svg,
                format="svg",
                bbox_inches="tight",
                transparent=True,
                metadata={"Creator": "plot_individual_vs_population_connectome"},
            )

        plt.show()
'''

#NEWWWW
def plot_individual_vs_population_connectome(
    indiv_dict,
    pop_df,
    line_color="tab:blue",
    title=None,
    *,
    metric="SynapseCount",
    mean_version="old",
    show_legend: bool = True,
    save_svg: str | Path | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,

    # colors
    individual_color_mode="same",   # "same" or "different"
    individual_color="gray",
    mean_line_color=None,
    dot_color=None,

    # line / dot sizes
    individual_alpha=0.45,
    individual_linewidth=1.0,
    mean_linewidth=4.0,
    dot_size=8.0,

    # figure
    figsize=(7, 5),

    # font sizes
    title_fontsize=16,
    xlabel_fontsize=14,
    ylabel_fontsize=14,
    xtick_fontsize=12,
    ytick_fontsize=12,
    legend_fontsize=10,

    # legend
    legend_ncol=1,
    legend_loc="upper center",
    legend_bbox=(0.5, 1.07),

    # axis label style
    ylabel_rotation=90,
    ylabel_labelpad=10,

    # axis/tick style
    axis_linewidth=1.5,
    tick_width=1.5,
    tick_length=5,
    axis_outward=6,

    # y axis
    y_max_pad_frac=0.08,
    y_tick_decimals=1,
    ymax_mode: str = "padded",   # "padded" or "exact"
    auto_y_scale: bool = True,

    # x axis
    x_margin=0.5,
    shorten_x_axis: bool = True,
):
    """
    Plot individual connectome traces and a population summary trace,
    with publication-style formatting and optional SVG export.
    """

    if not indiv_dict:
        raise ValueError("indiv_dict is empty.")

    metric = str(metric)
    if metric not in {"SynapseCount", "percentage_SynapseCount"}:
        raise ValueError("metric must be 'SynapseCount' or 'percentage_SynapseCount'.")

    # -----------------------------
    # choose population column
    # -----------------------------
    if metric == "SynapseCount":
        pop_col = "SynapseCount_mean"

        if pop_col not in pop_df.columns:
            raise ValueError(f"pop_df must contain '{pop_col}'.")

        y_pop_raw = pd.to_numeric(pop_df[pop_col], errors="coerce")
        pop_label = "population (SynapseCount_mean)"

    else:
        mean_version = str(mean_version).lower()

        if mean_version not in {"old", "new"}:
            raise ValueError("mean_version must be 'old' or 'new'.")

        if mean_version == "old":
            pop_col = "percentage_SynapseCount_mean"

            if pop_col not in pop_df.columns:
                raise ValueError(f"pop_df must contain '{pop_col}'.")

            y_pop_raw = pd.to_numeric(pop_df[pop_col], errors="coerce")
            pop_label = "population (percentage_SynapseCount_mean)"

        else:
            if "percentage_SynapseCount_mean_NEW" in pop_df.columns:
                pop_col = "percentage_SynapseCount_mean_NEW"
                y_pop_raw = pd.to_numeric(pop_df[pop_col], errors="coerce")
                pop_label = "population (percentage_SynapseCount_mean_NEW)"
            else:
                if "SynapseCount_mean" not in pop_df.columns:
                    raise ValueError(
                        "pop_df must contain 'SynapseCount_mean' to compute NEW version."
                    )

                pop_col = "SynapseCount_mean / max"
                scm = pd.to_numeric(pop_df["SynapseCount_mean"], errors="coerce")
                m = scm.max(skipna=True)
                y_pop_raw = (scm / m) if (np.isfinite(m) and m != 0) else scm * 0
                pop_label = "population (SynapseCount_mean / max)"

    # -----------------------------
    # colors
    # -----------------------------
    if mean_line_color is None:
        mean_line_color = line_color

    if dot_color is None:
        dot_color = mean_line_color

    individual_color_mode = str(individual_color_mode).lower()

    if individual_color_mode not in {"same", "different"}:
        raise ValueError("individual_color_mode must be 'same' or 'different'.")

    # -----------------------------
    # x-axis setup
    # -----------------------------
    lengths = [len(df) for df in indiv_dict.values()]
    lengths.append(len(pop_df))
    L = int(min(lengths))
    x = np.arange(L)

    if "glomerulus" in pop_df.columns:
        xtick_labels = pop_df["glomerulus"].astype(str).tolist()[:L]
    elif "Glomerulus" in pop_df.columns:
        xtick_labels = pop_df["Glomerulus"].astype(str).tolist()[:L]
    else:
        xtick_labels = [str(i) for i in range(L)]

    all_y = []

    # -----------------------------
    # plot
    # -----------------------------
    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=figsize)

        if individual_color_mode == "different":
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, (name, df) in enumerate(indiv_dict.items()):
            if metric not in df.columns:
                raise ValueError(f"'{metric}' not found in individual df '{name}'.")

            y_ind = pd.to_numeric(df[metric], errors="coerce").to_numpy()[:L]
            all_y.extend(y_ind[np.isfinite(y_ind)])

            color = (
                individual_color
                if individual_color_mode == "same"
                else color_cycle[i % len(color_cycle)]
            )

            ax.plot(
                x,
                y_ind,
                linewidth=individual_linewidth,
                alpha=individual_alpha,
                color=color,
                zorder=1,
            )

        if show_legend:
            legend_color = individual_color if individual_color_mode == "same" else "gray"

            ax.plot(
                [],
                [],
                linewidth=individual_linewidth,
                color=legend_color,
                alpha=individual_alpha,
                label=f"individuals ({metric})",
            )

        y_pop = y_pop_raw.to_numpy()[:L]
        all_y.extend(y_pop[np.isfinite(y_pop)])

        ax.plot(
            x,
            y_pop,
            linewidth=mean_linewidth,
            color=mean_line_color,
            marker="o",
            markersize=dot_size,
            markerfacecolor=dot_color,
            markeredgecolor=dot_color,
            label=(pop_label if show_legend else None),
            zorder=3,
        )

        # -----------------------------
        # x ticks
        # -----------------------------
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels)

        # -----------------------------
        # y-axis range
        # -----------------------------
        if len(all_y) == 0:
            ymax_use = 1
        else:
            ymax_data = np.nanmax(all_y)

            if ymax_mode == "exact":
                ymax_use = ymax_data
            elif ymax_mode == "padded":
                ymax_use = ymax_data * (1 + y_max_pad_frac)
            else:
                raise ValueError("ymax_mode must be 'exact' or 'padded'.")

            if ymax_use == 0:
                ymax_use = 1

        ax.set_ylim(0, ymax_use)

        # y ticks: always 0, middle, max
        y_ticks = np.linspace(0, ymax_use, 3)
        ax.set_yticks(y_ticks)

        # automatic y-axis scaling for display only
        if auto_y_scale and ymax_use != 0:
            y_scale_exp = int(np.floor(np.log10(abs(ymax_use))))

            # do not scale ordinary values like 0.1–999
            if -1 <= y_scale_exp <= 2:
                y_scale_exp = 0
        else:
            y_scale_exp = 0

        y_scale = 10 ** y_scale_exp

        ax.set_yticklabels(
            [f"{v / y_scale:.{y_tick_decimals}f}" for v in y_ticks]
        )

        if y_scale_exp != 0:
            ax.text(
                -0.03,
                1.02,
                rf"$\times 10^{{{y_scale_exp}}}$",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=ytick_fontsize,
            )

        # -----------------------------
        # tick styling
        # -----------------------------
        ax.tick_params(
            axis="x",
            labelsize=xtick_fontsize,
            direction="out",
            width=tick_width,
            length=tick_length,
        )

        ax.tick_params(
            axis="y",
            labelsize=ytick_fontsize,
            direction="out",
            width=tick_width,
            length=tick_length,
        )

        # -----------------------------
        # labels / title
        # -----------------------------
        ax.set_xlabel(
            xlabel if xlabel is not None else "Glomerulus",
            fontsize=xlabel_fontsize,
        )

        ax.set_ylabel(
            ylabel if ylabel is not None else metric.replace("_", " "),
            fontsize=ylabel_fontsize,
            rotation=ylabel_rotation,
            labelpad=ylabel_labelpad,
        )

        ax.set_title(
            title if title is not None else f"Individuals vs Population (n={len(indiv_dict)})",
            fontsize=title_fontsize,
            pad=20,
        )

        # -----------------------------
        # axis style
        # -----------------------------
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_linewidth(axis_linewidth)
        ax.spines["bottom"].set_linewidth(axis_linewidth)

        ax.spines["left"].set_position(("outward", axis_outward))
        ax.spines["bottom"].set_position(("outward", axis_outward))

        ax.spines["left"].set_bounds(0, ymax_use)

        if shorten_x_axis:
            ax.spines["bottom"].set_bounds(x[0], x[-1])

        ax.set_xlim(x[0] - x_margin, x[-1] + x_margin)

        # -----------------------------
        # legend
        # -----------------------------
        if show_legend:
            ax.legend(
                frameon=False,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox,
                ncol=legend_ncol,
                borderaxespad=0.0,
                fontsize=legend_fontsize,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.92])
        else:
            fig.tight_layout()

        # -----------------------------
        # save
        # -----------------------------
        if save_svg is not None:
            save_svg = Path(save_svg)
            save_svg.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_svg,
                format="svg",
                bbox_inches="tight",
                transparent=True,
                metadata={"Creator": "plot_individual_vs_population_connectome"},
            )

        plt.show()



















def plot_line_svg(
    df: pd.DataFrame,
    y_col: str,
    *,
    x_col: str = "glomerulus",
    filename: str | Path = "line.svg",
    line_color: str | None = None,
    marker: str = "o",
    figsize=(6, 4),
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    show_legend: bool = True,
) -> str:
    """
    Plot one column against an x-axis column and save the plot as an SVG.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot.

    y_col : str
        Column to plot on the y-axis.

    x_col : str
        Column to use as the x-axis. If missing or non-numeric, row index is used.

    filename : str or Path
        Output SVG path.

    line_color : str or None
        Line color. If None, uses black.

    marker : str
        Marker style.

    figsize : tuple
        Figure size.

    xlabel, ylabel, title : str or None
        Custom labels/title. Defaults are generated if None.

    show_legend : bool
        Whether to show legend.

    Returns
    -------
    str
        Saved SVG path.
    """
    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' not found in DataFrame.")

    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)

    if x_col in df.columns:
        x_num = pd.to_numeric(df[x_col], errors="coerce")
        x = (
            x_num.to_numpy(dtype=float)
            if x_num.notna().any()
            else np.arange(len(df), dtype=float)
        )
        x_name = x_col
    else:
        x = np.arange(len(df), dtype=float)
        x_name = "index"

    used = np.isfinite(x) & np.isfinite(y)

    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            x[used],
            y[used],
            marker=marker,
            linestyle="-",
            linewidth=2,
            color=line_color or "black",
            label=y_col,
        )

        ax.set_xlabel(xlabel if xlabel is not None else x_name)
        ax.set_ylabel(ylabel if ylabel is not None else y_col.replace("_", " "))
        ax.set_title(title if title is not None else f"{y_col} vs {x_name}")

        if show_legend:
            ax.legend(frameon=False)

        fig.tight_layout()

        out = Path(filename)
        out.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            out,
            format="svg",
            bbox_inches="tight",
            transparent=True,
            metadata={"Creator": "plot_line_svg"},
        )

        plt.close(fig)

    return str(out)





def plot_individual_vs_population_ihc_with_stats(
    indiv_dict,
    pop_df,
    metric="mean",
    channel="green",
    line_color=None,
    *,
    column=None,
    stat_comparisons=((4, 0), (4, 3), (0, 3)),
    stats_df=None,
    alternative="two-sided",
    method="exact",
    show_stats=True,
    stat_line_height=0.08,
    stat_text_offset=0.015,
    stat_fontsize=12,
    column_label_for_stats=None,
    figsize=(7, 5),
    show_legend: bool = True,
    save_svg: str | Path | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    individual_color_mode="same",
    individual_color="gray",
    individual_alpha=0.45,
    individual_linewidth=1,
):
    """
    Plot individual IHC traces + population average, with optional Wilcoxon
    significance bars/stars between selected x-index positions.
    """

    if not indiv_dict:
        raise ValueError("indiv_dict is empty.")

    signal = column if column is not None else f"{metric}_{channel}"
    signal_avg = f"{signal}_avg"

    if line_color is None:
        if "green" in signal:
            line_color = "green"
        elif "red" in signal:
            line_color = "red"
        else:
            line_color = "tab:blue"

    if signal_avg not in pop_df.columns:
        raise ValueError(f"pop_df must contain '{signal_avg}'.")

    individual_color_mode = str(individual_color_mode).lower()
    if individual_color_mode not in {"same", "different"}:
        raise ValueError("individual_color_mode must be 'same' or 'different'.")

    lengths = [len(df) for df in indiv_dict.values()]
    lengths.append(len(pop_df))
    L = min(lengths)
    x = np.arange(L)

    # ---------------- get stats stars ----------------
    stat_rows = []

    if show_stats:
        for idx1, idx2 in stat_comparisons:
            if stats_df is not None:
                stats_col = column_label_for_stats if column_label_for_stats is not None else signal

                match = stats_df[
                    (stats_df["column"] == stats_col)
                    & (stats_df["idx1"] == idx1)
                    & (stats_df["idx2"] == idx2)
                ]

                if match.empty:
                    match = stats_df[
                        (stats_df["column"] == stats_col)
                        & (stats_df["idx1"] == idx2)
                        & (stats_df["idx2"] == idx1)
                    ]

                if match.empty:
                    star = "NA"
                    p_value = np.nan
                else:
                    star_col = "star" if "star" in match.columns else "p_value_summary"
                    star = match.iloc[0][star_col]
                    p_value = match.iloc[0]["p_value"] if "p_value" in match.columns else np.nan

            else:
                res = paired_wilcoxon_from_dfs(
                    indiv_dict,
                    column=signal,
                    idx1=idx1,
                    idx2=idx2,
                    label1=f"idx{idx1}",
                    label2=f"idx{idx2}",
                    alternative=alternative,
                    method=method,
                )
                star = res["p_value_summary"]
                p_value = res["p_value"]

            stat_rows.append({
                "idx1": idx1,
                "idx2": idx2,
                "star": star,
                "p_value": p_value,
            })

    # ---------------- plot ----------------
    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=figsize)  # ✅ UPDATED

        if individual_color_mode == "different":
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, (name, df) in enumerate(indiv_dict.items()):
            if signal not in df.columns:
                raise ValueError(f"'{signal}' not found in '{name}'.")

            y = pd.to_numeric(df[signal], errors="coerce").to_numpy()[:L]

            color = (
                individual_color
                if individual_color_mode == "same"
                else color_cycle[i % len(color_cycle)]
            )

            ax.plot(
                x,
                y,
                linewidth=individual_linewidth,
                alpha=individual_alpha,
                color=color,
            )

        if show_legend:
            legend_color = individual_color if individual_color_mode == "same" else "gray"
            ax.plot(
                [],
                [],
                linewidth=individual_linewidth,
                color=legend_color,
                alpha=individual_alpha,
                label=f"individuals ({signal})",
            )

        y_pop = pd.to_numeric(pop_df[signal_avg], errors="coerce").to_numpy()[:L]

        ax.plot(
            x,
            y_pop,
            linewidth=4,
            color=line_color,
            marker="o",
            label=(f"population ({signal_avg})" if show_legend else None),
        )

        if "glomerulus" in pop_df.columns:
            xtick_labels = pop_df["glomerulus"].astype(str).tolist()[:L]
        else:
            xtick_labels = [str(i) for i in range(L)]

        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels)

        ax.set_xlabel(xlabel if xlabel is not None else "Glomerulus")
        ax.set_ylabel(ylabel if ylabel is not None else signal.replace("_", " "))
        ax.set_title(
            title if title is not None else f"Individuals vs Population (n={len(indiv_dict)})",
            pad=25,
        )

        # ---------------- significance bars ----------------
        if show_stats and stat_rows:
            all_y = []

            for df in indiv_dict.values():
                if signal in df.columns:
                    vals = pd.to_numeric(df[signal], errors="coerce").to_numpy()[:L]
                    all_y.extend(vals[np.isfinite(vals)])

            pop_vals = y_pop[np.isfinite(y_pop)]
            all_y.extend(pop_vals)

            y_min = np.nanmin(all_y)
            y_max = np.nanmax(all_y)
            y_range = y_max - y_min if y_max > y_min else 1

            base_y = y_max + stat_line_height * y_range

            for k, row in enumerate(stat_rows):
                idx1 = row["idx1"]
                idx2 = row["idx2"]
                star = row["star"]

                x1, x2 = idx1, idx2
                if x1 > x2:
                    x1, x2 = x2, x1

                y = base_y + k * stat_line_height * y_range
                h = 0.025 * y_range

                ax.plot(
                    [x1, x1, x2, x2],
                    [y, y + h, y + h, y],
                    color="black",
                    linewidth=1,
                )

                ax.text(
                    (x1 + x2) / 2,
                    y + h + stat_text_offset * y_range,
                    star,
                    ha="center",
                    va="bottom",
                    fontsize=stat_fontsize,
                )

            ax.set_ylim(
                y_min,
                base_y + len(stat_rows) * stat_line_height * y_range + 0.12 * y_range,
            )

        if show_legend:
            ax.legend(
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.12),
                ncol=2,
                columnspacing=1.5,
                handletextpad=0.5,
                borderaxespad=0.0,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.86])
        else:
            fig.tight_layout()

        if save_svg is not None:
            save_svg = Path(save_svg)
            save_svg.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_svg,
                format="svg",
                bbox_inches="tight",
                transparent=True,
                metadata={"Creator": "plot_individual_vs_population_ihc_with_stats"},
            )

        plt.show()











#-------------------------PART FOUR-------------------------

def cos_model(x, A, T, phi, B):
    """
    Cosine model.

    y = A * cos(2π/T * (x - phi)) + B
    """
    return A * np.cos(2 * np.pi / T * (x - phi)) + B


def compute_fit_metrics(y_obs, y_pred):
    """
    Compute R² and RMSE, ignoring NaN values in y_obs.
    """
    m = ~np.isnan(y_obs)
    y_o, y_p = y_obs[m], y_pred[m]

    if len(y_o) == 0:
        return np.nan, np.nan

    ss_res = np.sum((y_o - y_p) ** 2)
    ss_tot = np.sum((y_o - np.mean(y_o)) ** 2)

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(ss_res / len(y_o))

    return r2, rmse


def prepare_xy(df, y_col, x_col="glomerulus", ignore_idx=None):
    """
    Extract x and y arrays from a DataFrame.

    Optionally mask one or more y indices as NaN. The original unmasked y values
    are also returned as y_raw.

    Returns
    -------
    x : np.ndarray
    y : np.ndarray
        y values with ignored indices set to NaN.
    y_raw : np.ndarray
        Original y values before masking.
    """
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
    y_raw = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)

    y = y_raw.copy()

    if ignore_idx is not None:
        idxs = [int(ignore_idx)] if np.isscalar(ignore_idx) else list(map(int, ignore_idx))
        for i in idxs:
            if 0 <= i < len(y):
                y[i] = np.nan

    return x, y, y_raw


def fit_cosine(
    df,
    y_col: str,
    x_col: str = "glomerulus",
    *,
    ignore_idx=None,
    T_fixed: float = 9.0,
    estimate_T: bool = False,
    p0=None,
    bounds=None,
    maxfev: int = 10000,
):
    """
    Fit a cosine model to one DataFrame column.
    # key: the computation of making a cosine curve is made by me, 
    # but the process of computing the best-fit parameters (that will be input into the cosine curve making function) uses a "built-in" function (curve_fit).

    Model
    -----
    y = A * cos(2π/T * (x - phi)) + B

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing x_col and y_col.

    y_col : str
        Column to fit.

    x_col : str
        X-axis column.

    ignore_idx : int, list[int], or None
        Row index/indices to mask as NaN before fitting.

    T_fixed : float
        Fixed cosine period if estimate_T=False.

    estimate_T : bool
        If False, fit A, phi, and B while keeping T fixed.
        If True, fit A, T, phi, and B.

    p0 : list or None
        Optional initial parameter guesses.

    bounds : tuple or None
        Optional bounds for curve_fit when estimate_T=True.

    maxfev : int
        Maximum function evaluations for curve_fit.

    Returns
    -------
    dict
        Contains fitted parameters, predictions, original/masked data, fit mask,
        R², and RMSE.
    """
    x, y, y_raw = prepare_xy(
        df,
        y_col=y_col,
        x_col=x_col,
        ignore_idx=ignore_idx,
    )

    m = ~np.isnan(y)

    if m.sum() < 3:
        raise ValueError("Need at least 3 non-NaN points to fit.")

    x_used, y_used = x[m], y[m]

    A0 = max((np.nanmax(y_used) - np.nanmin(y_used)) / 2.0, 1e-6)
    B0 = np.nanmean(y_used)
    phi0 = 0.0

    if estimate_T:
        T0 = T_fixed if T_fixed is not None else (x_used.max() - x_used.min() + 1)

        def cos_free(x, A, T, phi, B):
            return cos_model(x, A, T, phi, B)

        if p0 is None:
            p0 = [A0, T0, phi0, B0]

        if bounds is None:
            bounds = ([0, 1e-6, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])

        popt, pcov = curve_fit(
            cos_free,
            x_used,
            y_used,
            p0=p0,
            bounds=bounds,
            maxfev=maxfev,
        )

        A, T, phi, B = popt
        y_pred = cos_model(x, A, T, phi, B)

    else:
        def cos_Tfixed(x, A, phi, B):
            return cos_model(x, A, T_fixed, phi, B)

        if p0 is None:
            p0 = [A0, phi0, B0]

        popt, pcov = curve_fit(
            cos_Tfixed,
            x_used,
            y_used,
            p0=p0,
            maxfev=maxfev,
        )

        A, phi, B = popt
        T = T_fixed
        y_pred = cos_Tfixed(x, A, phi, B)

    r2, rmse = compute_fit_metrics(y, y_pred)

    return {
        "params": {
            "A": float(A),
            "T": float(T),
            "phi": float(phi),
            "B": float(B),
        },
        "popt": popt,
        "pcov": pcov,
        "x": x,
        "y_obs": y,
        "y_raw": y_raw,
        "y_pred": y_pred,
        "mask_used": m,
        "r2": float(r2),
        "rmse": float(rmse),
    }


def plot_cosine_fit(
    x,
    y_obs,
    y_pred,
    title=None,
    y_title="Value",
    ignored_index=None,
    y_raw=None,
):
    """
    Plot observed data and cosine fit.

    If ignored_index and y_raw are provided, ignored points are shown as x markers.
    """
    plt.figure(figsize=(6, 4))

    m = ~np.isnan(y_obs)

    plt.plot(
        x[m],
        y_obs[m],
        marker="o",
        linestyle="-",
        label="Data used",
    )

    if ignored_index is not None and y_raw is not None:
        idxs = [ignored_index] if np.isscalar(ignored_index) else ignored_index

        for i in idxs:
            if 0 <= i < len(y_raw) and not np.isnan(y_raw[i]):
                plt.plot(
                    x[i],
                    y_raw[i],
                    marker="x",
                    linestyle="None",
                    markersize=9,
                    label="Ignored point",
                )

    plt.plot(x, y_pred, linestyle="--", label="Cosine fit")

    plt.title(title or "Cosine fit")
    plt.xlabel("Glomerulus")
    plt.ylabel(y_title)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_cosine_fit_svg(
    x,
    y_obs,
    y_pred,
    filename,
    *,
    show_full_fit: bool = True,
    xlabel: str = "Glomerulus",
    ylabel: str = "Value",
    title: str | None = None,
    show_legend: bool = False,
    data_color: str = "tab:blue",
    data_lw: float = 2.0,
    fit_color: str = "0.5",
    fit_lw: float = 3.0,
    fit_ls: str = "--",
    show_plot: bool = False,
) -> str:
    """
    Save a cosine-fit plot as an SVG.

    The observed data are plotted as a solid line. The cosine fit is plotted as
    a dashed line by default. If show_plot=True, the plot is also displayed in
    the notebook.

    Parameters
    ----------
    x, y_obs, y_pred : array-like
        X values, observed y values, and predicted y values.

    filename : str or Path
        Output SVG path.

    show_full_fit : bool
        If True, plot fit across all x values. If False, plot only positions
        where y_obs is finite.

    xlabel, ylabel, title : str
        Axis labels and optional title.

    show_legend : bool
        Whether to show the legend.

    data_color : str
        Color of observed data curve.

    fit_color : str
        Color of fitted cosine curve.

    show_plot : bool
        If True, display the plot in the notebook after saving.

    Returns
    -------
    str
        Saved SVG path.
    """
    x = np.asarray(x)
    y_obs = np.asarray(y_obs)
    y_pred = np.asarray(y_pred)

    used = np.isfinite(x) & np.isfinite(y_obs)

    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(
            x[used],
            y_obs[used],
            linestyle="-",
            linewidth=data_lw,
            color=data_color,
            label="Data (used)",
        )

        if show_full_fit:
            ax.plot(
                x,
                y_pred,
                linestyle=fit_ls,
                linewidth=fit_lw,
                color=fit_color,
                label="Cosine fit",
            )
        else:
            ax.plot(
                x[used],
                y_pred[used],
                linestyle=fit_ls,
                linewidth=fit_lw,
                color=fit_color,
                label="Cosine fit",
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title:
            ax.set_title(title, pad=12)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.tick_params(direction="out", length=5, width=1)
        ax.grid(True, axis="y", alpha=0.15, linewidth=0.8)

        if show_legend:
            ax.legend(frameon=False)

        fig.tight_layout()

        out = Path(filename)
        out.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            out,
            format="svg",
            bbox_inches="tight",
            transparent=True,
            metadata={"Creator": "plot_cosine_fit_svg"},
        )

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    return str(out)


def plot_cosine_fit_svg_from_result(res: dict, filename: str | Path, **kwargs) -> str:
    """
    Save a cosine-fit SVG from the dictionary returned by fit_cosine().
    """
    return plot_cosine_fit_svg(
        res["x"],
        res["y_obs"],
        res["y_pred"],
        filename,
        **kwargs,
    )








#-------------------------PART FIVE-------------------------

from scipy.stats import wilcoxon, spearmanr


def p_to_stars(p: float) -> str:
    """Convert p-value to Prism-style significance stars."""
    if pd.isna(p):
        return "nan"
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def paired_wilcoxon_from_dfs(
    dfs: dict[str, pd.DataFrame],
    column: str,
    idx1: int,
    idx2: int,
    *,
    label1: str | None = None,
    label2: str | None = None,
    alternative: str = "two-sided",
    method: str = "auto",
    alpha: float = 0.05,
) -> dict:
    """
    Run a paired Wilcoxon signed-rank test comparing two row indices across
    all DataFrames in a dictionary.

    Each DataFrame contributes one paired value:
    - value1 = df[column].iloc[idx1]
    - value2 = df[column].iloc[idx2]

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Dictionary of individual DataFrames.

    column : str
        Column to extract values from.

    idx1, idx2 : int
        Row indices to compare.

    label1, label2 : str or None
        Optional names for the two paired groups.

    alternative : {"two-sided", "greater", "less"}
        Alternative hypothesis passed to scipy.stats.wilcoxon.

    method : {"auto", "exact", "approx"}
        P-value calculation method passed to scipy.stats.wilcoxon.

    alpha : float
        Significance threshold.

    Returns
    -------
    dict
        Prism-like Wilcoxon test summary plus extracted paired data.
    """
    rows = []

    for name, df in dfs.items():
        if not isinstance(df, pd.DataFrame):
            continue
        if column not in df.columns:
            continue
        if len(df) <= max(idx1, idx2):
            continue

        s = pd.to_numeric(df[column], errors="coerce")

        v1 = s.iloc[idx1]
        v2 = s.iloc[idx2]

        if pd.notna(v1) and pd.notna(v2):
            rows.append(
                {
                    "name": name,
                    "idx1": idx1,
                    "idx2": idx2,
                    "value1": float(v1),
                    "value2": float(v2),
                    "difference": float(v1 - v2),
                }
            )

    paired_df = pd.DataFrame(rows)

    if paired_df.empty:
        raise ValueError("No valid paired values found.")

    x = paired_df["value1"].to_numpy()
    y = paired_df["value2"].to_numpy()
    diff = paired_df["difference"].to_numpy()

    nonzero_diff = diff[diff != 0]
    n_pairs = len(paired_df)
    n_ties = int(np.sum(diff == 0))

    if len(nonzero_diff) == 0:
        stat = np.nan
        p = np.nan
    else:
        result = wilcoxon(
            x,
            y,
            alternative=alternative,
            method=method,
            zero_method="wilcox",
        )
        stat = float(result.statistic)
        p = float(result.pvalue)

    # Rank sums, Prism-style-ish
    abs_diff = np.abs(nonzero_diff)
    ranks = pd.Series(abs_diff).rank(method="average").to_numpy()

    sum_positive_ranks = float(ranks[nonzero_diff > 0].sum())
    sum_negative_ranks = -float(ranks[nonzero_diff < 0].sum())
    signed_rank_sum = sum_positive_ranks + sum_negative_ranks

    median_difference = float(np.median(diff))

    # Pairing effectiveness: Spearman correlation between paired columns
    if n_pairs >= 3:
        rs, rs_p = spearmanr(x, y, alternative="greater")
        rs = float(rs)
        rs_p = float(rs_p)
    else:
        rs, rs_p = np.nan, np.nan

    return {
        "comparison": f"{label1 or f'idx{idx1}'} vs {label2 or f'idx{idx2}'}",
        "column": column,
        "idx1": idx1,
        "idx2": idx2,
        "label1": label1 or f"idx{idx1}",
        "label2": label2 or f"idx{idx2}",
        "alternative": alternative,
        "method": method,
        "statistic": stat,
        "p_value": p,
        "p_value_summary": p_to_stars(p),
        "significantly_different": bool(p < alpha) if pd.notna(p) else False,
        "alpha": alpha,
        "sum_positive_ranks": sum_positive_ranks,
        "sum_negative_ranks": sum_negative_ranks,
        "sum_signed_ranks": signed_rank_sum,
        "n_pairs": n_pairs,
        "n_ties_ignored": n_ties,
        "median_difference": median_difference,
        "spearman_rs": rs,
        "spearman_p_one_tailed": rs_p,
        "paired_values": paired_df,
    }





def plot_paired_index_comparison_svg(
    dfs: dict[str, pd.DataFrame],
    column: str,
    idx1: int,
    idx2: int,
    *,
    label1: str | None = None,
    label2: str | None = None,
    title: str | None = None,
    xlabel: str = "Glomerular Distance to Axonal Glomerulus",
    ylabel: str = "Value",
    stars: str | None = None,

    # --- colors ---
    line_color: str = "black",
    point_color: str = "black",
    mean_color: str = "black",

    # --- layout ---
    figsize=(4, 6),
    column_spacing: float = 1.0,
    side_margin: float = 0.45,
    jitter: float = 0.0,

    # --- y axis (NEW improved control) ---
    ymin: float | None = None,
    ymax: float | None = None,
    ylim: tuple[float, float] | None = None,
    ytick_step: float | None = None,

    # --- sizes ---
    point_size: float = 90,
    line_width: float = 2,
    mean_line_width: float = 3,
    mean_bar_width: float = 0.18,
    show_sem: bool = False,
    sem_line_width: float = 3,
    sem_cap_width: float = 0.08,
    mean_x_offset: float = 0.12,

    # --- significance bar ---
    bracket_y: float | None = None,
    bracket_h: float = 0.08,
    bracket_linewidth: float = 3,
    stars_y_offset: float = 0.02,

    # --- axis styling ---
    axis_linewidth: float = 3,
    tick_width: float = 3,
    tick_length: float = 8,
    y_ticks=None,

    # --- fonts ---
    title_fontsize: float = 28,
    xlabel_fontsize: float = 24,
    ylabel_fontsize: float = 24,
    xtick_fontsize: float = 22,
    ytick_fontsize: float = 22,
    stars_fontsize: float = 30,

    title_weight: str = "bold",
    label_weight: str = "bold",
    tick_weight: str = "bold",

    # --- behavior ---
    alpha: float = 1.0,
    show_mean: bool = False,
    show_plot: bool = True,
    save_svg: str | Path | None = None,
) -> pd.DataFrame:

    rows = []

    # --- extract paired values ---
    for name, df in dfs.items():
        if column not in df.columns:
            continue
        if len(df) <= max(idx1, idx2):
            continue

        s = pd.to_numeric(df[column], errors="coerce")
        v1 = s.iloc[idx1]
        v2 = s.iloc[idx2]

        if pd.notna(v1) and pd.notna(v2):
            rows.append({"name": name, "v1": float(v1), "v2": float(v2)})

    paired_df = pd.DataFrame(rows)

    if paired_df.empty:
        raise ValueError("No valid paired values found.")

    label1 = label1 or str(idx1)
    label2 = label2 or str(idx2)

    x1 = 0
    x2 = column_spacing
    x_positions = [x1, x2]

    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=figsize)

        jitter_vals = np.random.uniform(-jitter, jitter, size=len(paired_df))

        # --- lines ---
        for i, row in paired_df.iterrows():
            ax.plot(
                [x1 + jitter_vals[i], x2 + jitter_vals[i]],
                [row["v1"], row["v2"]],
                color=line_color,
                alpha=alpha,
                linewidth=line_width,
                zorder=1,
            )

        # --- points ---
        ax.scatter(x1 + jitter_vals, paired_df["v1"], color=point_color, s=point_size, zorder=2)
        ax.scatter(x2 + jitter_vals, paired_df["v2"], color=point_color, s=point_size, zorder=2)

        # --- mean bars ---
        '''
        if show_mean:
            m1 = paired_df["v1"].mean()
            m2 = paired_df["v2"].mean()

            ax.plot([x1 - mean_bar_width/2, x1 + mean_bar_width/2], [m1, m1],
                    color=mean_color, linewidth=mean_line_width)

            ax.plot([x2 - mean_bar_width/2, x2 + mean_bar_width/2], [m2, m2],
                    color=mean_color, linewidth=mean_line_width)
        '''    

        # --- mean bars + SEM ---
        m1 = paired_df["v1"].mean()
        m2 = paired_df["v2"].mean()

        sem1 = paired_df["v1"].sem()
        sem2 = paired_df["v2"].sem()

        if show_sem:
            ax.errorbar(
                x1 + mean_x_offset,
                m1,
                yerr=sem1,
                fmt="none",
                ecolor=mean_color,
                elinewidth=sem_line_width,
                capsize=sem_cap_width * 100,
                capthick=sem_line_width,
                zorder=4,
            )

            ax.errorbar(
                x2 + mean_x_offset,
                m2,
                yerr=sem2,
                fmt="none",
                ecolor=mean_color,
                elinewidth=sem_line_width,
                capsize=sem_cap_width * 100,
                capthick=sem_line_width,
                zorder=4,
            )

        if show_mean:
            mean_x1 = x1 + mean_x_offset
            mean_x2 = x2 + mean_x_offset
            ax.plot(
                [mean_x1 - mean_bar_width/2, mean_x1 + mean_bar_width/2],
                [m1, m1],
                color=mean_color,
                linewidth=mean_line_width,
                zorder=5,
            )

            ax.plot(
                [mean_x2 - mean_bar_width/2, mean_x2 + mean_bar_width/2],
                [m2, m2],
                color=mean_color,
                linewidth=mean_line_width,
                zorder=5,
            )



        # --- significance ---
        if stars is not None:
            if bracket_y is None:
                y_max = max(paired_df["v1"].max(), paired_df["v2"].max())
                bracket_y_use = y_max + 0.12
            else:
                bracket_y_use = bracket_y

            ax.plot(
                [x1, x1, x2, x2],
                [bracket_y_use, bracket_y_use + bracket_h,
                 bracket_y_use + bracket_h, bracket_y_use],
                color="black",
                linewidth=bracket_linewidth,
                clip_on=False,
            )

            ax.text(
                (x1 + x2) / 2,
                bracket_y_use + bracket_h + stars_y_offset,
                stars,
                ha="center",
                va="bottom",
                fontsize=stars_fontsize,
                weight="bold",
            )

        # --- labels ---
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, weight=label_weight)
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, weight=label_weight)

        if title:
            ax.set_title(title, fontsize=title_fontsize, weight=title_weight)

        # --- ticks ---
        ax.set_xticks(x_positions)
        ax.set_xticklabels([label1, label2], fontsize=xtick_fontsize, weight=tick_weight)

        for tick in ax.get_yticklabels():
            tick.set_fontsize(ytick_fontsize)
            tick.set_fontweight(tick_weight)

        ax.tick_params(width=tick_width, length=tick_length)

        # --- x spacing ---
        ax.set_xlim(x1 - side_margin, x2 + side_margin)

        # ✅ --- NEW y range logic ---
        if ylim is not None:
            ax.set_ylim(ylim)
        elif ymin is not None or ymax is not None:
            current = ax.get_ylim()
            ax.set_ylim(
                ymin if ymin is not None else current[0],
                ymax if ymax is not None else current[1],
            )

        # ticks
        if y_ticks is not None:
            ax.set_yticks(y_ticks)
        else:
            low, high = ax.get_ylim()
            ax.set_yticks(np.arange(low, high + ytick_step, ytick_step))

        # --- style ---
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(axis_linewidth)
        ax.spines["bottom"].set_linewidth(axis_linewidth)

        fig.subplots_adjust(left=0.28, right=0.95, bottom=0.25, top=0.82)

        # --- save ---
        if save_svg is not None:
            out = Path(save_svg)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, format="svg", bbox_inches="tight", transparent=True)

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    return paired_df











#-------------------------PART SIX-------------------------

def load_csv_folder_as_dict(
    data_dir: str | Path,
    *,
    pattern: str = "*.csv",
    sort_files: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Load all CSV files in a folder into a dictionary.

    Keys are file stems.
    Values are pandas DataFrames.
    """

    data_dir = Path(data_dir)

    csv_files = list(data_dir.glob(pattern))

    if sort_files:
        csv_files = sorted(csv_files)

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No files matching {pattern!r} found in: {data_dir}")

    return {
        f.stem: pd.read_csv(f)
        for f in csv_files
    }




def _extract_numeric_id(key) -> int | None:
    s = str(key)

    for tok in s.split("_"):
        if tok.isdigit() and len(tok) >= 7:
            return int(tok)

    m = re.search(r"(\d{7,})", s)
    return int(m.group(1)) if m else None


def _sum_synapse_counts(
    d: Dict,
    syn_col: str = "SynapseCount",
) -> dict[int, int]:
    """
    For each DataFrame in dict d, sum the SynapseCount column as an integer
    and map it to the parsed neuron id.

    If multiple entries map to the same id, sums are accumulated.
    """

    totals: dict[int, int] = {}

    for key, df in d.items():
        if not isinstance(df, pd.DataFrame) or syn_col not in df.columns:
            continue

        nid = _extract_numeric_id(key)

        if nid is None:
            continue

        s_num = pd.to_numeric(df[syn_col], errors="coerce").dropna().astype("Int64")
        total = int(s_num.sum()) if len(s_num) else 0

        totals[nid] = totals.get(nid, 0) + total

    return totals


def build_glutamate_upstream_summary(
    delta7_naming: pd.DataFrame,
    dict_d7_as_upstream: Dict[str, pd.DataFrame],
    dict_lpsp_as_upstream: Dict[str, pd.DataFrame],
    dict_p19_as_upstream: Dict[str, pd.DataFrame],
    dict_p68p9_as_upstream: Dict[str, pd.DataFrame],
    *,
    syn_col: str = "SynapseCount",
) -> pd.DataFrame:
    """
    Create a summary DataFrame with columns:
    id, d7, lpsp, p19, p68p9.

    Each value is the integer sum of `syn_col` for that neuron's DataFrame
    in the corresponding dictionary.
    """

    ids = pd.to_numeric(delta7_naming["id"], errors="coerce").dropna().astype(int)
    out = pd.DataFrame({"id": ids})

    map_d7 = _sum_synapse_counts(dict_d7_as_upstream, syn_col=syn_col)
    map_lpsp = _sum_synapse_counts(dict_lpsp_as_upstream, syn_col=syn_col)
    map_p19 = _sum_synapse_counts(dict_p19_as_upstream, syn_col=syn_col)
    map_p68p9 = _sum_synapse_counts(dict_p68p9_as_upstream, syn_col=syn_col)

    out["d7"] = out["id"].map(map_d7).fillna(0).astype(int)
    out["lpsp"] = out["id"].map(map_lpsp).fillna(0).astype(int)
    out["p19"] = out["id"].map(map_p19).fillna(0).astype(int)
    out["p68p9"] = out["id"].map(map_p68p9).fillna(0).astype(int)

    return out.sort_values("id").reset_index(drop=True)




def plot_upstream_mix(
    df: pd.DataFrame,
    *,
    cols: Sequence[str] = ("d7", "lpsp", "p19", "p68p9"),
    scale_to_100: bool = False,
    title: Optional[str] = "Upstream source mix",
    xlabel: Optional[str] = "Source",
    ylabel: Optional[str] = None,
    colors: Optional[Mapping[str, str] | Sequence[str]] = None,
    save: bool = False,
    filename: str | Path = "upstream_mix.svg",
    bar_width: float = 0.6,
    bar_gap: float = 0.0,
    edgecolor: str = "0.2",
    edge_lw: float = 0.8,
    show_labels: bool = True,
    label_fmt: str = "{:.1f}%",
    show_grid: bool = False,
) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    totals = (
        pd.to_numeric(df[list(cols)].stack(), errors="coerce")
        .unstack()
        .sum(axis=0)
    )

    grand_total = float(totals.sum())

    if not np.isfinite(grand_total) or grand_total <= 0:
        fractions = totals * 0.0
    else:
        fractions = totals / grand_total

    values = fractions * (100.0 if scale_to_100 else 1.0)

    summary = pd.DataFrame(
        {
            "source": list(cols),
            "total": totals.values,
            "percentage": values.values,
        }
    )

    if ylabel is None:
        ylabel = "Percentage (%)" if scale_to_100 else "Fraction of total"

    step = 1.0 + float(bar_gap)
    x = np.arange(len(cols)) * step

    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(5.5, 4))

        if colors is None:
            bar_colors = None
        elif isinstance(colors, dict):
            bar_colors = [colors.get(c, None) for c in cols]
        else:
            bar_colors = list(colors)

        used_width = min(bar_width, step * 0.9)

        bars = ax.bar(
            x,
            values.values,
            width=used_width,
            color=bar_colors,
            edgecolor=edgecolor,
            linewidth=edge_lw,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(list(cols))
        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel)

        if title:
            ax.set_title(title, pad=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if scale_to_100:
            ax.set_ylim(0, 100)

        if show_grid:
            ax.grid(True, axis="y", alpha=0.2, linewidth=0.8)

        ax.set_xlim(x[0] - step * 0.6, x[-1] + step * 0.6)

        if show_labels:
            label_vals = values.values if scale_to_100 else (values.values * 100.0)
            ylim = ax.get_ylim()
            y_off = 0.01 * (ylim[1] - ylim[0]) if ylim[1] > ylim[0] else 0.05

            for bar, v in zip(bars, label_vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + y_off,
                    label_fmt.format(v),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    clip_on=False,
                )

        fig.tight_layout()

        if save:
            out = Path(filename)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                out,
                format="svg",
                bbox_inches="tight",
                transparent=True,
                metadata={"Creator": "plot_upstream_mix"},
            )

        plt.show()

    return summary



def plot_upstream_type_totals(
    df: pd.DataFrame,
    *,
    type_col: str = "upstream_type",
    count_col: str = "total_synapse_count",
    priority_types: Sequence[str] = ("Delta7", "LPsP", "P1-9", "P6-8P9"),
    unknown_label: str = "unknown",
    scale_to_100: bool = True,
    colors: Optional[Mapping[str, str] | Sequence[str]] = None,
    default_color: str = "0.7",
    title: Optional[str] = "upstream source mix",
    xlabel: Optional[str] = "upstream type",
    ylabel: Optional[str] = None,
    ymin: float | None = None,
    ymax: float | None = None,
    n_yticks: int = 3,
    figsize: tuple[float, float] = (12, 4),
    bar_width: float = 0.55,
    bar_gap: float = 0.3,
    edgecolor: str = "0.2",
    edge_lw: float = 0.8,
    show_labels: bool = True,
    label_fmt: str = "{:.1f}%",
    label_fontsize: float = 8,
    label_rotation: float = 90,
    label_offset_frac: float = 0.01,
    xtick_rotation: float = 60,
    xtick_ha: str = "right",
    show_grid: bool = False,
    bracket_types: Sequence[str] | None = None,
    bracket_label: str | None = None,
    bracket_y_frac: float = -0.32,
    bracket_h_frac: float = 0.05,
    bracket_linewidth: float = 1.5,
    bracket_fontsize: float = 10,
    save: bool = False,
    filename: str | Path = "upstream_type_totals.svg",
    return_ordered_df: bool = True,
) -> pd.DataFrame | None:
    """
    Plot upstream neuron-type contribution as percentage of total synapse count.

    Ordering:
    1. priority_types appear first in the exact order provided
    2. all remaining types are sorted by count_col from high to low

    Parameters
    ----------
    df : pd.DataFrame
        Must contain type_col and count_col.

    type_col : str
        Column containing upstream neuron type names.

    count_col : str
        Column containing total synapse counts.

    priority_types : sequence of str
        Types to force to the left side of the plot.

    unknown_label : str
        Label used to replace NaN upstream type names.

    scale_to_100 : bool
        If True, plot percentage from 0-100.
        If False, plot fraction from 0-1.

    colors : mapping or sequence or None
        If dict, maps upstream type → color.
        If sequence, uses colors in order.
        If None, uses default_color.

    default_color : str
        Color for bars not listed in colors.

    title, xlabel, ylabel : str or None
        Plot labels.

    figsize : tuple
        Figure size.

    bar_width : float
        Bar width.

    bar_gap : float
        Gap between bars.

    edgecolor, edge_lw : str, float
        Bar edge styling.

    show_labels : bool
        Whether to place percentage labels above bars.

    label_fmt : str
        Format for labels. Usually "{:.1f}%".

    label_fontsize : float
        Label font size.

    label_rotation : float
        Rotation angle for labels above bars.

    label_offset_frac : float
        Label y-offset as fraction of y-axis range.

    xtick_rotation : float
        X tick label rotation.

    xtick_ha : str
        X tick horizontal alignment.

    show_grid : bool
        Whether to show y-axis grid.

    bracket_types : sequence of str or None
        Types to bracket with a horizontal line.

    bracket_label : str or None
        Label for the bracket.

    bracket_y_frac : float
        Y position of the bracket as a fraction of the y-axis range.

    bracket_h_frac : float
        Height of the bracket as a fraction of the y-axis range.

    bracket_linewidth : float
        Width of the bracket line.

    bracket_fontsize : float
        Font size of the bracket label.

    save : bool
        Whether to save SVG.

    filename : str or Path
        Output SVG path.

    return_ordered_df : bool
        If True, return the ordered plotting DataFrame.

    Returns
    -------
    pd.DataFrame or None
        Ordered plotting DataFrame if return_ordered_df=True.
    """

    if type_col not in df.columns:
        raise ValueError(f"Missing type column: {type_col}")

    if count_col not in df.columns:
        raise ValueError(f"Missing count column: {count_col}")

    plot_df = df[[type_col, count_col]].copy()

    plot_df[type_col] = plot_df[type_col].fillna(unknown_label)
    plot_df[count_col] = pd.to_numeric(plot_df[count_col], errors="coerce").fillna(0)

    total = plot_df[count_col].sum()

    if not np.isfinite(total) or total <= 0:
        plot_df["fraction"] = 0.0
    else:
        plot_df["fraction"] = plot_df[count_col] / total

    plot_df["percentage"] = plot_df["fraction"] * (100 if scale_to_100 else 1)

    priority_df = plot_df[plot_df[type_col].isin(priority_types)].copy()
    priority_df["__order"] = priority_df[type_col].map(
        {name: i for i, name in enumerate(priority_types)}
    )
    priority_df = priority_df.sort_values("__order").drop(columns="__order")

    rest_df = plot_df[~plot_df[type_col].isin(priority_types)].copy()
    rest_df = rest_df.sort_values(count_col, ascending=False)

    ordered_df = pd.concat([priority_df, rest_df], ignore_index=True)

    if ylabel is None:
        ylabel = "percentage of total synapse count" if scale_to_100 else "fraction of total synapse count"

    step = 1.0 + float(bar_gap)
    x = np.arange(len(ordered_df)) * step

    if colors is None:
        bar_colors = [default_color] * len(ordered_df)
    elif isinstance(colors, dict):
        bar_colors = [
            colors.get(t, default_color)
            for t in ordered_df[type_col]
        ]
    else:
        color_list = list(colors)
        bar_colors = (
            color_list * (len(ordered_df) // len(color_list) + 1)
        )[:len(ordered_df)]

    used_width = min(bar_width, step * 0.9)

    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.bar(
            x,
            ordered_df["percentage"],
            width=used_width,
            color=bar_colors,
            edgecolor=edgecolor,
            linewidth=edge_lw,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            ordered_df[type_col],
            rotation=xtick_rotation,
            ha=xtick_ha,
        )

        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel)

        if title:
            ax.set_title(title, pad=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # ---------- y-axis control ----------
        if ymin is None:
            ymin_use = 0
        else:
            ymin_use = ymin

        if ymax is not None:
            ymax_use = ymax
        else:
            _, current_high = ax.get_ylim()
            ymax_use = current_high

        ax.set_ylim(ymin_use, ymax_use)

        # 3 ticks: min, middle, max
        ax.set_yticks(
            np.linspace(ymin_use, ymax_use, n_yticks)
        )

        # shorten y-axis spine
        ax.spines["left"].set_bounds(
            ymin_use,
            ymax_use,
        )

        if show_grid:
            ax.grid(True, axis="y", alpha=0.2, linewidth=0.8)

        ax.set_xlim(x[0] - step * 0.6, x[-1] + step * 0.6)
        # ---------- optional bracket below x-axis ----------
        if bracket_types is not None and bracket_label is not None:
            type_to_x = dict(zip(ordered_df[type_col], x))

            missing = [t for t in bracket_types if t not in type_to_x]
            if missing:
                raise ValueError(f"bracket_types not found in plot: {missing}")

            x_start = type_to_x[bracket_types[0]]
            x_end = type_to_x[bracket_types[-1]]

            y0 = bracket_y_frac
            h = bracket_h_frac

            trans = ax.get_xaxis_transform()

            ax.plot(
                [x_start, x_start, x_end, x_end],
                [y0 + h, y0, y0, y0 + h],
                color="black",
                linewidth=bracket_linewidth,
                transform=trans,
                clip_on=False,
            )

            ax.text(
                (x_start + x_end) / 2,
                y0 - h * 0.9,
                bracket_label,
                ha="center",
                va="top",
                fontsize=bracket_fontsize,
                transform=trans,
                clip_on=False,
            )

        if show_labels:
            label_vals = ordered_df["percentage"].to_numpy()
            ylim = ax.get_ylim()
            y_off = label_offset_frac * (ylim[1] - ylim[0])

            for bar, v in zip(bars, label_vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_off,
                    label_fmt.format(v),
                    ha="center",
                    va="bottom",
                    fontsize=label_fontsize,
                    rotation=label_rotation,
                    clip_on=False,
                )

        fig.tight_layout()

        if save:
            out = Path(filename)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                out,
                format="svg",
                bbox_inches="tight",
                transparent=True,
                metadata={"Creator": "plot_upstream_type_totals"},
            )

        plt.show()

    if return_ordered_df:
        return ordered_df

    return None


def plot_upstream_type_totals_by_nt(
    df: pd.DataFrame,
    neurotransmitter_df: pd.DataFrame,
    *,
    type_col: str = "upstream_type",
    count_col: str = "total_synapse_count",
    nt_type_col: str = "neuron_type",
    nt_col: str = "neurotransmitter",
    nt_order: Sequence[str] = (
        "glutamate",
        "acetylcholine",
        "tyramine",
        "dopamine",
        "octopamine",
    ),
    unsure_label: str = "unsure",
    unknown_type_label: str = "unknown",
    scale_to_100: bool = True,
    colors: Optional[Mapping[str, str] | Sequence[str]] = None,
    nt_colors: Optional[Mapping[str, str]] = None,
    default_color: str = "0.7",
    title: Optional[str] = "upstream source mix by neurotransmitter",
    xlabel: Optional[str] = "upstream type",
    ylabel: Optional[str] = None,
    ymin: float | None = 0,
    ymax: float | None = None,
    n_yticks: int = 3,
    figsize: tuple[float, float] = (14, 5),
    bar_width: float = 0.55,
    bar_gap: float = 0.3,
    edgecolor: str = "0.2",
    edge_lw: float = 0.8,
    show_labels: bool = True,
    label_fmt: str = "{:.1f}%",
    label_fontsize: float = 8,
    label_rotation: float = 0,
    label_offset_frac: float = 0.01,
    xtick_rotation: float = 60,
    xtick_ha: str = "right",
    fontsize_axis_label: float = 12,
    fontsize_ticks: float = 10,
    fontsize_title: float = 14,
    show_grid: bool = False,
    group_gap: float = 0.9,
    show_group_brackets: bool = True,
    bracket_y_frac: float = -0.55,
    bracket_h_frac: float = 0.05,
    bracket_linewidth: float = 1.5,
    bracket_fontsize: float = 10,
    bottom_adjust: float = 0.45,
    save: bool = False,
    filename: str | Path = "upstream_type_totals_by_nt.svg",
    return_ordered_df: bool = True,
) -> pd.DataFrame | None:
    """
    Plot upstream neuron-type contribution grouped by neurotransmitter.

    Grouping rule:
    - Merge df with neurotransmitter_df by neuron type.
    - If neurotransmitter is exactly one of nt_order, assign that group.
    - Otherwise assign unsure_label.
    - Within each neurotransmitter group, sort bars by count_col high to low.
    - Draw one bracket below the x tick labels / x-axis title for each group.

    Notes:
    - Mixed labels such as 'acetylcholine/dopamine' are treated as unsure
      unless that exact string is included in nt_order.
    - Types missing from neurotransmitter_df are treated as unsure.
    """

    if type_col not in df.columns:
        raise ValueError(f"Missing type column: {type_col}")

    if count_col not in df.columns:
        raise ValueError(f"Missing count column: {count_col}")

    if nt_type_col not in neurotransmitter_df.columns:
        raise ValueError(f"Missing neurotransmitter type column: {nt_type_col}")

    if nt_col not in neurotransmitter_df.columns:
        raise ValueError(f"Missing neurotransmitter column: {nt_col}")

    # -----------------------------
    # prepare data
    # -----------------------------
    plot_df = df[[type_col, count_col]].copy()
    plot_df[type_col] = plot_df[type_col].fillna(unknown_type_label)
    plot_df[count_col] = pd.to_numeric(plot_df[count_col], errors="coerce").fillna(0)

    nt_lookup = neurotransmitter_df[[nt_type_col, nt_col]].copy()
    nt_lookup[nt_type_col] = nt_lookup[nt_type_col].astype(str)
    nt_lookup[nt_col] = nt_lookup[nt_col].astype(str)

    plot_df = plot_df.merge(
        nt_lookup,
        left_on=type_col,
        right_on=nt_type_col,
        how="left",
    )

    nt_order = list(nt_order)
    full_group_order = nt_order + [unsure_label]

    plot_df["nt_group"] = np.where(
        plot_df[nt_col].isin(nt_order),
        plot_df[nt_col],
        unsure_label,
    )

    total = plot_df[count_col].sum()

    if not np.isfinite(total) or total <= 0:
        plot_df["fraction"] = 0.0
    else:
        plot_df["fraction"] = plot_df[count_col] / total

    plot_df["percentage"] = plot_df["fraction"] * (100 if scale_to_100 else 1)

    # -----------------------------
    # order data by NT group, then synapse count high -> low
    # -----------------------------
    ordered_parts = []

    for group_name in full_group_order:
        group_df = plot_df[plot_df["nt_group"] == group_name].copy()

        if group_df.empty:
            continue

        group_df = group_df.sort_values(
            count_col,
            ascending=False,
        )

        ordered_parts.append(group_df)

    if not ordered_parts:
        raise ValueError("No rows available after neurotransmitter grouping.")

    ordered_df = pd.concat(ordered_parts, ignore_index=True)

    if ylabel is None:
        ylabel = (
            "percentage of total synapse count"
            if scale_to_100
            else "fraction of total synapse count"
        )

    # -----------------------------
    # x positions with larger gaps between NT groups
    # -----------------------------
    x_positions = []
    group_ranges = {}

    current_x = 0.0

    for group_name in full_group_order:
        group_indices = ordered_df.index[ordered_df["nt_group"] == group_name].tolist()

        if not group_indices:
            continue

        group_xs = []

        for idx in group_indices:
            x_positions.append(current_x)
            group_xs.append(current_x)
            current_x += 1.0 + float(bar_gap)

        group_ranges[group_name] = (min(group_xs), max(group_xs))

        current_x += float(group_gap)

    x = np.array(x_positions, dtype=float)

    used_width = min(bar_width, (1.0 + float(bar_gap)) * 0.9)

    # -----------------------------
    # colors
    # -----------------------------
    if nt_colors is None:
        nt_colors = {}

    if colors is None:
        bar_colors = [
            nt_colors.get(group, default_color)
            for group in ordered_df["nt_group"]
        ]
    elif isinstance(colors, dict):
        bar_colors = [
            colors.get(t, nt_colors.get(group, default_color))
            for t, group in zip(ordered_df[type_col], ordered_df["nt_group"])
        ]
    else:
        color_list = list(colors)
        bar_colors = (
            color_list * (len(ordered_df) // len(color_list) + 1)
        )[:len(ordered_df)]

    # -----------------------------
    # plot
    # -----------------------------
    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.bar(
            x,
            ordered_df["percentage"],
            width=used_width,
            color=bar_colors,
            edgecolor=edgecolor,
            linewidth=edge_lw,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            ordered_df[type_col],
            rotation=xtick_rotation,
            ha=xtick_ha,
            fontsize=fontsize_ticks,
        )

        ax.set_xlabel(xlabel or "", fontsize=fontsize_axis_label, labelpad=40,)
        ax.set_ylabel(ylabel, fontsize=fontsize_axis_label)

        if title:
            ax.set_title(title, fontsize=fontsize_title, pad=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # y-axis control
        ymin_use = 0 if ymin is None else ymin

        if ymax is not None:
            ymax_use = ymax
        else:
            _, current_high = ax.get_ylim()
            ymax_use = current_high

        ax.set_ylim(ymin_use, ymax_use)
        ax.set_yticks(np.linspace(ymin_use, ymax_use, n_yticks))
        ax.spines["left"].set_bounds(ymin_use, ymax_use)

        for tick in ax.get_yticklabels():
            tick.set_fontsize(fontsize_ticks)

        if show_grid:
            ax.grid(True, axis="y", alpha=0.2, linewidth=0.8)

        ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)

        # value labels
        if show_labels:
            label_vals = ordered_df["percentage"].to_numpy()
            ylim = ax.get_ylim()
            y_off = label_offset_frac * (ylim[1] - ylim[0])

            for bar, v in zip(bars, label_vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_off,
                    label_fmt.format(v),
                    ha="center",
                    va="bottom",
                    fontsize=label_fontsize,
                    rotation=label_rotation,
                    clip_on=False,
                )

        # group brackets all on the same level, below x tick labels and x-axis title
        if show_group_brackets:
            trans = ax.get_xaxis_transform()

            for group_name, (x_start, x_end) in group_ranges.items():
                y0 = bracket_y_frac
                h = bracket_h_frac

                ax.plot(
                    [x_start, x_start, x_end, x_end],
                    [y0 + h, y0, y0, y0 + h],
                    color="black",
                    linewidth=bracket_linewidth,
                    transform=trans,
                    clip_on=False,
                )

                ax.text(
                    (x_start + x_end) / 2,
                    y0 - h * 0.9,
                    group_name,
                    ha="center",
                    va="top",
                    fontsize=bracket_fontsize,
                    transform=trans,
                    clip_on=False,
                )

        fig.subplots_adjust(bottom=bottom_adjust)

        if save:
            out = Path(filename)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                out,
                format="svg",
                bbox_inches="tight",
                transparent=True,
                metadata={"Creator": "plot_upstream_type_totals_by_nt"},
            )

        plt.show()

    if return_ordered_df:
        return ordered_df

    return None


def plot_upstream_type_totals_by_nt_new(
    df: pd.DataFrame,
    neurotransmitter_df: pd.DataFrame,
    *,
    type_col: str = "upstream_type",
    count_col: str = "total_synapse_count",
    nt_type_col: str = "neuron_type",
    nt_col: str = "neurotransmitter",
    scale_to_100: bool = True,
    colors: Optional[Mapping[str, str] | Sequence[str]] = None,
    nt_colors: Optional[Mapping[str, str]] = None,
    default_color: str = "0.7",
    title: Optional[str] = "upstream source mix by neurotransmitter",
    xlabel: Optional[str] = "upstream type",
    ylabel: Optional[str] = None,
    ymin: float | None = 0,
    ymax: float | None = None,
    n_yticks: int = 3,
    figsize: tuple[float, float] = (14, 5),
    bar_width: float = 0.55,
    bar_gap: float = 0.3,
    edgecolor: str = "0.2",
    edge_lw: float = 0.8,
    show_labels: bool = True,
    label_fmt: str = "{:.1f}%",
    label_fontsize: float = 8,
    label_rotation: float = 0,
    label_offset_frac: float = 0.01,
    xtick_rotation: float = 60,
    xtick_ha: str = "right",
    fontsize_axis_label: float = 12,
    fontsize_ticks: float = 10,
    fontsize_title: float = 14,
    show_grid: bool = False,
    group_gap: float = 0.9,
    show_group_brackets: bool = True,
    bracket_y_frac: float = -0.55,
    bracket_h_frac: float = 0.05,
    bracket_linewidth: float = 1.5,
    bracket_fontsize: float = 10,
    bottom_adjust: float = 0.45,
    save: bool = False,
    filename: str | Path = "upstream_type_totals_by_nt.svg",
    return_ordered_df: bool = True,
) -> pd.DataFrame | None:

    if type_col not in df.columns:
        raise ValueError(f"Missing type column: {type_col}")

    if count_col not in df.columns:
        raise ValueError(f"Missing count column: {count_col}")

    if nt_type_col not in neurotransmitter_df.columns:
        raise ValueError(f"Missing neurotransmitter type column: {nt_type_col}")

    if nt_col not in neurotransmitter_df.columns:
        raise ValueError(f"Missing neurotransmitter column: {nt_col}")

    # -----------------------------
    # prepare data
    # -----------------------------
    plot_df = df[[type_col, count_col]].copy()
    plot_df[type_col] = plot_df[type_col].fillna("unknown")
    plot_df[count_col] = pd.to_numeric(plot_df[count_col], errors="coerce").fillna(0)

    nt_lookup = neurotransmitter_df[[nt_type_col, nt_col]].copy()
    nt_lookup[nt_type_col] = nt_lookup[nt_type_col].astype(str)
    nt_lookup[nt_col] = nt_lookup[nt_col].astype(str)

    plot_df = plot_df.merge(
        nt_lookup,
        left_on=type_col,
        right_on=nt_type_col,
        how="left",
    )

    # -----------------------------
    # NEW grouping
    # -----------------------------
    plot_df["nt_group"] = np.where(
        plot_df[nt_col] == "glutamate",
        "glutamate",
        np.where(
            plot_df[nt_col] == "acetylcholine",
            "acetylcholine",
            "Other",
        ),
    )

    full_group_order = [
        "glutamate",
        "acetylcholine",
        "Other",
    ]

    total = plot_df[count_col].sum()

    if not np.isfinite(total) or total <= 0:
        plot_df["fraction"] = 0.0
    else:
        plot_df["fraction"] = plot_df[count_col] / total

    plot_df["percentage"] = plot_df["fraction"] * (100 if scale_to_100 else 1)

    # -----------------------------
    # sort within each group
    # -----------------------------
    ordered_parts = []

    for group_name in full_group_order:
        group_df = plot_df[plot_df["nt_group"] == group_name].copy()

        if group_df.empty:
            continue

        group_df = group_df.sort_values(
            count_col,
            ascending=False,
        )

        ordered_parts.append(group_df)

    if not ordered_parts:
        raise ValueError("No rows available after grouping.")

    ordered_df = pd.concat(ordered_parts, ignore_index=True)

    if ylabel is None:
        ylabel = (
            "percentage of total synapse count"
            if scale_to_100
            else "fraction of total synapse count"
        )

    # -----------------------------
    # x positions
    # -----------------------------
    x_positions = []
    group_ranges = {}

    current_x = 0.0

    for group_name in full_group_order:
        group_indices = ordered_df.index[
            ordered_df["nt_group"] == group_name
        ].tolist()

        if not group_indices:
            continue

        xs = []

        for idx in group_indices:
            x_positions.append(current_x)
            xs.append(current_x)
            current_x += 1.0 + float(bar_gap)

        group_ranges[group_name] = (min(xs), max(xs))

        current_x += float(group_gap)

    x = np.asarray(x_positions)

    used_width = min(bar_width, (1.0 + bar_gap) * 0.9)

    # -----------------------------
    # colors
    # -----------------------------
    if nt_colors is None:
        nt_colors = {}

    if colors is None:
        bar_colors = [
            nt_colors.get(group, default_color)
            for group in ordered_df["nt_group"]
        ]
    elif isinstance(colors, dict):
        bar_colors = [
            colors.get(t, nt_colors.get(group, default_color))
            for t, group in zip(
                ordered_df[type_col],
                ordered_df["nt_group"],
            )
        ]
    else:
        color_list = list(colors)
        bar_colors = (
            color_list * (len(ordered_df) // len(color_list) + 1)
        )[:len(ordered_df)]

    # -----------------------------
    # plot
    # -----------------------------
    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.bar(
            x,
            ordered_df["percentage"],
            width=used_width,
            color=bar_colors,
            edgecolor=edgecolor,
            linewidth=edge_lw,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            ordered_df[type_col],
            rotation=xtick_rotation,
            ha=xtick_ha,
            fontsize=fontsize_ticks,
        )

        ax.set_xlabel(xlabel or "", fontsize=fontsize_axis_label, labelpad=40)
        ax.set_ylabel(ylabel, fontsize=fontsize_axis_label)

        if title:
            ax.set_title(title, fontsize=fontsize_title, pad=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ymin_use = 0 if ymin is None else ymin

        if ymax is None:
            _, ymax_use = ax.get_ylim()
        else:
            ymax_use = ymax

        ax.set_ylim(ymin_use, ymax_use)
        ax.set_yticks(np.linspace(ymin_use, ymax_use, n_yticks))
        ax.spines["left"].set_bounds(ymin_use, ymax_use)

        for tick in ax.get_yticklabels():
            tick.set_fontsize(fontsize_ticks)

        if show_grid:
            ax.grid(True, axis="y", alpha=0.2, linewidth=0.8)

        ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)

        if show_labels:
            ylim = ax.get_ylim()
            y_off = label_offset_frac * (ylim[1] - ylim[0])

            for bar, v in zip(bars, ordered_df["percentage"]):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + y_off,
                    label_fmt.format(v),
                    ha="center",
                    va="bottom",
                    fontsize=label_fontsize,
                    rotation=label_rotation,
                    clip_on=False,
                )

        if show_group_brackets:
            trans = ax.get_xaxis_transform()

            for group_name, (x_start, x_end) in group_ranges.items():
                y0 = bracket_y_frac
                h = bracket_h_frac

                ax.plot(
                    [x_start, x_start, x_end, x_end],
                    [y0+h, y0, y0, y0+h],
                    color="black",
                    linewidth=bracket_linewidth,
                    transform=trans,
                    clip_on=False,
                )

                ax.text(
                    (x_start+x_end)/2,
                    y0-h*0.9,
                    group_name,
                    ha="center",
                    va="top",
                    fontsize=bracket_fontsize,
                    transform=trans,
                    clip_on=False,
                )

        fig.subplots_adjust(bottom=bottom_adjust)

        if save:
            out = Path(filename)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                out,
                format="svg",
                bbox_inches="tight",
                transparent=True,
                metadata={"Creator": "plot_upstream_type_totals_by_nt"},
            )

        plt.show()

    if return_ordered_df:
        return ordered_df

    return None