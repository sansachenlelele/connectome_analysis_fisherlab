from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Sequence, Mapping

from kylie_lib import syn_specs
from scipy.optimize import curve_fit


def fetch_d7_upstream_per_cell(
    d7_names_df: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    conn_id: str = "Delta7",
    rois="PB",
    primary_only: bool = True,
) -> dict[str, pd.DataFrame]:

    all_results = {}

    # full list of Delta7 neurons that should always appear
    all_d7_cells = (
        d7_names_df[["id", "instance"]]
        .copy()
        .rename(columns={
            "id": "bodyId_pre",
            "instance": "instance_pre",
        })
    )

    all_d7_cells["bodyId_pre"] = all_d7_cells["bodyId_pre"].astype(int)
    all_d7_cells["type_pre"] = conn_id

    for _, row in d7_names_df.iterrows():
        neuron_id = int(row["id"])
        inst_name = row["instance"]

        result_name = f"delta7_{neuron_id}_{inst_name}_{conn_id}_upstream"

        spec = syn_specs(
            target_neuron=neuron_id,
            scale="type",
            conn_type="pre",
            conn_id=conn_id,
            rois=rois,
            lable_res="neuron",
            top=None,
            primary_only=primary_only,
        )

        df = spec.fetch_syn_conns()

        if isinstance(df, pd.DataFrame) and not df.empty:
            observed_counts = (
                df.groupby(["bodyId_pre"])
                .size()
                .reset_index(name="SynapseCount")
            )

            observed_counts["bodyId_pre"] = observed_counts["bodyId_pre"].astype(int)

        else:
            observed_counts = pd.DataFrame(
                columns=["bodyId_pre", "SynapseCount"]
            )

        summary = (
            all_d7_cells
            .merge(observed_counts, on="bodyId_pre", how="left")
            .fillna({"SynapseCount": 0})
        )

        summary["SynapseCount"] = summary["SynapseCount"].astype(int)

        summary = summary[
            ["bodyId_pre", "instance_pre", "type_pre", "SynapseCount"]
        ]

        all_results[result_name] = summary

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, df_out in all_results.items():
            df_out.to_csv(output_dir / f"{name}.csv", index=False)

    return all_results



def build_synapse_count_table(
    delta_name: pd.DataFrame,
    input_dict: dict[str, pd.DataFrame],
    *,
    id_col: str = "id",
    instance_col: str = "instance",
    pre_id_col: str = "bodyId_pre",
    syn_col: str = "SynapseCount",
) -> pd.DataFrame:
    """
    Build a table summarizing total, same-instance, and other-instance
    Delta7 input for each target Delta7 neuron.
    """

    rows = []

    delta_tmp = delta_name.copy()
    delta_tmp[id_col] = delta_tmp[id_col].astype(int)

    for _, row in delta_tmp.iterrows():
        target_id = int(row[id_col])
        target_instance = row[instance_col]

        # find the one df whose key contains this target id
        matches = [
            name for name in input_dict.keys()
            if str(target_id) in str(name)
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one matching df for target id {target_id}, "
                f"but found {len(matches)}: {matches}"
            )

        target_df = input_dict[matches[0]].copy()

        # all Delta7 ids with the same instance as the target neuron
        same_instance_ids = (
            delta_tmp.loc[
                delta_tmp[instance_col] == target_instance,
                id_col
            ]
            .astype(int)
            .tolist()
        )

        target_df[pre_id_col] = pd.to_numeric(
            target_df[pre_id_col],
            errors="coerce"
        ).astype("Int64")

        target_df[syn_col] = pd.to_numeric(
            target_df[syn_col],
            errors="coerce"
        ).fillna(0)

        same_instance_mask = target_df[pre_id_col].isin(same_instance_ids)

        total_input = target_df[syn_col].sum()
        axonal_input = target_df.loc[same_instance_mask, syn_col].sum()
        dendritic_input = target_df.loc[~same_instance_mask, syn_col].sum()

        rows.append(
            {
                id_col: target_id,
                instance_col: target_instance,
                "total_input": total_input,
                "axonal_input": axonal_input,
                "dendritic_input": dendritic_input,
            }
        )

    synapse_count_table = pd.DataFrame(rows)

    for col in ["total_input", "axonal_input", "dendritic_input"]:
        synapse_count_table[col] = synapse_count_table[col].astype(int)

    return synapse_count_table




def add_cosine_bump_columns(
    summary_df: pd.DataFrame,
    *,
    axonal_col: str = "axonal_input",
    dendritic_col: str = "dendritic_input",
) -> pd.DataFrame:
    """
    Add 9-position cosine bump columns for axonal_input and dendritic_input.

    Axonal cosine:
    - x = 1 to 9
    - peak at x = 5
    - y-min = 0
    - y-max = 1

    Dendritic cosine:
    - x = 1 to 9
    - peaks at x = 1 and x = 9
    - y-min = 0
    - y-max = 1
    """

    out = summary_df.copy()

    bump_cols = [
        "4_L", "3_L", "2_L", "1_L", "0",
        "1_R", "2_R", "3_R", "4_R"
    ]

    x = np.arange(1, 10)

    # axonal bump: peak at x=5
    axon_raw = np.cos(2 * np.pi * (x - 5) / 9)
    axon_bump = (axon_raw - axon_raw.min()) / (axon_raw.max() - axon_raw.min())

    # dendritic bump: peaks at x=1 and x=9
    dendrite_raw = np.cos(2 * np.pi * (x - 1) / 8)
    dendrite_bump = (
        dendrite_raw - dendrite_raw.min()
    ) / (
        dendrite_raw.max() - dendrite_raw.min()
    )

    for suffix, weight in zip(bump_cols, axon_bump):
        out[f"axon_{suffix}"] = out[axonal_col] * weight

    for suffix, weight in zip(bump_cols, dendrite_bump):
        out[f"dendrite_{suffix}"] = out[dendritic_col] * weight

    return out








def plot_axon_dendrite_bump_summary(
    df: pd.DataFrame,
    *,
    rows=None,
    source_col: str = "source",
    title: str | None = "Axon vs dendrite bump summary",
    xlabel: str = "Position",
    ylabel: str = "Summed input",
    axon_colors=None,
    dendrite_colors=None,
    marker: str = "o",
    linewidth: float = 2,
    figsize=(7, 5),
    legend_fontsize: float = 10,
    save_svg: str | Path | None = None,
):
    """
    Plot axon and dendrite bump-summary curves.

    Can plot one row, selected rows, or all rows.
    Each row produces 2 curves: axon and dendrite.
    """

    suffixes = [
        "4_L", "3_L", "2_L", "1_L", "0",
        "1_R", "2_R", "3_R", "4_R",
    ]

    axon_cols = [f"axon_{s}" for s in suffixes]
    dendrite_cols = [f"dendrite_{s}" for s in suffixes]

    missing = [
        c for c in axon_cols + dendrite_cols
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # choose rows
    if rows is None:
        plot_df = df.copy()
    else:
        if np.isscalar(rows):
            rows = [rows]

        if source_col in df.columns and all(isinstance(r, str) for r in rows):
            plot_df = df[df[source_col].isin(rows)].copy()
        else:
            plot_df = df.iloc[list(rows)].copy()

    if plot_df.empty:
        raise ValueError("No rows selected to plot.")

    n_rows = len(plot_df)

    # default colors
    if axon_colors is None:
        axon_colors = ["tab:blue"] * n_rows
    elif isinstance(axon_colors, str):
        axon_colors = [axon_colors] * n_rows

    if dendrite_colors is None:
        dendrite_colors = ["purple"] * n_rows
    elif isinstance(dendrite_colors, str):
        dendrite_colors = [dendrite_colors] * n_rows

    if len(axon_colors) != n_rows:
        raise ValueError("axon_colors length must match number of selected rows.")
    if len(dendrite_colors) != n_rows:
        raise ValueError("dendrite_colors length must match number of selected rows.")

    x = np.arange(len(suffixes))

    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=figsize)

        for i, (_, row) in enumerate(plot_df.iterrows()):
            label_base = (
                str(row[source_col])
                if source_col in plot_df.columns
                else f"row {i}"
            )

            axon_y = pd.to_numeric(row[axon_cols], errors="coerce").to_numpy(dtype=float)
            dendrite_y = pd.to_numeric(row[dendrite_cols], errors="coerce").to_numpy(dtype=float)

            ax.plot(
                x,
                axon_y,
                marker=marker,
                linewidth=linewidth,
                color=axon_colors[i],
                label=f"{label_base} axon",
            )

            ax.plot(
                x,
                dendrite_y,
                marker=marker,
                linewidth=linewidth,
                color=dendrite_colors[i],
                label=f"{label_base} dendrite",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(suffixes)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title:
            ax.set_title(title, pad=12)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(frameon=False, fontsize=legend_fontsize,)
        fig.tight_layout()

        if save_svg is not None:
            save_svg = Path(save_svg)
            save_svg.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_svg,
                format="svg",
                bbox_inches="tight",
                transparent=True,
                metadata={"Creator": "plot_axon_dendrite_bump_summary"},
            )

        plt.show()






def plot_individual_and_mean_axon_dendrite_bumps(
    individual_df: pd.DataFrame,
    sum_df: pd.DataFrame,
    *,
    sum_source: str | None = None,
    source_col: str = "source",
    axon_color: str = "tab:blue",
    dendrite_color: str = "purple",
    individual_alpha: float = 0.25,
    individual_linewidth: float = 1,
    mean_linewidth: float = 4,
    marker: str | None = "o",
    figsize=(7, 5),
    title: str | None = "Individual and mean axon/dendrite bumps",
    xlabel: str = "Position",
    ylabel: str = "Input",
    save_svg: str | Path | None = None,
):
    """
    Plot individual axon/dendrite bump curves plus selected mean curves.

    individual_df:
        Each row gives one axon curve and one dendrite curve.

    sum_df:
        Summary df containing summed axon/dendrite columns.
        If sum_df has more than one row, use sum_source to choose which row.

    sum_source:
        Value in source_col identifying which row of sum_df to use for the mean.
        If None, sum_df must have exactly one row.
    """

    suffixes = [
        "4_L", "3_L", "2_L", "1_L", "0",
        "1_R", "2_R", "3_R", "4_R"
    ]

    axon_cols = [f"axon_{s}" for s in suffixes]
    dendrite_cols = [f"dendrite_{s}" for s in suffixes]

    missing_indiv = [
        c for c in axon_cols + dendrite_cols
        if c not in individual_df.columns
    ]
    if missing_indiv:
        raise ValueError(f"Missing columns in individual_df: {missing_indiv}")

    missing_sum = [
        c for c in axon_cols + dendrite_cols
        if c not in sum_df.columns
    ]
    if missing_sum:
        raise ValueError(f"Missing columns in sum_df: {missing_sum}")

    if sum_source is not None:
        if source_col not in sum_df.columns:
            raise ValueError(f"sum_df must contain source_col={source_col!r}.")

        matched = sum_df[sum_df[source_col] == sum_source]

        if len(matched) != 1:
            raise ValueError(
                f"Expected exactly one row where {source_col} == {sum_source!r}, "
                f"but found {len(matched)}."
            )

        sum_row = matched.iloc[0]

    else:
        if len(sum_df) != 1:
            raise ValueError(
                "sum_df has more than one row, so you must provide sum_source."
            )

        sum_row = sum_df.iloc[0]

    n = len(individual_df)
    if n == 0:
        raise ValueError("individual_df has 0 rows.")

    x = np.arange(len(suffixes))

    axon_indiv = individual_df[axon_cols].apply(pd.to_numeric, errors="coerce")
    dendrite_indiv = individual_df[dendrite_cols].apply(pd.to_numeric, errors="coerce")

    axon_mean = (
        pd.to_numeric(sum_row[axon_cols], errors="coerce")
        .to_numpy(dtype=float)
        / n
    )

    dendrite_mean = (
        pd.to_numeric(sum_row[dendrite_cols], errors="coerce")
        .to_numpy(dtype=float)
        / n
    )

    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=figsize)

        # individual axon curves
        for _, row in axon_indiv.iterrows():
            ax.plot(
                x,
                row.to_numpy(dtype=float),
                color=axon_color,
                alpha=individual_alpha,
                linewidth=individual_linewidth,
            )

        # individual dendrite curves
        for _, row in dendrite_indiv.iterrows():
            ax.plot(
                x,
                row.to_numpy(dtype=float),
                color=dendrite_color,
                alpha=individual_alpha,
                linewidth=individual_linewidth,
            )

        # mean curves
        ax.plot(
            x,
            axon_mean,
            color=axon_color,
            linewidth=mean_linewidth,
            marker=marker,
            label=f"axon mean (n={n})",
        )

        ax.plot(
            x,
            dendrite_mean,
            color=dendrite_color,
            linewidth=mean_linewidth,
            marker=marker,
            label=f"dendrite mean (n={n})",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(suffixes)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title:
            ax.set_title(title, pad=12)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(frameon=False)
        fig.tight_layout()

        if save_svg is not None:
            save_svg = Path(save_svg)
            save_svg.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_svg,
                format="svg",
                bbox_inches="tight",
                transparent=True,
                metadata={
                    "Creator": "plot_individual_and_mean_axon_dendrite_bumps"
                },
            )

        plt.show()




def compute_glucl_scaling_values(
    summary_df: pd.DataFrame,
    *,
    columns=(
        "percentage_total_green",
        "percentage_total_green_divided_by_d7countall",
    ),
    rows=range(0, 9),
    exclude_idx: int = 4,
    center_idx: int = 4,
    divide_by: float = 24,
) -> dict:
    """
    For each column:
    - dendritemean = mean of selected rows excluding exclude_idx, then / divide_by
    - axon = center_idx value / divide_by
    """

    out = {}

    idxs = [i for i in rows if i != exclude_idx]

    for col in columns:
        if col not in summary_df.columns:
            raise ValueError(f"Column not found in summary_df: {col}")

        dendrite_vals = pd.to_numeric(
            summary_df.loc[idxs, col],
            errors="coerce",
        )

        center_val = pd.to_numeric(
            pd.Series([summary_df.loc[center_idx, col]]),
            errors="coerce",
        ).iloc[0]

        out[f"{col}_dendritemean"] = dendrite_vals.mean(skipna=True) / divide_by
        out[f"{col}_axon"] = center_val / divide_by

    return out



def build_multiplied_bump_summary(
    summary_df: pd.DataFrame,
    bump_summary_df: pd.DataFrame,
    *,
    green_col: str = "percentage_total_green",
    green_d7_col: str = "percentage_total_green_divided_by_d7countall",
    center_idx: int = 4,
    dendrite_rows=range(0, 9),
    dendrite_exclude_idx: int = 4,
    divide_by: float = 24,
) -> pd.DataFrame:
    """
    Build a 2-row bump summary after multiplying axon columns by center GluCl
    values and dendrite columns by dendrite-mean GluCl values.
    """

    if len(bump_summary_df) != 1:
        raise ValueError(
            f"Expected bump_summary_df to have exactly one row, got {len(bump_summary_df)}."
        )

    suffixes = [
        "4_L", "3_L", "2_L", "1_L", "0",
        "1_R", "2_R", "3_R", "4_R",
    ]

    axon_cols = [f"axon_{s}" for s in suffixes]
    dendrite_cols = [f"dendrite_{s}" for s in suffixes]
    bump_cols = axon_cols + dendrite_cols

    missing_bump = [c for c in bump_cols if c not in bump_summary_df.columns]
    if missing_bump:
        raise ValueError(f"Missing columns in bump_summary_df: {missing_bump}")

    for col in [green_col, green_d7_col]:
        if col not in summary_df.columns:
            raise ValueError(f"Column not found in summary_df: {col}")

    glucl_scalings = compute_glucl_scaling_values(
        summary_df=summary_df,
        columns=(green_col, green_d7_col),
        rows=dendrite_rows,
        exclude_idx=dendrite_exclude_idx,
        divide_by=divide_by,
    )

    base = bump_summary_df.iloc[0]

    rows_out = []

    for col in [green_col, green_d7_col]:

        axon_multiplier = glucl_scalings[f"{col}_axon"]
        dendrite_multiplier = glucl_scalings[f"{col}_dendritemean"]

        row_data = {}

        for axon_col in axon_cols:
            row_data[axon_col] = base[axon_col] * axon_multiplier

        for dendrite_col in dendrite_cols:
            row_data[dendrite_col] = base[dendrite_col] * dendrite_multiplier

        row_data["source"] = f"multiply_{col}"

        rows_out.append(row_data)

    out = pd.DataFrame(rows_out)

    out = out[["source"] + bump_cols]

    return out




def apply_glucl_scaling_to_individual_bumps(
    df: pd.DataFrame,
    glucl_scaling_values: dict,
    *,
    axon_key: str,
    dendrite_key: str,
) -> pd.DataFrame:
    """
    Apply GluCl scaling to individual Delta7 bump table.

    Axon scaling is applied to:
    - axonal_input
    - columns starting with 'axon'

    Dendrite scaling is applied to:
    - dendritic_input
    - columns starting with 'dendrite' or 'dendritic'
    """

    out = df.copy()

    axon_scale = glucl_scaling_values[axon_key]
    dendrite_scale = glucl_scaling_values[dendrite_key]

    axon_cols = [
        c for c in out.columns
        if c == "axonal_input" or c.startswith("axon_")
    ]

    dendrite_cols = [
        c for c in out.columns
        if c == "dendritic_input"
        or c.startswith("dendrite_")
        or c.startswith("dendritic_")
    ]

    for c in axon_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce") * axon_scale

    for c in dendrite_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce") * dendrite_scale

    return out


















def build_d7_input_matrix(
    delta_name: pd.DataFrame,
    input_dict: dict[str, pd.DataFrame],
    *,
    id_col: str = "id",
    instance_col: str = "instance",
    pre_id_col: str = "bodyId_pre",
    syn_col: str = "SynapseCount",
    label_with_instance: bool = True,
) -> pd.DataFrame:
    """
    Build a Delta7-to-Delta7 synapse-count matrix.

    Rows = presynaptic/sending Delta7 neurons.
    Columns = postsynaptic/target Delta7 neurons.

    Both axes follow the order in delta_name.
    """

    delta_tmp = delta_name[[id_col, instance_col]].copy()
    delta_tmp[id_col] = delta_tmp[id_col].astype(int)

    ordered_ids = delta_tmp[id_col].tolist()

    if label_with_instance:
        labels = (
            delta_tmp[id_col].astype(str)
            + "_"
            + delta_tmp[instance_col].astype(str)
        ).tolist()
    else:
        labels = [str(x) for x in ordered_ids]

    matrix = pd.DataFrame(
        0,
        index=ordered_ids,
        columns=ordered_ids,
        dtype=float,
    )

    for target_id in ordered_ids:
        matches = [
            name for name in input_dict.keys()
            if str(target_id) in str(name)
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one matching df for target id {target_id}, "
                f"but found {len(matches)}: {matches}"
            )

        target_df = input_dict[matches[0]].copy()

        target_df[pre_id_col] = pd.to_numeric(
            target_df[pre_id_col],
            errors="coerce",
        )

        target_df[syn_col] = pd.to_numeric(
            target_df[syn_col],
            errors="coerce",
        ).fillna(0)

        counts = (
            target_df
            .dropna(subset=[pre_id_col])
            .assign(**{pre_id_col: lambda d: d[pre_id_col].astype(int)})
            .groupby(pre_id_col)[syn_col]
            .sum()
        )

        for pre_id, count in counts.items():
            if pre_id in matrix.index:
                matrix.loc[pre_id, target_id] = count

    if label_with_instance:
        matrix.index = labels
        matrix.columns = labels

    return matrix



def _extract_center_glom(label: str) -> str:
    """
    Extract first 2 characters after the first underscore.

    Example:
    '911911004_L1L9R8_R' -> 'L1'
    '941810314_L7R3_L'   -> 'L7'
    """
    return str(label).split("_", 1)[1][:2]


def _signed_circular_position(center: str, glom: str, glomerulus_order: list[str]) -> int:
    """
    Assign signed circular position of glom relative to center.

    Example for 8 positions:
    offset 0 -> 0
    offset 1 -> -1
    offset 2 -> -2
    offset 3 -> -3
    offset 4 -> -4
    offset 5 -> 3
    offset 6 -> 2
    offset 7 -> 1
    """

    n = len(glomerulus_order)

    center_idx = glomerulus_order.index(center)
    glom_idx = glomerulus_order.index(glom)

    offset = (glom_idx - center_idx) % n

    if offset <= n // 2:
        return -offset
    else:
        return n - offset


def build_d7_relative_glomerulus_matrix(
    reference_matrix: pd.DataFrame,
    *,
    glomerulus_order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a matrix with same index/columns as reference_matrix.

    Rows = target Delta7 cells.
    Columns = presynaptic Delta7 cells.

    Each value is the signed circular glomerulus position of the presynaptic
    cell relative to the target cell center.

    Cell labels are expected to look like:
    '911911004_L1L9R8_R'
    """

    if glomerulus_order is None:
        glomerulus_order = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]

    out = pd.DataFrame(
        index=reference_matrix.index,
        columns=reference_matrix.columns,
        dtype=float,
    )

    for target_label in out.index:
        center = _extract_center_glom(target_label)

        if center not in glomerulus_order:
            raise ValueError(
                f"Target center {center!r} from {target_label!r} "
                f"not found in glomerulus_order."
            )

        for pre_label in out.columns:
            pre_glom = _extract_center_glom(pre_label)

            if pre_glom not in glomerulus_order:
                raise ValueError(
                    f"Presynaptic glomerulus {pre_glom!r} from {pre_label!r} "
                    f"not found in glomerulus_order."
                )

            out.loc[target_label, pre_label] = _signed_circular_position(
                center=center,
                glom=pre_glom,
                glomerulus_order=glomerulus_order,
            )

    return out



def map_relative_matrix_to_summary_values(
    relative_matrix: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    value_col: str = "percentage_total_green",
    glomerulus_col: str = "glomerulus",
    drop_glomerulus=4,
) -> pd.DataFrame:
    """
    Replace each value in a relative-position matrix with the corresponding
    value from summary_df.

    Example:
    matrix cell = -4
    → use summary_df[value_col] where summary_df[glomerulus_col] == -4
    """

    if value_col not in summary_df.columns:
        raise ValueError(f"Column not found in summary_df: {value_col}")

    if glomerulus_col not in summary_df.columns:
        raise ValueError(f"Column not found in summary_df: {glomerulus_col}")

    summary_use = summary_df.copy()

    if drop_glomerulus is not None:
        summary_use = summary_use[
            summary_use[glomerulus_col] != drop_glomerulus
        ]

    value_map = (
        summary_use
        .set_index(glomerulus_col)[value_col]
        .to_dict()
    )

    out = relative_matrix.copy()

    out = out.applymap(lambda x: value_map.get(x, np.nan))

    return out




def map_relative_matrix_to_cosine_bump(
    relative_matrix: pd.DataFrame,
    cos_bump_array,
) -> pd.DataFrame:
    """
    Convert a relative-position matrix into a cosine-bump-value matrix.

    Expected relative_matrix values for 8 glomeruli:
        0, -1, -2, -3, -4, 3, 2, 1

    cos_bump_array should be length 8 and ordered like:
        [center, one-step, two-step, three-step, opposite,
         three-step-other-side, two-step-other-side, one-step-other-side]

    For a standard 8-glom cosine bump centered at 0:
        relative  0  -> cos_bump_array[0]
        relative -1  -> cos_bump_array[1]
        relative -2  -> cos_bump_array[2]
        relative -3  -> cos_bump_array[3]
        relative -4  -> cos_bump_array[4]
        relative  3  -> cos_bump_array[5]
        relative  2  -> cos_bump_array[6]
        relative  1  -> cos_bump_array[7]
    """

    cos_bump_array = np.asarray(cos_bump_array, dtype=float)

    if len(cos_bump_array) != 8:
        raise ValueError(f"Expected cos_bump_array to have length 8, got {len(cos_bump_array)}.")

    relative_to_bump = {
        0: cos_bump_array[0],
        -1: cos_bump_array[1],
        -2: cos_bump_array[2],
        -3: cos_bump_array[3],
        -4: cos_bump_array[4],
        3: cos_bump_array[5],
        2: cos_bump_array[6],
        1: cos_bump_array[7],
    }

    out = relative_matrix.copy()

    out = out.applymap(lambda x: relative_to_bump.get(int(x), np.nan))

    return out




def roll_matrix_columns_to_center_zero(
    input_matrix: pd.DataFrame,
    relative_matrix: pd.DataFrame,
    *,
    center_idx: int | None = None,
    zero_value=0,
) -> pd.DataFrame:
    """
    For each column, circularly roll rows so that the row where
    relative_matrix == zero_value is moved to center_idx.

    The roll order is determined from relative_matrix, then applied to input_matrix.

    Parameters
    ----------
    input_matrix : pd.DataFrame
        Matrix to roll, e.g. d7_result_bump_summary.

    relative_matrix : pd.DataFrame
        Matrix with same shape/index/columns as input_matrix.
        Used to locate zero positions in each column.

    center_idx : int or None
        Target row index. If None, uses len(input_matrix)//2.

    zero_value : number
        Value in relative_matrix used as center marker. Default 0.

    Returns
    -------
    pd.DataFrame
        Rolled matrix with same columns as input_matrix.
        Row index is reset to integer positions because each column may be
        rolled differently.
    """

    if input_matrix.shape != relative_matrix.shape:
        raise ValueError(
            f"Shape mismatch: input_matrix {input_matrix.shape}, "
            f"relative_matrix {relative_matrix.shape}"
        )

    if not input_matrix.columns.equals(relative_matrix.columns):
        raise ValueError("input_matrix and relative_matrix must have same columns.")

    if not input_matrix.index.equals(relative_matrix.index):
        raise ValueError("input_matrix and relative_matrix must have same index.")

    n_rows = len(input_matrix)

    if center_idx is None:
        center_idx = n_rows // 2

    rolled_cols = {}

    for col in input_matrix.columns:
        rel_col = relative_matrix[col].to_numpy()

        zero_positions = np.where(rel_col == zero_value)[0]

        if len(zero_positions) == 0:
            raise ValueError(f"No zero_value={zero_value!r} found in column {col!r}.")

        # choose middle zero if multiple zero rows exist
        chosen_zero_pos = zero_positions[len(zero_positions) // 2]

        shift = center_idx - chosen_zero_pos

        rolled_cols[col] = np.roll(
            input_matrix[col].to_numpy(),
            shift,
        )

    out = pd.DataFrame(
        rolled_cols,
        index=np.arange(n_rows),
        columns=input_matrix.columns,
    )

    return out




# cosine fit + group by left glomerulus

def group_rowmean_by_left_glomerulus(
    rowmean_df: pd.DataFrame,
    labels,
    *,
    value_col: str = "row_mean",
) -> pd.DataFrame:
    """
    Group 42 row-mean values into 8 left-glomerulus groups.

    labels should be in the same order as rowmean_df rows.
    Example label: '911911004_L1L9R8_R' -> group '1'
    """

    y = pd.to_numeric(rowmean_df[value_col], errors="coerce").to_numpy(dtype=float)
    labels = list(labels)

    if len(y) != len(labels):
        raise ValueError(
            f"rowmean_df has {len(y)} values, but labels has {len(labels)} values."
        )

    rows = []

    for i, (label, value) in enumerate(zip(labels, y)):
        glom = str(label).split("_", 1)[1][:2]   # e.g. L1
        group = int(glom[1])                     # e.g. 1

        rows.append(
            {
                "raw_idx": i,
                "label": label,
                "left_glomerulus": group,
                value_col: value,
            }
        )

    long_df = pd.DataFrame(rows)

    grouped = (
        long_df
        .groupby("left_glomerulus", sort=True)
        .agg(
            n_cells=("raw_idx", "count"),
            raw_indices=("raw_idx", lambda x: list(x)),
            x_center=("raw_idx", "mean"),
            bump_sum=(value_col, "sum"),
            bump_mean=(value_col, "mean"),
        )
        .reset_index()
    )

    return grouped



def cosine_fixed_amp_offset(x, phi, A, B, T=8):
    """
    Cosine with fixed amplitude, offset, and period.
    Only phi is fit.
    """
    return A * np.cos(2 * np.pi / T * (x - phi)) + B


def fit_8point_cosine_phase_only(
    grouped_bump_df: pd.DataFrame,
    *,
    x_col: str = "left_glomerulus",
    y_col: str = "bump_mean",
    T: float = 8,
) -> dict:
    """
    Fit an 8-point cosine by fitting only phi.

    A = (max - min) / 2
    B = (max + min) / 2
    T is fixed.
    """

    x = pd.to_numeric(grouped_bump_df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(grouped_bump_df[y_col], errors="coerce").to_numpy(dtype=float)

    # fixed amplitude + offset
    A = (np.nanmax(y) - np.nanmin(y)) / 2
    B = (np.nanmax(y) + np.nanmin(y)) / 2

    def model_for_fit(x, phi):
        return cosine_fixed_amp_offset(x, phi, A=A, B=B, T=T)

    popt, pcov = curve_fit(
        model_for_fit,
        x,
        y,
        p0=[4.5],
        maxfev=10000,
    )

    phi = float(popt[0])

    y_fit = model_for_fit(x, phi)

    # ---------------- goodness of fit ----------------

    residuals = y - y_fit

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    r2 = 1 - (ss_res / ss_tot)

    rmse = np.sqrt(np.mean(residuals**2))

    return {
        "x": x,
        "y": y,
        "A": float(A),
        "B": float(B),
        "T": float(T),
        "phi": phi,
        "y_fit": y_fit,
        "residuals": residuals,
        "R2": float(r2),
        "RMSE": float(rmse),
        "popt": popt,
        "pcov": pcov,
    }



def plot_raw_grouped_and_cosine_fit(
    rowmean_df: pd.DataFrame,
    grouped_bump_df: pd.DataFrame,
    fit_result: dict,
    *,
    value_col: str = "row_mean",
    title: str = "Raw, grouped, and cosine fit",
    xlabel: str = "Rolled row index",
    ylabel: str = "Mean value",
    raw_color: str = "gray",
    grouped_color: str = "black",
    fit_color: str = "tab:red",
    raw_alpha: float = 0.5,
    raw_marker: str = "o",
    grouped_marker: str = "o",
    fit_linewidth: float = 3,
    figsize=(9, 4),
):
    """
    Plot:
    1. 42-point raw rowmean curve
    2. 8 grouped means, stretched onto raw x-axis
    3. cosine fit, stretched onto raw x-axis
    """

    raw_y = pd.to_numeric(rowmean_df[value_col], errors="coerce").to_numpy(dtype=float)
    raw_x = np.arange(len(raw_y))

    group_x = grouped_bump_df["x_center"].to_numpy(dtype=float)
    group_y = grouped_bump_df["bump_mean"].to_numpy(dtype=float)

    fit_y = np.asarray(fit_result["y_fit"], dtype=float)

    plt.figure(figsize=figsize)

    plt.plot(
        raw_x,
        raw_y,
        marker=raw_marker,
        color=raw_color,
        alpha=raw_alpha,
        label="42 raw points",
    )

    plt.plot(
        group_x,
        group_y,
        marker=grouped_marker,
        color=grouped_color,
        linewidth=2,
        label="8 grouped means",
    )

    plt.plot(
        group_x,
        fit_y,
        color=fit_color,
        linewidth=fit_linewidth,
        label="cosine fit",
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()



















# YF part


def plot_matrix_heatmap(
    matrix: pd.DataFrame,
    *,
    cmap: str = "coolwarm",
    center=None,
    square: bool = True,
    figsize=(8, 6),
    xlabel: str = "Columns",
    ylabel: str = "Rows",
    title: str | None = None,
    vmin=None,
    vmax=None,
    show_xticklabels: bool = True,
    show_yticklabels: bool = True,
    xtick_fontsize: float = 8,
    ytick_fontsize: float = 8,
    save_svg: str | Path | None = None,
):
    """
    Plot a heatmap for a matrix/DataFrame.
    """

    with mpl.rc_context({"svg.fonttype": "none"}):
        plt.figure(figsize=figsize)

        ax = sns.heatmap(
            matrix,
            cmap=cmap,
            center=center,
            square=square,
            vmin=vmin,
            vmax=vmax,
            xticklabels=show_xticklabels,
            yticklabels=show_yticklabels,
        )
        
        ax.tick_params(axis="x", labelsize=xtick_fontsize)
        ax.tick_params(axis="y", labelsize=ytick_fontsize)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if title:
            plt.title(title)

        plt.tight_layout()

        if save_svg is not None:
            save_svg = Path(save_svg)
            save_svg.parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(
                save_svg,
                format="svg",
                bbox_inches="tight",
                transparent=True,
                metadata={
                    "Creator": "plot_matrix_heatmap"
                },
            )

        plt.show()





def project_inhibition_onto_bump_columns(
    inhibition_matrix: pd.DataFrame,
    bump_matrix_colcondensed: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each column in inhibition_matrix:
    - multiply it elementwise with each column in bump_matrix_colcondensed
    - sum the multiplied values
    - store the 8 sums as one row

    Rows of output = columns of inhibition_matrix.
    Columns of output = = bump_matrix_colcondensed column names, e.g. L1, L2, ..., L8.
    """

    if len(inhibition_matrix) != len(bump_matrix_colcondensed):
        raise ValueError(
            f"Row count mismatch: inhibition_matrix has {len(inhibition_matrix)} rows, "
            f"bump_matrix_colcondensed has {len(bump_matrix_colcondensed)} rows."
        )

    out_rows = []

    for inhib_col in inhibition_matrix.columns:
        inhib_values = pd.to_numeric(
            inhibition_matrix[inhib_col],
            errors="coerce",
        ).to_numpy(dtype=float)

        row_data = {}

        for bump_col in bump_matrix_colcondensed.columns:
            bump_values = pd.to_numeric(
                bump_matrix_colcondensed[bump_col],
                errors="coerce",
            ).to_numpy(dtype=float)

            row_data[bump_col] = np.nansum(
                inhib_values * bump_values
            )

        row_data["source"] = inhib_col
        out_rows.append(row_data)

    out = pd.DataFrame(out_rows)
    out = out.set_index("source")

    return out



def roll_each_row_to_center_glomerulus(
    df: pd.DataFrame,
    *,
    target_idx: int = 4,
) -> pd.DataFrame:
    """
    For each row, extract the first 2 characters after the first underscore
    from the row name, then circularly roll that row's values so that the
    matching column lands at target_idx.

    Example:
    row name '911911004_L1L9R8_R' -> center column 'L1'
    """

    out_rows = []

    columns = list(df.columns)

    for row_name, row in df.iterrows():
        center_glom = str(row_name).split("_", 1)[1][:2]

        if center_glom not in columns:
            raise ValueError(
                f"Center glomerulus {center_glom!r} from row {row_name!r} "
                f"not found in df columns."
            )

        current_idx = columns.index(center_glom)
        shift = target_idx - current_idx

        rolled_values = np.roll(
            pd.to_numeric(row, errors="coerce").to_numpy(dtype=float),
            shift,
        )

        out_rows.append(rolled_values)
    
    relative_cols = [-4, -3, -2, -1, 0, 1, 2, 3]
    out = pd.DataFrame(
        out_rows,
        index=df.index,
        columns=relative_cols,
    )

    return out





def round_up_to_nice_number(value: float) -> float:
    if value <= 0:
        return value

    magnitude = 10 ** np.floor(np.log10(value))
    return np.ceil(value / magnitude) * magnitude

'''
OLD version, doesn't contain the beautifying features, but have the default plotting features.

def plot_each_row_as_line(
    df: pd.DataFrame,
    *,
    figsize=(7, 5),
    color="tab:blue",
    alpha: float = 0.3,
    linewidth: float = 1,
    mean_color: str | None = "black",
    mean_linewidth: float = 3,
    marker: str | None = None,
    xlabel: str = "Column",
    ylabel: str = "Value",
    title: str | None = None,
    show_legend: bool = False,
    save_svg: str | Path | None = None,
):
    """
    Plot each row of a DataFrame as one line.

    Assumes:
    - rows = observations
    - columns = ordered x positions
    """

    plot_df = df.apply(pd.to_numeric, errors="coerce")

    x = np.arange(len(plot_df.columns))

    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=figsize)

        # individual rows
        for idx, row in plot_df.iterrows():

            ax.plot(
                x,
                row.to_numpy(dtype=float),
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                marker=marker,
                label=str(idx) if show_legend else None,
            )

        # mean line
        if mean_color is not None:

            mean_vals = plot_df.mean(axis=0).to_numpy(dtype=float)

            ax.plot(
                x,
                mean_vals,
                color=mean_color,
                linewidth=mean_linewidth,
                marker=marker,
                label="mean",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(plot_df.columns)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title:
            ax.set_title(title)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if show_legend or mean_color is not None:
            ax.legend(frameon=False)

        fig.tight_layout()

        if save_svg is not None:
            save_svg = Path(save_svg)
            save_svg.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                save_svg,
                format="svg",
                bbox_inches="tight",
                transparent=True,
                metadata={
                    "Creator": "plot_each_row_as_line"
                },
            )

        plt.show()
'''

# NEWWWWWW
def plot_each_row_as_line(
    df: pd.DataFrame,
    *,
    figsize=(7, 5),
    color="tab:blue",
    alpha: float = 0.3,
    linewidth: float = 1,
    mean_color: str | None = "black",
    mean_linewidth: float = 3,
    marker: str | None = None,
    marker_size: float = 5,
    xlabel: str = "Column",
    ylabel: str = "Value",
    title: str | None = None,
    show_legend: bool = False,
    save_svg: str | Path | None = None,

    # new controls
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    xtick_rotation: float = 0,
    show_mean: bool = True,
    show_individual: bool = True,
    mean_label: str = "mean",
    title_fontsize: float = 14,
    axis_label_size: float = 12,
    tick_label_size: float = 10,
    hide_legend_if_only_mean: bool = True,
    save_png: str | Path | None = None,

    # y tick controls
    y_min_mode: str = "zero",   # "zero" or "data"
    y_max_pad_frac: float = 0.05,
    ytick_decimals: int | None = None,

    # global style controls
    font_size: float | None = None,
    legend_fontsize: float | None = None,
    axis_linewidth: float = 1.5,
    tick_length: float = 4,
    tick_width: float = 1.5,

    # axis spacing controls
    x_axis_y_offset_frac: float = 0.04,
    y_axis_x_offset_frac: float = 0.04,
):
    """
    Plot each row of a DataFrame as one line.

    Assumes:
    - rows = observations / cells
    - columns = ordered x positions
    """

    plot_df = df.apply(pd.to_numeric, errors="coerce")

    x = np.arange(len(plot_df.columns))

    all_vals = plot_df.to_numpy(dtype=float)
    data_min = np.nanmin(all_vals)
    data_max = np.nanmax(all_vals)

    if y_min_mode == "zero":
        y_bottom = 0
    elif y_min_mode == "data":
        y_bottom = data_min
    else:
        raise ValueError("y_min_mode must be either 'zero' or 'data'.")

    y_range = data_max - y_bottom

    if y_range == 0:
        y_range = abs(data_max) if data_max != 0 else 1

    y_top = data_max + y_max_pad_frac * y_range
    y_top = round_up_to_nice_number(y_top)
    y_middle = (y_bottom + y_top) / 2

    auto_ylim = (y_bottom, y_top)
    auto_yticks = [y_bottom, y_middle, y_top]

    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=figsize)

        # master font size
        if font_size is not None:
            title_fontsize = font_size
            axis_label_size = font_size
            tick_label_size = font_size

            if legend_fontsize is None:
                legend_fontsize = font_size

        # individual rows
        if show_individual:
            for idx, row in plot_df.iterrows():
                ax.plot(
                    x,
                    row.to_numpy(dtype=float),
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                    marker=marker,
                    markersize=marker_size,
                    label=str(idx) if show_legend else None,
                )

        # mean line
        if show_mean and mean_color is not None:
            mean_vals = plot_df.mean(axis=0).to_numpy(dtype=float)

            ax.plot(
                x,
                mean_vals,
                color=mean_color,
                linewidth=mean_linewidth,
                marker=marker,
                markersize=marker_size,
                label=mean_label,
            )

        # x axis
        if xlim is None:
            ax.set_xlim(x[0], x[-1])
        else:
            ax.set_xlim(xlim)

        # y axis
        if ylim is None:
            ax.set_ylim(auto_ylim)
            ax.set_yticks(auto_yticks)

            if ytick_decimals is not None:
                ax.set_yticklabels(
                    [f"{v:.{ytick_decimals}f}" for v in auto_yticks]
                )

        else:
            ax.set_ylim(ylim)

            y_bottom_manual, y_top_manual = ylim
            y_middle_manual = (y_bottom_manual + y_top_manual) / 2

            manual_ticks = [
                y_bottom_manual,
                y_middle_manual,
                y_top_manual,
            ]

            ax.set_yticks(manual_ticks)

            if ytick_decimals is not None:
                ax.set_yticklabels(
                    [f"{v:.{ytick_decimals}f}" for v in manual_ticks]
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            plot_df.columns,
            rotation=xtick_rotation,
            fontsize=tick_label_size,
        )

        ax.set_xlabel(xlabel, fontsize=axis_label_size)
        ax.set_ylabel(ylabel, fontsize=axis_label_size)

        if title:
            ax.set_title(title, fontsize=title_fontsize)

        # tick length, width, and tick-label font size
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=tick_label_size,
            length=tick_length,
            width=tick_width,
        )

        # spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_linewidth(axis_linewidth)
        ax.spines["bottom"].set_linewidth(axis_linewidth)

        # offset x and y axes so they do not touch at the corner
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        x_offset = y_axis_x_offset_frac * (x1 - x0)
        y_offset = x_axis_y_offset_frac * (y1 - y0)

        ax.spines["left"].set_position(("data", x0 - x_offset))
        ax.spines["bottom"].set_position(("data", y0 - y_offset))

        # keep ticks only on left/bottom
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

        # legend behavior
        if show_legend:
            ax.legend(
                frameon=False,
                fontsize=legend_fontsize,
            )

        elif show_mean and mean_color is not None and not hide_legend_if_only_mean:
            ax.legend(
                frameon=False,
                fontsize=legend_fontsize,
            )

        fig.tight_layout()

        if save_svg is not None:
            save_svg = Path(save_svg)
            save_svg.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                save_svg,
                format="svg",
                bbox_inches="tight",
                transparent=True,
                metadata={"Creator": "plot_each_row_as_line"},
            )

        if save_png is not None:
            save_png = Path(save_png)
            save_png.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                save_png,
                format="png",
                dpi=300,
                bbox_inches="tight",
                transparent=True,
                metadata={"Creator": "plot_each_row_as_line"},
            )

        plt.show()












'''
OLD version, doesn't contain the beautifying features, but have the default plotting features.
def plot_two_dfs_each_row_as_line(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    df1_color: str = "tab:blue",
    df2_color: str = "tab:purple",
    df1_label: str = "df1 mean",
    df2_label: str = "df2 mean",
    alpha: float = 0.25,
    linewidth: float = 1,
    mean_linewidth: float = 4,
    marker: str | None = None,
    figsize=(7, 5),
    xlabel: str = "Position",
    ylabel: str = "Value",
    title: str | None = None,
    save_svg: str | Path | None = None,
):
    """
    Plot all rows from two DataFrames as lines on the same plot.

    - all rows from df1 plotted in one color
    - all rows from df2 plotted in another color
    - mean line for each df also plotted
    """

    if not df1.columns.equals(df2.columns):
        raise ValueError("df1 and df2 must have identical columns.")

    plot_df1 = df1.apply(pd.to_numeric, errors="coerce")
    plot_df2 = df2.apply(pd.to_numeric, errors="coerce")

    x = np.arange(len(plot_df1.columns))

    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=figsize)

        # ---------------- df1 individual ----------------

        for _, row in plot_df1.iterrows():

            ax.plot(
                x,
                row.to_numpy(dtype=float),
                color=df1_color,
                alpha=alpha,
                linewidth=linewidth,
            )

        # ---------------- df2 individual ----------------

        for _, row in plot_df2.iterrows():

            ax.plot(
                x,
                row.to_numpy(dtype=float),
                color=df2_color,
                alpha=alpha,
                linewidth=linewidth,
            )

        # ---------------- mean lines ----------------

        df1_mean = plot_df1.mean(axis=0).to_numpy(dtype=float)
        df2_mean = plot_df2.mean(axis=0).to_numpy(dtype=float)

        ax.plot(
            x,
            df1_mean,
            color=df1_color,
            linewidth=mean_linewidth,
            marker=marker,
            label=df1_label,
        )

        ax.plot(
            x,
            df2_mean,
            color=df2_color,
            linewidth=mean_linewidth,
            marker=marker,
            label=df2_label,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(plot_df1.columns)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title:
            ax.set_title(title)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(frameon=False)

        fig.tight_layout()

        if save_svg is not None:
            save_svg = Path(save_svg)
            save_svg.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                save_svg,
                format="svg",
                bbox_inches="tight",
                transparent=True,
                metadata={
                    "Creator": "plot_two_dfs_each_row_as_line"
                },
            )

        plt.show()
'''


# NEWWW
def plot_two_dfs_each_row_as_line(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    df1_color: str = "tab:blue",
    df2_color: str = "purple",
    df1_label: str = "df1 mean",
    df2_label: str = "df2 mean",
    alpha: float = 0.25,
    linewidth: float = 1,
    mean_linewidth: float = 4,
    marker: str | None = None,
    marker_size: float = 5,
    figsize=(7, 5),
    xlabel: str = "Position",
    ylabel: str = "Value",
    title: str | None = None,
    save_svg: str | Path | None = None,

    # added controls
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    xtick_rotation: float = 0,
    show_mean: bool = True,
    show_individual: bool = True,
    show_legend: bool = True,
    title_fontsize: float = 14,
    axis_label_size: float = 12,
    tick_label_size: float = 10,
    save_png: str | Path | None = None,

    # y-axis tick controls
    y_min_mode: str = "zero",   # "zero" or "data"
    y_max_pad_frac: float = 0.05,
    ytick_decimals: int | None = None,

    # global style controls
    font_size: float | None = None,
    legend_fontsize: float | None = None,
    axis_linewidth: float = 1.5,
    tick_length: float = 4,
    tick_width: float = 1.5,

    # axis spacing controls
    x_axis_y_offset_frac: float = 0.04,
    y_axis_x_offset_frac: float = 0.04,
):
    """
    Plot all rows from two DataFrames as lines on the same plot.

    - all rows from df1 plotted in one color
    - all rows from df2 plotted in another color
    - mean line for each df also plotted
    """

    if not df1.columns.equals(df2.columns):
        raise ValueError("df1 and df2 must have identical columns.")

    plot_df1 = df1.apply(pd.to_numeric, errors="coerce")
    plot_df2 = df2.apply(pd.to_numeric, errors="coerce")

    x = np.arange(len(plot_df1.columns))

    all_vals = np.concatenate([
        plot_df1.to_numpy(dtype=float).ravel(),
        plot_df2.to_numpy(dtype=float).ravel(),
    ])

    data_min = np.nanmin(all_vals)
    data_max = np.nanmax(all_vals)

    if y_min_mode == "zero":
        y_bottom = 0
    elif y_min_mode == "data":
        y_bottom = data_min
    else:
        raise ValueError("y_min_mode must be either 'zero' or 'data'.")

    y_range = data_max - y_bottom

    if y_range == 0:
        y_range = abs(data_max) if data_max != 0 else 1

    y_top = data_max + y_max_pad_frac * y_range
    y_top = round_up_to_nice_number(y_top)
    y_middle = (y_bottom + y_top) / 2

    auto_ylim = (y_bottom, y_top)
    auto_yticks = [y_bottom, y_middle, y_top]

    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=figsize)

        # master font size
        if font_size is not None:
            title_fontsize = font_size
            axis_label_size = font_size
            tick_label_size = font_size

            if legend_fontsize is None:
                legend_fontsize = font_size

        # ---------------- df1 individual ----------------
        if show_individual:
            for _, row in plot_df1.iterrows():
                ax.plot(
                    x,
                    row.to_numpy(dtype=float),
                    color=df1_color,
                    alpha=alpha,
                    linewidth=linewidth,
                    marker=marker,
                    markersize=marker_size,
                )

        # ---------------- df2 individual ----------------
        if show_individual:
            for _, row in plot_df2.iterrows():
                ax.plot(
                    x,
                    row.to_numpy(dtype=float),
                    color=df2_color,
                    alpha=alpha,
                    linewidth=linewidth,
                    marker=marker,
                    markersize=marker_size,
                )

        # ---------------- mean lines ----------------
        if show_mean:
            df1_mean = plot_df1.mean(axis=0).to_numpy(dtype=float)
            df2_mean = plot_df2.mean(axis=0).to_numpy(dtype=float)

            ax.plot(
                x,
                df1_mean,
                color=df1_color,
                linewidth=mean_linewidth,
                marker=marker,
                markersize=marker_size,
                label=df1_label,
            )

            ax.plot(
                x,
                df2_mean,
                color=df2_color,
                linewidth=mean_linewidth,
                marker=marker,
                markersize=marker_size,
                label=df2_label,
            )

        # x axis
        if xlim is None:
            ax.set_xlim(x[0], x[-1])
        else:
            ax.set_xlim(xlim)

        # y axis
        if ylim is None:
            ax.set_ylim(auto_ylim)
            ax.set_yticks(auto_yticks)

            if ytick_decimals is not None:
                ax.set_yticklabels(
                    [f"{v:.{ytick_decimals}f}" for v in auto_yticks]
                )

        else:
            ax.set_ylim(ylim)

            y_bottom_manual, y_top_manual = ylim
            y_middle_manual = (y_bottom_manual + y_top_manual) / 2

            manual_ticks = [
                y_bottom_manual,
                y_middle_manual,
                y_top_manual,
            ]

            ax.set_yticks(manual_ticks)

            if ytick_decimals is not None:
                ax.set_yticklabels(
                    [f"{v:.{ytick_decimals}f}" for v in manual_ticks]
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            plot_df1.columns,
            rotation=xtick_rotation,
            fontsize=tick_label_size,
        )

        ax.set_xlabel(xlabel, fontsize=axis_label_size)
        ax.set_ylabel(ylabel, fontsize=axis_label_size)

        if title:
            ax.set_title(title, fontsize=title_fontsize)

        # tick length, width, and tick-label font size
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=tick_label_size,
            length=tick_length,
            width=tick_width,
        )

        # spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_linewidth(axis_linewidth)
        ax.spines["bottom"].set_linewidth(axis_linewidth)

        # offset x and y axes so they do not touch
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        x_offset = y_axis_x_offset_frac * (x1 - x0)
        y_offset = x_axis_y_offset_frac * (y1 - y0)

        ax.spines["left"].set_position(("data", x0 - x_offset))
        ax.spines["bottom"].set_position(("data", y0 - y_offset))

        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

        if show_legend and show_mean:
            ax.legend(
                frameon=False,
                fontsize=legend_fontsize,
            )

        fig.tight_layout()

        if save_svg is not None:
            save_svg = Path(save_svg)
            save_svg.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                save_svg,
                format="svg",
                bbox_inches="tight",
                transparent=True,
                metadata={
                    "Creator": "plot_two_dfs_each_row_as_line"
                },
            )

        if save_png is not None:
            save_png = Path(save_png)
            save_png.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                save_png,
                format="png",
                dpi=300,
                bbox_inches="tight",
                transparent=True,
                metadata={
                    "Creator": "plot_two_dfs_each_row_as_line"
                },
            )

        plt.show()






def plot_each_cell_two_lines(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    df1_color: str = "tab:blue",
    df2_color: str = "tab:purple",
    df1_label: str = "axon",
    df2_label: str = "dendrite",
    linewidth: float = 2,
    marker: str | None = "o",
    figsize=(5, 4),
    xlabel: str = "Position",
    ylabel: str = "Value",
    title_prefix: str = "",
    save_dir: str | Path | None = None,
    show_plots: bool = True,
):
    """
    For each matched row/cell in df1 and df2, make one plot with two lines.
    All plots use the same y-axis range based on the global min/max across both dfs.
    """

    if not df1.columns.equals(df2.columns):
        raise ValueError("df1 and df2 must have identical columns.")

    if not df1.index.equals(df2.index):
        raise ValueError("df1 and df2 must have identical row index/cell names.")

    x = np.arange(len(df1.columns))

    df1_num = df1.apply(pd.to_numeric, errors="coerce")
    df2_num = df2.apply(pd.to_numeric, errors="coerce")

    all_vals = np.concatenate([
        df1_num.to_numpy().ravel(),
        df2_num.to_numpy().ravel(),
    ])

    global_ymin = np.nanmin(all_vals)
    global_ymax = np.nanmax(all_vals)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for cell_name in df1.index:
        y1 = df1_num.loc[cell_name].to_numpy(dtype=float)
        y2 = df2_num.loc[cell_name].to_numpy(dtype=float)

        with mpl.rc_context({"svg.fonttype": "none"}):
            fig, ax = plt.subplots(figsize=figsize)

            ax.plot(
                x,
                y1,
                color=df1_color,
                linewidth=linewidth,
                marker=marker,
                label=df1_label,
            )

            ax.plot(
                x,
                y2,
                color=df2_color,
                linewidth=linewidth,
                marker=marker,
                label=df2_label,
            )

            ax.set_ylim(global_ymin, global_ymax)

            #yticks = np.linspace(global_ymin, global_ymax, 5)
            #ax.set_yticks(yticks)

            ax.set_xticks(x)
            ax.set_xticklabels(df1.columns)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            title = f"{title_prefix}{cell_name}" if title_prefix else str(cell_name)
            ax.set_title(title)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.legend(frameon=False)
            fig.tight_layout()

            if save_dir is not None:
                safe_name = str(cell_name).replace("/", "_").replace(" ", "_")
                fig.savefig(
                    save_dir / f"{safe_name}.svg",
                    format="svg",
                    bbox_inches="tight",
                    transparent=True,
                )

            if show_plots:
                plt.show()
            else:
                plt.close(fig)