from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from kylie_lib import syn_specs


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
    Plot individual axon/dendrite bump curves plus mean curves.

    individual_df:
        each row gives one axon curve and one dendrite curve.

    sum_df:
        one-row summary df containing summed axon/dendrite columns.
        Values are divided by n_rows of individual_df before plotting.
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

    if len(sum_df) != 1:
        raise ValueError(f"Expected sum_df to have exactly one row, got {len(sum_df)}.")

    n = len(individual_df)
    if n == 0:
        raise ValueError("individual_df has 0 rows.")

    x = np.arange(len(suffixes))

    axon_indiv = individual_df[axon_cols].apply(pd.to_numeric, errors="coerce")
    dendrite_indiv = individual_df[dendrite_cols].apply(pd.to_numeric, errors="coerce")

    axon_mean = (
        pd.to_numeric(sum_df.iloc[0][axon_cols], errors="coerce")
        .to_numpy(dtype=float)
        / n
    )

    dendrite_mean = (
        pd.to_numeric(sum_df.iloc[0][dendrite_cols], errors="coerce")
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