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
    title: str | None = "Axon vs dendrite bump summary",
    xlabel: str = "Position",
    ylabel: str = "Summed input",
    axon_label: str = "axon",
    dendrite_label: str = "dendrite",
    axon_color: str = "tab:blue",
    dendrite_color: str = "tab:orange",
    marker: str = "o",
    linewidth: float = 2,
    figsize=(6, 4),
    save_svg: str | Path | None = None,
):
    """
    Plot axon and dendrite bump-summary curves from a one-row DataFrame.
    """

    if len(df) != 1:
        raise ValueError(f"Expected df to have exactly one row, got {len(df)} rows.")

    suffixes = [
        "4_L", "3_L", "2_L", "1_L", "0",
        "1_R", "2_R", "3_R", "4_R"
    ]

    axon_cols = [f"axon_{s}" for s in suffixes]
    dendrite_cols = [f"dendrite_{s}" for s in suffixes]

    missing = [
        c for c in axon_cols + dendrite_cols
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    x = np.arange(len(suffixes))

    axon_y = pd.to_numeric(df.loc[df.index[0], axon_cols], errors="coerce").to_numpy(dtype=float)
    dendrite_y = pd.to_numeric(df.loc[df.index[0], dendrite_cols], errors="coerce").to_numpy(dtype=float)

    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            x,
            axon_y,
            marker=marker,
            linewidth=linewidth,
            color=axon_color,
            label=axon_label,
        )

        ax.plot(
            x,
            dendrite_y,
            marker=marker,
            linewidth=linewidth,
            color=dendrite_color,
            label=dendrite_label,
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
                metadata={"Creator": "plot_axon_dendrite_bump_summary"},
            )

        plt.show()