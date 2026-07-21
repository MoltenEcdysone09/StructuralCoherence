import pandas as pd
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import gc
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu, spearmanr
from statannotations.Annotator import Annotator
import itertools
import re
import matplotlib.lines as mlines

# =====================================================================
# STYLE & DESIGN PATTERNS (Nord Theme)
# =====================================================================

NORD_COLORS = {
    "dark": "#2e3440",
    "gray": "#3b4252",
    "red": "#bf616a",
    "blue": "#5e81ac",
    "green": "#a3be8c",
    "yellow": "#ebcb8b",
    "purple": "#b48ead",
    "orange": "#d08770",
    "cyan": "#8fbcbb",
    "light_blue": "#88c0d0",
}

NORD_PALETTE = [
    NORD_COLORS["red"],
    NORD_COLORS["blue"],
    NORD_COLORS["green"],
    NORD_COLORS["yellow"],
    NORD_COLORS["purple"],
    NORD_COLORS["cyan"],
    NORD_COLORS["orange"],
]


NETWORK_TO_ORGANISM = {
    "196627_v2020_s21_regNetwork_Strong": "Corynebacterium glutamicum",
    "83332_v2018_s15-16_regNetwork": "Mycobacterium tuberculosis",
    "224308_v2022_sSW22_regNetwork": "Bacillus subtilis",
    "511145_v2022_sRDB22_eStrong_regNetwork_Strong": "Escherichia coli",
    "208964_v2020_sRPA20_regNetwork_Strong": "Pseudomonas aeruginosa",
    "100226_v2019_sA22-DBSCR15_eStrong_regNetwork": "Streptomyces coelicolor",
}


def set_global_nord_style():
    """Configures Matplotlib global settings for the Nord aesthetic."""
    plt.style.use("default")

    # Let Seaborn dynamically scale all fonts, lines, and markers for a larger canvas.
    # 'talk' is usually perfect for presentations. Use 'poster' if it's still too small.
    sns.set_context("paper", font_scale=1.6)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Roboto", "Arial", "DejaVu Sans"],
            # You can remove the hardcoded font.size and labelsize here,
            # as sns.set_context handles the hierarchy automatically.
            "text.color": NORD_COLORS["dark"],
            "axes.labelcolor": NORD_COLORS["dark"],
            "axes.titlecolor": NORD_COLORS["dark"],
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": NORD_COLORS["gray"],
            "axes.linewidth": 1.5,
            "xtick.color": NORD_COLORS["dark"],
            "ytick.color": NORD_COLORS["dark"],
            # "grid.color": NORD_COLORS["gray"],
            # "grid.alpha": 0.3,
            "axes.grid": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": True,
        }
    )


#######################################################################################
### Pre-Processing
#######################################################################################


def load_metadata_dictionaries(network_name, coh_dir, go_dir):
    """
    Locates and loads the Teams and GO information for a specific network.
    Returns dictionaries mapping Node -> Metadata.
    """
    team_dict = {}
    tf_dict = {}

    # 1. Load Teams Data (Still sourced from coh_dir)
    team_path = coh_dir / network_name / f"{network_name}_Teams.csv"
    if team_path.exists():
        teams_df = pd.read_csv(team_path)
        team_dict = dict(
            zip(teams_df["Node"].astype(str).str.strip(), teams_df["PreSplitGroup"])
        )
    else:
        print(f"  -> Warning: Teams file not found at {team_path}")

    # 2. Load GO Data (TF Status) via Glob Search
    core_prefix = re.split(r"_eStrong|_regNetwork", network_name)[0]
    go_matches = list(go_dir.glob(f"{core_prefix}*_GOInformation.tsv"))

    if go_matches:
        go_path = go_matches[0]
        go_df = pd.read_csv(go_path, sep="\t")

        clean_nodes = go_df["Node"].astype(str).str.strip()
        is_tf_series = go_df["Is_TF_GO"].astype(str).str.strip().str.lower() == "true"
        tf_dict = dict(zip(clean_nodes, is_tf_series))
    else:
        print(
            f"  -> Warning: No GO Info file found for prefix '{core_prefix}' in {go_dir}"
        )

    return team_dict, tf_dict


def process_unique_mappings(unique_mappings, team_dict, tf_dict):
    """
    Parses node mappings (e.g., 'A:gene1, B:gene2'), extracts the ordered nodes,
    and queries the metadata dictionaries to generate the string identifiers.
    Returns mapping dictionaries for fast vectorization.
    """
    team_strs = {}
    tf_strs = {}
    tf_counts = {}

    for mapping in unique_mappings:
        parts = [p.strip() for p in mapping.split(",")]
        parts.sort(key=lambda x: x.split(":")[0].strip())

        nodes = [p.split(":")[1].strip() for p in parts]

        team_strs[mapping] = "_".join(str(team_dict.get(n, "Unknown")) for n in nodes)

        tf_bools = [tf_dict.get(n, False) for n in nodes]
        tf_strs[mapping] = "_".join("TF" if b else "NonTF" for b in tf_bools)
        tf_counts[mapping] = sum(tf_bools)

    return team_strs, tf_strs, tf_counts


def OLD_classify_motif_topology_OLD(man_code_str):
    """
    Classifies a triad MAN code string into 'Complete', 'Cyclic', or 'Feed-Forward'.
    Fails gracefully back to 'Complex' for dyads or invalid string inputs.
    """
    if not isinstance(man_code_str, str) or len(man_code_str) < 3:
        return "Complex"

    base_numeric = man_code_str[:3]

    # 1. Complete: All connections are mutual
    if base_numeric in ["300", "100"]:
        return "Complete"

    if base_numeric == "021" and "C" in man_code_str:
        return "Feed-Forward"

    # 3. Cyclic: Contains a 'C' suffix indicating a directional feedback loop (e.g., 030C)
    if "C" in man_code_str:
        return "Cyclic"

    # 4. Feed-Forward: Hierarchical layouts with shortcuts or clear acyclic splits
    # Captures 030T (FFL), 021D (Diverging), 021U (Converging), etc.
    if any(suffix in man_code_str for suffix in ["T", "D", "U"]):
        return "Feed-Forward"

    # Handle base sparse structures (e.g., single asymmetric edge 012 or two unlinked edges 021)
    if base_numeric in ["012", "021", "010"]:
        return "Feed-Forward"

    return "Complex"


def classify_motif_topology(man_code_str):
    if man_code_str == "120C":
        return "Complex"
    if not isinstance(man_code_str, str) or len(man_code_str) < 3:
        return "Complex"

    base_numeric = man_code_str[:3]

    if base_numeric in ["300", "100"]:
        return "Complete"

    # Intercept any motif containing mutual edges (first digit > 0)
    if int(base_numeric[0]) > 0:
        return "Complex"  # Can also be set to "Cyclic" depending on your grouping preference

    if base_numeric == "021" and "C" in man_code_str:
        return "Feed-Forward"

    if "C" in man_code_str:
        return "Cyclic"

    if any(suffix in man_code_str for suffix in ["T", "D", "U"]):
        return "Feed-Forward"

    if base_numeric in ["012", "021", "010"]:
        return "Feed-Forward"

    return "Complex"


def compile_extended_motif_counts():
    """
    Main pipeline: Iterates through each network, loads ONLY Topological Parquet files,
    calculates string dimensions efficiently, aggregates counts,
    and saves one comprehensive CSV per organism with structural class groupings.
    """
    INPUT_DIR = Path("./GRNMotifCounts_Targeted")
    COH_DIR = Path("./AbasyCohResults_Targeted")
    GO_DIR = Path("./GOInfo_Targeted")
    OUTPUT_DIR = Path("./AbasyNets_Extended_Counts")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting Extended Motif Pre-Processing (Topology Only)...")

    for net_dir in [d for d in INPUT_DIR.iterdir() if d.is_dir()]:
        network_name = net_dir.name
        organism_name = NETWORK_TO_ORGANISM.get(network_name, network_name)
        safe_org_name = organism_name.replace(" ", "_").replace("/", "_")

        raw_dir = net_dir / "MotifRawData"
        if not raw_dir.exists():
            continue

        parquet_files = list(raw_dir.glob("*_Topo_Raw.parquet"))
        if not parquet_files:
            continue

        print(f"\nProcessing Topological Network: {network_name} ({organism_name})")

        team_dict, tf_dict = load_metadata_dictionaries(network_name, COH_DIR, GO_DIR)
        network_dataframes = []

        for p_file in parquet_files:
            if pq.ParquetFile(p_file).metadata.num_rows == 0:
                continue

            df = pd.read_parquet(
                p_file,
                columns=[
                    "Motif",
                    "Level_String",
                    "NDA_String",
                    "Node_Mapping",
                ],
            )

            # Clean Circuit strings immediately
            df["Motif"] = df["Motif"].str.replace("_NS", "", regex=False)
            df["Motif"] = df["Motif"].str.replace("_", "-", regex=False)

            unique_mappings = df["Node_Mapping"].unique()
            team_map, tf_map, tf_count_map = process_unique_mappings(
                unique_mappings, team_dict, tf_dict
            )

            df["Team_String"] = df["Node_Mapping"].map(team_map)
            df["TF_String"] = df["Node_Mapping"].map(tf_map)
            df["Num_TFs"] = df["Node_Mapping"].map(tf_count_map)

            # Split out MAN components safely
            df["MAN_Code"] = df["Motif"].str.split("-").str[0]
            df["Topology_Class"] = df["MAN_Code"].map(classify_motif_topology)

            # --- NEW METRIC COLUMNS ---

            # 1. Structural size tracking from PN string length
            pn_strings = df["Motif"].str.split("-").str[1].fillna("")
            df["Motif_Size"] = np.where(pn_strings.str.len() == 6, 3, 2)

            # 2. Hierarchical structure role counts
            # Fill NaNs with empty strings to preserve safety during token counts
            level_filled = df["Level_String"].fillna("")
            df["Num_Input_Nodes"] = level_filled.str.count("Input")
            df["Num_Middle_Nodes"] = level_filled.str.count("Middle")
            df["Num_Output_Nodes"] = level_filled.str.count("Output")

            # 3. Unique NDA modules extraction
            # Splits strings like 'NDA1_NDA2_NDA1' at underscores to count distinct blocks
            df["Num_Unique_NDAs"] = (
                df["NDA_String"]
                .fillna("")
                .apply(lambda x: len(set(x.split(":"))) if x else 0)
            )
            df["Num_Unique_Teams"] = (
                df["Team_String"]
                .fillna("")
                .apply(lambda x: len(set(x.split("_"))) if x else 0)
            )

            df = df.rename(columns={"Motif": "Circuit"})

            group_cols = [
                "MAN_Code",
                "Topology_Class",
                "Circuit",
                "Motif_Size",
                "Level_String",
                "Num_Input_Nodes",
                "Num_Middle_Nodes",
                "Num_Output_Nodes",
                "NDA_String",
                "Num_Unique_NDAs",
                "Team_String",
                "Num_Unique_Teams",
                "TF_String",
                "Num_TFs",
            ]

            agg_df = df.groupby(group_cols).size().reset_index(name="Count")
            network_dataframes.append(agg_df)

            del df
            gc.collect()

        if network_dataframes:
            print(f"  -> Aggregating {len(network_dataframes)} Motif variants...")
            master_net_df = pd.concat(network_dataframes, ignore_index=True)

            final_group_cols = [
                "MAN_Code",
                "Topology_Class",
                "Circuit",
                "Motif_Size",
                "Level_String",
                "Num_Input_Nodes",
                "Num_Middle_Nodes",
                "Num_Output_Nodes",
                "NDA_String",
                "Num_Unique_NDAs",
                "Team_String",
                "Num_Unique_Teams",
                "TF_String",
                "Num_TFs",
            ]
            final_df = (
                master_net_df.groupby(final_group_cols)["Count"].sum().reset_index()
            )

            final_df = final_df.sort_values(
                by=["MAN_Code", "Circuit", "Count"],
                ascending=[True, True, False],
            ).reset_index(drop=True)

            out_file = OUTPUT_DIR / f"{safe_org_name}_Topo_Composition_Counts.csv"
            final_df.to_csv(out_file, index=False)

            print(
                f"  -> Saved {len(final_df)} compressed combinations to: {out_file.name}"
            )

            del master_net_df, final_df
            gc.collect()

    print("\nPre-processing complete. Topological Extended DataFrames generated.")


CHAR_MAP = {1: "P", -1: "N", 0: "0"}


def calculate_man_digits(edge_str):
    """
    Calculates the Mutual, Asymmetric, and Null edge pair counts
    directly from the flattened edge string.
    """
    if edge_str in ["Dropped_Node", "Invalid"]:
        return edge_str

    if len(edge_str) == 6:
        # Triad has 3 pairs: (A,B), (A,C), (B,C)
        pairs = [
            (edge_str[0], edge_str[1]),
            (edge_str[2], edge_str[3]),
            (edge_str[4], edge_str[5]),
        ]
    elif len(edge_str) == 2:
        # Dyad has 1 pair: (A,B)
        pairs = [(edge_str[0], edge_str[1])]
    else:
        return "Invalid"

    M = A = N = 0
    for p1, p2 in pairs:
        if p1 != "0" and p2 != "0":
            M += 1
        elif p1 == "0" and p2 == "0":
            N += 1
        else:
            A += 1

    return f"{M}{A}{N}"


def load_clean_coh_matrix(coh_path):
    """Loads and strictly binarizes the coherence matrix. Strips self-loops."""
    coh_df = pd.read_parquet(coh_path)

    if isinstance(coh_df.index, pd.MultiIndex):
        coh_df.index = coh_df.index.get_level_values(-1)
    if isinstance(coh_df.columns, pd.MultiIndex):
        coh_df.columns = coh_df.columns.get_level_values(-1)

    coh_df.index = coh_df.index.astype(str).str.strip()
    coh_df.columns = coh_df.columns.astype(str).str.strip()

    raw_vals = coh_df.values
    bin_mat = np.zeros_like(raw_vals, dtype=int)

    bin_mat[pd.isna(raw_vals)] = 0
    bin_mat[raw_vals > 0] = 1
    bin_mat[(raw_vals <= 0) & (~pd.isna(raw_vals))] = -1

    np.fill_diagonal(bin_mat, 0)

    return pd.DataFrame(bin_mat, index=coh_df.index, columns=coh_df.columns)


def parse_ordered_nodes(mapping_series):
    """Extracts nodes and sorts them strictly by their motif structural keys."""

    def _parse(val):
        if not isinstance(val, str):
            return tuple()
        parts = [p.strip().split(":") for p in val.split(",")]
        parts.sort(key=lambda x: x[0].strip())
        return tuple([p[1].strip() for p in parts])

    return mapping_series.map(_parse)


def execute_direct_transitions():
    INPUT_DIR = Path("./GRNMotifCounts_Targeted")
    COH_DIR = Path("./AbasyCohResults_Targeted")
    OUTPUT_DIR = Path("./AbasyNets_Extended_Counts")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Starting O(1) Matrix Transition Mapping ---")

    for net_dir in [d for d in INPUT_DIR.iterdir() if d.is_dir()]:
        network_name = net_dir.name
        organism_name = NETWORK_TO_ORGANISM.get(network_name, network_name)
        safe_org_name = organism_name.replace(" ", "_").replace("/", "_")

        raw_dir = net_dir / "MotifRawData"
        coh_path = COH_DIR / network_name / f"{network_name}_CohMat.parquet"

        if not raw_dir.exists() or not coh_path.exists():
            continue

        parquet_files = list(raw_dir.glob("*_Topo_Raw.parquet"))
        if not parquet_files:
            continue

        print(f"\nProcessing Transitions: {organism_name}")

        bin_df = load_clean_coh_matrix(coh_path)
        coh_nodes_set = set(bin_df.columns)

        topo_dfs = []
        for p_file in parquet_files:
            if pq.ParquetFile(p_file).metadata.num_rows == 0:
                continue
            df = pd.read_parquet(
                p_file, columns=["Motif", "Level_String", "Node_Mapping"]
            )
            topo_dfs.append(df)

        if not topo_dfs:
            continue

        master_df = pd.concat(topo_dfs, ignore_index=True)
        master_df["Node_Tuple"] = parse_ordered_nodes(master_df["Node_Mapping"])
        master_df = master_df.drop_duplicates(subset=["Node_Tuple"])

        master_df = master_df.rename(
            columns={"Motif": "Topo_Circuit", "Level_String": "Topo_Level_String"}
        )

        master_df["Topo_MAN_Code"] = master_df["Topo_Circuit"].str.split("_").str[0]
        master_df["Topo_EdgeString"] = master_df["Topo_Circuit"].str.split("_").str[1]

        coh_edge_str_list = []
        coh_man_list = []

        for topo_circuit, nodes in zip(
            master_df["Topo_Circuit"], master_df["Node_Tuple"]
        ):
            if not set(nodes).issubset(coh_nodes_set):
                coh_edge_str_list.append("Dropped_Node")
                coh_man_list.append("Dropped_Node")
                continue

            sub_mat = bin_df.loc[list(nodes), list(nodes)].values

            if len(nodes) == 3:
                edge_str = (
                    CHAR_MAP[sub_mat[0, 1]]
                    + CHAR_MAP[sub_mat[1, 0]]
                    + CHAR_MAP[sub_mat[0, 2]]
                    + CHAR_MAP[sub_mat[2, 0]]
                    + CHAR_MAP[sub_mat[1, 2]]
                    + CHAR_MAP[sub_mat[2, 1]]
                )
            elif len(nodes) == 2:
                edge_str = CHAR_MAP[sub_mat[0, 1]] + CHAR_MAP[sub_mat[1, 0]]
            else:
                edge_str = "Invalid"

            if edge_str in ["000000", "00"]:
                print("\n[ALERT - MOTIF DISAPPEARED]")
                print("Motif Name: {topo_circuit}")
                print("Nodes: {nodes}")
                print("---------------------------")

            coh_edge_str_list.append(edge_str)
            coh_man_list.append(calculate_man_digits(edge_str))

        master_df["Coh_EdgeString"] = coh_edge_str_list
        master_df["Coh_MAN_Base"] = coh_man_list

        conditions = [
            master_df["Coh_EdgeString"] == "Dropped_Node",
            master_df["Coh_EdgeString"].isin(["000000", "00"]),
            master_df["Topo_EdgeString"] == master_df["Coh_EdgeString"],
        ]
        choices = ["Dropped_Node", "Disappeared", "Preserved"]
        master_df["Status"] = np.select(conditions, choices, default="Altered")

        # Include Coh_MAN_Base in the final output aggregation
        group_cols = [
            "Topo_MAN_Code",
            "Topo_Circuit",
            "Topo_Level_String",
            "Coh_MAN_Base",
            "Coh_EdgeString",
            "Status",
        ]

        final_df = master_df.groupby(group_cols).size().reset_index(name="Count")
        final_df = final_df.sort_values(
            by=["Topo_MAN_Code", "Topo_Circuit", "Count"], ascending=[True, True, False]
        ).reset_index(drop=True)

        out_file = OUTPUT_DIR / f"{safe_org_name}_True_Transitions.csv"
        final_df.to_csv(out_file, index=False)
        print(f"  -> Processed {len(master_df)} circuits. Saved to {out_file.name}")

        del master_df, bin_df, final_df
        gc.collect()

    print("\nTransition Extraction Complete.")


def preprocess_shuffled_dataframe(shuffled_csv_path, run_index):
    """
    Loads a raw shuffled motif CSV, calculates structural proportions relative
    to that specific shuffle run's total counts first, and then collapses the
    proportions to the Topology_Class + MAN_Code level.
    """
    s_df = pd.read_csv(shuffled_csv_path)[["Motif", "Count"]]

    # Clean the raw shuffled motif strings to match formatting
    s_df["Motif"] = s_df["Motif"].astype(str).str.replace("_NS", "", regex=False)
    s_df["Motif"] = s_df["Motif"].str.replace("_", "-", regex=False)

    # Extract structural features
    s_df["MAN_Code"] = s_df["Motif"].str.split("-").str[0]
    s_df["Topology_Class"] = s_df["MAN_Code"].map(classify_motif_topology)

    # -------------------------------------------------------------------------
    # EARLY NORMALIZATION: Convert to proportions relative ONLY to this shuffle file
    # -------------------------------------------------------------------------
    total_shuffle_counts = s_df["Count"].sum()
    s_df["Run_Proportion"] = (
        s_df["Count"] / total_shuffle_counts if total_shuffle_counts > 0 else 0
    )

    # Collapse by summing up the independent proportions for this run
    collapsed_s_df = (
        s_df.groupby(["Topology_Class", "MAN_Code"])["Run_Proportion"]
        .sum()
        .reset_index()
    )

    # Keep the column naming consistent with the rest of your pipeline downstream
    collapsed_s_df = collapsed_s_df.rename(
        columns={"Run_Proportion": f"Random_Proportion_{run_index}"}
    )
    return collapsed_s_df


def analyse_pure_structural_enrichment(
    extended_counts_dir, motif_count_data_dir, output_csv
):
    """
    Calculates pure structural enrichment by normalizing actual and shuffled
    motifs to local proportions *before* collapsing and running statistical joins.
    """
    extended_dir = Path(extended_counts_dir)
    base_dir = Path(motif_count_data_dir)
    final_merged_frames = []

    net_list = [d.name for d in base_dir.iterdir() if d.is_dir()]
    print(f"Found {len(net_list)} networks to process for Structural Enrichment.")

    for net in net_list:
        print(f"\nProcessing: {net}")

        organism_name = NETWORK_TO_ORGANISM.get(net, net)
        safe_org_name = organism_name.replace(" ", "_").replace("/", "_")
        actual_csv_path = extended_dir / f"{safe_org_name}_Topo_Composition_Counts.csv"

        if not actual_csv_path.exists():
            print(
                f"  -> No actual topological counts found for {net} at {actual_csv_path.name}. Skipping."
            )
            continue

        # =====================================================================
        # 1. Load Actual Data & Compute Proportions Natively Before Grouping
        # =====================================================================
        actual_df = pd.read_csv(actual_csv_path)

        # Capture total actual counts across all configurations for this network
        total_actual_network_counts = actual_df["Count"].sum()

        # Map each individual row to its baseline share of the complete network
        actual_df["Raw_Proportion"] = (
            actual_df["Count"] / total_actual_network_counts
            if total_actual_network_counts > 0
            else 0
        )

        # Collapse strictly to structural properties by summing up the proportions
        actual_struct_df = (
            actual_df.groupby(["Topology_Class", "MAN_Code"])["Raw_Proportion"]
            .sum()
            .reset_index()
        )
        actual_struct_df = actual_struct_df.rename(
            columns={"Raw_Proportion": "Proportion"}
        )

        # Also preserve absolute counts for raw reporting in the final dataframe
        actual_counts_df = (
            actual_df.groupby(["Topology_Class", "MAN_Code"])["Count"]
            .sum()
            .reset_index()
        )
        actual_struct_df = pd.merge(
            actual_struct_df,
            actual_counts_df,
            on=["Topology_Class", "MAN_Code"],
            how="left",
        )

        # =====================================================================
        # 2. Load Preprocessed Shuffled Proportions
        # =====================================================================
        shuffled_motifcounts = sorted(
            list((base_dir / net / "ProcessedMotifCounts").glob("*MotifCounts.csv"))
        )

        if not shuffled_motifcounts:
            print("  -> No random shuffle files found. Skipping.")
            continue

        shuffled_df_list = []
        for idx, shuffled_csv_path in enumerate(shuffled_motifcounts, start=1):
            processed_s_df = preprocess_shuffled_dataframe(shuffled_csv_path, idx)
            shuffled_df_list.append(processed_s_df)

        # Merge processed randomized proportion profiles
        shuffled_df = reduce(
            lambda left, right: pd.merge(
                left, right, on=["Topology_Class", "MAN_Code"], how="outer"
            ),
            shuffled_df_list,
        )

        # Get all proportion column headers
        prop_cols = [cc for cc in shuffled_df.columns if "Random_Proportion_" in cc]

        # =====================================================================
        # 3. Merge Real vs Shuffled Proportions (Outer Join)
        # =====================================================================
        merged_net_df = pd.merge(
            actual_struct_df,
            shuffled_df,
            on=["Topology_Class", "MAN_Code"],
            how="outer",
        ).copy()

        # Handle missing properties smoothly
        merged_net_df["Count"] = merged_net_df["Count"].fillna(0)
        merged_net_df["Proportion"] = merged_net_df["Proportion"].fillna(0)
        merged_net_df["Network"] = net
        merged_net_df["Graph_Type"] = "Topo"

        fill_dict = {col: 0.0 for col in prop_cols}
        merged_net_df.fillna(value=fill_dict, inplace=True)

        # =====================================================================
        # 4. Calculate Empirical P-Values and Z-Scores Directly using Proportions
        # =====================================================================
        n_shuffles = len(prop_cols)

        # Metrics are calculated completely independent of raw network count scales
        emp_p_val_prop = (
            merged_net_df[prop_cols].ge(merged_net_df["Proportion"], axis=0).sum(axis=1)
            / n_shuffles
        )
        z_score_prop = (
            merged_net_df["Proportion"] - merged_net_df[prop_cols].mean(axis=1)
        ) / merged_net_df[prop_cols].std(axis=1).replace(0, pd.NA)

        # Backwards compatibility: Map proportion-derived metrics to both layout columns
        metrics_df = pd.DataFrame(
            {
                "Empirical_P_Value_Count": emp_p_val_prop,
                "Empirical_P_Value_Proportion": emp_p_val_prop,
                "Z_Score_Count": z_score_prop,
                "Z_Score_Proportion": z_score_prop,
            }
        )

        merged_net_df = pd.concat([merged_net_df, metrics_df], axis=1)

        # =====================================================================
        # 5. Final Output Ordering
        # =====================================================================
        base_cols = [
            "Network",
            "Graph_Type",
            "Topology_Class",
            "MAN_Code",
            "Count",
            "Proportion",
            "Empirical_P_Value_Count",
            "Empirical_P_Value_Proportion",
            "Z_Score_Count",
            "Z_Score_Proportion",
        ]

        all_cols = base_cols + prop_cols
        merged_net_df = merged_net_df[all_cols]

        merged_net_df = merged_net_df.sort_values(
            by=["Z_Score_Proportion", "Proportion"],
            ascending=[False, False],
            na_position="last",
        )

        final_merged_frames.append(merged_net_df)

    if final_merged_frames:
        master_enrichment_df = pd.concat(final_merged_frames, ignore_index=True)
        master_enrichment_df.to_csv(output_csv, index=False)
        print(
            f"\nSuccessfully compiled proportion-first structural enrichment data ({len(master_enrichment_df)} records)."
        )
        print(f"Saved to: {output_csv}")
        return master_enrichment_df
    else:
        print("\nNo shuffle data was processed.")
        return None


#######################################################################################
### Plotting
#######################################################################################


def plot_global_man_code_distribution_extremes_F(
    enrichment_csv, output_dir, metric="pvalue", mode="top", top_n=10, min_organisms=1
):
    """
    Plots the full cross-network distribution of individual MAN codes using a Boxplot + Stripplot.
    Filters for consistent motifs across organisms and sorts the x-axis by median values.
    Colors boxes and stripplot points by Topology Class and draws a legend.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(enrichment_csv)

    # 1. Process Chosen Analytical Metric Natively
    if metric == "pvalue":
        df["plot_metric"] = df["Empirical_P_Value_Proportion"]
        y_label = "Empirical $p$-value"
        cmap_list = [NORD_COLORS["purple"], NORD_COLORS["green"]]
    else:
        df["Z_Score_Proportion"] = df["Z_Score_Proportion"].replace(
            [np.inf, -np.inf], np.nan
        )
        max_real_z = df["Z_Score_Proportion"].dropna().max()
        df["plot_metric"] = df["Z_Score_Proportion"].fillna(
            max_real_z if not pd.isna(max_real_z) else 10.0
        )
        y_label = "Structural $Z$-Score"
        cmap_list = [NORD_COLORS["purple"], NORD_COLORS["green"]]

    # 2. Consistency Screening based on Top (Enrichment) vs Bottom (Depletion)
    shuffle_cols = [c for c in df.columns if "Random_Proportion_" in c]
    if mode == "top":
        valid_counts = df[df["Count"] > 0].groupby("MAN_Code")["Network"].nunique()
        consistent_motifs = valid_counts[valid_counts >= min_organisms].index.tolist()
    else:
        df["In_Background"] = df[shuffle_cols].sum(axis=1) > 0
        valid_backgrounds = (
            df[df["In_Background"]].groupby("MAN_Code")["Network"].nunique()
        )
        consistent_motifs = valid_backgrounds[
            valid_backgrounds >= min_organisms
        ].index.tolist()

    df_filtered = df[df["MAN_Code"].isin(consistent_motifs)].copy()

    if df_filtered.empty:
        print("Warning: No structural elements qualified with consistency settings.")
        return

    # 3. Global Imputation and Matrix Pivoting
    pivot_df = df_filtered.pivot_table(
        index="Network", columns="MAN_Code", values="plot_metric"
    ).fillna(0.0 if metric == "zscore" else 1.0)

    # 4. Extract Extreme Architectures Sorted Strictly by Median
    if metric == "pvalue":
        global_medians = pivot_df.median().sort_values(ascending=True)
        if mode == "top":
            selected_man_codes = global_medians.head(top_n).index.tolist()
        else:
            selected_man_codes = global_medians.tail(top_n).index.tolist()
    else:
        global_medians = pivot_df.median().sort_values(ascending=False)
        if mode == "top":
            selected_man_codes = global_medians.head(top_n).index.tolist()
        else:
            selected_man_codes = global_medians.tail(top_n).index.tolist()

    # 5. Generate Color Mappings Based on Topology Class
    topology_map = (
        df_filtered.drop_duplicates(subset=["MAN_Code"])
        .set_index("MAN_Code")["Topology_Class"]
        .to_dict()
    )

    class_color_rules = {
        "Complete": NORD_COLORS["red"],
        "Cyclic": NORD_COLORS["yellow"],
        "Feed-Forward": NORD_COLORS["green"],
        "Complex": NORD_COLORS["cyan"],
    }

    box_colors = [
        class_color_rules.get(topology_map.get(code), NORD_COLORS["dark"])
        for code in selected_man_codes
    ]

    # Melt selected components for plotting
    plot_df = (
        pivot_df[selected_man_codes]
        .reset_index()
        .melt(id_vars="Network", value_name="plot_metric")
    )

    # Map Topology Class back onto melted dataframe for stripplot hue matching
    plot_df["Topology_Class"] = plot_df["MAN_Code"].map(topology_map)

    # 6. Chart Setup
    fig, ax = plt.subplots(figsize=(6, 4))

    # 7. Render Symmetrical Reference Lines Behind Boxplots (zorder=0)
    if metric == "pvalue":
        ax.set_ylim(-0.05, 1.05)
        gradient = np.linspace(0, 1, 100).reshape(-1, 1)
        ax.imshow(
            gradient,
            aspect="auto",
            cmap=LinearSegmentedColormap.from_list(
                "bg_grad", [NORD_COLORS["green"], NORD_COLORS["purple"]]
            ),
            extent=[-0.5, len(selected_man_codes) - 0.5, 0.0, 1.00],
            origin="lower",
            alpha=0.85,
            zorder=0.0,
        )
    else:
        ax.axhline(
            0.0, color=NORD_COLORS["red"], linestyle="--", linewidth=1.5, zorder=0
        )

    # 8. Render Boxplot Base
    box_plot = sns.boxplot(
        data=plot_df,
        x="MAN_Code",
        y="plot_metric",
        order=selected_man_codes,
        width=0.6,
        showfliers=False,
        ax=ax,
    )

    for patch, color in zip(box_plot.patches, box_colors):
        patch.set_facecolor(color)

    # 9. Render Unlinked Stripplot colored by Topology Class
    sns.stripplot(
        data=plot_df,
        x="MAN_Code",
        y="plot_metric",
        hue="Topology_Class",
        hue_order=list(class_color_rules.keys()),
        palette=class_color_rules,
        order=selected_man_codes,
        size=8,
        alpha=0.75,
        jitter=0.18,
        edgecolor=NORD_COLORS["dark"],
        linewidth=0.6,
        legend=False,  # Suppress Seaborn's default legend to use the custom Patch layout instead
        ax=ax,
    )

    # 10. Frame Formatting & Legend Addition
    ax.set_xlabel("MAN Code")
    ax.set_ylabel(y_label)

    # Create custom legend handles using the rules dictionary mapping
    legend_elements = [
        Patch(facecolor=color, label=label, edgecolor=NORD_COLORS["dark"], alpha=0.9)
        for label, color in class_color_rules.items()
    ]
    ax.legend(
        handles=legend_elements,
        title="Topology Class",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        facecolor="white",
        edgecolor="none",
    )

    # Only add the gradient colorbar if we are in p-value mode
    if metric == "pvalue":
        # Create the same colormap used in your background gradient
        bg_cmap = LinearSegmentedColormap.from_list(
            "bg_grad", [NORD_COLORS["green"], NORD_COLORS["purple"]]
        )

        # Create a ScalarMappable for the colorbar mapping 0 to 1
        norm = plt.Normalize(vmin=0.0, vmax=1.0)
        sm = plt.cm.ScalarMappable(cmap=bg_cmap, norm=norm)
        sm.set_array([])

        # Add an inset axis for the colorbar directly below the legend bounding box
        # Coordinates are relative to the main axis: [left, bottom, width, height]
        # Placed at X=1.05 (matching legend), Y=0.35, Width=0.15, Height=0.04
        cax = ax.inset_axes([1.15, -0.08, 0.05, 0.50], transform=ax.transAxes)

        # Draw the horizontal colorbar
        cb = fig.colorbar(sm, cax=cax, orientation="vertical")

        # Format the colorbar ticks to match your metric narrative
        cb.set_ticks([0.0, 0.5, 1.0])
        cb.set_ticklabels(
            ["0.0 (Enriched)", "0.5\n(Random\nExpectation)", "1.0 (Depleted)"]
        )
        cb.ax.tick_params(rotation=0)
        cb.ax.set_title("Enrichment", pad=10, loc="Center")

    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()

    # 11. Save
    save_filename = f"Global_MAN_Extremes_{mode}_{metric}.png"
    save_path = out_path / save_filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(
        save_path.with_suffix(".svg"), dpi=300, bbox_inches="tight", transparent=True
    )
    print(f"Saved Median-Sorted MAN Code Extreme Distribution Plot to: {save_path}")

    plt.close()


def plot_circuit_compositions_for_extremes_F(
    extended_counts_dir, enrichment_csv, output_dir, top_n_motifs=5, min_proportion=0.01
):
    """

    Plots a single comprehensive chart showing the internal relative circuit composition

    of the top N most enriched structural MAN codes. Mirrors the Boxplot + Stripplot

    architecture of the global extremes distribution function.

    """

    out_path = Path(output_dir)

    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Isolate the top N enriched structural MAN codes via mean P-values

    enrich_df = pd.read_csv(enrichment_csv)

    topology_map = (
        enrich_df.drop_duplicates(subset=["MAN_Code"])
        .set_index("MAN_Code")["Topology_Class"]
        .to_dict()
    )

    class_color_rules = {
        "Complete": NORD_COLORS["red"],
        "Cyclic": NORD_COLORS["yellow"],
        "Feed-Forward": NORD_COLORS["green"],
        "Complex": NORD_COLORS["cyan"],
    }

    pivot_df = enrich_df.pivot_table(
        index="Network", columns="MAN_Code", values="Empirical_P_Value_Proportion"
    ).fillna(1.0)

    global_medians = pivot_df.mean().sort_values(ascending=True)

    enriched_man_targets = global_medians.head(top_n_motifs).index.tolist()

    print(
        f"Target Enriched Structures for Composition Breakdown: {enriched_man_targets}"
    )

    # 2. Gather, Aggregate, and Calculate Proportions for Individual Networks

    extended_path = Path(extended_counts_dir)

    composition_files = list(extended_path.glob("*_Topo_Composition_Counts.csv"))

    if not composition_files:
        print(f"Error: No composition CSV files found in {extended_counts_dir}")

        return

    all_org_frames = []

    for c_file in composition_files:
        org_df = pd.read_csv(c_file)

        if org_df.empty or "Count" not in org_df.columns:
            continue

        # Assign network label

        network_name = c_file.name.replace("_Topo_Composition_Counts.csv", "")

        org_df["Network"] = network_name

        agg_df = (
            org_df.groupby(["Network", "MAN_Code", "Circuit"])["Count"]
            .sum()
            .reset_index()
        )

        # CALCULATE TRUE PROPORTIONS

        total_network_count = agg_df["Count"].sum()

        if total_network_count == 0:
            continue

        agg_df["Network_Proportion"] = agg_df["Count"] / total_network_count

        # Filter to target MAN codes to save processing

        agg_df = agg_df[agg_df["MAN_Code"].isin(enriched_man_targets)].copy()

        # Calculate Relative Proportion (Portion of the Portion)

        cluster_prop_totals = agg_df.groupby("MAN_Code")[
            "Network_Proportion"
        ].transform("sum")

        agg_df["Relative_Proportion"] = (
            agg_df["Network_Proportion"] / cluster_prop_totals
        ).fillna(0.0)

        all_org_frames.append(agg_df)

    # Combine all processed matrices

    master_comp_df = pd.concat(all_org_frames, ignore_index=True)

    if master_comp_df.empty:
        print(
            "Warning: No granular circuit metrics matched the target enriched structural classes."
        )

        return

    # 3. Local Cluster Filter (Based on Median Relative Proportion)

    summary_df = (
        master_comp_df.groupby(["MAN_Code", "Circuit"])["Relative_Proportion"]
        .median()
        .reset_index(name="Median_Proportion")
    )

    valid_circuits = summary_df[summary_df["Median_Proportion"] >= min_proportion][
        "Circuit"
    ].tolist()

    plot_df = master_comp_df[master_comp_df["Circuit"].isin(valid_circuits)].copy()

    if plot_df.empty:
        print(
            f"Warning: No circuits remained after applying the >= {min_proportion} filter."
        )

        return

    # 4. Extract Extreme Architectures Sorted Strictly by Cluster and Median

    plot_df["MAN_Rank"] = plot_df["MAN_Code"].apply(
        lambda x: enriched_man_targets.index(x)
    )

    plot_df = pd.merge(plot_df, summary_df, on=["MAN_Code", "Circuit"], how="left")

    plot_df = plot_df.sort_values(
        by=["MAN_Rank", "Median_Proportion"], ascending=[True, False]
    ).reset_index(drop=True)

    # Establish explicit categorical ordering

    ordered_circuits = plot_df["Circuit"].drop_duplicates().tolist()

    plot_df["Circuit"] = pd.Categorical(
        plot_df["Circuit"], categories=ordered_circuits, ordered=True
    )

    # 6. Chart Setup
    fig_width = max(5, len(ordered_circuits) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))

    # 7. Render Boxplot Base
    box_plot = sns.boxplot(
        data=plot_df,
        x="Circuit",
        y="Relative_Proportion",
        order=ordered_circuits,
        width=0.4,
        showfliers=False,
        ax=ax,
    )

    circuit_colors = []
    for c in ordered_circuits:
        man_code = plot_df[plot_df["Circuit"] == c]["MAN_Code"].iloc[0]
        topo_class = topology_map.get(man_code, "Complex")
        circuit_colors.append(class_color_rules.get(topo_class, NORD_COLORS["dark"]))

    # Fill the boxes with the class colors and force edges/lines to a neutral dark tone
    for patch, color in zip(box_plot.patches, circuit_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(NORD_COLORS["dark"])
        patch.set_linewidth(1.1)

    for line in box_plot.lines:
        line.set_color(NORD_COLORS["dark"])
        line.set_linewidth(1.1)

    # 8. Render Unlinked Stripplot
    sns.stripplot(
        data=plot_df,
        x="Circuit",
        y="Relative_Proportion",
        order=ordered_circuits,
        color=NORD_COLORS["dark"],
        size=8,
        alpha=0.75,
        jitter=0.18,
        edgecolor=NORD_COLORS["dark"],
        ax=ax,
    )

    # 9. Draw Demarcation Region Lines
    boundary_indices = []
    unique_layout = (
        plot_df[["MAN_Code", "Circuit"]]
        .drop_duplicates()
        .sort_values("Circuit")
        .reset_index(drop=True)
    )

    current_man = unique_layout.loc[0, "MAN_Code"]

    for idx, row in unique_layout.iterrows():
        if row["MAN_Code"] != current_man:
            boundary_indices.append(idx - 0.5)

            current_man = row["MAN_Code"]

    for boundary in boundary_indices:
        ax.axvline(
            x=boundary,
            color=NORD_COLORS["gray"],
            linestyle="-.",
            linewidth=1.2,
            zorder=1,
        )

    ax.set_ylim(-0.05, 1.05)

    # bg_cmap = LinearSegmentedColormap.from_list(
    #     "bg_grad", [NORD_COLORS["green"], NORD_COLORS["purple"]]
    # )
    #
    # gradient = np.linspace(0, 1, 100).reshape(-1, 1)
    #
    # ax.imshow(
    #     gradient,
    #     aspect="auto",
    #     cmap=bg_cmap,
    #     extent=[-0.5, len(ordered_circuits) - 0.5, 0, 1],
    #     origin="lower",
    #     alpha=0.15,
    #     zorder=0.0,
    # )
    #
    # norm = plt.Normalize(vmin=0.0, vmax=1.0)
    # sm = plt.cm.ScalarMappable(cmap=bg_cmap, norm=norm)
    # sm.set_array([])
    # cax = ax.inset_axes([1.08, 0.02, 0.02, 0.40], transform=ax.transAxes)
    #
    # # Draw and format the vertical colorbar
    # cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    # cb.set_ticks([0.0, 0.5, 1.0])
    # cb.set_ticklabels(
    #     ["0.0 (Enriched)", "0.5\n(Random\nExpectation)", "1.0 (Depleted)"]
    # )
    # cb.ax.set_title("Enrichment", pad=10, loc="center")
    legend_elements = [
        Patch(
            facecolor=color, edgecolor=NORD_COLORS["dark"], linewidth=1.0, label=label
        )
        for label, color in class_color_rules.items()
    ]
    ax.legend(
        handles=legend_elements,
        title="Topology Class",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),  # Adjust anchor to clear the inset colorbar
        frameon=True,
        facecolor="white",
        edgecolor="none",
    )

    # 10. Frame Formatting
    ax.set_title(
        f"Granular Relative Circuit Composition\n(Top {top_n_motifs} Enriched Families, Local Medians $\geq$ {min_proportion * 100:.0f}%)"
    )
    ax.set_xlabel(
        # "Granular Signed Circuit Variant (Grouped Sequentially by Structural Class)"
        ""
    )
    ax.set_ylabel("Relative Frequency")
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()

    # 11. Save Output
    save_path = out_path / "Global_Granular_Circuit_Enriched_Composition_Sequenced.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(
        save_path.with_suffix(".svg"), dpi=300, bbox_inches="tight", transparent=True
    )
    print(f"Saved Filtered and Demarcated Enriched Composition Plot to: {save_path}")
    plt.close()


def plot_coherence_vs_relative_proportion_F(
    extended_counts_dir, enrichment_csv, coh_csv, output_dir, min_organisms=1
):
    """
    Calculates the mean relative proportion of circuits across consistent organisms,
    and plots this against their absolute structural coherence (|MeanCoh|).
    Underlays a fully opaque, screen-filling blue-to-purple KDE density map clipped between 0 and 1.
    Points are hollow with thick edges, styled consistently by Topology_Class.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Topology Class Mapping from Master Enrichment Data
    enrich_df = pd.read_csv(enrichment_csv)
    topo_map = (
        enrich_df.drop_duplicates("MAN_Code")
        .set_index("MAN_Code")["Topology_Class"]
        .to_dict()
    )

    # 2. Gather & Calculate Relative Proportions from Composition Files
    extended_path = Path(extended_counts_dir)
    composition_files = list(extended_path.glob("*_Topo_Composition_Counts.csv"))

    if not composition_files:
        print(f"Error: No composition CSV files found in {extended_counts_dir}")
        return

    all_org_frames = []
    for c_file in composition_files:
        org_df = pd.read_csv(c_file)
        if org_df.empty or "Count" not in org_df.columns:
            continue

        network_name = c_file.name.replace("_Topo_Composition_Counts.csv", "")
        org_df["Network"] = network_name

        # Aggregate locally to prevent duplicate row entries
        agg_df = (
            org_df.groupby(["Network", "MAN_Code", "Circuit"])["Count"]
            .sum()
            .reset_index()
        )

        total_network_count = agg_df["Count"].sum()
        if total_network_count == 0:
            continue

        agg_df["Network_Proportion"] = agg_df["Count"] / total_network_count
        all_org_frames.append(agg_df)

    master_df = pd.concat(all_org_frames, ignore_index=True)

    # 3. Consistency Filter (MAN Code must appear in >= min_organisms)
    man_org_counts = (
        master_df[master_df["Count"] > 0].groupby("MAN_Code")["Network"].nunique()
    )
    consistent_mans = man_org_counts[man_org_counts >= min_organisms].index.tolist()

    master_df = master_df[master_df["MAN_Code"].isin(consistent_mans)].copy()

    if master_df.empty:
        print("Warning: No structural elements qualified with consistency settings.")
        return

    # 4. Calculate Relative Proportions (Proportion of the Proportion)
    org_cluster_prop_totals = master_df.groupby(["Network", "MAN_Code"])[
        "Network_Proportion"
    ].transform("sum")
    master_df["Relative_Proportion"] = (
        master_df["Network_Proportion"] / org_cluster_prop_totals
    ).fillna(0.0)

    # Extract the cross-organism mean relative proportion for each circuit
    circuit_props = (
        master_df.groupby(["MAN_Code", "Circuit"])["Relative_Proportion"]
        .mean()
        .reset_index()
    )

    # Map the Topology Class to the grouped circuits
    circuit_props["Topology_Class"] = circuit_props["MAN_Code"].map(topo_map)
    circuit_props = circuit_props.dropna(subset=["Topology_Class"])

    # 5. Load & Format Coherence Data
    coh_df = pd.read_csv(coh_csv)

    # Filter strictly for Non-Self-Activating (NS) motifs
    coh_df = coh_df[coh_df["TopoName"].str.endswith("_NS", na=False)].copy()

    # Convert TopoName to Circuit string format (e.g., 010_N0_NS -> 010-N0)
    coh_df["Circuit"] = coh_df["TopoName"].str.replace("_NS", "").str.replace("_", "-")

    # Retain only circuits that have a valid computed MeanCoh value
    coh_df = coh_df[["Circuit", "MeanCoh"]].dropna()

    # Calculate Absolute Coherence
    coh_df["AbsMeanCoh"] = coh_df["MeanCoh"].abs()

    # 6. Merge Data Streams
    plot_df = pd.merge(circuit_props, coh_df, on="Circuit", how="inner")

    if plot_df.empty:
        print("Warning: Merged dataframe is empty. Check mapping string equivalence.")
        return

    # Sort to ensure stable layering/rendering
    plot_df = plot_df.sort_values(
        by=["Topology_Class", "Relative_Proportion"], ascending=[True, False]
    )

    # 7. Chart Geometry & Plotting
    fig, ax = plt.subplots(figsize=(6, 4))

    # Underlay the fully opaque, screen-filling KDE Density Map
    kde_cmap = LinearSegmentedColormap.from_list(
        "nord_blue_purple",
        [NORD_COLORS.get("blue", "#81a1c1"), NORD_COLORS.get("purple", "#b48ead")],
    )

    # clip=((-np.inf, np.inf), (0.0, 1.0)) locks the KDE to 0 and 1 on the Y-axis
    sns.kdeplot(
        data=plot_df,
        x="AbsMeanCoh",
        y="Relative_Proportion",
        fill=True,
        cmap=kde_cmap,
        alpha=1.0,
        thresh=0,  # Forces the lowest contour to fill the plot bounds
        levels=15,
        clip=((0.0, 1.0), (0.0, 1.0)),
        ax=ax,
        zorder=1,
    )

    # Explicit, consistent style mapping dictionary
    style_mapping = {
        "Feed-Forward": {
            "color": NORD_COLORS["green"],
            "marker": "^",
            "zorder": 3,
        },  # Green triangle
        "Cyclic": {
            "color": NORD_COLORS["yellow"],
            "marker": "s",
            "zorder": 3,
        },  # Yellow square
        "Complete": {
            "color": NORD_COLORS["red"],
            "marker": "o",
            "zorder": 3,
        },  # Red circle
        "Complex": {
            "color": NORD_COLORS["cyan"],
            "marker": "D",
            "zorder": 2,
        },  # Purple diamond
    }

    # Fallback palette for unmapped or generic topology classes
    fallback_colors = (
        NORD_PALETTE if "NORD_PALETTE" in globals() else ["#88c0d0", "#5e81ac"]
    )
    fallback_markers = ["v", "<", ">", "p", "*", "X"]

    unique_classes = plot_df["Topology_Class"].unique()

    # Plot each Topology Class with explicit style mappings
    for i, t_class in enumerate(unique_classes):
        class_df = plot_df[plot_df["Topology_Class"] == t_class]

        if t_class in style_mapping:
            class_color = style_mapping[t_class]["color"]
            class_marker = style_mapping[t_class]["marker"]
            class_zorder = style_mapping[t_class]["zorder"]
        else:
            # Dynamically style any outlier classes safely
            class_color = fallback_colors[i % len(fallback_colors)]
            class_marker = fallback_markers[i % len(fallback_markers)]
            class_zorder = 3

        ax.scatter(
            class_df["AbsMeanCoh"],
            class_df["Relative_Proportion"],
            marker=class_marker,
            facecolors="none",  # Keeps the marker hollow
            edgecolors=class_color,  # Colors the thick edge
            linewidths=2.8,
            s=75,
            alpha=0.95,
            label=t_class,
            zorder=class_zorder,
        )

    # Constrain axes padding so markers at 0.0 or 1.0 do not get cropped
    x_min, x_max = plot_df["AbsMeanCoh"].min(), plot_df["AbsMeanCoh"].max()
    y_min, y_max = (
        plot_df["Relative_Proportion"].min(),
        plot_df["Relative_Proportion"].max(),
    )
    ax.set_xlim(x_min - 0.03, x_max + 0.03)
    ax.set_ylim(y_min - 0.05, y_max + 0.05)

    ax.set_title(
        "Relative Motif Composition vs. Absolute Structural Coherence\n(Consistent Elements $\geq$ 1 Organisms)"
    )
    ax.set_xlabel(r"$| C_{\mathrm{struct}} |$")
    ax.set_ylabel("Mean Relative Frequency")

    # Legend formatting
    ax.legend(
        title="Topology Class",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        facecolor="#f2f4f8",
        edgecolor=NORD_COLORS.get("gray", "#4c566a"),
        framealpha=0.75,
        ncol=2 if plot_df["Topology_Class"].nunique() > 10 else 1,
    )

    plt.tight_layout()

    # 8. Save Output
    save_path = out_path / "Global_RelativeProportion_vs_AbsCoherence.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(
        save_path.with_suffix(".svg"), dpi=300, bbox_inches="tight", transparent=True
    )
    print(f"Saved Coherence vs. Proportion Plot to: {save_path}")

    plt.close()


def plot_organism_mean_middle_nodes_by_fate_F(transitions_df, output_dir):
    """
    Calculates the mean number of 'Middle' nodes per motif for each Organism,
    separated by transition fate (Preserved vs Altered).
    Plots a boxplot where each dot represents an individual organism's mean.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = transitions_df.copy()

    # 1. Extract number of Middle nodes
    df["Num_Middle_Nodes"] = df["Topo_Level_String"].astype(str).str.count("Middle")

    # 2. Filter for only Preserved and Altered fates
    df = df[df["Status"].isin(["Preserved", "Altered"])]

    if df.empty:
        print("Warning: No Preserved or Altered transitions found.")
        return

    # 3. Calculate the weighted mean per Organism per Status
    df["Total_Middle"] = df["Num_Middle_Nodes"] * df["Count"]

    agg_df = (
        df.groupby(["Organism", "Status"])
        .agg(Total_Motifs=("Count", "sum"), Total_Middle_Nodes=("Total_Middle", "sum"))
        .reset_index()
    )

    # The mean number of middle nodes per motif in this organism/status category
    agg_df["Mean_Middle_Nodes"] = agg_df["Total_Middle_Nodes"] / agg_df["Total_Motifs"]

    # 4. Plotting Setup
    fig, ax = plt.subplots(figsize=(3.5, 4))

    status_order = ["Preserved", "Altered"]
    status_palette = {
        "Preserved": NORD_COLORS["green"],
        "Altered": NORD_COLORS[
            "orange"
        ],  # Alternatively, use 'red' depending on exact Nord preference
    }

    # Base Boxplot
    sns.boxplot(
        data=agg_df,
        x="Status",
        y="Mean_Middle_Nodes",
        order=status_order,
        palette=status_palette,
        width=0.4,
        showfliers=False,
        linewidth=1.5,
        ax=ax,
    )

    # Stripplot overlaid (Dots represent Organisms)
    sns.stripplot(
        data=agg_df,
        x="Status",
        y="Mean_Middle_Nodes",
        order=status_order,
        color=NORD_COLORS["dark"],
        dodge=False,
        size=9,
        alpha=0.8,
        jitter=0.1,
        edgecolor=NORD_COLORS["dark"],
        linewidth=1.3,
        ax=ax,
    )

    # 5. Statistical Annotations
    try:
        annotator = Annotator(
            ax,
            [("Preserved", "Altered")],
            data=agg_df,
            x="Status",
            y="Mean_Middle_Nodes",
            order=status_order,
        )
        annotator.configure(
            test="Mann-Whitney",
            text_format="star",
            loc="inside",
            hide_non_significant=True,
            verbose=False,
            color=NORD_COLORS["dark"],
            line_width=1.5,
        )
        annotator.apply_and_annotate()
    except Exception as e:
        print(f"Stats annotation failed: {e}")

    # 6. Formatting
    # ax.set_title(
    #     "Global Mean 'Middle' Hierarchy Nodes by Motif Fate\n(Dots represent individual Organisms)"
    # )
    ax.set_xlabel("Transition Fate")
    ax.set_ylabel("Mean Middle Nodes per Motif")

    bottom, top = ax.get_ylim()
    ax.set_ylim(-0.05, top * 1.15)  # Give room for the stats annotation

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])

    plt.tight_layout()

    # 7. Save
    save_path = out_path / "Global_Mean_Middle_Nodes_by_Fate_Boxplot.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(
        save_path.with_suffix(".svg"), dpi=300, bbox_inches="tight", transparent=True
    )
    print(f"Saved Organism Mean Middle Nodes Plot to: {save_path}")
    plt.close()


# =====================================================================
# Analysis & Plotting for Matrix Transitions
# =====================================================================


def plot_topology_composition_by_dimensions_barplot_F(extended_counts_dir, output_dir):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_list = []
    for file in Path(extended_counts_dir).glob("*_Topo_Composition_Counts.csv"):
        df = pd.read_csv(file)
        df["Network"] = file.stem.replace("_Topo_Composition_Counts", "")
        df_list.append(df)

    if not df_list:
        print(f"No Topo Composition CSVs found in {extended_counts_dir}")
        return

    master_df = pd.concat(df_list, ignore_index=True)

    # Expanded to include Input, Middle, and Output node columns
    for col in [
        "Num_TFs",
        "Num_Unique_NDAs",
        "Num_Unique_Teams",
        "Num_Input_Nodes",
        "Num_Middle_Nodes",
        "Num_Output_Nodes",
    ]:
        master_df[col] = (
            pd.to_numeric(master_df[col], errors="coerce").fillna(0).astype(int)
        )

    # Expanded dimensions dictionary for plot labeling
    dimensions = {
        "Num_TFs": "Number of\nTranscription Factors (TFs)",
        "Num_Unique_NDAs": "Number of\nUnique NDAs",
        "Num_Unique_Teams": "Number of\nUnique Teams",
        "Num_Input_Nodes": "Number of\nInput Nodes",
        "Num_Middle_Nodes": "Number of\nMiddle Nodes",
        "Num_Output_Nodes": "Number of\nOutput Nodes",
    }

    topology_order = ["Feed-Forward", "Complete", "Cyclic", "Complex"]

    for dim_col, dim_label in dimensions.items():
        raw_agg = (
            master_df.groupby(["Network", "Topology_Class", dim_col])["Count"]
            .sum()
            .reset_index()
        )

        if raw_agg.empty:
            continue

        pivot_agg = raw_agg.pivot_table(
            index=["Network", "Topology_Class"],
            columns=dim_col,
            values="Count",
            fill_value=0,
        ).reset_index()

        agg_df = pivot_agg.melt(
            id_vars=["Network", "Topology_Class"], var_name=dim_col, value_name="Count"
        )

        network_class_totals = agg_df.groupby(["Network", "Topology_Class"])[
            "Count"
        ].transform("sum")
        agg_df = agg_df[network_class_totals > 0].copy()

        totals = agg_df.groupby(["Network", "Topology_Class"])["Count"].transform("sum")
        agg_df["Percentage"] = (agg_df["Count"] / totals) * 100
        agg_df[dim_col] = agg_df[dim_col].astype(str)

        hue_order = sorted(agg_df[dim_col].unique(), key=lambda x: int(x))
        current_order = [
            tc for tc in topology_order if tc in agg_df["Topology_Class"].values
        ]

        # Calculate dynamic width based on the number of active Topology Classes
        # num_x_items = len(current_order)

        # Minimum width of 6 inches, adding 2.0 inches per class
        # fig_width = max(5, num_x_items * 2.0)

        fig, ax = plt.subplots(figsize=(7, 5.5))

        sns.barplot(
            data=agg_df,
            x="Topology_Class",
            y="Percentage",
            hue=dim_col,
            order=current_order,
            hue_order=hue_order,
            palette=NORD_PALETTE,
            edgecolor=NORD_COLORS["dark"],
            capsize=0.1,
            width=0.6,
            err_kws={"linewidth": 1.5, "color": NORD_COLORS["dark"]},
            ax=ax,
        )

        sns.stripplot(
            data=agg_df,
            x="Topology_Class",
            y="Percentage",
            hue=dim_col,
            order=current_order,
            hue_order=hue_order,
            dodge=True,
            color=NORD_COLORS["dark"],
            size=7,
            alpha=0.8,
            jitter=0.15,
            ax=ax,
        )

        pairs = []
        for tc in current_order:
            valid_hues = []
            for h in hue_order:
                subset = agg_df[
                    (agg_df["Topology_Class"] == tc) & (agg_df[dim_col] == h)
                ]
                if len(subset) > 1:
                    valid_hues.append(h)

            for h1, h2 in itertools.combinations(valid_hues, 2):
                pairs.append(((tc, h1), (tc, h2)))

        if pairs:
            try:
                annotator = Annotator(
                    ax,
                    pairs,
                    data=agg_df,
                    x="Topology_Class",
                    y="Percentage",
                    order=current_order,
                    hue=dim_col,
                    hue_order=hue_order,
                )
                annotator.configure(
                    test="Mann-Whitney",
                    text_format="star",
                    loc="inside",
                    hide_non_significant=True,
                    verbose=False,
                )
                annotator.apply_and_annotate()
            except Exception as e:
                print(
                    f"Statannotations skipping annotations for {dim_col} due to internal variance limits: {e}"
                )

        ax.set_title(
            f"Per-Organism Composition of {dim_label}\nwithin Topology Classes"
        )
        ax.set_xlabel("Topology Class")
        ax.set_ylabel("Relative Composition (%)")

        handles, labels = ax.get_legend_handles_labels()
        n_hues = len(hue_order)

        ax.legend(
            handles[:n_hues],
            labels[:n_hues],
            title=dim_label,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=True,
            facecolor="#f2f4f8",
            edgecolor=NORD_COLORS["gray"],
        )

        current_bottom, current_top = ax.get_ylim()
        ax.set_ylim(-5, current_top)

        plt.tight_layout()

        save_path = out_path / f"Global_Topology_Barplot_by_{dim_col}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
        plt.savefig(
            save_path.with_suffix(".svg"),
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        print(f"Saved Grouped Topology Barplot to: {save_path}")
        plt.close()


def plot_topology_composition_by_dimensions_heatmap_F(extended_counts_dir, output_dir):
    """
    Plots a heatmap showing the mean relative composition of node attributes
    (0, 1, 2, 3... nodes) across different Topology Classes.
    Colors scale from Nord #4c566a to Nord Green.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_list = []
    for file in Path(extended_counts_dir).glob("*_Topo_Composition_Counts.csv"):
        df = pd.read_csv(file)
        df["Network"] = file.stem.replace("_Topo_Composition_Counts", "")
        df_list.append(df)

    if not df_list:
        print(f"No Topo Composition CSVs found in {extended_counts_dir}")
        return

    master_df = pd.concat(df_list, ignore_index=True)

    # Ensure targeted dimension columns are integers
    for col in [
        "Num_TFs",
        "Num_Unique_NDAs",
        "Num_Unique_Teams",
        "Num_Input_Nodes",
        "Num_Middle_Nodes",
        "Num_Output_Nodes",
    ]:
        master_df[col] = (
            pd.to_numeric(master_df[col], errors="coerce").fillna(0).astype(int)
        )

    dimensions = {
        "Num_TFs": "Transcription Factors (TFs)",
        "Num_Unique_NDAs": "Unique NDAs",
        "Num_Unique_Teams": "Unique Teams",
        "Num_Input_Nodes": "Input Nodes",
        "Num_Middle_Nodes": "Middle Nodes",
        "Num_Output_Nodes": "Output Nodes",
    }

    topology_order = ["Feed-Forward", "Complete", "Cyclic", "Complex"]

    # Custom Colormap: #4c566a -> Nord Green
    custom_cmap = LinearSegmentedColormap.from_list(
        "NordDarkGreyToGreen", ["#3b4252", NORD_COLORS["yellow"]]
    )

    for dim_col, dim_label in dimensions.items():
        # 1. Base Aggregation
        raw_agg = (
            master_df.groupby(["Network", "Topology_Class", dim_col])["Count"]
            .sum()
            .reset_index()
        )

        if raw_agg.empty:
            continue

        # 2. Pivot & Melt to zero-fill missing dimension counts per network
        pivot_agg = raw_agg.pivot_table(
            index=["Network", "Topology_Class"],
            columns=dim_col,
            values="Count",
            fill_value=0,
        ).reset_index()

        agg_df = pivot_agg.melt(
            id_vars=["Network", "Topology_Class"], var_name=dim_col, value_name="Count"
        )

        # 3. Remove network/topology combinations that have 0 motifs entirely
        network_class_totals = agg_df.groupby(["Network", "Topology_Class"])[
            "Count"
        ].transform("sum")
        agg_df = agg_df[network_class_totals > 0].copy()

        # 4. Calculate Percentage Composition per Network
        totals = agg_df.groupby(["Network", "Topology_Class"])["Count"].transform("sum")
        agg_df["Percentage"] = (agg_df["Count"] / totals) * 100

        # 5. Average the percentages across all networks
        mean_df = (
            agg_df.groupby(["Topology_Class", dim_col])["Percentage"]
            .mean()
            .reset_index()
        )

        # 6. Pivot into a 2D Heatmap Matrix
        heatmap_data = mean_df.pivot(
            index="Topology_Class", columns=dim_col, values="Percentage"
        ).fillna(0)

        # Reorder rows and columns logically
        current_rows = [tc for tc in topology_order if tc in heatmap_data.index]
        current_cols = sorted(heatmap_data.columns, key=lambda x: int(x))
        heatmap_data = heatmap_data.loc[current_rows, current_cols]

        # 7. Plot Setup
        fig, ax = plt.subplots(figsize=(4.5, 4))

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".1f",  # 1 decimal place for percentages
            cmap=custom_cmap,
            cbar_kws={"label": "Mean Relative Composition (%)"},
            linewidths=1.0,
            linecolor="white",  # Crisp separation
            ax=ax,
            vmin=0.0,
            vmax=100.0,
        )

        # 8. Formatting
        ax.set_title(f"{dim_label}")
        ax.set_xlabel(f"Number of {dim_label.replace('NDAs', 'Modules')}")
        # ax.set_ylabel("Topology Class")
        ax.set_ylabel("")

        plt.xticks(rotation=0)
        plt.yticks(rotation=0)

        # Draw frame borders
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(NORD_COLORS["dark"])

        plt.tight_layout()

        # 9. Save
        save_path = out_path / f"Global_Topology_Heatmap_by_{dim_col}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
        plt.savefig(
            save_path.with_suffix(".svg"),
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        print(f"Saved Grouped Topology Heatmap to: {save_path}")
        plt.close()


def plot_node_positions_by_topology_class_F(extended_counts_dir, output_dir):
    """
    Plots grouped bar charts (one per Topology Class) showing the relative composition
    of Node counts (0, 1, 2, 3...) across Input, Middle, and Output positions.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load and concatenate all composition data WITH Network tracking
    df_list = []
    for file in Path(extended_counts_dir).glob("*_Topo_Composition_Counts.csv"):
        df = pd.read_csv(file)
        df["Network"] = file.stem.replace("_Topo_Composition_Counts", "")
        df_list.append(df)

    if not df_list:
        print(f"No Topo Composition CSVs found in {extended_counts_dir}")
        return

    master_df = pd.concat(df_list, ignore_index=True)

    # Ensure integer types for the node count dimensions
    node_cols = {
        "Num_Input_Nodes": "Input Nodes",
        "Num_Middle_Nodes": "Middle Nodes",
        "Num_Output_Nodes": "Output Nodes",
    }

    for col in node_cols.keys():
        master_df[col] = (
            pd.to_numeric(master_df[col], errors="coerce").fillna(0).astype(int)
        )

    # 2. Process each Node Position type independently and concatenate
    all_node_data = []

    for dim_col, dim_label in node_cols.items():
        raw_agg = (
            master_df.groupby(["Network", "Topology_Class", dim_col])["Count"]
            .sum()
            .reset_index()
        )

        if raw_agg.empty:
            continue

        # Zero-fill missing counts to preserve valid 0% samples for stats
        pivot_agg = raw_agg.pivot_table(
            index=["Network", "Topology_Class"],
            columns=dim_col,
            values="Count",
            fill_value=0,
        ).reset_index()

        melted = pivot_agg.melt(
            id_vars=["Network", "Topology_Class"],
            var_name="Node_Count",
            value_name="Count",
        )

        # Filter out networks that have 0 motifs for this Topology Class overall
        totals = melted.groupby(["Network", "Topology_Class"])["Count"].transform("sum")
        melted = melted[totals > 0].copy()

        # Recalculate normalized percentage (must sum to 100% per Network/Topology)
        totals_clean = melted.groupby(["Network", "Topology_Class"])["Count"].transform(
            "sum"
        )
        melted["Percentage"] = (melted["Count"] / totals_clean) * 100
        melted["Node_Type"] = dim_label

        all_node_data.append(melted)

    if not all_node_data:
        return

    combined_df = pd.concat(all_node_data, ignore_index=True)
    combined_df["Node_Count"] = combined_df["Node_Count"].astype(str)

    topology_order = ["Feed-Forward", "Complete", "Cyclic", "Complex"]

    # 3. Generate one distinct plot PER Topology Class
    for tc in topology_order:
        tc_df = combined_df[combined_df["Topology_Class"] == tc].copy()

        if tc_df.empty:
            continue

        hue_order = sorted(tc_df["Node_Count"].unique(), key=lambda x: int(x))
        x_order = ["Input Nodes", "Middle Nodes", "Output Nodes"]
        current_x = [x for x in x_order if x in tc_df["Node_Type"].values]

        # 4. Plotting Setup
        fig, ax = plt.subplots(figsize=(6, 4))

        sns.barplot(
            data=tc_df,
            x="Node_Type",
            y="Percentage",
            hue="Node_Count",
            order=current_x,
            hue_order=hue_order,
            palette=NORD_PALETTE,
            edgecolor=NORD_COLORS["dark"],
            capsize=0.1,
            err_kws={"linewidth": 1.5, "color": NORD_COLORS["dark"]},
            ax=ax,
        )

        sns.stripplot(
            data=tc_df,
            x="Node_Type",
            y="Percentage",
            hue="Node_Count",
            order=current_x,
            hue_order=hue_order,
            dodge=True,
            color=NORD_COLORS["dark"],
            size=8,
            alpha=0.7,
            jitter=0.15,
            ax=ax,
        )

        # 5. Statistical Annotations
        pairs = []
        for nx in current_x:
            valid_hues = []
            for h in hue_order:
                # Require > 1 data point to run Mann-Whitney U test
                subset = tc_df[(tc_df["Node_Type"] == nx) & (tc_df["Node_Count"] == h)]
                if len(subset) > 1:
                    valid_hues.append(h)

            for h1, h2 in itertools.combinations(valid_hues, 2):
                pairs.append(((nx, h1), (nx, h2)))

        if pairs:
            try:
                annotator = Annotator(
                    ax,
                    pairs,
                    data=tc_df,
                    x="Node_Type",
                    y="Percentage",
                    order=current_x,
                    hue="Node_Count",
                    hue_order=hue_order,
                )
                annotator.configure(
                    test="Mann-Whitney",
                    text_format="star",
                    loc="inside",
                    hide_non_significant=True,
                    verbose=False,
                )
                annotator.apply_and_annotate()
            except Exception as e:
                print(
                    f"Statannotations skipping annotations for {tc} due to internal variance limits: {e}"
                )

        # 6. Formatting & Cleanup (Relying on set_global_nord_style)
        ax.set_title(f"Node Position Distribution within {tc} Topologies")
        ax.set_xlabel("Node Position in Hierarchy")
        ax.set_ylabel("Relative Composition (%)")

        # Deduplicate Legend
        handles, labels = ax.get_legend_handles_labels()
        n_hues = len(hue_order)

        ax.legend(
            handles[:n_hues],
            labels[:n_hues],
            title="Number of Nodes",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=True,
            facecolor="#f2f4f8",
            edgecolor=NORD_COLORS["gray"],
        )

        # Extend Y-axis slightly below 0 so dots at 0% are not clipped
        current_bottom, current_top = ax.get_ylim()
        ax.set_ylim(-5, current_top)

        plt.tight_layout()

        # 7. Save Output
        save_path = out_path / f"Topology_{tc}_NodePositions.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
        plt.savefig(
            save_path.with_suffix(".svg"),
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        print(f"Saved {tc} Node Position Plot to: {save_path}")
        plt.close()


def plot_node_positions_by_topology_class_heatmap_F(extended_counts_dir, output_dir):
    """
    Plots heatmaps (one per Topology Class) showing the mean relative composition
    of Node counts (0, 1, 2, 3...) across Input, Middle, and Output positions.
    Colors scale from Nord #3b4252 to Nord Yellow.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load and concatenate all composition data WITH Network tracking
    df_list = []
    for file in Path(extended_counts_dir).glob("*_Topo_Composition_Counts.csv"):
        df = pd.read_csv(file)
        df["Network"] = file.stem.replace("_Topo_Composition_Counts", "")
        df_list.append(df)

    if not df_list:
        print(f"No Topo Composition CSVs found in {extended_counts_dir}")
        return

    master_df = pd.concat(df_list, ignore_index=True)

    # Ensure integer types for the node count dimensions
    node_cols = {
        "Num_Input_Nodes": "Input",
        "Num_Middle_Nodes": "Middle",
        "Num_Output_Nodes": "Output",
    }

    for col in node_cols.keys():
        master_df[col] = (
            pd.to_numeric(master_df[col], errors="coerce").fillna(0).astype(int)
        )

    # 2. Process each Node Position type independently and concatenate
    all_node_data = []

    for dim_col, dim_label in node_cols.items():
        raw_agg = (
            master_df.groupby(["Network", "Topology_Class", dim_col])["Count"]
            .sum()
            .reset_index()
        )

        if raw_agg.empty:
            continue

        # Zero-fill missing counts to preserve valid 0% samples for stats
        pivot_agg = raw_agg.pivot_table(
            index=["Network", "Topology_Class"],
            columns=dim_col,
            values="Count",
            fill_value=0,
        ).reset_index()

        melted = pivot_agg.melt(
            id_vars=["Network", "Topology_Class"],
            var_name="Node_Count",
            value_name="Count",
        )

        # Filter out networks that have 0 motifs for this Topology Class overall
        totals = melted.groupby(["Network", "Topology_Class"])["Count"].transform("sum")
        melted = melted[totals > 0].copy()

        # Recalculate normalized percentage (must sum to 100% per Network/Topology)
        totals_clean = melted.groupby(["Network", "Topology_Class"])["Count"].transform(
            "sum"
        )
        melted["Percentage"] = (melted["Count"] / totals_clean) * 100
        melted["Node_Type"] = dim_label

        all_node_data.append(melted)

    if not all_node_data:
        return

    combined_df = pd.concat(all_node_data, ignore_index=True)
    combined_df["Node_Count"] = combined_df["Node_Count"].astype(str)

    # 3. Average the percentages across all networks
    mean_df = (
        combined_df.groupby(["Topology_Class", "Node_Type", "Node_Count"])["Percentage"]
        .mean()
        .reset_index()
    )

    topology_order = ["Feed-Forward", "Complete", "Cyclic", "Complex"]
    y_order = ["Input", "Middle", "Output"]

    # Custom Colormap: #3b4252 -> Nord Yellow
    custom_cmap = LinearSegmentedColormap.from_list(
        "NordDarkGreyToYellow", ["#3b4252", NORD_COLORS["light_blue"]]
    )

    # 4. Generate one distinct heatmap PER Topology Class
    for tc in topology_order:
        tc_df = mean_df[mean_df["Topology_Class"] == tc].copy()

        if tc_df.empty:
            continue

        # Pivot into a 2D Heatmap Matrix
        heatmap_data = tc_df.pivot(
            # index="Node_Type", columns="Node_Count", values="Percentage"
            index="Node_Count",
            columns="Node_Type",
            values="Percentage",
        ).fillna(0)

        # # Reorder rows and columns logically
        # current_rows = [y for y in y_order if y in heatmap_data.index]
        # current_cols = sorted(heatmap_data.columns, key=lambda x: int(x))
        # heatmap_data = heatmap_data.loc[current_rows, current_cols]

        # 5. Plotting Setup
        fig, ax = plt.subplots(figsize=(4, 4))

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".1f",  # 1 decimal place for percentages
            cmap=custom_cmap,
            cbar_kws={"label": "Mean Relative Composition (%)"},
            linewidths=1.0,
            linecolor="white",  # Crisp separation
            ax=ax,
            vmin=0.0,
            vmax=100.0,
        )

        # 6. Formatting
        ax.set_title(f"{tc}")
        ax.set_ylabel("Number of Nodes")
        ax.set_xlabel("Node Position in Hierarchy")

        plt.xticks(rotation=0)
        plt.yticks(rotation=0)

        # Draw frame borders
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(NORD_COLORS["dark"])

        plt.tight_layout()

        # 7. Save Output
        save_path = out_path / f"Topology_{tc}_NodePositions_Heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
        plt.savefig(
            save_path.with_suffix(".svg"),
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        print(f"Saved {tc} Node Position Heatmap Plot to: {save_path}")
        plt.close()


def plot_transition_fates_by_topology_class_barplot_F(transitions_df, output_dir):
    """
    Plots grouped bar charts with overlaid stripplots showing the relative composition
    of transition fates (Preserved, Altered, etc.) within each Topology Class across organisms.
    Includes statistical annotations (Mann-Whitney) comparing fates within classes.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Map Topo_MAN_Code to Topology_Class
    df = transitions_df.copy()
    df["Topology_Class"] = df["Topo_MAN_Code"].apply(classify_motif_topology)

    # 2. Aggregate counts by Organism, Topology Class, and Status
    # Note: load_all_transition_data assigns the network name to the 'Organism' column
    raw_agg = (
        df.groupby(["Organism", "Topology_Class", "Status"])["Count"]
        .sum()
        .reset_index()
    )

    if raw_agg.empty:
        print("Warning: No transition data available for topology mapping.")
        return

    # 3. Zero-fill missing statuses to preserve valid 0% samples for statistics
    pivot_agg = raw_agg.pivot_table(
        index=["Organism", "Topology_Class"],
        columns="Status",
        values="Count",
        fill_value=0,
    ).reset_index()

    agg_df = pivot_agg.melt(
        id_vars=["Organism", "Topology_Class"], var_name="Status", value_name="Count"
    )

    # Filter out Topology Classes that have 0 motifs across ALL statuses in a specific organism
    network_class_totals = agg_df.groupby(["Organism", "Topology_Class"])[
        "Count"
    ].transform("sum")
    agg_df = agg_df[network_class_totals > 0].copy()

    # 4. Normalize to 100% *within* each Organism and Topology Class
    totals = agg_df.groupby(["Organism", "Topology_Class"])["Count"].transform("sum")
    agg_df["Percentage"] = (agg_df["Count"] / totals) * 100

    # 5. Order definitions and palette mapping
    topology_order = ["Feed-Forward", "Complete", "Cyclic", "Complex"]
    current_order = [
        tc for tc in topology_order if tc in agg_df["Topology_Class"].values
    ]

    status_order = ["Preserved", "Altered", "Disappeared", "Dropped_Node"]
    current_hue_order = [s for s in status_order if s in agg_df["Status"].values]

    status_palette = {
        "Preserved": NORD_COLORS["green"],
        "Altered": NORD_COLORS["orange"],
        "Disappeared": NORD_COLORS["red"],
        "Dropped_Node": NORD_COLORS["gray"],
    }

    # 6. Plotting setup
    # Dynamic width based on the number of Topology Classes present
    # fig_width = max(5.8, len(current_order) * 0.9)
    fig, ax = plt.subplots(figsize=(6, 4))

    # Barplot Base
    sns.barplot(
        data=agg_df,
        x="Topology_Class",
        y="Percentage",
        hue="Status",
        order=current_order,
        hue_order=current_hue_order,
        palette=status_palette,
        edgecolor=NORD_COLORS["dark"],
        capsize=0.1,
        width=0.6,
        err_kws={"linewidth": 1.5, "color": NORD_COLORS["dark"]},
        ax=ax,
    )

    # Unlinked Stripplot (dodge=True aligns dots with the grouped bars)
    sns.stripplot(
        data=agg_df,
        x="Topology_Class",
        y="Percentage",
        hue="Status",
        order=current_order,
        hue_order=current_hue_order,
        dodge=True,
        color=NORD_COLORS["dark"],
        size=8,
        alpha=0.7,
        jitter=0.15,
        ax=ax,
    )

    # 7. Statistical Annotations
    pairs = []
    for tc in current_order:
        valid_hues = []
        for h in current_hue_order:
            # Require > 1 data point to run Mann-Whitney U test
            subset = agg_df[(agg_df["Topology_Class"] == tc) & (agg_df["Status"] == h)]
            if len(subset) > 1:
                valid_hues.append(h)

        for h1, h2 in itertools.combinations(valid_hues, 2):
            pairs.append(((tc, h1), (tc, h2)))

    if pairs:
        try:
            annotator = Annotator(
                ax,
                pairs,
                data=agg_df,
                x="Topology_Class",
                y="Percentage",
                order=current_order,
                hue="Status",
                hue_order=current_hue_order,
            )
            annotator.configure(
                test="Mann-Whitney", text_format="star", loc="inside", verbose=False
            )
            annotator.apply_and_annotate()
        except Exception as e:
            print(
                f"Statannotations skipping annotations due to internal variance limits: {e}"
            )

    # 8. Formatting
    ax.set_title("Global Motif Transition Fates by Topology Class")
    ax.set_xlabel("Topology Class")
    ax.set_ylabel("Relative Fate Composition (%)")

    # Deduplicate Legend
    handles, labels = ax.get_legend_handles_labels()
    n_hues = len(current_hue_order)

    ax.legend(
        handles[:n_hues],
        labels[:n_hues],
        title="Transition Status",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        edgecolor=NORD_COLORS["gray"],
    )

    # Extend Y-axis slightly below 0 so dots at 0% are not clipped
    current_bottom, current_top = ax.get_ylim()
    ax.set_ylim(-5, current_top)

    plt.tight_layout()

    # 9. Save Output
    save_path = out_path / "Global_Transition_Fates_Barplot_by_Topology_Class.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(
        save_path.with_suffix(".svg"), dpi=300, bbox_inches="tight", transparent=True
    )
    print(f"Saved Grouped Transition Fates Plot to: {save_path}")
    plt.close()


def plot_altered_transition_destinations_heatmap_F(
    transitions_df, output_dir, min_global_occurrences=1
):
    """
    Plots a row-normalized heatmap showing the destination structures (Coh_MAN_Base)
    for motifs that underwent an 'Altered' transition. Values sum to 1.0 per row.
    Source motifs are condensed to their 3-digit base.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Filter for only altered motifs globally
    altered_df = transitions_df[transitions_df["Status"] == "Altered"].copy()

    if altered_df.empty:
        print("Warning: No altered transitions found.")
        return

    # Condense Topo_MAN_Code to its 3-digit base to match Coh_MAN_Base format
    altered_df["Topo_MAN_Code"] = altered_df["Topo_MAN_Code"].astype(str).str[:3]

    # 2. Aggregate counts for Topo -> Coh transitions
    trans_counts = (
        altered_df.groupby(["Topo_MAN_Code", "Coh_MAN_Base"])["Count"]
        .sum()
        .reset_index()
    )

    # 3. Filter out rare starting Topo_MAN_Codes to avoid noisy 1.0 artifacts from single occurrences
    topo_totals = (
        trans_counts.groupby("Topo_MAN_Code")["Count"]
        .sum()
        .reset_index(name="Total_Altered")
    )
    valid_topos = topo_totals[topo_totals["Total_Altered"] >= min_global_occurrences][
        "Topo_MAN_Code"
    ]

    trans_counts = trans_counts[trans_counts["Topo_MAN_Code"].isin(valid_topos)]

    if trans_counts.empty:
        print(
            f"Warning: No altered transitions met the minimum occurrence threshold of {min_global_occurrences}."
        )
        return

    # 4. Pivot to transition matrix (Rows: Original, Cols: Destination)
    trans_matrix = trans_counts.pivot(
        index="Topo_MAN_Code", columns="Coh_MAN_Base", values="Count"
    ).fillna(0)

    # 5. Row-normalize to get fractions between 0.0 and 1.0
    trans_matrix_pct = trans_matrix.div(trans_matrix.sum(axis=1), axis=0)

    # 6. Plotting Setup
    fig, ax = plt.subplots(figsize=(6, 4.5))

    cmap = LinearSegmentedColormap.from_list(
        "nord_heatmap",
        [
            NORD_COLORS.get("dark", "#2e3440"),
            NORD_COLORS["red"],
        ],
    )

    sns.heatmap(
        trans_matrix_pct,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Fraction of Altered Fates"},
        linewidths=0.8,
        linecolor=NORD_COLORS.get("gray", "#3b4252"),
        ax=ax,
    )

    # 7. Formatting
    ax.set_title(
        "Transition Destinations of Altered Motifs\n(Row-Normalized Fractions)"
    )
    ax.set_xlabel("Destination Structure")
    ax.set_ylabel("Original Structure")

    plt.xticks(rotation=90, ha="center")
    plt.yticks(rotation=0)

    plt.tight_layout()

    # 8. Save Output
    save_path = out_path / "Global_Altered_Destinations_Heatmap.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(
        save_path.with_suffix(".svg"), dpi=300, bbox_inches="tight", transparent=True
    )
    print(f"Saved Altered Transitions Heatmap to: {save_path}")
    plt.close()


def load_all_transition_data(extended_counts_dir):
    """
    Loads and concatenates all *_True_Transitions.csv files into a single DataFrame.
    Enforces string typing for MAN codes to prevent pandas from dropping leading zeros (e.g., '030' -> 30).
    """
    df_list = []
    for file in Path(extended_counts_dir).glob("*_True_Transitions.csv"):
        # Reverse the safe_org_name formatting back to human-readable format
        org_name = file.stem.replace("_True_Transitions", "").replace("_", " ")

        # Explicitly declare dtypes to prevent integer casting of 030 -> 30
        df = pd.read_csv(
            file,
            dtype={
                "Topo_MAN_Code": str,
                "Coh_MAN_Base": str,
                "Topo_EdgeString": str,
                "Coh_EdgeString": str,
            },
        )
        df["Organism"] = org_name
        df_list.append(df)

    if not df_list:
        print(f"No transition CSVs found in {extended_counts_dir}")
        return pd.DataFrame()

    master_df = pd.concat(df_list, ignore_index=True)

    # Fallback safety: explicitly zero-pad string structures back to 3 characters
    for col in ["Topo_MAN_Code", "Coh_MAN_Base"]:
        master_df[col] = master_df[col].apply(
            lambda x: str(x).zfill(3)
            if pd.notna(x) and str(x) not in ["Dropped_Node", "Invalid", "nan"]
            else x
        )

    return master_df


# =====================================================================

if __name__ == "__main__":
    INPUT_DIR = Path("./GRNMotifCounts_Targeted/")
    OUTPUT_FILE = INPUT_DIR / "Master_Global_Motif_Counts.csv"
    PLOT_OUTPUT_DIR = Path("./GRN_Plots/Fig8")
    # COHERENCE_CSV_FILE = Path("./MotifCohResults/CompiledMotifSummary.csv")
    EXTENDED_COUNTS_DIR = Path("./AbasyNets_Extended_Counts")
    SHUFFLED_DIR = Path("./WTvsShuffledAnalysis_AbasyNets_Targeted/")

    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # set_global_nord_style()

    # # #####################################################################################
    # # ### Pre-Processing RUN ONCE
    # # #####################################################################################
    # # Compiling different types of motifs
    # compile_extended_motif_counts()
    # execute_direct_transitions()
    # analyse_pure_structural_enrichment(
    #     EXTENDED_COUNTS_DIR,
    #     SHUFFLED_DIR,
    #     EXTENDED_COUNTS_DIR / "Master_Global_Structural_Enrichment.csv",
    # )
    # # #####################################################################################
    # # #####################################################################################

    ENRICHMENT_CSV = EXTENDED_COUNTS_DIR / "Master_Global_Structural_Enrichment.csv"
    plot_global_man_code_distribution_extremes_F(
        enrichment_csv=ENRICHMENT_CSV,
        output_dir=PLOT_OUTPUT_DIR,
        metric="pvalue",
        mode="top",
        top_n=20,
        min_organisms=1,
    )
    plot_circuit_compositions_for_extremes_F(
        extended_counts_dir=EXTENDED_COUNTS_DIR,
        enrichment_csv=ENRICHMENT_CSV,
        output_dir=PLOT_OUTPUT_DIR,
        top_n_motifs=3,
    )
    COHERENCE_CSV = Path("./MotifCohResults/CompiledMotifSummary.csv")
    plot_coherence_vs_relative_proportion_F(
        extended_counts_dir=EXTENDED_COUNTS_DIR,
        enrichment_csv=ENRICHMENT_CSV,
        coh_csv=COHERENCE_CSV,
        output_dir=PLOT_OUTPUT_DIR,
        min_organisms=1,
    )

    # plot_topology_composition_by_dimensions_barplot_F(
    #     extended_counts_dir=EXTENDED_COUNTS_DIR, output_dir=PLOT_OUTPUT_DIR
    # )

    plot_topology_composition_by_dimensions_heatmap_F(
        extended_counts_dir=EXTENDED_COUNTS_DIR, output_dir=PLOT_OUTPUT_DIR
    )

    # plot_node_positions_by_topology_class_F(
    #     extended_counts_dir=EXTENDED_COUNTS_DIR, output_dir=PLOT_OUTPUT_DIR
    # )

    plot_node_positions_by_topology_class_heatmap_F(
        extended_counts_dir=EXTENDED_COUNTS_DIR, output_dir=PLOT_OUTPUT_DIR
    )

    transitions_df = load_all_transition_data(EXTENDED_COUNTS_DIR)

    plot_transition_fates_by_topology_class_barplot_F(transitions_df, PLOT_OUTPUT_DIR)

    plot_altered_transition_destinations_heatmap_F(
        transitions_df=transitions_df,
        output_dir=PLOT_OUTPUT_DIR,
        min_global_occurrences=1,
    )

    plot_organism_mean_middle_nodes_by_fate_F(
        transitions_df=transitions_df, output_dir=PLOT_OUTPUT_DIR
    )
