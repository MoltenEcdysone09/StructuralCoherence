import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path
from goatools.obo_parser import GODag
from goatools.go_enrichment import GOEnrichmentStudy
from bioservices import UniProt
from tqdm import tqdm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator
import ast
import io
import time
import re
import os


###################################################################################################
### Plotting Settings
###################################################################################################

# --- GLOBAL CONSTANTS ---
# Nord Palette Definition
NORD_COLORS = {
    "dark": "#2e3440",  # nord0: Darkest black/grey
    "gray": "#3b4252",  # nord1: Dark gray
    "red": "#bf616a",  # nord11
    "blue": "#5e81ac",  # nord10
    "green": "#a3be8c",  # nord14
    "yellow": "#ebcb8b",  # nord13
    "purple": "#b48ead",  # nord15
}

# Define the categorical palette list for use in all plots
NORD_PALETTE = [
    NORD_COLORS["red"],
    NORD_COLORS["blue"],
    NORD_COLORS["green"],
    NORD_COLORS["yellow"],
    NORD_COLORS["purple"],
]


def set_global_nord_style():
    """
    Configures Matplotlib global settings for the entire script run.
    """
    # Reset to default first to clear any conflicting states
    plt.style.use("default")

    # Update global parameters
    plt.rcParams.update(
        {
            # 1. Fonts
            "font.family": "sans-serif",
            "font.sans-serif": ["Roboto", "Arial", "DejaVu Sans"],
            "font.size": 16,
            # "font.weight": "bold",
            # 2. Text Colors
            "text.color": NORD_COLORS["dark"],
            "axes.labelcolor": NORD_COLORS["dark"],
            "axes.titlecolor": NORD_COLORS["dark"],
            # "axes.labelweight": "bold",
            # "axes.titleweight": "bold",
            # 3. Axes & Spines (The "Closed Box" Look)
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": NORD_COLORS["dark"],
            "axes.linewidth": 2.0,  # Thick Spines
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,  # No Grid
            # 4. Ticks
            "xtick.color": NORD_COLORS["dark"],
            "ytick.color": NORD_COLORS["dark"],
            "xtick.major.width": 2.0,
            "ytick.major.width": 2.0,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            # 5. Legend Defaults
            "legend.edgecolor": NORD_COLORS["dark"],
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": False,
            # 6. Global Line/Patch widths (helps with consistency)
            "lines.linewidth": 2.0,
            "patch.linewidth": 2.0,
        }
    )


# --- APPLY THE SETTINGS ---
set_global_nord_style()

###################################################################################################
###################################################################################################


# Ensure GO ontology is loaded globally
if "go_ontology" not in globals():
    go_ontology = GODag("go.obo")


def get_priority(filename):
    # Priority 1: Must contain BOTH "eStrong" AND "Strong"
    if "_eStrong" in filename and "_Strong" in filename:
        return 1
    # Priority 2: Contains "Strong" (but not "eStrong", as it was caught above)
    if "_Strong" in filename:
        return 2
    # Priority 3: Contains neither
    return 3


# Function to classify nodes
def classify_node(row):
    # Treat NaN as 0 for this logic
    out_val = row["MeanAbsOutCoh"] if pd.notna(row["MeanAbsOutCoh"]) else 0
    in_val = row["MeanAbsInCoh"] if pd.notna(row["MeanAbsInCoh"]) else 0

    if out_val == 0 and in_val != 0:
        return "Output"
    elif in_val == 0 and out_val != 0:
        return "Input"
    else:
        # This catches cases where both are 0/NaN
        # or both are non-zero
        return "Middle"


# ----------------------------------------------------------
# GO ontology classification
# ----------------------------------------------------------


def classify_go_annotations_generic(df, go_targets_dict, go_ontology_dag):
    """
    Adds boolean columns to the DataFrame indicating ancestry to target GO terms.

    Args:
        df (pd.DataFrame): DataFrame containing the list column 'GO_terms'.
        go_targets_dict (dict): Maps {Classification_Name: Target_GO_ID}.
        go_ontology_dag (GODag): The loaded GO ontology object.

    Returns:
        pd.DataFrame: DataFrame with new boolean columns (e.g., Is_TF_GO).
    """

    # Core logic: check ancestry to the target GO term
    def is_descendant_of(go_terms, target_go):
        if not isinstance(go_terms, list) or not go_terms:
            return False

        for term in go_terms:
            if term in go_ontology_dag:
                anc = go_ontology_dag[term].get_all_parents()
                if target_go in anc or term == target_go:
                    return True
        return False

    for name, go_id in go_targets_dict.items():
        # Construct sanitized column name (e.g., 'Signaling' -> 'Is_Signaling_GO')
        column_name = f"Is_{name.replace(' ', '_')}_GO"

        print(f"-> Classifying for: {name} (GO: {go_id}) into column '{column_name}'")

        tqdm.pandas(desc=f"Checking {name} Ancestry")

        # Apply the classification logic
        df[column_name] = df["GO_terms"].progress_apply(
            lambda terms: is_descendant_of(terms, go_id)
        )

    return df


# ----------------------------------------------------------
# Fetch GO terms directly from UniProt accession
# ----------------------------------------------------------


def fetch_go_terms_batched(uniprot_ids, batch_size=200):
    """
    Fetch GO terms from UniProt using search() in TSV mode.
    Returns: dict {uniprot_id: [GO terms]}
    """
    u = UniProt()

    ids = [str(x) for x in uniprot_ids if isinstance(x, str) and x not in ("", "nan")]
    all_results = {uid: [] for uid in ids}

    for i in tqdm(range(0, len(ids), batch_size), desc="Fetching GO (batched)"):
        batch = ids[i : i + batch_size]

        query = " OR ".join([f"accession:{x}" for x in batch])

        try:
            # res = u.search(query=query, frmt="tsv", columns="accession,go_id")
            res = u.search(query=query, frmt="tsv", columns="accession,go_f")
            # print(res)
        except Exception as e:
            print(f"[ERROR] UniProt batch failed for {batch[:5]}: {e}")
            continue

            # # Handle NONE response
            # if not res or not isinstance(res, str) or res.strip() == "":
            #     print(f"[WARNING] Empty response for batch {batch[:3]}...")
            #     continue

        # Check if response is valid (not empty, not just whitespace)
        if res and isinstance(res, str) and len(res.strip()) > 0:
            # 3. LOAD DIRECTLY INTO PANDAS
            # This replaces: lines = res.strip().split("\n") ...
            try:
                df_batch = pd.read_csv(io.StringIO(res), sep="\t")
            except pd.errors.EmptyDataError:
                continue  # Skip if batch was empty/malformed

            # Ensure we have data and the expected columns
            if not df_batch.empty and df_batch.shape[1] >= 2:
                # Rename columns to be safe (Accession is col 0, GO is col 1)
                df_batch.columns = ["Node", "GO_Raw"]

                # Handle missing values (NaN becomes empty string)
                df_batch["GO_Raw"] = df_batch["GO_Raw"].fillna("")

                # Extract GO terms into a list using regex
                df_batch["GO_List"] = df_batch["GO_Raw"].apply(
                    lambda x: re.findall(r"GO:\d+", str(x))
                )
                # print(df_batch)
                # print(df_batch.columns)

                # Convert to dictionary: { 'P123': ['GO:001', 'GO:002'], ... }
                # We discard the dataframe immediately to save memory
                batch_dict = pd.Series(
                    df_batch.GO_List.values, index=df_batch.Node
                ).to_dict()

                # Update master dictionary
                all_results.update(batch_dict)

        # Sleep to avoid rate limits
        time.sleep(0.5)
        # break
    return all_results


# Function to remove outliers based on IQR (matching boxplot logic)
def remove_outliers(df, x_col, y_col):
    clean_rows = []
    for group in df[x_col].unique():
        # Get data for this specific group
        group_data = df[df[x_col] == group]

        # Calculate Boxplot Stats
        q1 = group_data[y_col].quantile(0.10)
        q3 = group_data[y_col].quantile(0.90)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        # Keep only points within bounds
        clean_rows.append(
            group_data[
                (group_data[y_col] >= lower_bound) & (group_data[y_col] <= upper_bound)
            ]
        )

    return pd.concat(clean_rows)


# test = fetch_go_terms_batched(["O53661"], batch_size=1)
# print(test)

# Abasy network data
absy_dir = Path("./AbasyCohResults/")
gene_info_dir = Path("./AbasyNets/")
topos_dir = Path("./AbasyTOPOS/")

# PLto direcotry
plots_dir = absy_dir / "Plots_GO"
plots_dir.mkdir(exist_ok=True)


################################################################################################
## Coherence - Heirarchy - GO Related
################################################################################################


# cohmat_list = list(absy_dir.glob("*/*_Strong_CohMat.parquet"))
cohmat_list = sorted(list(absy_dir.glob("*/*_CohMat.parquet")))

# Getting the unique netowrk code
net_codes = set([str(nc.stem).split("_")[0] for nc in cohmat_list])
print(set(net_codes))
print(len(net_codes))

# nunduplicate cohmat list
nondup_cohmat_list = []

for nc in net_codes:
    print(nc)
    # Subet the networks which have th same names
    sub_list = sorted([n for n in cohmat_list if n.stem.startswith(nc)])
    if not sub_list:
        print(f"No networks found for {nc}")
        continue
    sel_network = min(sub_list, key=lambda path: get_priority(path.stem))
    nondup_cohmat_list.append(sel_network)
    print(f"Selected Network: {sel_network.name}")


# Create Plots folder if it doesn't exist
plots_dir = absy_dir / "Plots_GO"
plots_dir.mkdir(exist_ok=True)


# for cm in nondup_cohmat_list:
#     print(cm.stem)
#     # Read coherence matrix
#     cmat = pd.read_parquet(cm)
#     num_mat = cmat.select_dtypes(include="number")
#
#     # Row-wise and column-wise absolute sums
#     # row_sums = num_mat.abs().sum(axis=1)
#     # col_sums = num_mat.abs().sum(axis=0)
#     row_sums = np.nanmean(num_mat.abs(), axis=1)
#     col_sums = np.nanmean(num_mat.abs(), axis=0)
#
#     # Create summary DataFrame
#     inout_coh = pd.DataFrame(
#         # {"Node": num_mat.index, "OutCoh": row_sums.values, "InCoh": col_sums.values}
#         {
#             "Node": num_mat.index.get_level_values(1),
#             "MeanAbsOutCoh": row_sums,
#             "MeanAbsInCoh": col_sums,
#         }
#     )
#
#     # Extract only the gene name
#     inout_coh["Node"] = inout_coh["Node"].apply(
#         lambda x: x[1] if isinstance(x, tuple) else str(x)
#     )
#     inout_coh["NodeLevel"] = inout_coh.apply(classify_node, axis=1)
#
#     # Extractin diagnoal values from the dataframe
#     diag_vals = pd.Series(np.diag(cmat), index=cmat.index, name="DiagCohval").to_frame()
#     diag_vals = diag_vals.reset_index()
#     diag_vals = diag_vals.drop(columns="Group")
#
#     inout_coh = pd.merge(
#         inout_coh, diag_vals, how="left", left_on="Node", right_on="SourceNode"
#     )
#     inout_coh = inout_coh.drop(columns="SourceNode")
#
#     clean_stem = re.sub(
#         r"_regNetwork(?:_Strong)?_CohMat$",  # matches both Strong and non-Strong
#         "",
#         cm.stem,
#     )
#
#     gene_info_path = (
#         gene_info_dir
#         / f"{clean_stem}_regNet-genes-modules"
#         / f"{clean_stem}_geneInformation.tsv"
#     )
#
#     gene_info_df = pd.read_csv(gene_info_path, sep="\t", engine="python").rename(
#         columns={"Gene_name": "Node"}
#     )
#
#     # Merge the node info witht he inout_coh dataframe
#     inout_coh = pd.merge(inout_coh, gene_info_df, how="left", on="Node")
#
#     print(inout_coh)
#     print(inout_coh.dtypes)
#
#     # print(inout_coh["Uniprot_ID"].head(20))
#     # print(inout_coh["Uniprot_ID"].dropna().unique()[:20])
#
#     # Clean UniProt IDs (remove empty, 'nan', None)
#     inout_coh["Uniprot_ID_clean"] = inout_coh["Uniprot_ID"].replace(
#         ["", "nan", np.nan], None
#     )
#
#     unique_ids = inout_coh["Uniprot_ID_clean"].dropna().astype(str).unique()
#
#     go_map = fetch_go_terms_batched(unique_ids, batch_size=50)
#
#     if "go_ontology" not in locals():
#         print("Loading GO DAG...")
#         go_ontology = GODag("go.obo")
#
#     # 1. Define all target GO terms for classification
#     classification_targets = {
#         "TF": "GO:0140110",
#         "SignalingReceptor": "GO:0038023",
#         "HistKinase": "GO:0009927",
#         "SignallingTransducer": "GO:0060089",
#         # "Ribosome": "GO:0005840"
#     }
#
#     # 2. Map the dictionary to the DataFrame (Same as original Step 3)
#     print("Mapping GO terms to DataFrame...")
#     inout_coh["GO_terms"] = inout_coh["Uniprot_ID_clean"].apply(
#         lambda uid: go_map.get(uid, [])
#     )
#
#     # 3. Apply all classifications using the generic function
#     inout_coh = classify_go_annotations_generic(
#         inout_coh,
#         classification_targets,
#         go_ontology,  # The globally defined GODag object
#     )
#
#     # 4. Save the updated DataFrame
#     go_info_path = cm.parent / f"{clean_stem}_GOInformation.tsv"
#     inout_coh.to_csv(go_info_path, sep="\t", index=False)
#     print(f"Saved updated TSV: {go_info_path}")
#
#     # 5. Check the results (Now checks all new columns)
#     print("-" * 30)
#     print("Classification Results (Boolean Counts):")
#     for col in [
#         c for c in inout_coh.columns if c.startswith("Is_") and c.endswith("_GO")
#     ]:
#         print(f"{col}:\n", inout_coh[col].value_counts())
#     print("-" * 30)

#################################################################################################
# Heirarchy - TF Enrichment - GO Related
#################################################################################################

# ----------------------------------------------------------
# Modular Enrichment & Plotting Pipeline
# ----------------------------------------------------------


def calculate_enrichment(file_list, target_col):
    """
    Iterates through all network files and calculates Fisher's Exact Test
    for a specific boolean column (e.g., 'Is_TF_GO').
    """
    enrichment_results = []

    for io_fl in file_list:
        # Read Data
        io_df = pd.read_csv(io_fl, sep="\t", engine="python")

        # Skip if column doesn't exist in this specific file
        if target_col not in io_df.columns:
            continue

        # Ensure boolean type
        if io_df[target_col].dtype == object:
            io_df[target_col] = io_df[target_col].astype(str) == "True"

        network_name = io_fl.stem.replace("_GOInformation", "")

        # Iterate through NodeLevels (Input, Middle, Output)
        for level in io_df["NodeLevel"].dropna().unique():
            is_level = io_df["NodeLevel"] == level
            is_target = io_df[target_col]

            # Contingency Table
            a = len(io_df[is_level & is_target])  # Target Inside Level
            b = len(io_df[is_level & ~is_target])  # Non-Target Inside Level
            c = len(io_df[~is_level & is_target])  # Target Outside Level
            d = len(io_df[~is_level & ~is_target])  # Non-Target Outside Level

            if (a + b) == 0:
                continue

            # Fisher's Exact Test
            oddsratio, pvalue = stats.fisher_exact(
                [[a, b], [c, d]], alternative="greater"
            )

            enrichment_results.append(
                {
                    "Network": network_name,
                    "NodeLevel": level,
                    "Target_Col": target_col,
                    "Odds_Ratio": oddsratio,
                    "P_value": pvalue,
                }
            )

    return pd.DataFrame(enrichment_results)


def plot_enrichment(enrichment_df, target_name, output_dir):
    """
    Generates Boxplot + Stripplot with Statistical Annotations for a given enrichment DataFrame.
    Enforces Nord Palette and Input -> Middle -> Output ordering.
    """
    if enrichment_df.empty:
        print(f"Skipping plot for {target_name}: No data found.")
        return

    # 1. Setup Order and Colors
    # Enforce strict consistency: Input (Red), Middle (Blue), Output (Green)
    level_order = ["Input", "Middle", "Output"]

    # Map specifically to the first three colors of the global NORD_PALETTE
    # (Red, Blue, Green based on your definition)
    nord_level_palette = {
        "Input": NORD_PALETTE[0],  # Red
        "Middle": NORD_PALETTE[1],  # Blue
        "Output": NORD_PALETTE[2],  # Green
    }

    # Palette for Significance (Black for Sig, White for Non-Sig)
    # Using 'dark' from your NORD_COLORS dict for consistency
    sig_palette = {True: NORD_COLORS["dark"], False: "white"}

    # 2. Process Data
    df = enrichment_df.copy()

    # Filter to ensure we only have the expected levels (optional safety step)
    df = df[df["NodeLevel"].isin(level_order)]

    # Convert to Categorical to enforce sorting order in plots
    df["NodeLevel"] = pd.Categorical(
        df["NodeLevel"], categories=level_order, ordered=True
    )

    # Sort by Level then Odds Ratio
    df = df.sort_values(["NodeLevel", "Odds_Ratio"], ascending=[True, False])

    # BH-FDR Correction
    reject, pvals_corrected, _, _ = multipletests(df["P_value"], method="fdr_bh")
    df["Significant FDR"] = pvals_corrected < 0.05

    # 3. Setup Plot
    plt.figure(figsize=(6.5, 5))

    # Boxplot (Colored by Node Level using Nord Theme)
    sns.boxplot(
        data=df,
        x="NodeLevel",
        y="Odds_Ratio",
        order=level_order,  # Enforce X-axis order
        palette=nord_level_palette,  # Apply Nord colors
        showcaps=True,
        showfliers=False,
        # boxprops=dict(alpha=0.7),  # Slight transparency to let points pop
    )

    # Stripplot (Colored by Significance)
    sns.stripplot(
        data=df,
        x="NodeLevel",
        y="Odds_Ratio",
        order=level_order,
        hue="Significant FDR",
        palette=sig_palette,
        dodge=False,
        jitter=True,
        # alpha=0.7,
        edgecolor=NORD_COLORS["dark"],
        linewidth=1.2,
        # size=8,
        marker="o",
    )

    plt.ylabel("Odds Ratio")
    plt.xlabel("Node Level")
    plt.title(f"{target_name} Enrichment Across Node Levels")

    # 4. Statistical Annotations (Mann-Whitney between Levels)
    # Get the levels present in the data for correct pairing
    present_levels = [l for l in level_order if l in df["NodeLevel"].unique()]

    pairs = [
        (present_levels[i], present_levels[j])
        for i in range(len(present_levels))
        for j in range(i + 1, len(present_levels))
    ]

    if pairs:
        try:
            annotator = Annotator(
                ax=plt.gca(),
                pairs=pairs,
                data=df,
                x="NodeLevel",
                y="Odds_Ratio",
                order=level_order,
            )
            annotator.configure(
                test="Mann-Whitney", text_format="star", loc="inside", verbose=0
            )
            annotator.apply_and_annotate()
        except Exception as e:
            print(f"Could not annotate stats for {target_name}: {e}")

    # 5. Legend & Save
    # We need to manually handle the legend because we have two color schemes (Box vs Strip)

    # Create manual handles for Significance
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="p < 0.05",
            markerfacecolor=sig_palette[True],
            markeredgecolor=NORD_COLORS["dark"],
            # markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="p >= 0.05",
            markerfacecolor=sig_palette[False],
            markeredgecolor=NORD_COLORS["dark"],
            # markersize=10,
        ),
    ]

    plt.legend(
        handles=legend_elements,
        title="Significance (FDR)",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=True,  # Clean look
    )

    plt.tight_layout()
    save_path = output_dir / f"{target_name}_Enrichment.png"
    save_path_svg = save_path.with_suffix(".svg")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    # plt.show()
    plt.close()
    print(f"Saved plot: {save_path}")


# ----------------------------------------------------------
# Main Execution Block
# ----------------------------------------------------------

inout_df_list = sorted(list(absy_dir.glob("*/*_GOInformation.tsv")))

if inout_df_list:
    # 1. Detect all available "Is_..._GO" columns from the first file
    # (Assuming all files share the same structure)
    first_df = pd.read_csv(inout_df_list[0], sep="\t", engine="python")
    target_columns = [
        col for col in first_df.columns if col.startswith("Is_") and col.endswith("_GO")
    ]

    print(f"Found target columns: {target_columns}")

    # 2. Loop through each target and run the pipeline
    for col in target_columns:
        # Clean name for plotting (e.g., "Is_Signaling_GO" -> "Signaling")
        clean_name = col.replace("Is_", "").replace("_GO", "").replace("_", " ")

        print(f"Processing: {clean_name}...")

        # Step A: Calculate
        res_df = calculate_enrichment(inout_df_list, col)

        # Step B: Plot
        plot_enrichment(res_df, clean_name, plots_dir)

else:
    print("No _GOInformation.tsv files found to process.")

#################################################################################################
#################################################################################################


#################################################################################################
# Heirarchy - Module - GO Related
#################################################################################################

inout_df_list = sorted(list(absy_dir.glob("*/*_GOInformation.tsv")))

module_enrichments = []

for io_fl in tqdm(inout_df_list, desc="Processing Networks"):
    # Load data
    io_df = pd.read_csv(io_fl, sep="\t", engine="python")
    network_name = io_fl.stem.replace("_GOInformation", "")
    # Check if NDA_Component column exists
    if "NDA_component" not in io_df.columns:
        print(f"Skipping {network_name}: 'NDA_component' column missing.")
        continue

    # Clean up: drop nodes that have no assigned level or component (if any)
    clean_df = io_df.dropna(subset=["NodeLevel", "NDA_component"])

    # Get list of unique levels and unique components in this network
    unique_levels = clean_df["NodeLevel"].unique()
    unique_components = clean_df["NDA_component"].unique()

    # --- DOUBLE LOOP: LEVEL vs COMPONENT ---
    for comp in unique_components:
        for level in unique_levels:
            # Define Boolean Masks
            is_level = clean_df["NodeLevel"] == level
            is_comp = clean_df["NDA_component"] == comp

            a = len(clean_df[is_comp & is_level])  # Component X inside Level Y
            b = len(clean_df[is_comp & ~is_level])  # Other components inside Level Y
            c = len(clean_df[~is_comp & is_level])  # Component X outside Level Y
            d = len(clean_df[~is_comp & ~is_level])  # Other components outside Level Y

            # Skip if the component doesn't appear at all in this split (optional optimization)
            if (a + c) == 0:
                continue

            # Fisher Exact Test
            oddsratio, pvalue = stats.fisher_exact(
                [[a, b], [c, d]], alternative="greater"
            )

            module_enrichments.append(
                {
                    "Network": network_name,
                    "NodeLevel": level,
                    "Module": comp,
                    "Nodes_in_Intersection": a,
                    "Total_Nodes_in_Module": a + b,
                    "P_value": pvalue,
                    "Odds_Ratio": oddsratio,
                    "Percentage_in_Module": (a / (a + b) * 100) if (a + b) > 0 else 0,
                }
            )

# Convert to DataFrame
module_enrich_df = pd.DataFrame(module_enrichments)

# FDR Correction (Benjamini-Hochberg)
# We correct across ALL tests (All networks * All Levels * All Components)
module_enrich_df["P_value_FDR"] = multipletests(
    module_enrich_df["P_value"], method="fdr_bh"
)[1]
module_enrich_df["Significant FDR"] = module_enrich_df["P_value_FDR"] < 0.05

# Remove the outliers
module_enrich_df = remove_outliers(module_enrich_df, "NodeLevel", "Odds_Ratio")

print(module_enrich_df["Significant FDR"].value_counts(normalize=True))


# Palette for significance (True = sig)
palette = {True: "black", False: "white"}

plt.figure(figsize=(8, 6))

# Create the Scatter Plot
sns.scatterplot(
    data=module_enrich_df,
    x="P_value_FDR",
    y="Odds_Ratio",
    hue="Significant FDR",  # Color points by significance (True/False)
    style="NodeLevel",  # Different shapes for Input, Middle, Output
    palette={True: "red", False: "grey"},  # Red for sig, Grey for non-sig
    markers=True,  # Ensure markers are used for styles
    s=60,  # Size of dots
    alpha=0.8,
    edgecolor="black",
)

# Placed at 0.05 on the X-axis
plt.axvline(x=0.05, color="black", linestyle="--", linewidth=1.5, label="FDR = 0.05")

plt.xscale("log")
# plt.yscale("log")

# Labels and Titles
plt.xlabel("FDR p-value (log scale)")
plt.ylabel("Odds Ratio")
plt.title("Module Enrichment Volcano Plot")

# Move legend outside
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()

# Save and Show
plt.savefig(plots_dir / "Module_Volcano_Plot.png", dpi=300)
plt.savefig(plots_dir / "Module_Volcano_Plot.svg", dpi=300, transparent=True)
plt.close()
# plt.show()

#################################################################################################

level_order = ["Input", "Middle", "Output"]

# Ensure categorical ordering
module_enrich_df["NodeLevel"] = pd.Categorical(
    module_enrich_df["NodeLevel"], categories=level_order, ordered=True
)

# Sort for consistency
module_enrich_df = module_enrich_df.sort_values(["NodeLevel", "Odds_Ratio"])

# 2. Define Palettes
# Map Levels to Nord Colors for the Boxplot
nord_level_palette = {
    "Input": NORD_PALETTE[0],  # Red
    "Middle": NORD_PALETTE[1],  # Blue
    "Output": NORD_PALETTE[2],  # Green
}

# Map Significance to Black/White for the Stripplot
# Using 'dark' from NORD_COLORS (nord0) for consistency
sig_palette = {True: NORD_COLORS["dark"], False: "white"}

# 3. Plotting
plt.figure(figsize=(6.5, 5))

# --- Boxplot (Nord Colors by Level) ---
sns.boxplot(
    data=module_enrich_df,
    x="NodeLevel",
    y="Odds_Ratio",
    order=level_order,  # Enforce X-axis order
    palette=nord_level_palette,  # Apply Nord colors
    showcaps=True,
    width=0.6,
    showfliers=False,
    # boxprops=dict(alpha=0.7),  # Slight transparency
)

# --- Overlay Stripplot (Black/White by Significance) ---
sns.stripplot(
    data=module_enrich_df,
    x="NodeLevel",
    y="Odds_Ratio",
    order=level_order,  # Must match boxplot order
    hue="Significant FDR",  # Color by Significance
    palette=sig_palette,
    dodge=False,
    jitter=True,
    alpha=1.0,
    edgecolor=NORD_COLORS["dark"],
    linewidth=0.9,
    size=7,
    marker="o",
)

# plt.yscale("log") # Uncomment if needed
plt.ylabel("Odds Ratio")
plt.xlabel("Node Level")
plt.title("Module Enrichment Across Node Levels")

# 4. Statistical Annotations
# Filter pairs to only include levels present in the data
present_levels = [l for l in level_order if l in module_enrich_df["NodeLevel"].unique()]

if len(present_levels) >= 2:
    print(f"Annotating stats for pairs in: {present_levels}")

    # Generate pairs respecting the strict order
    pairs = [
        (present_levels[i], present_levels[j])
        for i in range(len(present_levels))
        for j in range(i + 1, len(present_levels))
    ]

    # Setup the Annotator
    try:
        annotator = Annotator(
            ax=plt.gca(),
            pairs=pairs,
            data=module_enrich_df,
            x="NodeLevel",
            y="Odds_Ratio",
            order=level_order,  # Critical to match plot order
        )
        annotator.configure(
            test="Mann-Whitney", text_format="star", loc="inside", verbose=0
        )
        annotator.apply_and_annotate()
    except Exception as e:
        print(f"Error during stat annotation drawing: {e}")

else:
    print("Skipping statistical annotation: insufficient unique levels.")


legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="p < 0.05",
        markerfacecolor=sig_palette[True],
        markeredgecolor=NORD_COLORS["dark"],
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="p >= 0.05",
        markerfacecolor=sig_palette[False],
        markeredgecolor=NORD_COLORS["dark"],
        markersize=10,
    ),
]

plt.legend(
    handles=legend_elements,
    title="Significance (FDR)",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    frameon=True,
    # alignment="center",
)

plt.tight_layout()
save_path = plots_dir / "Module_Enrichment.png"
save_path_svg = save_path.with_suffix(".svg")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.savefig(save_path_svg, dpi=300, transparent=True)
# plt.show()
plt.close()
print(f"Saved plot: {save_path}")


#################################################################################################
#################################################################################################


#################################################################################################
#### Node Level Enrichment
#################################################################################################


def get_level_enrichment_from_tsv(io_df, network_name, go_ontology):
    """
    Performs GO enrichment analysis for all NodeLevels using data already loaded from TSV.
    Returns a list of dictionaries with enrichment results for one network.
    """
    level_enrichment_results = []

    # Filter out rows with missing GO terms or UIDs
    valid_data = io_df.dropna(subset=["Uniprot_ID_clean", "GO_terms"]).copy()

    # 1. Define the GO Association Map (UID -> GO Terms)
    # This uses the cleaned data column 'GO_terms'
    # go_associations = dict(zip(valid_data["Uniprot_ID_clean"], valid_data["GO_terms"]))
    go_associations = {
        uid: set(terms)
        for uid, terms in zip(valid_data["Uniprot_ID_clean"], valid_data["GO_terms"])
        if terms  # Ensures we skip empty lists/none
    }

    # 2. Define the Population (Background) - All UIDs in the current network
    population_set = set(valid_data["Uniprot_ID_clean"].unique())

    # 3. Iterate through each Node Level
    for level in valid_data["NodeLevel"].dropna().unique():
        # Define the Study Set (This contains the UIDs)
        study_set = set(
            valid_data[valid_data["NodeLevel"] == level]["Uniprot_ID_clean"]
        )

        # Skip if the study set is empty
        if not study_set:
            continue

        # 4. Initialize the GO Enrichment Study Object (The FIX)
        # This object is where the 'find_enrichment' method resides.
        goea_obj = GOEnrichmentStudy(
            population_set,  # The background set of UIDs
            go_associations,  # The UID-to-GO term map
            go_ontology,  # The GODag object
            propagate_counts=True,
            alpha=0.05,  # Initial alpha level
            methods=["fdr_bh"],  # Methods for multiple testing correction
        )

        # 5. Run GO Enrichment Analysis
        # The find_enrichment function is a method of the GOEA object, not the GODag object.
        goea_results = goea_obj.run_study(
            study_set,
            verbose=False,
        )

        # Get the total counts for the denominator (Study size and Population size)
        study_total = len(study_set)
        pop_total = len(population_set)

        # Filter for significantly enriched terms
        significant_results = [r for r in goea_results if r.p_fdr_bh < 0.05]

        # Print selected information for significant terms
        for result in significant_results:
            calc_study_ratio = result.study_count / study_total
            calc_pop_ratio = result.pop_count / pop_total
            level_enrichment_results.append(
                {
                    "Network": network_name,
                    "NodeLevel": level,
                    "GO_ID": result.GO,
                    "GO_Term": result.name,
                    "FDR": result.p_fdr_bh,
                    "P_value": result.p_uncorrected,
                    "Study_Count": result.study_count,
                    "Pop_Count": result.pop_count,
                    "Study_Ratio": calc_study_ratio,
                    "Pop_Ratio": calc_pop_ratio,
                }
            )

    return level_enrichment_results


print("\n--- Starting GO Enrichment from Saved TSV Files ---")

inout_df_list = sorted(list(absy_dir.glob("*/*_GOInformation.tsv")))
master_enrichment_list = []


for io_fl in tqdm(inout_df_list, desc="Processing TSV Files"):
    # Load data
    io_df = pd.read_csv(io_fl, sep="\t", engine="python")

    # CRITICAL: Convert the string representation of lists back into actual lists
    # This happens because the 'GO_terms' column was saved as text.
    io_df["GO_terms"] = io_df["GO_terms"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

    network_name = io_fl.stem.replace("_GOInformation", "")

    # Run the efficient, TSV-based enrichment
    level_results = get_level_enrichment_from_tsv(io_df, network_name, go_ontology)
    master_enrichment_list.extend(level_results)

# Final step: Save all combined results
if master_enrichment_list:
    final_enrichment_df = pd.DataFrame(master_enrichment_list)
    final_enrichment_df.to_csv(
        absy_dir / "GO_Level_Enrichment.tsv", sep="\t", index=False
    )
    print(
        f"\nSuccessfully saved Master GO Enrichment Results ({len(final_enrichment_df)} total results)."
    )
else:
    print("\nNo GO enrichment results were generated.")


def filter_generic_go_terms(
    df, go_ontology_dag, min_depth=5, max_depth=None, exclude_namespaces=None
):
    """
    Removes GO terms based on namespace, minimum depth (too generic), and
    maximum depth (too specific).

    Args:
        df (pd.DataFrame): The enrichment results DataFrame.
        go_ontology_dag (GODag): The loaded GO ontology object.
        min_depth (int): Minimum depth required for a term (excludes generic).
        max_depth (int | None): Maximum depth allowed for a term (excludes specific).
        exclude_namespaces (list): List of namespaces to remove.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if exclude_namespaces is None:
        exclude_namespaces = []

    print(f"Initial terms: {len(df)}")

    # Helper function to check all criteria for a single GO ID
    def check_term_specificity(go_id):
        # Check if the GO ID exists in the loaded DAG
        if go_id not in go_ontology_dag:
            return False, 0

        term_record = go_ontology_dag[go_id]
        term_level = term_record.level

        # 1. Check Namespace Exclusion
        if term_record.namespace in exclude_namespaces:
            return False, term_level

        # 2. Check MIN Depth (Too Generic)
        if term_level < min_depth:
            return False, term_level

        if max_depth is not None and term_level > max_depth:
            return False, term_level

        return True, term_level  # Passed all checks

    # Apply the check to the DataFrame
    # Creates two temporary columns: Pass_Filter (bool) and Term_Level (int)
    df[["Pass_Filter", "Term_Level"]] = df["GO_ID"].apply(
        lambda x: pd.Series(check_term_specificity(x))
    )

    # Filter down to the passing terms
    filtered_df = df[df["Pass_Filter"]].drop(columns=["Pass_Filter", "Term_Level"])

    print(
        f"Terms remaining after filtering (Depth {min_depth}-{max_depth} / Excl NS {exclude_namespaces}): {len(filtered_df)}"
    )

    return filtered_df


# Read the master enrichment_df
final_enrichment_df = pd.read_csv(absy_dir / "GO_Level_Enrichment.tsv", sep="\t")
print(final_enrichment_df.columns)
final_enrichment_df = final_enrichment_df[
    final_enrichment_df["Study_Count"] >= 3
].copy()
final_enrichment_df["Fold_Enrichment"] = (
    final_enrichment_df["Study_Ratio"] / final_enrichment_df["Pop_Ratio"]
)
# Create specificity score (Information content style)
final_enrichment_df["Specificity"] = -np.log(final_enrichment_df["Pop_Ratio"])

# Combined score = effect size × specificity
final_enrichment_df["Combined_Score"] = (
    final_enrichment_df["Fold_Enrichment"] * final_enrichment_df["Specificity"]
)

# Rank within each NodeLevel and select top 15
ranked_terms = (
    final_enrichment_df.sort_values("Combined_Score", ascending=False)
    .groupby("NodeLevel")
    .head(15)
)

# Better label
ranked_terms["Label"] = ranked_terms["GO_Term"] + " (" + ranked_terms["GO_ID"] + ")"

# ---------------------------------------------------------
# 0. Enforce consistent NodeLevel ordering
# ---------------------------------------------------------
level_order = ["Input", "Middle", "Output"]

for df in [final_enrichment_df, ranked_terms]:
    df["NodeLevel"] = pd.Categorical(
        df["NodeLevel"],
        categories=level_order,
        ordered=True,
    )

# ---------------------------------------------------------
#################################################################################################
# Plot: Top GO Terms per NodeLevel (Lollipop Plot)
#################################################################################################
# ---------------------------------------------------------


def plot_lollipop(ranked_terms, output_dir):
    """
    Generates a Lollipop plot with special highlighting.
    - Legend is placed at the BOTTOM with larger text.
    """
    # 1. Deduplication
    df_clean = ranked_terms.sort_values(
        ["NodeLevel", "Combined_Score"], ascending=[True, False]
    ).copy()
    df_clean = df_clean.drop_duplicates(subset=["NodeLevel", "Label"])

    # 2. Setup Palettes
    level_order = ["Input", "Middle", "Output"]
    nord_level_palette = {
        "Input": NORD_PALETTE[0],  # Red
        "Middle": NORD_PALETTE[1],  # Blue
        "Output": NORD_PALETTE[2],  # Green
    }

    color_class_1 = NORD_COLORS["purple"]  # DNA/Transcription
    color_class_2 = "#d08770"  # Aurora Orange

    # 3. Helper to determine style
    def get_term_style(term):
        t = term.lower()
        if (
            ("dna" in t and "binding" in t)
            or ("transcription" in t and "activity" in t)
            or ("transcription" in t and "binding" in t)
            or ("sigma" in t and "activity" in t)
        ):
            return "Class1"
        if "phosphorelay" in t or "phosporelay" in t or "molecular transducer" in t:
            return "Class2"
        return "Normal"

    # 4. Plotting
    fig, axes = plt.subplots(
        ncols=1,
        nrows=len(level_order),
        figsize=(14, 15),  # Increased height slightly for the bottom legend
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )

    for ax, level in zip(axes, level_order):
        subset = df_clean[df_clean["NodeLevel"] == level].copy()

        if subset.empty:
            ax.set_visible(False)
            continue

        subset = subset.head(10).sort_values("Combined_Score", ascending=True)
        color_bar = nord_level_palette.get(level, "black")

        # --- Base Plot ---
        ax.grid(axis="x", linestyle="--", alpha=0.5, color=NORD_COLORS["gray"])

        ax.hlines(
            y=subset["Label"],
            xmin=0,
            xmax=subset["Combined_Score"],
            color=color_bar,
            alpha=0.6,
            linewidth=2.5,
        )

        ax.scatter(
            subset["Combined_Score"],
            subset["Label"],
            s=100,
            color=color_bar,
            edgecolor=NORD_COLORS["dark"],
            linewidth=1.5,
            zorder=3,
        )

        # --- Highlight Labels ---
        y_labels = ax.get_yticklabels()
        for label in y_labels:
            text_val = label.get_text()
            style = get_term_style(text_val)

            if style == "Class1":
                label.set_color(color_class_1)
                label.set_fontweight("bold")
            elif style == "Class2":
                label.set_color(color_class_2)
                label.set_fontweight("bold")
            else:
                label.set_color(NORD_COLORS["dark"])

        # --- Styling ---
        ax.set_title(
            f"{level} Layer",
            loc="left",
            fontsize=18,
            color=color_bar,
            fontweight="bold",
        )
        ax.set_xlabel(
            "Combined Score\n(Fold Enrichment x $-\log_{10}(\mathrm{Population\ Ratio})$)"
            if level == "Output"
            else ""
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(NORD_COLORS["dark"])
            spine.set_linewidth(2.0)

    # fig.suptitle(
    #     "Top Enriched GO Terms by Network Layer", fontsize=22, fontweight="bold"
    # )

    # --- Bottom Legend ---
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="w",
            label="DNA Binding / Transcription",
            marker="s",
            markerfacecolor=color_class_1,
            markersize=14,
        ),
        Line2D(
            [0],
            [0],
            color="w",
            label="Phosphorelay / Transducer",
            marker="s",
            markerfacecolor=color_class_2,
            markersize=14,
        ),
    ]

    # Place legend at bottom center, outside the plots
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),  # Push it slightly below the figure area
        ncol=2,
        frameon=True,
        fontsize=16,  # Bigger font
    )

    # Adjust layout to make room for the legend at the bottom
    # (Note: constrained_layout usually handles this, but bbox_to_anchor can pull it out)
    save_path = output_dir / "GO_lollipop.png"
    save_path_svg = save_path.with_suffix(".svg")
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight"
    )  # bbox_inches='tight' will include the outside legend
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    # plt.show()
    plt.close()
    print(f"Saved plot: {save_path}")


plot_lollipop(ranked_terms, plots_dir)

#################################################################################################
#################################################################################################

#################################################################################################
# Plot: Scatter Study_Ratio vs Pop_Ratio
#################################################################################################


def plot_study_vs_pop_ratio(final_enrichment_df, output_dir):
    """
    Scatter plot of Study Ratio vs Population Ratio for TOP 200 terms per level.
    Uses Nord styling and strict color ordering.
    """

    # 2. Setup Order & Colors
    level_order = ["Input", "Middle", "Output"]

    # Ensure correct sorting for the plot z-order/legend
    ranked_terms = final_enrichment_df.copy()
    ranked_terms["NodeLevel"] = pd.Categorical(
        ranked_terms["NodeLevel"], categories=level_order, ordered=True
    )
    ranked_terms = ranked_terms.sort_values("NodeLevel")

    # Strict Color List (Red -> Blue -> Green)
    nord_color_list = [
        NORD_PALETTE[0],  # Red
        NORD_PALETTE[1],  # Blue
        NORD_PALETTE[2],  # Green
    ]

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(9.5, 8))

    # --- Scatter Plot ---
    sns.scatterplot(
        data=ranked_terms,
        x="Study_Ratio",
        y="Pop_Ratio",
        hue="NodeLevel",
        hue_order=level_order,
        palette=nord_color_list,
        s=40,
        alpha=1.0,
        zorder=5,
        edgecolor=NORD_COLORS["dark"],
        # linewidth=0.8,  # Slightly thinner edge for small dots looks cleaner
        ax=ax,
    )

    # --- Diagonal Line ---
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        # linewidth=1.5,
        color=NORD_COLORS["dark"],
        alpha=0.8,
        zorder=6,
    )

    # --- Region Shading ---
    # Depleted (Nord Red)
    ax.fill_between(
        [0, 1], [0, 1], [1, 1], color=NORD_COLORS["red"], alpha=0.10, zorder=1
    )
    # Enriched (Nord Green)
    ax.fill_between(
        [0, 1], [0, 0], [0, 1], color=NORD_COLORS["green"], alpha=0.10, zorder=1
    )

    # --- Styling ---
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    ax.set_xlabel("Study Ratio")
    ax.set_ylabel("Population Ratio")
    ax.set_title(
        "Study Ratio vs Population Ratio (Top 200 Ranked GO Terms)",
        pad=12,
    )

    # Add Box Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])
        spine.set_linewidth(2.0)

    # --- Legend Management ---

    # 1. Main Legend (Levels)
    handles, labels = ax.get_legend_handles_labels()
    first_legend = ax.legend(
        handles,
        labels,
        title="Network Layer",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
    )
    ax.add_artist(first_legend)

    # 2. Region Legend
    region_handles = [
        Patch(
            facecolor=NORD_COLORS["green"],
            alpha=0.3,
            edgecolor=NORD_COLORS["dark"],
            label="Enriched",
        ),
        Patch(
            facecolor=NORD_COLORS["red"],
            alpha=0.3,
            edgecolor=NORD_COLORS["dark"],
            label="Depleted",
        ),
    ]
    ax.legend(
        handles=region_handles,
        title="Region Status",
        bbox_to_anchor=(1.02, 0.8),
        loc="upper left",
        frameon=False,
    )

    plt.tight_layout()

    save_path = output_dir / "GO_study_vs_pop_ratio.png"
    save_path_svg = save_path.with_suffix(".svg")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    # plt.show()
    plt.close()
    print(f"Saved plot: {save_path}")


plot_study_vs_pop_ratio(final_enrichment_df, plots_dir)


#################################################################################################
### 3. Stripplot: Full GO Set Fold Enrichment
#################################################################################################


def plot_fold_enrichment_distribution(final_enrichment_df, output_dir):
    """
    Overlaid Boxplot + Stripplot for Fold Enrichment across all terms.
    Uses consistent Nord Colors (Input=Red, Middle=Blue, Output=Green).
    """

    # 1. Setup Order & Colors
    level_order = ["Input", "Middle", "Output"]

    # Strict Color List
    nord_color_list = [
        NORD_PALETTE[0],  # Red
        NORD_PALETTE[1],  # Blue
        NORD_PALETTE[2],  # Green
    ]

    # 2. Setup Plot
    plt.figure(figsize=(5, 5))

    # --- Layer 1: Boxplot (Background Distribution) ---
    ax = sns.boxplot(
        data=final_enrichment_df,
        x="NodeLevel",
        y="Fold_Enrichment",
        order=level_order,
        palette=nord_color_list,
        showfliers=False,  # Hide outliers (stripplot will show them)
        width=0.5,
    )

    # --- Layer 2: Stripplot (Individual Points) ---
    sns.stripplot(
        data=final_enrichment_df,
        x="NodeLevel",
        y="Fold_Enrichment",
        order=level_order,
        hue="NodeLevel",
        hue_order=level_order,
        palette=nord_color_list,
        jitter=True,
        alpha=0.8,  # Darker points
        edgecolor=NORD_COLORS["dark"],
        linewidth=1.0,  # Slight outline to make them pop
        dodge=False,
        size=5,
        zorder=3,
    )

    pairs = [("Input", "Middle"), ("Input", "Output"), ("Middle", "Output")]

    try:
        annotator = Annotator(
            ax=ax,
            pairs=pairs,
            data=df,
            x="NodeLevel",
            y="Fold_Enrichment",
            order=level_order,
        )

        annotator.configure(
            test="Mann-Whitney",
            text_format="star",  # Display stars (*, **, ***)
            loc="inside",
            verbose=0,
            color=NORD_COLORS["dark"],  # Match text color
            line_width=1.5,
        )

        annotator.apply_and_annotate()

    except Exception as e:
        print(f"Stats annotation failed: {e}")

    # 3. Styling
    plt.title("Fold Enrichment Across All GO Terms", pad=15)
    plt.xlabel("Network Layer")
    plt.ylabel("Fold Enrichment")

    # Y-scale log often helps with enrichment values (optional, uncomment if needed)
    # plt.yscale('log')

    # Box Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])
        spine.set_linewidth(2.0)

    # Remove the redundant legend (since x-axis labels cover it)
    if ax.get_legend():
        ax.get_legend().remove()

    plt.tight_layout()

    save_path = output_dir / "GO_fold_enrichment_all_terms_boxplot.png"
    save_path_svg = save_path.with_suffix(".svg")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    # plt.show()
    plt.close()
    print(f"Saved plot: {save_path}")


plot_fold_enrichment_distribution(final_enrichment_df, plots_dir)
