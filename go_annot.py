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

# --- Organism Name Mapping ---
NETWORK_TO_ORGANISM = {
    "196627_v2020_s21_regNetwork_Strong": "Corynebacterium glutamicum",
    "83332_v2018_s15-16_regNetwork": "Mycobacterium tuberculosis",
    "224308_v2022_sSW22_regNetwork": "Bacillus subtilis",
    "511145_v2022_sRDB22_eStrong_regNetwork_Strong": "Escherichia coli",
    "208964_v2020_sRPA20_regNetwork_Strong": "Pseudomonas aeruginosa",
    "100226_v2019_sA22-DBSCR15_eStrong_regNetwork": "Streptomyces coelicolor",
}


###################################################################################################
### Plotting Settings
###################################################################################################

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
    NORD_COLORS["light_blue"],
]


def set_global_nord_style():
    """Configures Matplotlib global settings for the Nord aesthetic."""
    plt.style.use("default")
    sns.set_context("paper", font_scale=1.6)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Roboto", "Arial", "DejaVu Sans"],
            "text.color": NORD_COLORS["dark"],
            "axes.labelcolor": NORD_COLORS["dark"],
            "axes.titlecolor": NORD_COLORS["dark"],
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": NORD_COLORS["gray"],
            "axes.linewidth": 1.5,
            "xtick.color": NORD_COLORS["dark"],
            "ytick.color": NORD_COLORS["dark"],
            "grid.color": NORD_COLORS["gray"],
            "grid.alpha": 0.3,
            "legend.frameon": False,
        }
    )


set_global_nord_style()

###################################################################################################
### Utility Functions
###################################################################################################


def get_priority(filename):
    if "_eStrong" in filename and "_Strong" in filename:
        return 1
    if "_Strong" in filename:
        return 2
    return 3


def classify_node(row):
    out_val = row["MeanAbsOutCoh"] if pd.notna(row["MeanAbsOutCoh"]) else 0
    in_val = row["MeanAbsInCoh"] if pd.notna(row["MeanAbsInCoh"]) else 0

    if out_val == 0 and in_val != 0:
        return "Output"
    elif in_val == 0 and out_val != 0:
        return "Input"
    else:
        return "Middle"


def classify_go_annotations_generic(df, go_targets_dict, go_ontology_dag):
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
        column_name = f"Is_{name.replace(' ', '_')}_GO"
        print(f"-> Classifying for: {name} (GO: {go_id}) into column '{column_name}'")
        tqdm.pandas(desc=f"Checking {name} Ancestry")
        df[column_name] = df["GO_terms"].progress_apply(
            lambda terms: is_descendant_of(terms, go_id)
        )
    return df


def fetch_go_terms_batched(uniprot_ids, batch_size=200):
    u = UniProt()
    ids = [str(x) for x in uniprot_ids if isinstance(x, str) and x not in ("", "nan")]
    all_results = {uid: [] for uid in ids}

    for i in tqdm(range(0, len(ids), batch_size), desc="Fetching GO (batched)"):
        batch = ids[i : i + batch_size]
        query = " OR ".join([f"accession:{x}" for x in batch])

        try:
            res = u.search(query=query, frmt="tsv", columns="accession,go_f")
        except Exception as e:
            print(f"[ERROR] UniProt batch failed for {batch[:5]}: {e}")
            continue

        if res and isinstance(res, str) and len(res.strip()) > 0:
            try:
                df_batch = pd.read_csv(io.StringIO(res), sep="\t")
            except pd.errors.EmptyDataError:
                continue

            if not df_batch.empty and df_batch.shape[1] >= 2:
                df_batch.columns = ["Node", "GO_Raw"]
                df_batch["GO_Raw"] = df_batch["GO_Raw"].fillna("")
                df_batch["GO_List"] = df_batch["GO_Raw"].apply(
                    lambda x: re.findall(r"GO:\d+", str(x))
                )
                batch_dict = pd.Series(
                    df_batch.GO_List.values, index=df_batch.Node
                ).to_dict()
                all_results.update(batch_dict)

        time.sleep(0.5)
    return all_results


def remove_outliers(df, x_col, y_col):
    clean_rows = []
    for group in df[x_col].unique():
        group_data = df[df[x_col] == group]
        q1 = group_data[y_col].quantile(0.10)
        q3 = group_data[y_col].quantile(0.90)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        clean_rows.append(
            group_data[
                (group_data[y_col] >= lower_bound) & (group_data[y_col] <= upper_bound)
            ]
        )
    return pd.concat(clean_rows)


def filter_generic_go_terms(
    df, go_ontology_dag, min_depth=5, max_depth=None, exclude_namespaces=None
):
    if exclude_namespaces is None:
        exclude_namespaces = []
    print(f"Initial terms: {len(df)}")

    def check_term_specificity(go_id):
        if go_id not in go_ontology_dag:
            return False, 0
        term_record = go_ontology_dag[go_id]
        term_level = term_record.level

        if term_record.namespace in exclude_namespaces:
            return False, term_level
        if term_level < min_depth:
            return False, term_level
        if max_depth is not None and term_level > max_depth:
            return False, term_level
        return True, term_level

    df[["Pass_Filter", "Term_Level"]] = df["GO_ID"].apply(
        lambda x: pd.Series(check_term_specificity(x))
    )
    filtered_df = df[df["Pass_Filter"]].drop(columns=["Pass_Filter", "Term_Level"])
    print(f"Terms remaining after filtering: {len(filtered_df)}")
    return filtered_df


###################################################################################################
### Preprocessing Functions
###################################################################################################


def calculate_enrichment(file_list, target_col):
    """Fisher's Exact Test for a specific boolean column (e.g., 'Is_TF_GO')."""
    enrichment_results = []
    for io_fl in file_list:
        io_df = pd.read_csv(io_fl, sep="\t", engine="python")
        if target_col not in io_df.columns:
            continue
        if io_df[target_col].dtype == object:
            io_df[target_col] = io_df[target_col].astype(str) == "True"

        network_name = io_fl.stem.replace("_GOInformation", "")
        for level in io_df["NodeLevel"].dropna().unique():
            is_level = io_df["NodeLevel"] == level
            is_target = io_df[target_col]

            a = len(io_df[is_level & is_target])
            b = len(io_df[is_level & ~is_target])
            c = len(io_df[~is_level & is_target])
            d = len(io_df[~is_level & ~is_target])

            if (a + b) == 0:
                continue

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


def preprocess_hierarchy_tf_enrichment(absy_dir, plots_dir):
    """Orchestrates TF Enrichment preprocessing and plotting."""
    inout_df_list = sorted(list(absy_dir.glob("*/*_GOInformation.tsv")))
    if inout_df_list:
        first_df = pd.read_csv(inout_df_list[0], sep="\t", engine="python")
        target_columns = [
            col
            for col in first_df.columns
            if col.startswith("Is_") and col.endswith("_GO")
        ]
        print(f"Found target columns: {target_columns}")

        for col in target_columns:
            clean_name = col.replace("Is_", "").replace("_GO", "").replace("_", " ")
            print(f"Processing: {clean_name}...")
            res_df = calculate_enrichment(inout_df_list, col)
            plot_enrichment_F(res_df, clean_name, plots_dir)
    else:
        print("No _GOInformation.tsv files found to process.")


def preprocess_module_enrichments(absy_dir):
    """Calculates Fisher exact test for module/component enrichments across levels."""
    inout_df_list = sorted(list(absy_dir.glob("*/*_GOInformation.tsv")))
    module_enrichments = []

    for io_fl in tqdm(inout_df_list, desc="Processing Networks"):
        io_df = pd.read_csv(io_fl, sep="\t", engine="python")
        network_name = io_fl.stem.replace("_GOInformation", "")
        if "NDA_component" not in io_df.columns:
            print(f"Skipping {network_name}: 'NDA_component' column missing.")
            continue

        clean_df = io_df.dropna(subset=["NodeLevel", "NDA_component"])
        unique_levels = clean_df["NodeLevel"].unique()
        unique_components = clean_df["NDA_component"].unique()

        for comp in unique_components:
            for level in unique_levels:
                is_level = clean_df["NodeLevel"] == level
                is_comp = clean_df["NDA_component"] == comp

                a = len(clean_df[is_comp & is_level])
                b = len(clean_df[is_comp & ~is_level])
                c = len(clean_df[~is_comp & is_level])
                d = len(clean_df[~is_comp & ~is_level])

                if (a + c) == 0:
                    continue

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
                        "Percentage_in_Module": (a / (a + b) * 100)
                        if (a + b) > 0
                        else 0,
                    }
                )

    module_enrich_df = pd.DataFrame(module_enrichments)
    module_enrich_df["P_value_FDR"] = multipletests(
        module_enrich_df["P_value"], method="fdr_bh"
    )[1]
    module_enrich_df["Significant FDR"] = module_enrich_df["P_value_FDR"] < 0.05

    level_order = ["Input", "Middle", "Output"]
    module_enrich_df["NodeLevel"] = pd.Categorical(
        module_enrich_df["NodeLevel"], categories=level_order, ordered=True
    )
    module_enrich_df = module_enrich_df.sort_values(["NodeLevel", "Odds_Ratio"])

    return module_enrich_df


def get_level_enrichment_from_tsv(io_df, network_name, go_ontology):
    level_enrichment_results = []
    valid_data = io_df.dropna(subset=["Uniprot_ID_clean", "GO_terms"]).copy()

    go_associations = {
        uid: set(terms)
        for uid, terms in zip(valid_data["Uniprot_ID_clean"], valid_data["GO_terms"])
        if terms
    }

    population_set = set(valid_data["Uniprot_ID_clean"].unique())

    for level in valid_data["NodeLevel"].dropna().unique():
        study_set = set(
            valid_data[valid_data["NodeLevel"] == level]["Uniprot_ID_clean"]
        )
        if not study_set:
            continue

        goea_obj = GOEnrichmentStudy(
            population_set,
            go_associations,
            go_ontology,
            propagate_counts=True,
            alpha=0.05,
            methods=["fdr_bh"],
        )

        goea_results = goea_obj.run_study(study_set, verbose=False)
        study_total = len(study_set)
        pop_total = len(population_set)

        significant_results = [r for r in goea_results if r.p_fdr_bh < 0.05]

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


def generate_master_go_enrichment(absy_dir, go_ontology):
    """Reads TSV files and generates the master GO enrichment dataset."""
    print("\n--- Starting GO Enrichment from Saved TSV Files ---")
    inout_df_list = sorted(list(absy_dir.glob("*/*_GOInformation.tsv")))
    master_enrichment_list = []

    for io_fl in tqdm(inout_df_list, desc="Processing TSV Files"):
        io_df = pd.read_csv(io_fl, sep="\t", engine="python")
        io_df["GO_terms"] = io_df["GO_terms"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
        network_name = io_fl.stem.replace("_GOInformation", "")
        level_results = get_level_enrichment_from_tsv(io_df, network_name, go_ontology)
        master_enrichment_list.extend(level_results)

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


def preprocess_final_enrichment_data(absy_dir):
    """Loads and scores the master enrichment TSV to generate dataframes for plotting."""
    final_enrichment_df = pd.read_csv(absy_dir / "GO_Level_Enrichment.tsv", sep="\t")
    final_enrichment_df = final_enrichment_df[
        final_enrichment_df["Study_Count"] >= 3
    ].copy()
    final_enrichment_df["Fold_Enrichment"] = (
        final_enrichment_df["Study_Ratio"] / final_enrichment_df["Pop_Ratio"]
    )
    final_enrichment_df["Specificity"] = -np.log(final_enrichment_df["Pop_Ratio"])
    final_enrichment_df["Combined_Score"] = (
        final_enrichment_df["Fold_Enrichment"] * final_enrichment_df["Specificity"]
    )

    ranked_terms = (
        final_enrichment_df.sort_values("Combined_Score", ascending=False)
        .groupby("NodeLevel")
        .head(15)
    )
    ranked_terms["Label"] = ranked_terms["GO_Term"] + " (" + ranked_terms["GO_ID"] + ")"

    level_order = ["Input", "Middle", "Output"]
    for df in [final_enrichment_df, ranked_terms]:
        df["NodeLevel"] = pd.Categorical(
            df["NodeLevel"],
            categories=level_order,
            ordered=True,
        )

    return final_enrichment_df, ranked_terms


###################################################################################################
### Plotting Functions
###################################################################################################


def plot_enrichment_F(enrichment_df, target_name, output_dir):
    if enrichment_df.empty:
        print(f"Skipping plot for {target_name}: No data found.")
        return

    level_order = ["Input", "Middle", "Output"]
    nord_level_palette = {
        "Input": NORD_PALETTE[0],
        "Middle": NORD_PALETTE[1],
        "Output": NORD_PALETTE[2],
    }
    sig_palette = {True: NORD_COLORS["dark"], False: "white"}

    df = enrichment_df.copy()
    df = df[df["NodeLevel"].isin(level_order)]
    df["NodeLevel"] = pd.Categorical(
        df["NodeLevel"], categories=level_order, ordered=True
    )
    df = df.sort_values(["NodeLevel", "Odds_Ratio"], ascending=[True, False])

    reject, pvals_corrected, _, _ = multipletests(df["P_value"], method="fdr_bh")
    df["Significant FDR"] = pvals_corrected < 0.05

    plt.figure(figsize=(6, 5))
    sns.boxplot(
        data=df,
        x="NodeLevel",
        y="Odds_Ratio",
        order=level_order,
        palette=nord_level_palette,
        showcaps=True,
        showfliers=False,
        width=0.5,
    )
    sns.stripplot(
        data=df,
        x="NodeLevel",
        y="Odds_Ratio",
        order=level_order,
        hue="Significant FDR",
        palette=sig_palette,
        dodge=False,
        jitter=True,
        alpha=0.9,
        edgecolor=NORD_COLORS["dark"],
        linewidth=1.5,
        size=10,
        marker="o",
    )

    plt.ylabel("Odds Ratio")
    plt.xlabel("Node Level")
    plt.title(f"{target_name} Enrichment Across Node Levels")

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

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="p < 0.05",
            markerfacecolor=sig_palette[True],
            markeredgecolor=NORD_COLORS["dark"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="p >= 0.05",
            markerfacecolor=sig_palette[False],
            markeredgecolor=NORD_COLORS["dark"],
        ),
    ]

    plt.legend(
        handles=legend_elements,
        title="Significance (FDR)",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
    )

    plt.tight_layout()
    save_path = output_dir / f"{target_name}_Enrichment.png"
    save_path_svg = save_path.with_suffix(".svg")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_module_enrichment_F(module_enrich_df, plots_dir):
    level_order = ["Input", "Middle", "Output"]
    nord_level_palette = {
        "Input": NORD_PALETTE[0],
        "Middle": NORD_PALETTE[1],
        "Output": NORD_PALETTE[2],
    }
    sig_palette = {True: NORD_COLORS["dark"], False: "white"}

    plt.figure(figsize=(6, 5))
    sns.boxplot(
        data=module_enrich_df,
        x="NodeLevel",
        y="Odds_Ratio",
        order=level_order,
        palette=nord_level_palette,
        showcaps=True,
        width=0.5,
        showfliers=False,
    )
    sns.stripplot(
        data=remove_outliers(module_enrich_df, "NodeLevel", "Odds_Ratio"),
        x="NodeLevel",
        y="Odds_Ratio",
        order=level_order,
        hue="Significant FDR",
        palette=sig_palette,
        dodge=False,
        jitter=True,
        alpha=0.7,
        edgecolor=NORD_COLORS["dark"],
        linewidth=0.8,
        size=8,
        marker="o",
    )

    plt.ylabel("Odds Ratio")
    plt.xlabel("Node Level")
    plt.title("Module Enrichment Across Node Levels")

    present_levels = [
        l for l in level_order if l in module_enrich_df["NodeLevel"].unique()
    ]
    if len(present_levels) >= 2:
        pairs = [
            (present_levels[i], present_levels[j])
            for i in range(len(present_levels))
            for j in range(i + 1, len(present_levels))
        ]
        try:
            annotator = Annotator(
                ax=plt.gca(),
                pairs=pairs,
                data=module_enrich_df,
                x="NodeLevel",
                y="Odds_Ratio",
                order=level_order,
            )
            annotator.configure(
                test="Mann-Whitney", text_format="star", loc="inside", verbose=0
            )
            annotator.apply_and_annotate()
        except Exception as e:
            print(f"Error during stat annotation drawing: {e}")

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
    )

    plt.tight_layout()
    save_path = plots_dir / "Module_Enrichment.png"
    save_path_svg = save_path.with_suffix(".svg")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_lollipop(ranked_terms, output_dir):
    df_clean = ranked_terms.sort_values(
        ["NodeLevel", "Combined_Score"], ascending=[True, False]
    ).copy()
    df_clean = df_clean.drop_duplicates(subset=["NodeLevel", "Label"])

    level_order = ["Input", "Middle", "Output"]
    nord_level_palette = {
        "Input": NORD_PALETTE[0],
        "Middle": NORD_PALETTE[1],
        "Output": NORD_PALETTE[2],
    }

    color_class_1 = NORD_COLORS["purple"]
    color_class_2 = "#d08770"

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

    fig, axes = plt.subplots(
        ncols=1,
        nrows=len(level_order),
        figsize=(14, 15),
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

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=2,
        frameon=True,
        fontsize=16,
    )

    save_path = output_dir / "GO_lollipop.png"
    save_path_svg = save_path.with_suffix(".svg")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_lollipop_split_F(ranked_terms, output_dir):
    df_clean = ranked_terms.sort_values(
        ["NodeLevel", "Combined_Score"], ascending=[True, False]
    ).copy()
    df_clean = df_clean.drop_duplicates(subset=["NodeLevel", "Label"])

    nord_level_palette = {
        "Input": NORD_PALETTE[0],
        "Middle": NORD_PALETTE[1],
        "Output": NORD_PALETTE[2],
    }

    color_class_1 = NORD_COLORS["purple"]
    color_class_2 = NORD_COLORS["orange"]

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

    plot_data = {}
    global_x_max = 0
    for level in ["Input", "Middle", "Output"]:
        subset = df_clean[df_clean["NodeLevel"] == level].copy()
        if not subset.empty:
            subset = subset.head(10).sort_values("Combined_Score", ascending=True)
            plot_data[level] = subset
            global_x_max = max(global_x_max, subset["Combined_Score"].max())

    x_lim_max = global_x_max * 1.05

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="w",
            label="DNA Binding / Transcription",
            marker="s",
            markerfacecolor=color_class_1,
            markersize=16,
        ),
        Line2D(
            [0],
            [0],
            color="w",
            label="Phosphorelay / Transducer",
            marker="s",
            markerfacecolor=color_class_2,
            markersize=16,
        ),
    ]

    def draw_lollipop_axis(ax, level, subset):
        color_bar = nord_level_palette.get(level, "black")
        ax.grid(axis="x", linestyle="--", alpha=0.5, color=NORD_COLORS["gray"])
        ax.hlines(
            y=subset["Label"],
            xmin=0,
            xmax=subset["Combined_Score"],
            color=color_bar,
            alpha=0.8,
            linewidth=3.5,
        )
        ax.scatter(
            subset["Combined_Score"],
            subset["Label"],
            s=120,
            color=color_bar,
            edgecolor=NORD_COLORS["dark"],
            linewidth=1.5,
            zorder=3,
        )

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

        ax.set_title(f"{level} Layer", loc="left", fontsize=18, color=color_bar)
        ax.set_xlim(0, x_lim_max)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(NORD_COLORS["dark"])

    # Figure 1: Input and Middle Layers
    fig1, axes1 = plt.subplots(
        ncols=1, nrows=2, figsize=(12, 8.5), sharex=True, constrained_layout=True
    )
    for ax, level in zip(axes1, ["Input", "Middle"]):
        if level in plot_data:
            draw_lollipop_axis(ax, level, plot_data[level])
        else:
            ax.set_visible(False)

    axes1[-1].set_xlabel(
        "Combined Score\n(Fold Enrichment x $-\\log_{10}(\\mathrm{Population\\ Ratio})$)"
    )
    fig1.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        frameon=True,
    )

    save_path1 = output_dir / "GO_lollipop_InputMiddle.png"
    plt.savefig(save_path1, dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(
        save_path1.with_suffix(".svg"), dpi=300, transparent=True, bbox_inches="tight"
    )
    plt.close(fig1)
    print(f"Saved plot: {save_path1}")

    # Figure 2: Output Layer
    if "Output" in plot_data:
        fig2, ax2 = plt.subplots(
            ncols=1, nrows=1, figsize=(12, 5.5), constrained_layout=True
        )
        draw_lollipop_axis(ax2, "Output", plot_data["Output"])
        ax2.set_xlabel(
            "Combined Score\n(Fold Enrichment x $-\\log_{10}(\\mathrm{Population\\ Ratio})$)"
        )
        fig2.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=2,
            frameon=True,
            fontsize=16,
        )

        save_path2 = output_dir / "GO_lollipop_Output.png"
        plt.savefig(save_path2, dpi=300, bbox_inches="tight", transparent=True)
        plt.savefig(
            save_path2.with_suffix(".svg"),
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )
        plt.close(fig2)
        print(f"Saved plot: {save_path2}")


def plot_study_vs_pop_ratio_F(final_enrichment_df, output_dir):
    level_order = ["Input", "Middle", "Output"]
    ranked_terms = final_enrichment_df.copy()
    ranked_terms["NodeLevel"] = pd.Categorical(
        ranked_terms["NodeLevel"], categories=level_order, ordered=True
    )
    ranked_terms = ranked_terms.sort_values("NodeLevel")

    nord_color_list = [NORD_PALETTE[0], NORD_PALETTE[1], NORD_PALETTE[2]]

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.scatterplot(
        data=ranked_terms,
        x="Study_Ratio",
        y="Pop_Ratio",
        hue="NodeLevel",
        hue_order=level_order,
        palette=nord_color_list,
        s=100,
        alpha=0.9,
        zorder=5,
        edgecolor=NORD_COLORS["dark"],
        linewidth=1.0,
        ax=ax,
    )

    ax.plot(
        [0, 1], [0, 1], linestyle="--", color=NORD_COLORS["dark"], alpha=0.8, zorder=6
    )
    ax.fill_between(
        [0, 1], [0, 1], [1, 1], color=NORD_COLORS["red"], alpha=0.10, zorder=1
    )
    ax.fill_between(
        [0, 1], [0, 0], [0, 1], color=NORD_COLORS["green"], alpha=0.10, zorder=1
    )

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("Study Ratio")
    ax.set_ylabel("Population Ratio")
    ax.set_title("Study Ratio vs Population Ratio (Top 200 Ranked GO Terms)", pad=12)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])
        spine.set_linewidth(2.0)

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
        bbox_to_anchor=(1.02, 0.7),
        loc="upper left",
        frameon=False,
    )

    plt.tight_layout()
    save_path = output_dir / "GO_study_vs_pop_ratio.png"
    save_path_svg = save_path.with_suffix(".svg")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_fold_enrichment_distribution_F(final_enrichment_df, output_dir):
    level_order = ["Input", "Middle", "Output"]
    nord_color_list = [NORD_PALETTE[0], NORD_PALETTE[1], NORD_PALETTE[2]]

    plt.figure(figsize=(5, 5))
    ax = sns.boxplot(
        data=final_enrichment_df,
        x="NodeLevel",
        y="Fold_Enrichment",
        order=level_order,
        palette=nord_color_list,
        showfliers=False,
        width=0.4,
        zorder=2,
    )
    sns.stripplot(
        data=final_enrichment_df,
        x="NodeLevel",
        y="Fold_Enrichment",
        order=level_order,
        hue="NodeLevel",
        hue_order=level_order,
        palette=nord_color_list,
        jitter=True,
        alpha=0.8,
        edgecolor=NORD_COLORS["dark"],
        linewidth=1.5,
        dodge=False,
        size=10,
        zorder=1,
    )

    pairs = [("Input", "Middle"), ("Input", "Output"), ("Middle", "Output")]

    try:
        annotator = Annotator(
            ax=ax,
            pairs=pairs,
            data=final_enrichment_df,
            x="NodeLevel",
            y="Fold_Enrichment",
            order=level_order,
        )
        annotator.configure(
            test="Mann-Whitney",
            text_format="star",
            loc="inside",
            verbose=0,
            color=NORD_COLORS["dark"],
            line_width=1.5,
        )
        annotator.apply_and_annotate()
    except Exception as e:
        print(f"Stats annotation failed: {e}")

    plt.title("Fold Enrichment Across All GO Terms", pad=15)
    plt.xlabel("Network Layer")
    plt.ylabel("Fold Enrichment")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NORD_COLORS["dark"])
        spine.set_linewidth(2.0)

    if ax.get_legend():
        ax.get_legend().remove()

    plt.tight_layout()
    save_path = output_dir / "GO_fold_enrichment_all_terms_boxplot.png"
    save_path_svg = save_path.with_suffix(".svg")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(save_path_svg, dpi=300, transparent=True)
    plt.close()
    print(f"Saved plot: {save_path}")


###################################################################################################
### Execution Block
###################################################################################################

if __name__ == "__main__":
    # 1. Define Execution Directories
    absy_dir = Path("../MotifFinding/AbasyCohResults_Targeted/")
    gene_info_dir = Path("../MotifFinding/AbasyNets/")
    topos_dir = Path("../MotifFinding/AbasyTOPOS_Targeted/")

    plots_dir = Path("../MotifFinding/GRN_Plots/Fig7/")
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Note: For actual execution, ensure go.obo exists in the specified path
    try:
        go_ontology = GODag(Path("../MotifFinding/GOInfo_Targeted/go.obo"))
    except Exception as e:
        print(f"Warning: Could not load GO Ontology. {e}")
        go_ontology = None

    # --- Part 1: Hierarchy TF Enrichment Processing & Plotting ---
    preprocess_hierarchy_tf_enrichment(absy_dir, plots_dir)

    # --- Part 2: Module Enrichment Processing & Plotting ---
    module_enrich_df = preprocess_module_enrichments(absy_dir)
    if not module_enrich_df.empty:
        plot_module_enrichment_F(module_enrich_df, plots_dir)

    # --- Part 3: Node Level GO Enrichment Processing ---
    if go_ontology:
        # Note: This will rewrite the GO_Level_Enrichment.tsv file based on currently available TSVs.
        # Comment this out if you just want to plot the existing file.
        generate_master_go_enrichment(absy_dir, go_ontology)

    # --- Part 4: Final Plotting using Master Enrichment Data ---
    if (absy_dir / "GO_Level_Enrichment.tsv").exists():
        final_enrichment_df, ranked_terms = preprocess_final_enrichment_data(absy_dir)

        plot_lollipop_split_F(ranked_terms, plots_dir)
        plot_study_vs_pop_ratio_F(final_enrichment_df, plots_dir)
        plot_fold_enrichment_distribution_F(final_enrichment_df, plots_dir)
    else:
        print("GO_Level_Enrichment.tsv not found. Run Part 3 first.")
