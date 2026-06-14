import pandas as pd
import numpy as np
import networkx as nx
from dotmotif import Motif, GrandIsoExecutor
from pathlib import Path
import multiprocessing as mp
import re
import gc

# =============================================================================
# Core Utility & Data Loading Functions
# =============================================================================


def build_metadata_index(go_dir):
    index = {}
    for tsv_path in go_dir.rglob("*_GOInformation.tsv"):
        match = re.search(r"^(.*?)_GOInformation", tsv_path.name)
        if match:
            clean_core = re.sub(r"_eStrong$", "", match.group(1))
            index[clean_core] = tsv_path
    return index


def load_metadata(tsv_path):
    if tsv_path and tsv_path.exists():
        meta_df = pd.read_csv(tsv_path, sep="\t")
        level_map = dict(
            zip(meta_df["Node"].astype(str).str.strip(), meta_df["NodeLevel"])
        )
        nda_map = dict(
            zip(meta_df["Node"].astype(str).str.strip(), meta_df["NDA_component"])
        )
        return level_map, nda_map
    return {}, {}


def load_topo_graph(network_path):
    if not network_path or not network_path.exists():
        return None
    net_df = pd.read_csv(network_path, sep=r"\s+")
    net_df["Type"] = net_df["Type"].replace(2, -1)
    net_df = net_df.rename(columns={"Type": "sign"})
    net_df["Source"] = net_df["Source"].astype(str).str.strip()
    net_df["Target"] = net_df["Target"].astype(str).str.strip()
    net_df = net_df[net_df["Source"] != net_df["Target"]]
    return nx.from_pandas_edgelist(
        net_df,
        source="Source",
        target="Target",
        edge_attr="sign",
        create_using=nx.DiGraph(),
    )


def topo_to_dotmotif_dsl(filepath):
    df = pd.read_csv(filepath, sep=r"\s+")
    df["Type"] = df["Type"].replace(2, -1)
    edge_lines, null_lines = [], []
    nodes_with_edges = set()
    all_nodes = set(df["Source"]).union(set(df["Target"]))

    for _, row in df.iterrows():
        src, tgt, t_type = row["Source"], row["Target"], int(row["Type"])
        if t_type == 0:
            null_lines.append(f"{src} !> {tgt}")
        else:
            edge_lines.append(f"{src} -> {tgt} [sign={t_type}]")
            nodes_with_edges.update([src, tgt])

    if all_nodes != nodes_with_edges or not edge_lines:
        return None
    return Motif(
        "\n".join(edge_lines + null_lines),
        enforce_inequality=True,
        exclude_automorphisms=True,
    )


def _process_single_motif_def(path):
    return path.name.replace(".topo", ""), topo_to_dotmotif_dsl(path)


# =============================================================================
# Optimized Worker Engine (Worker-level Caching + Direct Parquet Dumps)
# =============================================================================

# Each worker process holds its own isolated cache in RAM
worker_cache = {"net_stem": None}


def _search_and_export_raw(
    G_target,
    m_name,
    motif_obj,
    net_stem,
    level_map,
    nda_map,
    out_dir,
):
    if G_target.number_of_edges() == 0:
        return 0

    executor = GrandIsoExecutor(graph=G_target)
    matches = list(executor.find(motif_obj))

    if not matches:
        return 0

    results = []
    keys = sorted(matches[0].keys())

    for match in matches:
        target_nodes = [str(match[k]) for k in keys]

        lvl = "_".join([str(level_map.get(n, "Unknown")) for n in target_nodes])
        nda = " : ".join([str(nda_map.get(n, "Unknown")) for n in target_nodes])

        row = {
            "Network": net_stem,
            "Motif": m_name,
            "Graph_Type": "Topo",
            "Node_Mapping": ", ".join([f"{k}:{match[k]}" for k in keys]),
            "Level_String": lvl,
            "NDA_String": nda,
        }
        results.append(row)

    df = pd.DataFrame(results)
    file_path = out_dir / f"{net_stem}_{m_name}_Raw.parquet"
    df.to_parquet(file_path, index=False)

    return len(results)


def worker_process_single_motif(args):
    (
        random_net_stem,
        topo_path,
        tsv_path,
        m_name,
        motif_obj,
        output_target_dir,
    ) = args

    if worker_cache["net_stem"] != random_net_stem:
        worker_cache.clear()
        worker_cache["net_stem"] = random_net_stem
        worker_cache["G_topo"] = load_topo_graph(topo_path)
        worker_cache["level_map"], worker_cache["nda_map"] = load_metadata(tsv_path)

    G_topo = worker_cache["G_topo"]
    level_map = worker_cache["level_map"]
    nda_map = worker_cache["nda_map"]

    stats = {
        "Network": random_net_stem,
        "Motif": m_name,
        "Topo_Raw_Dumps": 0,
    }

    if motif_obj is None or not level_map or not G_topo:
        return stats

    output_target_dir.mkdir(exist_ok=True, parents=True)

    stats["Topo_Raw_Dumps"] = _search_and_export_raw(
        G_topo,
        m_name,
        motif_obj,
        random_net_stem,
        level_map,
        nda_map,
        output_target_dir,
    )

    return stats


# =============================================================================
# Aggregation function to save the motif counts as a CSV
# =============================================================================


def compile_shuffled_motif_counts(shuffled_base_dir):
    """
    Reads raw Parquet match dumps from shuffled networks, calculates total
    motif counts per randomized run, and exports formatted CSVs for downstream analysis.
    """
    base_dir = Path(shuffled_base_dir)
    print("Aggregating raw shuffled parquet files into count CSVs...")
    print(list(base_dir.glob("*/")))

    for base_net_dir in [d for d in base_dir.iterdir() if d.is_dir()]:
        motif_counts_dir = base_net_dir / "MotifsCounts"
        if not motif_counts_dir.exists():
            continue

        # Make the processed count direcotry
        motif_counts_proc_dir = base_net_dir / "ProcessedMotifCounts"
        motif_counts_proc_dir.mkdir(exist_ok=True)

        print(f"Processing shuffles for: {base_net_dir.name}")
        parquet_files = list(motif_counts_dir.glob("*_Raw.parquet"))

        # Organize files by the shuffle run stem (e.g., BaseNet_Random001)
        run_dict = {}
        for p_file in parquet_files:
            # Matches the filename up to the random seed: e.g., NetworkName_Random050
            match = re.search(r"(.*_Random\d{3})_(.*)_Raw\.parquet", p_file.name)
            if match:
                random_stem = match.group(1)
                if random_stem not in run_dict:
                    run_dict[random_stem] = []
                run_dict[random_stem].append(p_file)

        # Aggregate counts for each specific shuffle run
        for random_stem, files in run_dict.items():
            run_counts = []

            for p_file in files:
                try:
                    # Load only the Motif column to save memory
                    df = pd.read_parquet(p_file, columns=["Motif"])
                except Exception:
                    # Handle cases where the file is corrupted or lacks a readable schema
                    continue

                if df.empty:
                    continue

                # Calculate counts for this specific motif
                counts = df.groupby("Motif").size().reset_index(name="Count")
                run_counts.append(counts)

            if run_counts:
                final_run_df = pd.concat(run_counts, ignore_index=True)

                # Aggregate again to handle any potential multi-file splits
                final_run_df = (
                    final_run_df.groupby("Motif")["Count"].sum().reset_index()
                )

                # Export to CSV formatted for preprocess_shuffled_dataframe
                out_csv = motif_counts_proc_dir / f"{random_stem}_MotifCounts.csv"
                final_run_df.to_csv(out_csv, index=False)

            del run_counts
            gc.collect()

    print("Shuffled network aggregation complete.")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Base directory containing the previously generated shuffled networks
    SHUFFLE_BASE_DIR = Path("./WTvsShuffledAnalysis_AbasyNets_Targeted")
    MOTIF_DIR = Path("./AllUniqueNets/Topologies/")
    GOINFO_DIR = Path("./GOInfo_Targeted")

    MOTIF_PATH_LIST = sorted(list(MOTIF_DIR.glob("*NS*.topo")))
    print(f"Identified {len(MOTIF_PATH_LIST)} motif paths.")

    num_cpus = max(1, mp.cpu_count() - 5)

    with mp.Pool(processes=num_cpus) as temp_pool:
        motif_dsl_dict = dict(temp_pool.map(_process_single_motif_def, MOTIF_PATH_LIST))

    print("Building metadata index...")
    metadata_index = build_metadata_index(GOINFO_DIR)

    flat_tasks = []

    # Iterate through the base network directories
    for base_net_dir in sorted([d for d in SHUFFLE_BASE_DIR.iterdir() if d.is_dir()]):
        base_net_name = base_net_dir.name

        # Extract the core name for metadata mapping
        clean_core = re.sub(r"_regNetwork.*$", "", base_net_name)
        clean_core = re.sub(r"_eStrong$", "", clean_core)
        tsv_path = metadata_index.get(clean_core)

        shuffle_dir = base_net_dir / "Shuffled_Networks"
        output_target_dir = base_net_dir / "MotifsCounts"

        if not shuffle_dir.exists():
            continue

        # Enqueue tasks for every random network generated for this base network
        for shuffled_topo in sorted(list(shuffle_dir.glob("*.topo"))):
            random_net_stem = shuffled_topo.stem

            for m_name, motif_obj in motif_dsl_dict.items():
                flat_tasks.append(
                    (
                        random_net_stem,
                        shuffled_topo,
                        tsv_path,
                        m_name,
                        motif_obj,
                        output_target_dir,
                    )
                )

    flat_tasks.sort(key=lambda x: x[0])
    print(f"Queue flattened: {len(flat_tasks)} tasks queued for {num_cpus} cores.")

    if flat_tasks:
        with mp.Pool(processes=num_cpus) as pool:
            for i, stats in enumerate(
                pool.imap_unordered(
                    worker_process_single_motif, flat_tasks, chunksize=10
                )
            ):
                if (i + 1) % 500 == 0:
                    print(f"Processed {i + 1}/{len(flat_tasks)} tasks...")

        print("All shuffled motifs successfully exported.")
    else:
        print("No tasks found.")

    compile_shuffled_motif_counts(SHUFFLE_BASE_DIR)
