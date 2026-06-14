import pandas as pd
import numpy as np
import networkx as nx
from dotmotif import Motif, GrandIsoExecutor
from pathlib import Path
import multiprocessing as mp
import re

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


def load_coh_graph(matrix_path, teams_path):
    if not matrix_path or not matrix_path.exists():
        return None, None

    coh_matrix = pd.read_parquet(matrix_path)
    group_map = {}

    if teams_path and teams_path.exists():
        teams_df = pd.read_csv(teams_path)
        group_map = dict(
            zip(teams_df["Node"].astype(str).str.strip(), teams_df["PreSplitGroup"])
        )

    if coh_matrix.index.nlevels == 2:
        coh_matrix.index.names = ["SourceGroup", "SourceNode"]
        coh_matrix.columns.names = ["TargetGroup", "TargetNode"]
        if not group_map:
            group_map = dict(
                zip(
                    coh_matrix.index.get_level_values("SourceNode")
                    .astype(str)
                    .str.strip(),
                    coh_matrix.index.get_level_values("SourceGroup"),
                )
            )
        coh_df = coh_matrix.stack(level=["TargetGroup", "TargetNode"]).reset_index()
    else:
        coh_matrix.index.name = "SourceNode"
        coh_matrix.columns.name = "TargetNode"
        coh_df = coh_matrix.stack().reset_index()

    coh_df = coh_df.rename(columns={0: "Value"})
    coh_df = coh_df.dropna(subset=["Value"])
    coh_df["sign"] = np.where(coh_df["Value"] <= 0, -1, 1)
    coh_df["SourceNode"] = coh_df["SourceNode"].astype(str).str.strip()
    coh_df["TargetNode"] = coh_df["TargetNode"].astype(str).str.strip()
    coh_df = coh_df[coh_df["SourceNode"] != coh_df["TargetNode"]]

    return nx.from_pandas_edgelist(
        coh_df,
        source="SourceNode",
        target="TargetNode",
        edge_attr="sign",
        create_using=nx.DiGraph(),
    ), group_map


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
    graph_type,
    level_map,
    nda_map,
    group_map,
    out_dir,
):
    """Executes GrandIso and dumps every raw match directly to an isolated Parquet file."""
    if G_target.number_of_edges() == 0:
        return 0

    executor = GrandIsoExecutor(graph=G_target)
    # Convert generator to list safely
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
            "Graph_Type": graph_type,
            "Node_Mapping": ", ".join([f"{k}:{match[k]}" for k in keys]),
            "Level_String": lvl,
            "NDA_String": nda,
        }

        if group_map:
            row["Group_String"] = "_".join(
                [str(group_map.get(n, "Unknown")) for n in target_nodes]
            )

        results.append(row)

    # Thread-safe write to disk (Filename guarantees unique write lock)
    df = pd.DataFrame(results)
    file_path = out_dir / f"{net_stem}_{m_name}_{graph_type}_Raw.parquet"
    df.to_parquet(file_path, index=False)

    return len(results)


def worker_process_single_motif(args):
    """
    Worker function. Receives ONE motif for ONE network.
    """
    (
        net_stem,
        topo_path,
        coh_path,
        teams_path,
        tsv_path,
        m_name,
        motif_obj,
        output_base_dir,
    ) = args

    # Smart Caching: Only read from SSD if we switched to a new network
    if worker_cache["net_stem"] != net_stem:
        worker_cache.clear()
        worker_cache["net_stem"] = net_stem
        worker_cache["G_topo"] = load_topo_graph(topo_path)
        worker_cache["G_coh"], worker_cache["group_map"] = load_coh_graph(
            coh_path, teams_path
        )
        worker_cache["level_map"], worker_cache["nda_map"] = load_metadata(tsv_path)

    G_topo = worker_cache["G_topo"]
    # G_coh = worker_cache["G_coh"]
    level_map = worker_cache["level_map"]
    nda_map = worker_cache["nda_map"]
    # group_map = worker_cache["group_map"]

    stats = {
        "Network": net_stem,
        "Motif": m_name,
        "Topo_Raw_Dumps": 0,
        # "Coh_Raw_Dumps": 0,
    }

    if motif_obj is None or not level_map:
        return stats

    out_dir = Path(output_base_dir) / net_stem / "MotifRawData"
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1. Dump Topo Matches
    if G_topo:
        stats["Topo_Raw_Dumps"] = _search_and_export_raw(
            G_topo,
            m_name,
            motif_obj,
            net_stem,
            "Topo",
            level_map,
            nda_map,
            None,
            out_dir,
        )

    # # 2. Dump Coh Matches
    # if G_coh:
    #     stats["Coh_Raw_Dumps"] = _search_and_export_raw(
    #         G_coh,
    #         m_name,
    #         motif_obj,
    #         net_stem,
    #         "Coh",
    #         level_map,
    #         nda_map,
    #         group_map,
    #         out_dir,
    #     )

    return stats


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    OUTPUT_DIR = Path("./GRNMotifCounts_Targeted/")
    MOTIF_DIR = Path("./AllUniqueNets/Topologies/")
    GRN_DIR = Path("./AbasyTOPOS_Targeted/")
    GOINFO_DIR = Path("./GOInfo_Targeted")
    ABASY_COH_DIR = Path("./AbasyCohResults_Targeted")

    MOTIF_PATH_LIST = sorted(list(MOTIF_DIR.glob("*NS*.topo")))
    print(f"Identified {len(MOTIF_PATH_LIST)} motif paths.")

    num_cpus = max(1, mp.cpu_count() - 5)

    with mp.Pool(processes=num_cpus) as temp_pool:
        motif_dsl_dict = dict(temp_pool.map(_process_single_motif_def, MOTIF_PATH_LIST))

    print("Building metadata index...")
    metadata_index = build_metadata_index(GOINFO_DIR)

    # 1. Flatten the Queue
    flat_tasks = []

    for gp in sorted(list(GRN_DIR.glob("*.topo"))):
        net_base = gp.stem
        clean_core = re.sub(r"_regNetwork.*$", "", net_base)
        clean_core = re.sub(r"_eStrong$", "", clean_core)

        tsv_path = metadata_index.get(clean_core)
        # coh_matrix_path = ABASY_COH_DIR / net_base / f"{net_base}_CohMat.parquet"
        coh_matrix_path = None
        teams_csv_path = ABASY_COH_DIR / net_base / f"{net_base}_Teams.csv"

        # if not coh_matrix_path.exists():
        #     coh_matrix_path = None
        if not teams_csv_path.exists():
            teams_csv_path = None

        for m_name, motif_obj in motif_dsl_dict.items():
            flat_tasks.append(
                (
                    net_base,
                    gp,
                    coh_matrix_path,
                    teams_csv_path,
                    tsv_path,
                    m_name,
                    motif_obj,
                    OUTPUT_DIR,
                )
            )

    flat_tasks.sort(key=lambda x: x[0])
    print(
        f"Queue flattened: {len(flat_tasks)} independent global motif tasks queued for {num_cpus} cores."
    )

    # 2. Process Everything Simultaneously
    if flat_tasks:
        with mp.Pool(processes=num_cpus) as pool:
            for i, stats in enumerate(
                pool.imap_unordered(
                    worker_process_single_motif, flat_tasks, chunksize=5
                )
            ):
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(flat_tasks)} tasks...")

        print("All raw motifs successfully exported to Parquet files.")
    else:
        print("No tasks found.")
