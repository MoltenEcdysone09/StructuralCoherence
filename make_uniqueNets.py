import pandas as pd
import numpy as np
from pathlib import Path
import concurrent.futures
from functools import partial
import re


# Function to get the edgelist from the triad code
DYAD_PAIRS = [("A", "B"), ("B", "A")]
TRIAD_PAIRS = [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("B", "C"), ("C", "B")]
VAL_MAP = {"P": 1, "N": -1, "0": 0}


# Function to create the adjmatrix from the troad code
def get_triad_adjmat(triad_code):
    print(triad_code)
    # Value map for replacing
    val_map = {"P": 1, "N": -1, "0": 0}
    # Constructing the adjacency matrix
    sign_string = triad_code.split("-")[1]
    adjmat = np.array(
        [
            0 if i is None else val_map[sign_string[i]]
            for i in [None, 0, 2, 1, None, 4, 3, 5, None]
        ]
    ).reshape(3, 3)
    return {triad_code: adjmat}


def get_edgelist(motif_code):
    sign_string = motif_code.split("-")[1]

    # Check if it's a dyad (length 2) or triad (length 6)
    if len(sign_string) == 2:
        pairs = DYAD_PAIRS
    else:
        pairs = TRIAD_PAIRS

    valid_edges = [
        (pairs[i][0], pairs[i][1], VAL_MAP[sign]) for i, sign in enumerate(sign_string)
    ]
    return pd.DataFrame(valid_edges, columns=["Source", "Target", "Type"])


dyad_matrices = {
    "001-00": np.array(
        [
            [0, 0],  # Null
            [0, 0],
        ]
    ),
    "010-P0": np.array(
        [
            [0, 1],  # Asymmetric Positive
            [0, 0],
        ]
    ),
    "010-N0": np.array(
        [
            [0, -1],  # Asymmetric Negative
            [0, 0],
        ]
    ),
    "100-PP": np.array(
        [
            [0, 1],  # Mutual Positive
            [1, 0],
        ]
    ),
    "100-NN": np.array(
        [
            [0, -1],  # Mutual Negative
            [-1, 0],
        ]
    ),
    "100-PN": np.array(
        [
            [0, 1],  # Mutual Mixed (Positive / Negative)
            [-1, 0],
        ]
    ),
}


def process_and_save_triad(motif_code, output_dir):
    """The worker function that each CPU core will execute."""
    df_edges = get_edgelist(motif_code)

    if not df_edges.empty:
        # Format base filename: replace '-' with '_'
        base_name = motif_code.replace("-", "_")

        # --- 1. Save standard version: No Self-Activation (_NS) ---
        file_name_ns = f"{base_name}_NS.topo"
        file_path_ns = output_dir / file_name_ns
        df_edges.to_csv(file_path_ns, sep=" ", index=False)

        # --- 2. Save modified version: With Self-Activation (_SA) ---
        # Identify non-isolated nodes (nodes involved in an edge where Type != 0)
        active_edges = df_edges[df_edges["Type"] != 0]
        active_nodes = set(active_edges["Source"]).union(set(active_edges["Target"]))

        # Create self-activation edges (Type 1 / Positive) for those active nodes
        sa_edges = [(node, node, 1) for node in active_nodes]
        df_sa_edges = pd.DataFrame(sa_edges, columns=["Source", "Target", "Type"])

        # Combine the original edges with the new self-activation edges
        df_combined = pd.concat([df_edges, df_sa_edges], ignore_index=True)

        file_name_sa = f"{base_name}_SA.topo"
        file_path_sa = output_dir / file_name_sa
        df_combined.to_csv(file_path_sa, sep=" ", index=False)

        return True  # Indicates files were saved

    return False


# Function to filter only networks which are capable of having a heirarchy
def label_all_motifs(triad_csv_path, dyad_data):
    """
    Labels both Triads (from CSV) and True Dyads (from dictionary)
    using consistent Middle Node and Category logic.
    """

    # --- 1. PROCESS TRIADS (3-node systems) ---
    df_triads = pd.read_csv(triad_csv_path)
    df_triads["CircuitType"] = "Triad"

    # MAN codes for Triads (strictly feedforward or fully integrated cycles)
    holistic_triads = [
        # "003",
        # "012",
        # "021D",
        # "021U",
        # "021C",
        "030T",
        "030C",
        # "120C",
        # "210",
        "300",
    ]

    def analyze_triad(val):
        prefix, signs = val.split("-")
        # Edges: A->B, B->A, A->C, C->A, B->C, C->B
        e = [1 if s != "0" else 0 for s in signs]
        # Nodes: A, B, C
        nodes = [
            (e[1] + e[3], e[0] + e[2]),
            (e[0] + e[5], e[1] + e[4]),
            (e[2] + e[4], e[3] + e[5]),
        ]

        has_middle = any(n[0] > 0 and n[1] > 0 for n in nodes)
        category = "Holistic" if prefix in holistic_triads else "Pendant"
        return category, True if has_middle else False

    triad_labels = df_triads["x"].apply(analyze_triad)
    df_triads["Category"], df_triads["MiddleNode"] = zip(*triad_labels)

    # --- 2. PROCESS DYADS (True 2-node systems) ---
    dyad_list = []
    for code, matrix in dyad_data.items():
        # Calculate degrees for a 2x2 matrix
        # Node 0: In = matrix[1,0], Out = matrix[0,1] (ignoring signs for presence)
        # Node 1: In = matrix[0,1], Out = matrix[1,0]
        n0_in, n0_out = abs(matrix[1, 0]), abs(matrix[0, 1])
        n1_in, n1_out = abs(matrix[0, 1]), abs(matrix[1, 0])

        # A true dyad has a middle node if at least one node has In > 0 and Out > 0
        # This only happens in 'Mutual' (100) dyads.
        has_middle = (n0_in > 0 and n0_out > 0) or (n1_in > 0 and n1_out > 0)

        # Category Logic: All true dyads are 'Holistic' because there is no
        # 3rd node to be "pendant" or isolated from.
        dyad_list.append(
            {
                "x": code,
                "CircuitType": "Dyad",
                "Category": "Holistic",
                "MiddleNode": True if has_middle else False,
            }
        )

    df_dyads = pd.DataFrame(dyad_list)

    # --- 3. COMBINE AND SAVE ---
    df_final = pd.concat([df_triads, df_dyads], ignore_index=True)
    df_final = df_final.rename(columns={"x": "MAN-PNStr"})
    temp_split = df_final["MAN-PNStr"].str.split("-", expand=True)
    df_final["MAN"] = temp_split[0]
    df_final["NumPosEdges"] = temp_split[1].str.count("P")
    df_final["NumNegEdges"] = temp_split[1].str.count("N")
    df_final["NumNullEdges"] = (
        temp_split[1].str.len() - df_final["NumPosEdges"] - df_final["NumNegEdges"]
    )
    df_final.to_csv("unified_labeled_circuits.csv", index=False)
    return df_final


def generate_unified_labels_from_topos(
    topo_dir, output_csv="unified_labeled_circuits.csv"
):
    """
    Scans the Topologies directory for .topo files and generates a labeled dataframe
    based on the filename convention: {MAN}_{EdgeString}_{SA|NS}.topo
    """
    topo_dir = Path(topo_dir)
    topo_files = list(topo_dir.glob("*.topo"))

    # MAN codes for Holistic Triads
    holistic_triads = [
        # "021D",
        # "021U",
        "030T",
        "030C",
        # "120C",
        # "210",
        "300",
    ]

    # Regex to extract components from the new naming scheme (e.g., 030T_PNN000_SA)
    pattern = re.compile(r"^([0-9A-Z]+)_([0PN]+)_(SA|NS)$")

    records = []

    for topo_path in topo_files:
        match = pattern.match(topo_path.stem)
        if not match:
            continue

        man_code, pn_str, sa_status = match.groups()

        # Determine Circuit Type based on EdgeString length (2 for Dyad, 6 for Triad)
        if len(pn_str) == 2:
            circuit_type = "Dyad"
        elif len(pn_str) == 6:
            circuit_type = "Triad"
        else:
            circuit_type = "Unknown"

        # Determine Category
        if circuit_type == "Dyad":
            category = "Holistic"
        else:
            category = "Holistic" if man_code in holistic_triads else "Pendant"

        # Calculate MiddleNode strictly structurally (ignoring self-activation status)
        e = [1 if s != "0" else 0 for s in pn_str]

        if circuit_type == "Dyad":
            # Nodes A and B. A_out=e[0], B_out=e[1], A_in=e[1], B_in=e[0]
            # Has middle if any node acts as both receiver and sender
            has_middle = e[0] > 0 and e[1] > 0

        elif circuit_type == "Triad":
            # Nodes A, B, C.
            nodes = [
                (e[1] + e[3], e[0] + e[2]),  # Node A: (In, Out)
                (e[0] + e[5], e[1] + e[4]),  # Node B: (In, Out)
                (e[2] + e[4], e[3] + e[5]),  # Node C: (In, Out)
            ]
            has_middle = any(n_in > 0 and n_out > 0 for n_in, n_out in nodes)

        records.append(
            {
                "TopoName": topo_path.stem,
                "MAN-PNStr": f"{man_code}-{pn_str}",
                "MAN": man_code,
                "EdgeString": pn_str,
                "SelfActivation": sa_status,
                "CircuitType": circuit_type,
                "Category": category,
                "MiddleNode": has_middle,
                "NumPosEdges": pn_str.count("P"),
                "NumNegEdges": pn_str.count("N"),
                "NumNullEdges": pn_str.count("0"),
            }
        )

    df_final = pd.DataFrame(records)

    if not df_final.empty:
        # Sort values to maintain a clean, organized output file
        df_final = df_final.sort_values(
            by=["CircuitType", "MAN", "EdgeString", "SelfActivation"], ignore_index=True
        )
        df_final.to_csv(output_csv, index=False)
        print(f"Processed {len(df_final)} network topologies. Saved to {output_csv}")
    else:
        print(f"No valid .topo files matching the convention were found in {topo_dir}")

    return df_final


# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Setup paths and load data
    triad_info_path = Path("./AllUniqueNets/signnet_138_codes.csv")
    triad_info_df = pd.read_csv(triad_info_path)
    triad_string_list = list(triad_info_df["x"])
    # Extract Dyad codes from dictionary keys
    dyad_string_list = list(dyad_matrices.keys())
    # Create list of dyads and triads
    full_motif_list = triad_string_list + dyad_string_list

    # 2. Create the output directory
    output_dir = Path("./AllUniqueNets/Topologies")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Create a 'partial' function so we only have to pass 'triad_code' in the map
    worker_func = partial(process_and_save_triad, output_dir=output_dir)

    # 4. Spin up the Multiprocessing Pool
    # ProcessPoolExecutor automatically uses all available CPU cores by default
    print(f"Starting multiprocessing pool across {len(triad_string_list)} networks...")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # executor.map distributes the triad_string_list across all CPU cores
        results = list(executor.map(worker_func, full_motif_list))

    # Count how many returned True (files actually saved)
    files_saved = sum(results) * 2

    print(f"Success! Saved {files_saved} topology files to: {output_dir.absolute()}")

    # Run the motif labelling function to get required motifs
    # df = label_all_motifs(triad_info_path, dyad_matrices)
    df = generate_unified_labels_from_topos(Path("./AllUniqueNets/Topologies"))
    print(df)
