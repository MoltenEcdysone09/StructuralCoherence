import numpy as np
import random
import sys
import os
import multiprocessing
from collections import defaultdict
import glob
import pandas as pd

# Check for required libraries
try:
    import networkx as nx
except ImportError:
    print("NetworkX library not found. Please install it using: pip install networkx")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Pandas library not found. Please install it using: pip install pandas")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ImportError:
    print(
        "Matplotlib library not found. Please install it using: pip install matplotlib"
    )
    sys.exit(1)

# --- 1. CORE UTILITIES ---


def generate_block_matrix(base_matrix, scale):
    base = np.array(base_matrix)
    if base.shape[0] != base.shape[1]:
        raise ValueError("The base matrix must be square.")
    ones_block = np.ones((scale, scale), dtype=base.dtype)
    return np.kron(base, ones_block)


def generate_node_names(base_matrix_shape, scale):
    node_names = []
    num_types = base_matrix_shape[0]
    for i in range(num_types):
        type_prefix = f"T{i + 1}"
        for j in range(1, scale + 1):
            node_suffix = f"{j:02d}"
            node_names.append(f"{type_prefix}_{node_suffix}")
    return node_names


def convert_matrices_to_edgelists(matrix_sequence_with_density, node_names):
    edgelist_sequence = []
    for _, matrix in matrix_sequence_with_density:
        df = pd.DataFrame(matrix, index=node_names, columns=node_names)
        stacked = df.stack()
        edgelist = stacked[stacked != 0].reset_index()
        edgelist.columns = ["Source", "Target", "Type"]
        edgelist_sequence.append(edgelist)
    return edgelist_sequence


def load_base_matrices_from_csvs(csv_paths):
    base_matrices_info = []
    for path in csv_paths:
        try:
            base_name = os.path.basename(path)
            matrix_name, _ = os.path.splitext(base_name)
            df = pd.read_csv(path, index_col=0)
            matrix_array = df.to_numpy()
            matrix_array[matrix_array == 2] = -1  # Convert 2s to -1s
            base_matrices_info.append({"name": matrix_name, "matrix": matrix_array})
        except FileNotFoundError:
            print(f"Warning: The file was not found at {path}. Skipping.")
        except Exception as e:
            print(f"Warning: Could not process file {path}. Error: {e}. Skipping.")
    return base_matrices_info


def ensure_connected_topology(current_mask, allowed_topology_mask):
    """
    Enforces Weak Connectivity by SWAPPING edges within the allowed topology.
    Does not change the total number of edges.
    """
    G = nx.from_numpy_array(current_mask, create_using=nx.DiGraph)
    if nx.is_weakly_connected(G):
        return current_mask

    matrix = current_mask.copy()
    max_repairs = matrix.shape[0] * 2
    repairs = 0

    while not nx.is_weakly_connected(G) and repairs < max_repairs:
        repairs += 1
        comps = sorted(list(nx.weakly_connected_components(G)), key=len, reverse=True)
        main_comp = list(comps[0])
        merged_this_round = False

        for i in range(1, len(comps)):
            target_comp = list(comps[i])
            valid_bridges = []

            for u in main_comp:
                for v in target_comp:
                    if allowed_topology_mask[u, v] == 1 and matrix[u, v] == 0:
                        valid_bridges.append((u, v))
            for u in target_comp:
                for v in main_comp:
                    if allowed_topology_mask[u, v] == 1 and matrix[u, v] == 0:
                        valid_bridges.append((u, v))

            if valid_bridges:
                u_add, v_add = random.choice(valid_bridges)
                matrix[u_add, v_add] = 1
                existing_edges = np.argwhere(matrix == 1)
                candidates_for_removal = [
                    tuple(e)
                    for e in existing_edges
                    if not (e[0] == u_add and e[1] == v_add)
                ]

                if candidates_for_removal:
                    u_rem, v_rem = random.choice(candidates_for_removal)
                    matrix[u_rem, v_rem] = 0
                    merged_this_round = True
                    break

        if not merged_this_round:
            break

        G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)

    return matrix


# --- 2. PIPELINE HELPERS ---


def apply_and_save_masks(
    base_matrix, base_matrix_name, scale, mask_dict, replicate_number, output_folder
):
    base_dim = np.array(base_matrix).shape[0]
    dense_matrix = generate_block_matrix(base_matrix, scale)
    node_names = generate_node_names(
        base_matrix_shape=(base_dim, base_dim), scale=scale
    )
    scale_str = str(scale).zfill(3)
    rep_str = str(replicate_number).zfill(3)

    for mask_label, mask_array in mask_dict.items():
        if mask_array.shape != dense_matrix.shape:
            continue

        masked_matrix = dense_matrix * mask_array
        df = convert_matrices_to_edgelists([(1.0, masked_matrix)], node_names)[0]

        filename = f"{scale_str}N_{base_matrix_name}TN_{mask_label}_{rep_str}R.topo"
        filepath = os.path.join(output_folder, filename)
        df.to_csv(filepath, sep=" ", index=False)


def _simulation_worker(args):
    (
        rep,
        base_matrices_info,
        scales,
        output_folder,
        mask_generator_func,
        mask_gen_kwargs,
    ) = args
    grouped_matrices = defaultdict(list)
    for info in base_matrices_info:
        base_dim = np.array(info["matrix"]).shape[0]
        grouped_matrices[base_dim].append(info)

    for scale in scales:
        for base_dim, info_list in grouped_matrices.items():
            current_mask_kwargs = mask_gen_kwargs.copy()
            current_mask_kwargs["base_dim"] = base_dim
            try:
                mask_dict = mask_generator_func(scale, rep, **current_mask_kwargs)
            except Exception as e:
                print(
                    f"ERROR: Mask generation failed for Scale={scale}, Rep={rep}. Error: {e}"
                )
                continue
            if not mask_dict:
                continue

            print(
                f"Processing: Rep={rep}, Scale={scale}, Dim={base_dim}, {len(mask_dict)} masks."
            )
            for info in info_list:
                apply_and_save_masks(
                    info["matrix"], info["name"], scale, mask_dict, rep, output_folder
                )


def run_simulation_from_generated_masks(
    base_matrices_info,
    scales,
    replicates,
    output_folder,
    mask_generator_func,
    mask_gen_kwargs={},
):
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving final networks to: {os.path.abspath(output_folder)}")
    tasks = [
        (
            rep,
            base_matrices_info,
            scales,
            output_folder,
            mask_generator_func,
            mask_gen_kwargs,
        )
        for rep in replicates
    ]
    for task in tasks:
        _simulation_worker(task)
    print("\nSimulation complete.")


# def generate_dense_networks_only(base_matrices_info, scales, output_folder):
#     os.makedirs(output_folder, exist_ok=True)
#     print(f"\n--- Saving 100D networks in: {os.path.abspath(output_folder)} ---")
#     for info in base_matrices_info:
#         base_matrix = info["matrix"]
#         base_matrix_name = info["name"]
#         for scale in scales:
#             dense_matrix = generate_block_matrix(base_matrix, scale)
#             node_names = generate_node_names(np.array(base_matrix).shape, scale)
#             edgelist_df = convert_matrices_to_edgelists(
#                 [(1.0, dense_matrix)], node_names
#             )[0]
#             filename = f"{str(scale).zfill(3)}N_{base_matrix_name}TN_100D_ER_000R.topo"
#             filepath = os.path.join(output_folder, filename)
#             edgelist_df.to_csv(filepath, sep=" ", index=False)


def generate_dense_networks_only(
    base_matrices_info,
    scales,
    output_folder,
    sub_group_proportions=None,
    h_matrices_dict=None,
):
    """
    Generates:
    1. 100D ER (The fully dense block matrix).
    2. 100D HI (The fully dense version of the Hierarchical Topology), if dict provided.
    """
    os.makedirs(output_folder, exist_ok=True)
    print(
        f"\n--- Saving 100D networks (ER + HI Baselines) in: {os.path.abspath(output_folder)} ---"
    )

    for info in base_matrices_info:
        base_matrix = info["matrix"]
        base_matrix_name = info["name"]
        base_dim = base_matrix.shape[0]

        for scale in scales:
            # --- 1. Generate ER (Global 100D) ---
            dense_matrix = generate_block_matrix(base_matrix, scale)
            node_names = generate_node_names(np.array(base_matrix).shape, scale)

            # Save ER
            edgelist_df = convert_matrices_to_edgelists(
                [(1.0, dense_matrix)], node_names
            )[0]
            filename = f"{str(scale).zfill(3)}N_{base_matrix_name}TN_100D_ER_000R.topo"
            filepath = os.path.join(output_folder, filename)
            edgelist_df.to_csv(filepath, sep=" ", index=False)

            # --- 2. Generate HI Baselines (Topology 100D) ---
            if h_matrices_dict and sub_group_proportions:
                total_size = base_dim * scale
                shape = (total_size, total_size)
                num_hei_lvls = len(sub_group_proportions)

                # Reuse the robust index calculator to ensure alignment matches the simulation
                grouped_indices, corrected_node_counts = _calculate_sub_block_indices(
                    base_dim, scale, sub_group_proportions
                )
                P, P_inv = _get_permutation_indices(grouped_indices, total_size)

                # Calculate Global Boundaries for the permuted view
                rearranged_boundaries_global = []
                curr = 0
                for c in corrected_node_counts:
                    rearranged_boundaries_global.append(curr)
                    curr += c * base_dim
                rearranged_boundaries_global.append(curr)

                for h_label, h_matrix in h_matrices_dict.items():
                    # Build Allowed Map (This IS the 100D HI Matrix)
                    hi_allowed_rearranged = np.zeros(shape, dtype=float)
                    for i in range(num_hei_lvls):
                        for j in range(num_hei_lvls):
                            # 0.0 means Allowed, 1.0 means Blocked (Sparsity Logic)
                            if h_matrix[i, j] < 1.0:
                                r_start = rearranged_boundaries_global[i]
                                r_end = rearranged_boundaries_global[i + 1]
                                c_start = rearranged_boundaries_global[j]
                                c_end = rearranged_boundaries_global[j + 1]
                                hi_allowed_rearranged[r_start:r_end, c_start:c_end] = (
                                    1.0
                                )

                    # Permute back to node space
                    hi_allowed_map = hi_allowed_rearranged[P_inv, :][:, P_inv]

                    # Apply to Dense Matrix (Intersection of Dense Block & HI Topology)
                    hi_dense_matrix = dense_matrix * hi_allowed_map

                    # Save HI
                    edgelist_df_hi = convert_matrices_to_edgelists(
                        [(1.0, hi_dense_matrix)], node_names
                    )[0]
                    filename_hi = f"{str(scale).zfill(3)}N_{base_matrix_name}TN_100D_{h_label}_000R.topo"
                    filepath_hi = os.path.join(output_folder, filename_hi)
                    edgelist_df_hi.to_csv(filepath_hi, sep=" ", index=False)


# --- 5. CONNECTED & STRUCTURED MASK GENERATORS ---


def _calculate_sub_block_indices(base_dim, scale, proportions):
    """
    Calculates node counts for subgroups with a safety guarantee.
    RETURNS THE CORRECTED NODE COUNTS.
    """
    raw_counts = np.array(proportions) * scale
    node_counts = np.round(raw_counts).astype(int)

    # Safety: Ensure at least 1 node if proportion > 0
    for i in range(len(node_counts)):
        if node_counts[i] == 0 and proportions[i] > 0:
            node_counts[i] = 1

    # Re-balance
    current_sum = node_counts.sum()
    diff = scale - current_sum

    if diff != 0:
        target_idx = np.argmax(node_counts)
        if node_counts[target_idx] + diff >= 1:
            node_counts[target_idx] += diff
        else:
            for i in range(len(node_counts)):
                if node_counts[i] + diff >= 1:
                    node_counts[i] += diff
                    break

    boundaries_within_block = np.cumsum([0] + list(node_counts))
    grouped_indices = [[] for _ in proportions]

    for k in range(len(proportions)):
        start_node, end_node = (
            boundaries_within_block[k],
            boundaries_within_block[k + 1],
        )
        for i in range(base_dim):
            block_offset = i * scale
            grouped_indices[k].extend(
                range(block_offset + start_node, block_offset + end_node)
            )

    return grouped_indices, node_counts


def _get_permutation_indices(grouped_indices, total_size):
    P = np.concatenate(grouped_indices).astype(int)
    P_inv = np.empty_like(P)
    P_inv[P] = np.arange(total_size, dtype=int)
    return P, P_inv


def generate_progressive_eruthi_mask_dict(
    scale,
    replicate_number,
    base_dim,
    sparsity_levels,
    sub_group_proportions,
    h_matrices_dict,
    block_density_constraints=None,
):
    """
    Generates ER and HI masks INDEPENDENTLY.
    - ER: Pure random, follows global density.
    - HI: Structured, follows Block Density Overrides (or global if -1).
    - Disconnected: HI is NOT a subset of ER. They are generated separately.
    """
    random.seed(replicate_number)
    np.random.seed(replicate_number)

    total_size = base_dim * scale
    shape = (total_size, total_size)
    num_hei_lvls = len(sub_group_proportions)

    if block_density_constraints is None:
        block_density_constraints = np.full((num_hei_lvls, num_hei_lvls), -1.0)
    else:
        block_density_constraints = np.array(block_density_constraints)

    # --- 1. SETUP TOPOLOGY MAPS ---
    er_allowed = np.ones(shape, dtype=float)

    # Use CORRECTED counts
    grouped_indices, corrected_node_counts = _calculate_sub_block_indices(
        base_dim, scale, sub_group_proportions
    )
    P, P_inv = _get_permutation_indices(grouped_indices, total_size)

    # Global Boundaries
    rearranged_boundaries_global = []
    curr = 0
    for c in corrected_node_counts:
        rearranged_boundaries_global.append(curr)
        curr += c * base_dim
    rearranged_boundaries_global.append(curr)

    # Block Maps (for HI construction)
    block_binary_maps = {}
    for i in range(num_hei_lvls):
        for j in range(num_hei_lvls):
            rearranged = np.zeros(shape, dtype=float)
            r_start = rearranged_boundaries_global[i]
            r_end = rearranged_boundaries_global[i + 1]
            c_start = rearranged_boundaries_global[j]
            c_end = rearranged_boundaries_global[j + 1]
            rearranged[r_start:r_end, c_start:c_end] = 1.0
            block_binary_maps[(i, j)] = rearranged[P_inv, :][:, P_inv]

    # HI Allowed Maps
    h_allowed_maps = {}
    for h_label, h_matrix in h_matrices_dict.items():
        hi_allowed_rearranged = np.zeros(shape, dtype=float)
        for i in range(num_hei_lvls):
            for j in range(num_hei_lvls):
                # 0.0 means Allowed, 1.0 means Blocked
                if h_matrix[i, j] < 1.0:
                    r_start = rearranged_boundaries_global[i]
                    r_end = rearranged_boundaries_global[i + 1]
                    c_start = rearranged_boundaries_global[j]
                    c_end = rearranged_boundaries_global[j + 1]
                    hi_allowed_rearranged[r_start:r_end, c_start:c_end] = 1.0

        h_allowed_maps[h_label] = hi_allowed_rearranged[P_inv, :][:, P_inv]

    final_mask_dict = {}

    # Sort High -> Low Sparsity (Low -> High Density)
    sorted_sparsities = sorted(sparsity_levels, reverse=True)

    # INDEPENDENT STATE TRACKERS
    prev_er_mask = np.zeros(shape, dtype=float)
    prev_hi_masks = {k: np.zeros(shape, dtype=float) for k in h_matrices_dict.keys()}

    for sparsity in sorted_sparsities:
        global_density = 1.0 - sparsity
        label_prefix = str(int(round(global_density * 100))).zfill(3) + "D"

        # --- TRACK A: ER GENERATION (Pure Random) ---
        curr_er = prev_er_mask.copy()
        target_er_edges = int((total_size**2) * global_density)
        current_er_count = np.count_nonzero(curr_er)
        needed_er = target_er_edges - current_er_count

        if needed_er > 0:
            empty_indices = np.where((curr_er.ravel() == 0))[0]
            if len(empty_indices) > 0:
                count = min(needed_er, len(empty_indices))
                chosen = np.random.choice(empty_indices, size=count, replace=False)
                curr_er.ravel()[chosen] = 1.0

        # Connectivity Check for ER
        curr_er = ensure_connected_topology(curr_er, er_allowed)

        final_mask_dict[f"{label_prefix}_ER"] = curr_er.copy()
        prev_er_mask = curr_er  # Save state for next density level

        # --- TRACK B: HI GENERATION (Structured & Independent) ---
        for h_label, allowed_map in h_allowed_maps.items():
            curr_hi = prev_hi_masks[h_label].copy()

            # Fill each block according to overrides or global defaults
            for i in range(num_hei_lvls):
                for j in range(num_hei_lvls):
                    # Check if block is allowed in topology (Sparsity < 1.0)
                    if h_matrices_dict[h_label][i, j] >= 1.0:
                        continue  # Block is forbidden

                    # Determine Target Density for this block
                    override_val = block_density_constraints[i, j]

                    if override_val >= 0.0:
                        target_density_for_block = override_val
                    else:
                        target_density_for_block = global_density

                    # Fill edges in this block
                    block_map = block_binary_maps[(i, j)]
                    block_slots_count = np.count_nonzero(block_map)

                    target_block_count = int(
                        block_slots_count * target_density_for_block
                    )

                    # Calculate edges ONLY within this block
                    current_block_edges = curr_hi * block_map
                    current_block_count = np.count_nonzero(current_block_edges)
                    needed_block = target_block_count - current_block_count

                    if needed_block > 0:
                        empty_block_indices = np.where(
                            (block_map.ravel() == 1) & (curr_hi.ravel() == 0)
                        )[0]
                        if len(empty_block_indices) > 0:
                            count_add = min(needed_block, len(empty_block_indices))
                            chosen_add = np.random.choice(
                                empty_block_indices, size=count_add, replace=False
                            )
                            curr_hi.ravel()[chosen_add] = 1.0

            # Connectivity Check for HI
            curr_hi = ensure_connected_topology(curr_hi, allowed_map)

            final_mask_dict[f"{label_prefix}_{h_label}"] = curr_hi.copy()
            prev_hi_masks[h_label] = curr_hi  # Save state for next density level

    return final_mask_dict


def generate_progressive_team_eruthi_mask_dict(
    scale,
    replicate_number,
    base_dim,
    sparsity_levels,
    sub_group_proportions,
    h_matrices_dict,
    block_density_constraints=None,
):
    # Set seeds for reproducibility
    random.seed(replicate_number)
    np.random.seed(replicate_number)

    total_size = base_dim * scale
    shape = (total_size, total_size)
    num_hei_lvls = len(sub_group_proportions)

    # Default constraints if none provided
    if block_density_constraints is None:
        block_density_constraints = np.full((num_hei_lvls, num_hei_lvls), -1.0)
    else:
        block_density_constraints = np.array(block_density_constraints)

    # --- 1. SETUP TOPOLOGY MAPS ---
    er_allowed = np.ones(shape, dtype=float)

    # Calculate Group Indices (Input / Middle / Output)
    grouped_indices, corrected_node_counts = _calculate_sub_block_indices(
        base_dim, scale, sub_group_proportions
    )
    # P maps: Sorted Index -> Real Node Index
    P, P_inv = _get_permutation_indices(grouped_indices, total_size)

    # Helper to identify which Team a node belongs to
    # Because of np.kron, Team 0 is 0..scale-1, Team 1 is scale..2*scale-1, etc.
    get_team = lambda node_idx: node_idx // scale

    # Calculate Boundaries in the Sorted (Grouped) Space
    rearranged_boundaries_global = []
    curr = 0
    for c in corrected_node_counts:
        rearranged_boundaries_global.append(curr)
        curr += c * base_dim
    rearranged_boundaries_global.append(curr)

    # Pre-compute Block Maps (Binary masks for Layer i -> Layer j)
    # stored in REAL NODE SPACE
    block_binary_maps = {}
    for i in range(num_hei_lvls):
        for j in range(num_hei_lvls):
            # Create block in Sorted Space
            rearranged = np.zeros(shape, dtype=float)
            r_start = rearranged_boundaries_global[i]
            r_end = rearranged_boundaries_global[i + 1]
            c_start = rearranged_boundaries_global[j]
            c_end = rearranged_boundaries_global[j + 1]
            rearranged[r_start:r_end, c_start:c_end] = 1.0

            # Convert to Real Node Space
            block_binary_maps[(i, j)] = rearranged[P_inv, :][:, P_inv]

    # Pre-compute Global Topology Constraints (HI Masks)
    h_allowed_maps = {}
    for h_label, h_matrix in h_matrices_dict.items():
        hi_allowed_rearranged = np.zeros(shape, dtype=float)
        for i in range(num_hei_lvls):
            for j in range(num_hei_lvls):
                if h_matrix[i, j] < 1.0:  # If allowed
                    r_start = rearranged_boundaries_global[i]
                    r_end = rearranged_boundaries_global[i + 1]
                    c_start = rearranged_boundaries_global[j]
                    c_end = rearranged_boundaries_global[j + 1]
                    hi_allowed_rearranged[r_start:r_end, c_start:c_end] = 1.0
        h_allowed_maps[h_label] = hi_allowed_rearranged[P_inv, :][:, P_inv]

    final_mask_dict = {}
    sorted_sparsities = sorted(sparsity_levels, reverse=True)

    # State trackers to maintain edges across density levels
    prev_er_mask = np.zeros(shape, dtype=float)
    prev_hi_masks = {k: np.zeros(shape, dtype=float) for k in h_matrices_dict.keys()}

    # --- MAIN LOOP ---
    for sparsity in sorted_sparsities:
        global_density = 1.0 - sparsity
        label_prefix = str(int(round(global_density * 100))).zfill(3) + "D"

        # --- TRACK A: ER GENERATION (Standard) ---
        curr_er = prev_er_mask.copy()
        target_er_edges = int((total_size**2) * global_density)
        needed_er = target_er_edges - np.count_nonzero(curr_er)

        if needed_er > 0:
            empty_indices = np.where((curr_er.ravel() == 0))[0]
            if len(empty_indices) > 0:
                count = min(needed_er, len(empty_indices))
                chosen = np.random.choice(empty_indices, size=count, replace=False)
                curr_er.ravel()[chosen] = 1.0

        curr_er = ensure_connected_topology(curr_er, er_allowed)
        final_mask_dict[f"{label_prefix}_ER"] = curr_er.copy()
        prev_er_mask = curr_er

        # --- TRACK B: HI GENERATION (With Same-Team Rescue) ---
        for h_label, allowed_map in h_allowed_maps.items():
            curr_hi = prev_hi_masks[h_label].copy()

            for i in range(num_hei_lvls):
                for j in range(num_hei_lvls):
                    # Skip forbidden blocks
                    if h_matrices_dict[h_label][i, j] >= 1.0:
                        continue

                    # 1. Calculate Budget for this Block
                    override_val = block_density_constraints[i, j]
                    target_density = (
                        override_val if override_val >= 0.0 else global_density
                    )

                    block_map = block_binary_maps[(i, j)]
                    total_slots = np.count_nonzero(block_map)
                    target_edges = int(total_slots * target_density)

                    current_edges = np.count_nonzero(curr_hi * block_map)
                    needed = target_edges - current_edges

                    if needed <= 0:
                        continue

                    # --- 2. SAME-TEAM RESCUE OPERATION ---
                    # Identify real indices for Source Layer (i) and Target Layer (j)
                    # We use the sorted boundaries + P mapping
                    src_indices = P[
                        rearranged_boundaries_global[i] : rearranged_boundaries_global[
                            i + 1
                        ]
                    ]
                    tgt_indices = P[
                        rearranged_boundaries_global[j] : rearranged_boundaries_global[
                            j + 1
                        ]
                    ]

                    # Shuffle targets to prevent index bias (saving Node 0 first)
                    tgt_indices_shuffled = tgt_indices.copy()
                    np.random.shuffle(tgt_indices_shuffled)

                    for t_node in tgt_indices_shuffled:
                        if needed <= 0:
                            break

                        t_team = get_team(t_node)

                        # Check if t_node is an "Orphan" (Has NO upstream edges from same team)
                        # Look at entire column in curr_hi
                        incoming_srcs = np.where(curr_hi[:, t_node] == 1)[0]

                        has_same_team_parent = False
                        for src in incoming_srcs:
                            if get_team(src) == t_team:
                                has_same_team_parent = True
                                break

                        if not has_same_team_parent:
                            # Attempt Rescue: Find a valid parent in CURRENT source layer
                            # Must be Same Team AND not already connected
                            valid_parents = [
                                s
                                for s in src_indices
                                if get_team(s) == t_team and curr_hi[s, t_node] == 0
                            ]

                            if valid_parents:
                                chosen_parent = np.random.choice(valid_parents)
                                curr_hi[chosen_parent, t_node] = 1.0
                                needed -= 1  # Deduct from budget

                    # --- 3. RANDOM FILL (For remaining budget) ---
                    if needed > 0:
                        # Find all empty slots in this block
                        empty_slots = np.where(
                            (block_map.ravel() == 1) & (curr_hi.ravel() == 0)
                        )[0]
                        if len(empty_slots) > 0:
                            count = min(needed, len(empty_slots))
                            chosen = np.random.choice(
                                empty_slots, size=count, replace=False
                            )
                            curr_hi.ravel()[chosen] = 1.0

            curr_hi = ensure_connected_topology(curr_hi, allowed_map)
            final_mask_dict[f"{label_prefix}_{h_label}"] = curr_hi.copy()
            prev_hi_masks[h_label] = curr_hi

    return final_mask_dict


def generate_progressive_team_eruthi_mask_dict(
    scale,
    replicate_number,
    base_dim,
    sparsity_levels,
    sub_group_proportions,
    h_matrices_dict,
    block_density_constraints=None,
):
    import numpy as np
    import random

    # ------------------ SETUP ------------------
    random.seed(replicate_number)
    np.random.seed(replicate_number)

    total_size = base_dim * scale
    shape = (total_size, total_size)
    num_hei_lvls = len(sub_group_proportions)

    if block_density_constraints is None:
        block_density_constraints = np.full((num_hei_lvls, num_hei_lvls), -1.0)
    else:
        block_density_constraints = np.array(block_density_constraints)

    er_allowed = np.ones(shape, dtype=float)
    get_team = lambda node: node // scale

    # ------------------ GROUPING ------------------
    grouped_indices, corrected_node_counts = _calculate_sub_block_indices(
        base_dim, scale, sub_group_proportions
    )
    P, P_inv = _get_permutation_indices(grouped_indices, total_size)

    boundaries = [0]
    for c in corrected_node_counts:
        boundaries.append(boundaries[-1] + c * base_dim)

    # ------------------ BLOCK MAPS ------------------
    block_binary_maps = {}
    for i in range(num_hei_lvls):
        for j in range(num_hei_lvls):
            rearranged = np.zeros(shape, dtype=float)
            r0, r1 = boundaries[i], boundaries[i + 1]
            c0, c1 = boundaries[j], boundaries[j + 1]
            rearranged[r0:r1, c0:c1] = 1.0
            block_binary_maps[(i, j)] = rearranged[P_inv, :][:, P_inv]

    # ------------------ TOPOLOGY MAPS ------------------
    h_allowed_maps = {}
    for h_label, h_matrix in h_matrices_dict.items():
        allowed = np.zeros(shape, dtype=float)
        for i in range(num_hei_lvls):
            for j in range(num_hei_lvls):
                if h_matrix[i, j] < 1.0:
                    r0, r1 = boundaries[i], boundaries[i + 1]
                    c0, c1 = boundaries[j], boundaries[j + 1]
                    allowed[r0:r1, c0:c1] = 1.0
        h_allowed_maps[h_label] = allowed[P_inv, :][:, P_inv]

    # ------------------ STATE ------------------
    final_mask_dict = {}
    sorted_sparsities = sorted(sparsity_levels, reverse=True)

    prev_er_mask = np.zeros(shape, dtype=float)
    prev_hi_masks = {k: np.zeros(shape, dtype=float) for k in h_matrices_dict}

    # ================== MAIN LOOP ==================
    for sparsity in sorted_sparsities:
        global_density = 1.0 - sparsity
        label_prefix = f"{int(round(global_density * 100)):03d}D"

        # =====================================================
        # TRACK A: ER (PURE RANDOM, PROGRESSIVE)
        # =====================================================
        curr_er = prev_er_mask.copy()
        target_edges = int((total_size**2) * global_density)
        needed = target_edges - np.count_nonzero(curr_er)

        if needed > 0:
            empty = np.where(curr_er.ravel() == 0)[0]
            count = min(needed, len(empty))
            if count > 0:
                chosen = np.random.choice(empty, size=count, replace=False)
                curr_er.ravel()[chosen] = 1.0

        curr_er = ensure_connected_topology(curr_er, er_allowed)
        final_mask_dict[f"{label_prefix}_ER"] = curr_er.copy()
        prev_er_mask = curr_er

        # =====================================================
        # TRACK B: HI (STRUCTURED + TEAM-AWARE)
        # =====================================================
        for h_label, h_matrix in h_matrices_dict.items():
            curr_hi = prev_hi_masks[h_label].copy()
            allowed_map = h_allowed_maps[h_label]

            # ---- Compute per-block budgets ----
            block_budget = {}
            for i in range(num_hei_lvls):
                for j in range(num_hei_lvls):
                    if h_matrix[i, j] >= 1.0:
                        continue

                    density = (
                        block_density_constraints[i, j]
                        if block_density_constraints[i, j] >= 0
                        else global_density
                    )

                    block_map = block_binary_maps[(i, j)]
                    target = int(np.count_nonzero(block_map) * density)
                    current = np.count_nonzero(curr_hi * block_map)
                    block_budget[(i, j)] = max(0, target - current)

            # ---- TARGET-CENTRIC, POOLED UPSTREAM RESCUE ----
            for j in range(1, num_hei_lvls):
                targets = grouped_indices[j].copy()
                random.shuffle(targets)

                upstream = [i for i in range(j) if (i, j) in block_budget]

                for v in targets:
                    if any(
                        get_team(u) == get_team(v)
                        for u in np.where(curr_hi[:, v] == 1)[0]
                    ):
                        continue

                    feasible = []
                    for i in upstream:
                        if block_budget[(i, j)] <= 0:
                            continue
                        for u in grouped_indices[i]:
                            if (
                                get_team(u) == get_team(v)
                                and curr_hi[u, v] == 0
                                and block_binary_maps[(i, j)][u, v] == 1
                            ):
                                feasible.append((i, u))

                    if feasible:
                        i, u = random.choice(feasible)
                        curr_hi[u, v] = 1.0
                        block_budget[(i, j)] -= 1

            # ---- RANDOM FILL ----
            for (i, j), needed in block_budget.items():
                if needed <= 0:
                    continue
                block_map = block_binary_maps[(i, j)]
                empty = np.where((block_map.ravel() == 1) & (curr_hi.ravel() == 0))[0]
                count = min(needed, len(empty))
                if count > 0:
                    chosen = np.random.choice(empty, size=count, replace=False)
                    curr_hi.ravel()[chosen] = 1.0

            curr_hi = ensure_connected_topology(curr_hi, allowed_map)
            final_mask_dict[f"{label_prefix}_{h_label}"] = curr_hi.copy()
            prev_hi_masks[h_label] = curr_hi

    return final_mask_dict


def plot_adjacency_heatmaps_sorted_by_role(
    output_folder, plot_subfolder_name="_Heatmaps_Sorted"
):
    plot_folder = os.path.join(output_folder, plot_subfolder_name)
    os.makedirs(plot_folder, exist_ok=True)
    print(f"\n--- Generating Sorted Heatmaps in: {os.path.abspath(plot_folder)} ---")

    cmap = mcolors.ListedColormap(["#FF3333", "#FFFFFF", "#3366FF"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    all_files = sorted(
        glob.glob(os.path.join(output_folder, "010N_2_3_0TN_*D_*_00*R.topo"))
    )
    if not all_files:
        return

    count = 0
    for filepath in all_files:
        filename = os.path.basename(filepath)
        try:
            try:
                df = pd.read_csv(filepath, sep=" ")
            except pd.errors.EmptyDataError:
                continue
            if df.empty or len(df) == 0:
                continue

            all_nodes = set(df["Source"]).union(set(df["Target"]))
            if not all_nodes:
                continue

            out_degrees = df["Source"].value_counts()
            in_degrees = df["Target"].value_counts()
            node_data = []
            for node in all_nodes:
                n_out, n_in = out_degrees.get(node, 0), in_degrees.get(node, 0)
                if n_in == 0 and n_out == 0:
                    role, lbl = 3, "(Iso)"
                elif n_in == 0:
                    role, lbl = 0, "(I)"
                elif n_out == 0:
                    role, lbl = 2, "(O)"
                else:
                    role, lbl = 1, "(M)"
                node_data.append({"node": node, "role": role, "label": f"{lbl} {node}"})

            sorted_node_data = sorted(node_data, key=lambda x: (x["role"], x["node"]))
            sorted_nodes = [d["node"] for d in sorted_node_data]
            sorted_labels = [d["label"] for d in sorted_node_data]

            adj_df = pd.DataFrame(
                0, index=sorted_nodes, columns=sorted_nodes, dtype=float
            )
            for _, row in df.iterrows():
                adj_df.at[row["Source"], row["Target"]] = row["Type"]

            num_nodes = len(sorted_nodes)
            fig_dim = max(10, num_nodes * 0.2)
            font_size = max(4, 12 - (num_nodes // 25))

            plt.figure(figsize=(fig_dim, fig_dim * 0.85))
            img = plt.imshow(
                adj_df.to_numpy(),
                cmap=cmap,
                norm=norm,
                origin="upper",
                interpolation="nearest",
            )
            plt.colorbar(img, ticks=[-1, 0, 1], shrink=0.7)
            plt.title(
                f"Sorted Adjacency: {filename}\n(Inputs Top-Left -> Outputs Bottom-Right)",
                fontsize=14,
            )
            plt.xticks(range(num_nodes), sorted_labels, rotation=90, fontsize=font_size)
            plt.yticks(range(num_nodes), sorted_labels, fontsize=font_size)
            plt.tight_layout()

            plt.savefig(
                os.path.join(plot_folder, filename.replace(".topo", "_sorted.png")),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
            plt.show()
            count += 1
            if count % 10 == 0:
                print(f"  Plots: {count}...")
        except Exception as e:
            print(f"Error {filename}: {e}")
    print(f"Done. {count} plots generated.")


if __name__ == "__main__":
    base_matrix_folder = "../../BaseMats/"
    # final_output_folder = "../../ArtiToposV3/"
    final_output_folder = "../../ArtiTopos_1N/"
    # base_matrix_folder = "./BaseMats/"
    # final_output_folder = "/mnt/4TB/Pradyumna/CohModTroph/ArtiToposV3/"

    base_matrix_files = sorted(
        [
            os.path.join(base_matrix_folder, f)
            for f in os.listdir(base_matrix_folder)
            if f.endswith(".csv")
        ]
    )
    base_matrices = load_base_matrices_from_csvs(base_matrix_files)

    if not base_matrices:
        print("Error: No base matrix CSV files found. Exiting.")
        sys.exit(1)

    # scales = [10, 20, 30, 40, 50]
    # replicates_to_run = range(1, 21)
    scales = [10, 50]
    replicates_to_run = range(1, 21)

    # --- DEFINE PARAMETERS FIRST ---

    # 1. Proportions
    proportions = [0.05, 0.05, 0.90]

    # 2. Topology (Density Logic)
    input_density_matrix = np.array(
        [
            [0.0, 1.0, 1.0],  # G0 -> G0, G1
            [0.0, 1.0, 1.0],  # G1 -> G1, G2
            [0.0, 0.0, 0.0],  # G2 -> Sink
        ]
    )
    h_mats = {
        "HI": 1.0 - input_density_matrix
    }  # Convert to Sparsity for internal logic

    # 3. Overrides
    density_overrides = np.array(
        [
            [-1.0, 0.5, -1.0],
            [-1.0, 1.0, 0.85],
            [-1.0, -1.0, -1.0],
        ]
    )

    # --- GENERATE 100D BASELINES (ER & HI) ---
    generate_dense_networks_only(
        base_matrices,
        [1, 2],
        final_output_folder,
        # sub_group_proportions=proportions,
        # h_matrices_dict=h_mats,
    )

    # # --- RUN SIMULATION (SPARSER NETWORKS) ---
    # combined_kwargs = {
    #     "sparsity_levels": np.linspace(0.05, 0.9, num=18, endpoint=True),
    #     "sub_group_proportions": proportions,
    #     "h_matrices_dict": h_mats,
    #     "block_density_constraints": density_overrides,
    # }
    #
    # print("\n### Running Simulation: ER and Hierarchical (Decoupled & Connected) ###")
    # run_simulation_from_generated_masks(
    #     base_matrices,
    #     scales,
    #     replicates_to_run,
    #     final_output_folder,
    #     generate_progressive_team_eruthi_mask_dict,
    #     combined_kwargs,
    # )

    # plot_adjacency_heatmaps_sorted_by_role(final_output_folder)
