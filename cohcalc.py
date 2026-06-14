# import asyncio
# import os
import re
import warnings
from functools import partial
from pathlib import Path
from typing import Union, Tuple, List
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd
from jax import jit, lax
from jax.scipy.linalg import expm
from jax.scipy.special import factorial
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import shutil


# Force CPU as the default backend and enable 64-bit precision for stable matrix math
# jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# # Semaphore to limit the number of concurrent asynchronous file writes
# WRITE_SEMAPHORE = asyncio.Semaphore(10)

#################################################################
# 1. Parsing
#################################################################


def parse_topodf(topo_file_path: Union[str, Path]) -> Tuple[jnp.ndarray, List[str]]:
    """
    Parses a network topology file into a JAX numpy adjacency matrix.
    Automatically sniffs the first line to detect if the delimiter is a comma or whitespace.

    Args:
        topo_file_path: Path to the network topology file.

    Returns:
        topo_net: A JAX numpy array representing the adjacency matrix of the network.
        node_list: A list of node names preserving the structural order of the matrix.
    """
    topo_file_path = Path(topo_file_path)

    # Read the first line to sniff the separator
    with topo_file_path.open("r") as f:
        first_line = f.readline()

    # If a comma is present in the header, assume CSV, otherwise fallback to whitespace regex
    delimiter = "," if "," in first_line else r"\s+"

    topo_df = pd.read_csv(topo_file_path, sep=delimiter)
    topo_df = topo_df.replace({2: -1})

    topo_net = nx.from_pandas_edgelist(
        topo_df,
        source="Source",
        target="Target",
        edge_attr="Type",
        create_using=nx.DiGraph,
    )
    node_list = list(topo_net.nodes())
    topo_net = nx.to_numpy_array(topo_net, weight="Type")
    topo_net = jnp.array(topo_net).astype(jnp.float64)

    return topo_net, node_list


#################################################################
# 2. Coherence and Walk Fraction Calculations
#################################################################


@jit
def calc_walk_fraction(mod_adj: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the continuous Walk Fraction matrix using the matrix exponential.
    Dynamically scales the matrix to prevent float64 overflow in dense networks.
    Preserves NaNs where no topological paths exist.
    """
    N = mod_adj.shape[0]
    max_degree = jnp.max(jnp.sum(mod_adj, axis=1))

    scale_factor = jnp.where(
        jnp.maximum(max_degree, N) > 700.0,
        700.0 / (jnp.maximum(max_degree, N) + 1e-8),
        1.0,
    )

    safe_mod_adj = mod_adj * scale_factor
    safe_ones = jnp.ones(mod_adj.shape) * scale_factor

    exp_safe_mod_adj = expm(safe_mod_adj)
    exp_safe_ones = expm(safe_ones)

    walk_mat = jnp.divide(exp_safe_mod_adj, exp_safe_ones)

    no_path_mask = exp_safe_mod_adj == 0.0
    walk_mat = jnp.where(no_path_mask, jnp.nan, walk_mat)

    return walk_mat


@partial(jit, static_argnums=0)
def calc_fast_coh_scan(
    upto_length: int, adj: jnp.ndarray, mod_adj: jnp.ndarray
) -> jnp.ndarray:
    """
    Ultra-fast discrete coherence matrix generation utilizing the mathematical equivalence
    of A^L / |A|^L. Scales efficiently by bypassing detailed walk-type tracking.
    """
    num_nodes = adj.shape[0]

    def step(carry, length_idx):
        A_L, mod_A_L = carry

        next_A_L = jnp.dot(A_L, adj)
        next_mod_A_L = jnp.dot(mod_A_L, mod_adj)

        C_L = jnp.where(next_mod_A_L > 0, next_A_L / next_mod_A_L, jnp.nan)
        Mask_L = (next_mod_A_L > 0).astype(jnp.float64)

        fact_L = factorial(length_idx)
        weighted_C_L = jnp.where(Mask_L > 0, C_L / fact_L, 0.0)
        weighted_Mask_L = Mask_L / fact_L

        return (next_A_L, next_mod_A_L), (weighted_C_L, weighted_Mask_L)

    init_A = jnp.eye(num_nodes)
    init_mod_A = jnp.eye(num_nodes)

    _, (all_weighted_C, all_weighted_Mask) = lax.scan(
        step, (init_A, init_mod_A), xs=jnp.arange(1, upto_length + 1)
    )

    sum_weighted_C = jnp.sum(all_weighted_C, axis=0)
    sum_weighted_Mask = jnp.sum(all_weighted_Mask, axis=0)

    final_coh = jnp.where(
        sum_weighted_Mask > 0, sum_weighted_C / sum_weighted_Mask, jnp.nan
    )
    return final_coh


@jit
def get_weightnorm_coh(walk_mats: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the normalized coherence matrix from detailed walk tracking matrices,
    bounded between -1.0 and 1.0.
    """
    num_nodes = walk_mats.shape[1] // 8
    upto_length = walk_mats.shape[0] // num_nodes

    num_walk_mat = walk_mats[:, :num_nodes].reshape(upto_length, num_nodes, num_nodes)
    frac_poswalk_mat = walk_mats[:, -(num_nodes):].reshape(
        upto_length, num_nodes, num_nodes
    )

    nowalk_mask = jnp.where(num_walk_mat == 0, jnp.nan, 1)
    frac_poswalk_mat = (2 * frac_poswalk_mat) - 1
    frac_poswalk_mat = jnp.where(num_walk_mat == 0, jnp.nan, frac_poswalk_mat)

    scaling_factors = factorial(jnp.arange(1, frac_poswalk_mat.shape[0] + 1)).reshape(
        frac_poswalk_mat.shape[0], 1, 1
    )

    frac_poswalk_mat = jnp.nansum(jnp.divide(frac_poswalk_mat, scaling_factors), axis=0)
    scaling_factors_sum = jnp.nansum(jnp.divide(nowalk_mask, scaling_factors), axis=0)

    nowalk_mask = jnp.sum(nowalk_mask, axis=0)

    frac_poswalk_mat = jnp.divide(frac_poswalk_mat, scaling_factors_sum)
    frac_poswalk_mat = jnp.where(nowalk_mask == 0, jnp.nan, frac_poswalk_mat)

    return frac_poswalk_mat


@jit
def next_walk_matrices_opt(prev_num_walks, prev_num_poswalks, adj, modadj):
    """Internal step function for detailed walk variations tracking."""
    prev_num_negwalk = prev_num_walks - prev_num_poswalks
    num_walks = jnp.dot(modadj, prev_num_walks)
    max_cnsrvd_poswalks = jnp.dot(modadj, prev_num_poswalks)
    max_cnsrvd_negwalks = num_walks - max_cnsrvd_poswalks

    net_num_poswalks = jnp.dot(adj, prev_num_poswalks)
    net_num_negwalks = jnp.dot(adj, prev_num_negwalk)

    cnsrvd_num_poswalks = (net_num_poswalks + max_cnsrvd_poswalks) / 2.0
    cnsrvd_num_negwalks = (net_num_negwalks + max_cnsrvd_negwalks) / 2.0

    flip_num_posneg_walks = max_cnsrvd_poswalks - cnsrvd_num_poswalks
    flip_num_negpos_walks = max_cnsrvd_negwalks - cnsrvd_num_negwalks

    tot_num_poswalks = cnsrvd_num_poswalks + flip_num_negpos_walks
    tot_num_negwalks = cnsrvd_num_negwalks + flip_num_posneg_walks

    frac_poswalks = jnp.where(
        num_walks > 0, tot_num_poswalks / jnp.maximum(num_walks, 1e-15), jnp.nan
    )

    req_mats = jnp.concatenate(
        (
            num_walks,
            tot_num_poswalks,
            tot_num_negwalks,
            cnsrvd_num_poswalks,
            cnsrvd_num_negwalks,
            flip_num_posneg_walks,
            flip_num_negpos_walks,
            frac_poswalks,
        ),
        axis=1,
    )
    return req_mats


@partial(jit, static_argnums=0)
def calc_walk_matrices_opt(upto_length, adj, mod_adj):
    """Internal scan execution for tracking detailed walks."""
    prev_num_poswalks = (adj == 1).astype(jnp.float64)
    num_nodes = adj.shape[0]

    def step(carry, _):
        prev_num_walks, prev_num_poswalks = carry
        req_mats = next_walk_matrices_opt(
            prev_num_walks, prev_num_poswalks, adj, mod_adj
        )
        new_num_walks = req_mats[:, :num_nodes]
        new_num_poswalks = req_mats[:, num_nodes : 2 * num_nodes]
        return (new_num_walks, new_num_poswalks), req_mats

    _, stacked_results = lax.scan(
        step, (mod_adj, prev_num_poswalks), xs=None, length=upto_length - 1
    )
    return stacked_results


#################################################################
# 3. Team Identification
#################################################################


def find_teams_graph_linkage(
    df: pd.DataFrame, split_components: bool = True
) -> pd.DataFrame:
    """
    Extracts teams directly from a coherence matrix DataFrame.
    Pre-processes self-inhibitory nodes as individual teams, computes a boolean
    Weighted Structural Similarity proxy, groups via Complete Linkage, and optionally
    splits nodes into connected components based on activating topology.

    Args:
        df: A Pandas DataFrame representing the pairwise network coherence matrix.
        split_components: A boolean toggle. If True, identified groups are further
                          split into connected components based on activating edges.

    Returns:
        groups_dict: A DataFrame with columns ['Group', 'Node'] assigning each node
                     to its calculated team.
    """
    if "Group" in df.index.names:
        df = df.droplevel("Group", axis=0)
    if "Group" in df.columns.names:
        df = df.droplevel("Group", axis=1)

    mat = np.sign(df.values)
    nodes = np.array(df.columns)

    diag = mat.diagonal()
    self_inhib_mask = diag <= 0
    reg_mask = ~self_inhib_mask

    self_inhib_nodes = nodes[self_inhib_mask].tolist()
    regular_nodes = nodes[reg_mask]
    regular_indices = np.where(reg_mask)[0]

    final_groups = [[node] for node in self_inhib_nodes]

    n_reg = len(regular_nodes)
    if n_reg > 0:
        if n_reg == 1:
            final_groups.append(regular_nodes.tolist())
        else:
            sub_mat = mat[np.ix_(regular_indices, regular_indices)]
            sub_mat_T = sub_mat.T

            sim_matrix = np.ones((n_reg, n_reg))
            sim_matrix[(sub_mat <= 0) | (sub_mat_T <= 0)] = -1
            sim_matrix[np.isnan(sub_mat) & np.isnan(sub_mat_T)] = 1
            np.fill_diagonal(sim_matrix, 1)

            dist_matrix = 1 - sim_matrix
            np.fill_diagonal(dist_matrix, 0)
            condensed_dist = squareform(dist_matrix)

            Z = linkage(condensed_dist, method="complete")
            labels = fcluster(Z, t=0, criterion="distance")

            initial_groups = {}
            for idx, label in enumerate(labels):
                initial_groups.setdefault(label, []).append(idx)

            for label, grp_local_indices in initial_groups.items():
                if len(grp_local_indices) <= 1:
                    final_groups.append([regular_nodes[grp_local_indices[0]]])
                    continue

                grp_local_indices = np.array(grp_local_indices)
                grp_nodes = regular_nodes[grp_local_indices]

                if split_components:
                    grp_sub_mat = sub_mat[np.ix_(grp_local_indices, grp_local_indices)]
                    grp_sub_mat_T = grp_sub_mat.T

                    adj = (grp_sub_mat > 0) | (grp_sub_mat_T > 0)
                    np.fill_diagonal(adj, False)

                    sub_graph = nx.from_numpy_array(adj)
                    sub_graph = nx.relabel_nodes(sub_graph, dict(enumerate(grp_nodes)))

                    for comp in nx.connected_components(sub_graph):
                        final_groups.append(list(comp))
                else:
                    final_groups.append(grp_nodes.tolist())

    final_groups = sorted(final_groups, key=len, reverse=True)

    groups_dict = [
        {"Group": group_id, "Node": node}
        for group_id, grp_nodes in enumerate(final_groups, 1)
        for node in sorted(grp_nodes)
    ]

    return pd.DataFrame(groups_dict)


def find_teams_kernel_expansion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts teams using a Seed-and-Expand Block Influence Model.
    Forms initial groups via complete linkage, extracts the largest connected
    component as the 'Kernel', and assigns disconnected 'Orphan' nodes by
    calculating the Cosine Distance of their aggregate block-level influence profiles.
    """
    if "Group" in df.index.names:
        df = df.droplevel("Group", axis=0)
    if "Group" in df.columns.names:
        df = df.droplevel("Group", axis=1)

    mat = np.sign(df.values)
    C = df.values  # Raw coherence values for influence magnitude
    nodes = np.array(df.columns)

    diag = mat.diagonal()
    self_inhib_mask = diag <= 0
    reg_mask = ~self_inhib_mask

    self_inhib_nodes = nodes[self_inhib_mask].tolist()
    regular_nodes = nodes[reg_mask]
    regular_indices = np.where(reg_mask)[0]

    final_groups = [[node] for node in self_inhib_nodes]

    n_reg = len(regular_nodes)
    if n_reg > 0:
        if n_reg == 1:
            final_groups.append(regular_nodes.tolist())
        else:
            # 1. INITIAL GROUPING (Original Linkage Logic)
            sub_mat = mat[np.ix_(regular_indices, regular_indices)]
            sub_mat_T = sub_mat.T

            sim_matrix = np.ones((n_reg, n_reg))
            sim_matrix[(sub_mat <= 0) | (sub_mat_T <= 0)] = -1
            sim_matrix[np.isnan(sub_mat) & np.isnan(sub_mat_T)] = 1
            np.fill_diagonal(sim_matrix, 1)

            dist_matrix = 1 - sim_matrix
            np.fill_diagonal(dist_matrix, 0)
            condensed_dist = squareform(dist_matrix)

            Z = linkage(condensed_dist, method="complete")
            labels = fcluster(Z, t=0, criterion="distance")

            initial_groups = {}
            for idx, label in enumerate(labels):
                initial_groups.setdefault(label, []).append(idx)

            # 2. EXTRACT KERNELS AND ORPHANS
            kernels = []
            orphans = []

            for label, grp_local_indices in initial_groups.items():
                if len(grp_local_indices) <= 1:
                    kernels.append(grp_local_indices)
                    continue

                grp_sub_mat = sub_mat[np.ix_(grp_local_indices, grp_local_indices)]
                grp_sub_mat_T = grp_sub_mat.T

                adj = (grp_sub_mat > 0) | (grp_sub_mat_T > 0)
                np.fill_diagonal(adj, False)

                sub_graph = nx.from_numpy_array(adj)
                comps = list(nx.connected_components(sub_graph))
                comps.sort(key=len, reverse=True)

                # Largest component acts as the definitive team Kernel
                kernel_nodes = list(comps[0])
                kernels.append([grp_local_indices[i] for i in kernel_nodes])

                # All other disconnected nodes in this initial group become Orphans
                for comp in comps[1:]:
                    orphans.extend([grp_local_indices[i] for i in comp])

            # 3. BLOCK INFLUENCE VECTOR ASSIGNMENT
            if orphans:
                C_reg = C[np.ix_(regular_indices, regular_indices)]

                # Function to generate a node's aggregate influence on all Kernels
                def get_block_vector(node_idx):
                    vec = []
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        for k_indices in kernels:
                            if len(k_indices) == 0:
                                vec.extend([0.0, 0.0])
                                continue

                            # Extract the raw coherence slices
                            out_slice = C_reg[node_idx, k_indices]
                            in_slice = C_reg[k_indices, node_idx]

                            # ZERO-TOLERANCE CHECK: If ANY value is negative, penalize heavily
                            if np.any(out_slice < 0) or np.any(in_slice < 0):
                                vec.extend([-1.0, -1.0])
                                continue

                            # Mean outgoing and incoming coherence to/from the Kernel
                            out_inf = np.nanmean(out_slice)
                            in_inf = np.nanmean(in_slice)

                            vec.extend([np.nan_to_num(out_inf), np.nan_to_num(in_inf)])
                    return np.array(vec)

                # # Function to generate a node's aggregate influence on all Kernels
                # def get_block_vector(node_idx):
                #     vec = []
                #     with warnings.catch_warnings():
                #         warnings.simplefilter("ignore", category=RuntimeWarning)
                #         for k_indices in kernels:
                #             if len(k_indices) == 0:
                #                 vec.extend([0.0, 0.0])
                #                 continue
                #
                #             # Mean outgoing and incoming coherence to/from the Kernel
                #             out_inf = np.nanmean(C_reg[node_idx, k_indices])
                #             in_inf = np.nanmean(C_reg[k_indices, node_idx])
                #
                #             vec.extend([np.nan_to_num(out_inf), np.nan_to_num(in_inf)])
                #     return np.array(vec)

                # Calculate Centroid Vectors for each Kernel
                kernel_vectors = []
                for k_indices in kernels:
                    k_vecs = [get_block_vector(i) for i in k_indices]
                    kernel_vectors.append(np.mean(k_vecs, axis=0))

                # Temporary list to hold new kernels to prevent mid-loop shape mutations
                new_isolated_kernels = []

                # Assign Orphans based on Cosine Similarity to Kernel Centroids
                for o_idx in orphans:
                    o_vec = get_block_vector(o_idx)
                    norm_o = np.linalg.norm(o_vec)

                    best_k = -1
                    max_sim = -float("inf")

                    for k_id, k_vec in enumerate(kernel_vectors):
                        norm_k = np.linalg.norm(k_vec)
                        if norm_o == 0 or norm_k == 0:
                            sim = 0.0
                        else:
                            sim = np.dot(o_vec, k_vec) / (norm_o * norm_k)

                        if sim > max_sim:
                            max_sim = sim
                            best_k = k_id

                    # Assign Orphan to best matching Kernel, or queue for isolation
                    if best_k != -1 and max_sim > 0:
                        kernels[best_k].append(o_idx)
                    else:
                        new_isolated_kernels.append([o_idx])

                # Safely extend the main list after the loop completes
                kernels.extend(new_isolated_kernels)

            # Map local indices back to global node names
            for k_indices in kernels:
                if len(k_indices) > 0:
                    final_groups.append([regular_nodes[i] for i in k_indices])

    final_groups = sorted(final_groups, key=len, reverse=True)

    groups_dict = [
        {"Group": group_id, "Node": node}
        for group_id, grp_nodes in enumerate(final_groups, 1)
        for node in sorted(grp_nodes)
    ]

    return pd.DataFrame(groups_dict)


def find_teams_benefit_ranking_old(
    df: pd.DataFrame, final_split: bool = True
) -> pd.DataFrame:
    """
    Extracts teams using Linkage + Connected Components to find safe cores.
    Executes a Global 2-Tier Dynamic Reassignment.
    Args:
        final_split: If True, splits final teams by positive paths and outputs
                     an additional 'PreSplitGroup' column tracking the team
                     assignments before the final split.
    """
    if "Group" in df.index.names:
        df = df.droplevel("Group", axis=0)
    if "Group" in df.columns.names:
        df = df.droplevel("Group", axis=1)

    mat = np.sign(df.values)
    nodes = np.array(df.columns)

    diag = mat.diagonal()
    self_inhib_mask = diag <= 0
    reg_mask = ~self_inhib_mask

    self_inhib_nodes = nodes[self_inhib_mask].tolist()
    regular_nodes = nodes[reg_mask]
    regular_indices = np.where(reg_mask)[0]

    pre_split_groups = [[node] for node in self_inhib_nodes]
    post_split_groups = [[node] for node in self_inhib_nodes]

    n_reg = len(regular_nodes)
    if n_reg > 0:
        if n_reg == 1:
            pre_split_groups.append(regular_nodes.tolist())
            post_split_groups.append(regular_nodes.tolist())
        else:
            # --- PHASE 1: INITIAL SAFE LINKAGE ---
            sub_mat = mat[np.ix_(regular_indices, regular_indices)]
            sub_mat_T = sub_mat.T

            sim_matrix = np.ones((n_reg, n_reg))
            sim_matrix[(sub_mat <= 0) | (sub_mat_T <= 0)] = -1
            sim_matrix[np.isnan(sub_mat) & np.isnan(sub_mat_T)] = 1
            np.fill_diagonal(sim_matrix, 1)

            dist_matrix = 1 - sim_matrix
            np.fill_diagonal(dist_matrix, 0)
            condensed_dist = squareform(dist_matrix)

            Z = linkage(condensed_dist, method="complete")
            labels = fcluster(Z, t=0, criterion="distance")

            initial_groups = {}
            for idx, label in enumerate(labels):
                initial_groups.setdefault(label, []).append(idx)

            # --- PHASE 2: CONNECTED COMPONENT SPLIT (HARDCODED) ---
            groups_list = []
            for label, grp_local_indices in initial_groups.items():
                if len(grp_local_indices) <= 1:
                    groups_list.append(grp_local_indices)
                    continue

                grp_local_indices = np.array(grp_local_indices)

                grp_sub_mat = sub_mat[np.ix_(grp_local_indices, grp_local_indices)]
                grp_sub_mat_T = grp_sub_mat.T

                adj = (grp_sub_mat > 0) | (grp_sub_mat_T > 0)
                np.fill_diagonal(adj, False)

                sub_graph = nx.from_numpy_array(adj)
                for comp in nx.connected_components(sub_graph):
                    groups_list.append([grp_local_indices[i] for i in comp])

            # --- PHASE 3: GLOBAL 2-TIER GREEDY REASSIGNMENT ---
            C_reg = sub_mat

            def get_best_global_move(tier):
                best_node = -1
                best_source_idx = -1
                best_target_idx = -1
                max_score = 0

                for source_idx, source_group in enumerate(groups_list):
                    for node in source_group:
                        for target_idx, target_group in enumerate(groups_list):
                            if source_idx == target_idx:
                                continue

                            # Only allow nodes to migrate to LARGER groups, or equal-sized groups with a lower index
                            if len(target_group) < len(source_group) or (
                                len(target_group) == len(source_group)
                                and target_idx > source_idx
                            ):
                                continue

                            out_paths = C_reg[node, target_group]
                            in_paths = C_reg[target_group, node]

                            # 1. THE ABSOLUTE VETO
                            if np.any(out_paths < 0) or np.any(in_paths < 0):
                                continue

                            # 2. SCORING
                            if tier == 1:  # Direct Activation Focus
                                score = np.sum(out_paths > 0) + np.sum(in_paths > 0)
                            elif tier == 2:  # Indirect Inhibition Focus
                                opposing_mask = np.ones(n_reg, dtype=bool)
                                opposing_mask[target_group] = False
                                opposing_mask[node] = False

                                score = np.sum(C_reg[node, opposing_mask] < 0) + np.sum(
                                    C_reg[opposing_mask, node] < 0
                                )

                            # 3. GLOBAL MAX COMPARISON
                            if score > max_score:
                                max_score = score
                                best_node = node
                                best_source_idx = source_idx
                                best_target_idx = target_idx

                return best_node, best_source_idx, best_target_idx, max_score

            # EXECUTE TIER 1
            while True:
                node, src_idx, tgt_idx, score = get_best_global_move(tier=1)
                if score > 0:
                    groups_list[src_idx].remove(node)
                    groups_list[tgt_idx].append(node)
                    if not groups_list[src_idx]:
                        del groups_list[src_idx]
                else:
                    break

            # EXECUTE TIER 2
            while True:
                node, src_idx, tgt_idx, score = get_best_global_move(tier=2)
                if score > 0:
                    groups_list[src_idx].remove(node)
                    groups_list[tgt_idx].append(node)
                    if not groups_list[src_idx]:
                        del groups_list[src_idx]
                else:
                    break

            # --- PHASE 4: FINAL CONNECTED COMPONENT SPLIT (TOGGLABLE) ---
            for grp_local_indices in groups_list:
                if len(grp_local_indices) > 0:
                    # Store the pre-split state
                    pre_split_groups.append(
                        [regular_nodes[i] for i in grp_local_indices]
                    )

                    if final_split:
                        grp_sub_mat = sub_mat[
                            np.ix_(grp_local_indices, grp_local_indices)
                        ]
                        grp_sub_mat_T = grp_sub_mat.T

                        adj = (grp_sub_mat > 0) | (grp_sub_mat_T > 0)
                        np.fill_diagonal(adj, False)

                        sub_graph = nx.from_numpy_array(adj)
                        for comp in nx.connected_components(sub_graph):
                            post_split_groups.append(
                                [regular_nodes[grp_local_indices[i]] for i in comp]
                            )
                    else:
                        post_split_groups.append(
                            [regular_nodes[i] for i in grp_local_indices]
                        )

    # Sort both group sets by size descending
    pre_split_groups = sorted(pre_split_groups, key=len, reverse=True)
    post_split_groups = sorted(post_split_groups, key=len, reverse=True)

    # Create mapping of Node -> PreSplitGroup ID
    pre_split_dict = {
        node: group_id
        for group_id, grp_nodes in enumerate(pre_split_groups, 1)
        for node in grp_nodes
    }

    # Build final output
    groups_dict = []
    for group_id, grp_nodes in enumerate(post_split_groups, 1):
        for node in sorted(grp_nodes):
            row = {"Group": group_id, "Node": node}
            if final_split:
                row["PreSplitGroup"] = pre_split_dict[node]
            groups_dict.append(row)

    df_out = pd.DataFrame(groups_dict)

    # if final_split:
    # Reorder columns so PreSplitGroup is on the left
    df_out = df_out[["PreSplitGroup", "Group", "Node"]]

    return df_out


def find_teams_benefit_ranking(
    df: pd.DataFrame, final_split: bool = False
) -> pd.DataFrame:
    if "Group" in df.index.names:
        df = df.droplevel("Group", axis=0)
    if "Group" in df.columns.names:
        df = df.droplevel("Group", axis=1)

    mat = np.sign(df.values)
    nodes = np.array(df.columns)

    diag = mat.diagonal()
    self_inhib_mask = diag <= 0
    reg_mask = ~self_inhib_mask

    self_inhib_nodes = nodes[self_inhib_mask].tolist()
    regular_nodes = nodes[reg_mask]
    regular_indices = np.where(reg_mask)[0]

    pre_split_groups = [[node] for node in self_inhib_nodes]
    post_split_groups = [[node] for node in self_inhib_nodes]

    n_reg = len(regular_nodes)
    if n_reg > 0:
        if n_reg == 1:
            pre_split_groups.append(regular_nodes.tolist())
            post_split_groups.append(regular_nodes.tolist())
        else:
            # --- PHASE 1: INITIAL SAFE LINKAGE ---
            sub_mat = mat[np.ix_(regular_indices, regular_indices)]
            sub_mat_T = sub_mat.T

            sim_matrix = np.ones((n_reg, n_reg))
            sim_matrix[(sub_mat <= 0) | (sub_mat_T <= 0)] = -1
            sim_matrix[np.isnan(sub_mat) & np.isnan(sub_mat_T)] = 1
            np.fill_diagonal(sim_matrix, 1)

            dist_matrix = 1 - sim_matrix
            np.fill_diagonal(dist_matrix, 0)
            condensed_dist = squareform(dist_matrix)

            Z = linkage(condensed_dist, method="complete")
            labels = fcluster(Z, t=0, criterion="distance")

            # --- PHASE 2: GLOBAL CONNECTED COMPONENT SPLIT (No Loops!) ---
            # Create a mask where True means nodes share the same initial label
            same_group_mask = labels[:, None] == labels[None, :]

            # Global undirected adjacency
            adj_global = (sub_mat > 0) | (sub_mat_T > 0)

            # Sever all edges that cross group boundaries
            adj_global = adj_global & same_group_mask
            np.fill_diagonal(adj_global, False)

            # Run connected components EXACTLY ONCE on the entire matrix
            n_components, comp_labels = connected_components(
                csgraph=csr_matrix(adj_global), directed=False, return_labels=True
            )

            # Fast numpy grouping: sort indices by their new component label
            order = np.argsort(comp_labels)
            sorted_labels = comp_labels[order]
            changes = np.where(sorted_labels[:-1] != sorted_labels[1:])[0] + 1
            splits = np.split(order, changes)

            groups_list = [split.tolist() for split in splits]

            # --- PHASE 3: GLOBAL 2-TIER GREEDY REASSIGNMENT (Vectorized) ---
            C_pos = (sub_mat > 0).astype(int)
            C_neg = (sub_mat < 0).astype(int)

            n_groups = len(groups_list)
            group_sizes = np.zeros(n_groups, dtype=int)
            node_to_group = np.full(n_reg, -1, dtype=int)

            for g_idx, grp_nodes in enumerate(groups_list):
                group_sizes[g_idx] = len(grp_nodes)
                for u in grp_nodes:
                    node_to_group[u] = g_idx

            G_mat = np.zeros((n_reg, n_groups), dtype=int)
            G_mat[np.arange(n_reg), node_to_group] = 1

            pos_out_count = C_pos @ G_mat
            pos_in_count = C_pos.T @ G_mat
            neg_out_count = C_neg @ G_mat
            neg_in_count = C_neg.T @ G_mat

            total_neg_out = C_neg.sum(axis=1)
            total_neg_in = C_neg.sum(axis=0)
            self_neg = np.diag(C_neg)

            def run_vectorized_tier(tier):
                while True:
                    u_groups = node_to_group
                    u_sizes = group_sizes[u_groups]

                    valid_mask = (group_sizes[None, :] > u_sizes[:, None]) | (
                        (group_sizes[None, :] == u_sizes[:, None])
                        & (np.arange(n_groups)[None, :] > u_groups[:, None])
                    )

                    veto_mask = (neg_out_count == 0) & (neg_in_count == 0)
                    valid_mask = valid_mask & veto_mask

                    if tier == 1:
                        score_mat = pos_out_count + pos_in_count
                    else:
                        score_mat = (
                            total_neg_out[:, None] - neg_out_count - self_neg[:, None]
                        ) + (total_neg_in[:, None] - neg_in_count - self_neg[:, None])

                    valid_mask = valid_mask & (score_mat > 0)

                    if not np.any(valid_mask):
                        break

                    score_mat = np.where(valid_mask, score_mat, -1)
                    best_flat_idx = np.argmax(score_mat)
                    max_score = score_mat.flat[best_flat_idx]

                    if max_score <= 0:
                        break

                    best_u = best_flat_idx // n_groups
                    best_g_tgt = best_flat_idx % n_groups
                    best_g_src = node_to_group[best_u]

                    node_to_group[best_u] = best_g_tgt
                    group_sizes[best_g_src] -= 1
                    group_sizes[best_g_tgt] += 1

                    pos_out_count[:, best_g_src] -= C_pos[:, best_u]
                    pos_out_count[:, best_g_tgt] += C_pos[:, best_u]
                    pos_in_count[:, best_g_src] -= C_pos[best_u, :]
                    pos_in_count[:, best_g_tgt] += C_pos[best_u, :]

                    neg_out_count[:, best_g_src] -= C_neg[:, best_u]
                    neg_out_count[:, best_g_tgt] += C_neg[:, best_u]
                    neg_in_count[:, best_g_src] -= C_neg[best_u, :]
                    neg_in_count[:, best_g_tgt] += C_neg[best_u, :]

            run_vectorized_tier(tier=1)
            run_vectorized_tier(tier=2)

            # --- PHASE 4: GLOBAL FINAL COMPONENT SPLIT ---
            # Append current state to pre-split
            for g_idx in range(n_groups):
                if group_sizes[g_idx] > 0:
                    members = np.where(node_to_group == g_idx)[0]
                    pre_split_groups.append([regular_nodes[i] for i in members])

            if final_split:
                same_group_mask = node_to_group[:, None] == node_to_group[None, :]
                adj_global = (sub_mat > 0) | (sub_mat_T > 0)
                adj_global = adj_global & same_group_mask
                np.fill_diagonal(adj_global, False)

                n_components, comp_labels = connected_components(
                    csgraph=csr_matrix(adj_global), directed=False, return_labels=True
                )

                order = np.argsort(comp_labels)
                sorted_labels = comp_labels[order]
                changes = np.where(sorted_labels[:-1] != sorted_labels[1:])[0] + 1
                splits = np.split(order, changes)

                for split in splits:
                    post_split_groups.append([regular_nodes[i] for i in split])
            else:
                for g_idx in range(n_groups):
                    if group_sizes[g_idx] > 0:
                        members = np.where(node_to_group == g_idx)[0]
                        post_split_groups.append([regular_nodes[i] for i in members])

    pre_split_groups = sorted(pre_split_groups, key=len, reverse=True)
    post_split_groups = sorted(post_split_groups, key=len, reverse=True)

    pre_split_dict = {
        node: group_id
        for group_id, grp_nodes in enumerate(pre_split_groups, 1)
        for node in grp_nodes
    }

    groups_dict = []
    for group_id, grp_nodes in enumerate(post_split_groups, 1):
        for node in sorted(grp_nodes):
            row = {
                "PreSplitGroup": pre_split_dict[node],
                "Group": group_id,
                "Node": node,
            }
            groups_dict.append(row)

    df_out = pd.DataFrame(groups_dict)
    df_out = df_out[["PreSplitGroup", "Group", "Node"]]

    return df_out


#################################################################
# 4. Processing Pipeline
#################################################################


def process_topology(
    topo_file_path: Union[str, Path],
    upto_length: int = 10,
    fast_mode: bool = True,
    extract_teams: bool = True,  # NEW TOGGLE
    final_split: bool = False,
):
    """
    Computes coherence matrices, walk fractions, and optionally extracts teams.
    """
    topo_file_path = Path(topo_file_path)
    adj, node_list = parse_topodf(topo_file_path)
    mod_adj = jnp.absolute(adj).astype(jnp.float64)

    walk_mats_df = None

    if fast_mode:
        coh_mat = calc_fast_coh_scan(upto_length, adj, mod_adj)
    else:
        # ... (Your existing slow mode code remains exactly the same) ...
        pass

    walk_frac_mat = calc_walk_fraction(mod_adj)

    # Format DataFrames
    coh_df = pd.DataFrame(coh_mat, index=node_list, columns=node_list)
    coh_df.index.name = "SourceNode"
    coh_df.columns.name = "TargetNode"

    walk_frac_df = pd.DataFrame(walk_frac_mat, index=node_list, columns=node_list)

    groups_df = None

    # Group Execution (Togglable)
    if extract_teams:
        groups_df = find_teams_benefit_ranking(coh_df, final_split=final_split)

        # Convert Indices to MultiIndex utilizing Team Assignments
        group_lookup = groups_df.set_index("Node")["Group"]
        row_multiindex = pd.MultiIndex.from_tuples(
            [(group_lookup.get(node, np.nan), node) for node in coh_df.index],
            names=["Group", "SourceNode"],
        )
        col_multiindex = pd.MultiIndex.from_tuples(
            [(group_lookup.get(node, np.nan), node) for node in coh_df.columns],
            names=["Group", "TargetNode"],
        )

        coh_df.index = row_multiindex.copy()
        coh_df.columns = col_multiindex.copy()
        walk_frac_df.index = row_multiindex.copy()
        walk_frac_df.columns = col_multiindex.copy()

    return walk_mats_df, coh_df, walk_frac_df, groups_df


def get_group_coherence(
    coh_mat_df: pd.DataFrame, groups_df: pd.DataFrame, group_col: str = "Group"
) -> Tuple[float, float]:
    """
    Computes the mean and median coherence strictly within the intra-group blocks
    of the coherence matrix for a specified grouping column.

    Args:
        coh_mat_df: The flattened NxN coherence matrix DataFrame.
        groups_df: DataFrame mapping 'Node' to teams.
        group_col: The target column in groups_df to use for grouping ("Group" or "PreSplitGroup").

    Returns:
        mean_res: The aggregated mean intra-group coherence.
        median_res: The aggregated median intra-group coherence.
    """
    if groups_df.empty or group_col not in groups_df.columns:
        return np.nan, np.nan

    groups_dict = groups_df.groupby(group_col)["Node"].apply(list).to_dict()
    mean_coh_list, median_coh_list = [], []

    for grp, nodes in groups_dict.items():
        if len(nodes) > 0:
            # Filter to ensure we only look up nodes that exist in the matrix index
            valid_nodes = [n for n in nodes if n in coh_mat_df.index]
            if not valid_nodes:
                continue

            submat = coh_mat_df.loc[valid_nodes, valid_nodes].values

            # Suppress RuntimeWarnings for slices that are entirely NaN
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if not np.all(np.isnan(submat)):
                    mean_coh_list.append(np.nanmean(submat))
                    median_coh_list.append(np.nanmedian(submat))

    mean_res = np.mean(mean_coh_list) if mean_coh_list else np.nan
    median_res = np.mean(median_coh_list) if median_coh_list else np.nan

    return mean_res, median_res


def compile_coherence_results(
    results_dir: Union[str, Path],
    output_filepath: Union[str, Path],
    parse_artinet: bool = False,
    parse_scaled: bool = False,
) -> pd.DataFrame:
    """
    Traverses the generated MotifCohResults directory, reads the saved matrices,
    re-calculates the summary statistics (including pre-split and post-split metrics),
    and outputs a consolidated DataFrame.
    """
    results_dir = Path(results_dir)
    output_filepath = Path(output_filepath)

    # Regex for base motifs (e.g., 021C_0NN000)
    signet_pattern = re.compile(r"^(?P<MAN_code>[0-9A-Z]+)_(?P<EdgeString>[0-9A-Z]+)$")

    # Regex for scaled networks (e.g., 030T_0P0PN0_NS_010N_050D_HI_001R)
    scaled_pattern = re.compile(
        r"^(?P<MAN_code>[0-9A-Z]+)_(?P<EdgeString>[0-9A-Z]+)_(?P<SelfActivation>SA|NS)_"
        r"(?P<Scale>[0-9]{3})N_(?P<Density>[0-9]{3})D_(?P<NetType>ER|HI)_(?P<Rep>[0-9]{3})R$"
    )

    all_records = []

    for topo_folder in tqdm(sorted(results_dir.iterdir()), desc="Compiling Results"):
        if not topo_folder.is_dir():
            continue

        topo_name = topo_folder.name
        coh_path = topo_folder / f"{topo_name}_CohMat.parquet"
        walk_path = topo_folder / f"{topo_name}_WalkFracMat.parquet"
        groups_path = topo_folder / f"{topo_name}_Teams.csv"

        if not (coh_path.exists() and walk_path.exists() and groups_path.exists()):
            continue

        try:
            # 1. Load DataFrames
            coh_df = pd.read_parquet(coh_path)
            walk_df = pd.read_parquet(walk_path)
            groups_df = pd.read_csv(groups_path)

            # 2. Flatten MultiIndex
            if isinstance(coh_df.index, pd.MultiIndex):
                coh_df.index = coh_df.index.get_level_values(-1)
            if isinstance(coh_df.columns, pd.MultiIndex):
                coh_df.columns = coh_df.columns.get_level_values(-1)
            if isinstance(walk_df.index, pd.MultiIndex):
                walk_df.index = walk_df.index.get_level_values(-1)
            if isinstance(walk_df.columns, pd.MultiIndex):
                walk_df.columns = walk_df.columns.get_level_values(-1)

            coh_vals = coh_df.values
            walk_vals = walk_df.values

            # 3. Calculate core statistical metrics
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                abs_mean_walk = np.nanmean(np.abs(walk_vals))
                abs_median_walk = np.nanmedian(np.abs(walk_vals))

                record = {
                    "TopoName": topo_name,
                    "NumNodes": coh_df.shape[0],
                    "NumGroups": groups_df["Group"].nunique()
                    if not groups_df.empty
                    else 0,
                    "NumPreSplitGroups": groups_df["PreSplitGroup"].nunique()
                    if "PreSplitGroup" in groups_df.columns
                    else np.nan,
                    "AbsMeanCohVal": np.nanmean(np.abs(coh_vals)),
                    "AbsMedianCohVal": np.nanmedian(np.abs(coh_vals)),
                    "CohMatMean": np.nanmean(coh_vals),
                    "CohMatMedian": np.nanmedian(coh_vals),
                    "AbsMeanWalkVal": np.log10(abs_mean_walk)
                    if abs_mean_walk > 0
                    else np.nan,
                    "AbsMedianWalkVal": np.log10(abs_median_walk)
                    if abs_median_walk > 0
                    else np.nan,
                }

            # 4. Calculate Group Coherence (Post-Split Final Groups)
            mean_coh, median_coh = get_group_coherence(
                coh_df, groups_df, group_col="Group"
            )
            record["MeanCoh"] = mean_coh
            record["MedianCoh"] = median_coh

            # 4b. Calculate Group Coherence (Pre-Split Groups)
            if "PreSplitGroup" in groups_df.columns:
                pre_mean_coh, pre_median_coh = get_group_coherence(
                    coh_df, groups_df, group_col="PreSplitGroup"
                )
                record["PreSplitMeanCoh"] = pre_mean_coh
                record["PreSplitMedianCoh"] = pre_median_coh
            else:
                record["PreSplitMeanCoh"] = np.nan
                record["PreSplitMedianCoh"] = np.nan

            # 5. Extract Network Convention Attributes
            if parse_scaled:
                match = scaled_pattern.match(topo_name)
                if match:
                    record.update(match.groupdict())
                    record["Scale"] = int(record["Scale"])
                    record["Density"] = float(record["Density"]) / 100.0
                    record["Rep"] = int(record["Rep"])
                else:
                    record.update(
                        {
                            "MAN_code": np.nan,
                            "EdgeString": np.nan,
                            "SelfActivation": "Unknown",
                            "Scale": np.nan,
                            "Density": np.nan,
                            "NetType": "Unknown",
                            "Rep": np.nan,
                        }
                    )

            elif parse_artinet:
                if topo_name.endswith("_SA"):
                    record["SelfActivation"] = "SA"
                    base_name = topo_name[:-3]
                elif topo_name.endswith("_NS"):
                    record["SelfActivation"] = "NS"
                    base_name = topo_name[:-3]
                else:
                    record["SelfActivation"] = "Unknown"
                    base_name = topo_name

                match = signet_pattern.match(base_name)
                if match:
                    record["MAN_code"] = match.group("MAN_code")
                    record["EdgeString"] = match.group("EdgeString")
                else:
                    record["MAN_code"] = base_name
                    record["EdgeString"] = np.nan

            all_records.append(record)

        except Exception as e:
            print(f"\nError processing {topo_name}: {e}")

    # 6. Format and save the final compilation
    final_df = pd.DataFrame(all_records)
    if final_df.empty:
        print("No valid results found. Empty DataFrame returned.")
        return final_df

    # 7. Add ordered meta columns
    if parse_scaled:
        meta_cols = [
            "TopoName",
            "MAN_code",
            "EdgeString",
            "SelfActivation",
            "NetType",
            "Scale",
            "Density",
            "Rep",
            "NumPreSplitGroups",
            "NumGroups",
            "NumNodes",
        ]
    elif parse_artinet:
        meta_cols = [
            "TopoName",
            "MAN_code",
            "EdgeString",
            "SelfActivation",
            "NumPreSplitGroups",
            "NumGroups",
            "NumNodes",
        ]
    else:
        meta_cols = ["TopoName", "NumPreSplitGroups", "NumGroups", "NumNodes"]

    stat_cols = [c for c in final_df.columns if c not in meta_cols]
    final_df = final_df[meta_cols + stat_cols]

    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_filepath.with_suffix(".parquet"), engine="pyarrow")
    final_df.to_csv(output_filepath.with_suffix(".csv"), index=False)

    print(f"\nCompilation complete. Saved {len(final_df)} records to:")
    print(f" -> {output_filepath.with_suffix('.parquet')}")

    return final_df


def process_and_save_serial(
    topo_pattern: Union[str, Path],
    save_dir: Union[str, Path],
    upto_length: int = 15,
    fast_mode: bool = True,
    extract_teams: bool = True,
    final_split: bool = False,
):
    """
    Synchronously computes matrices and optionally extracts teams.
    """
    topo_pattern = Path(topo_pattern)

    if "*" in topo_pattern.name:
        topo_list = sorted(topo_pattern.parent.glob(topo_pattern.name))
    else:
        topo_list = sorted(topo_pattern.glob("*.topo"))

    print(f"Found {len(topo_list)} topo files. Starting serial processing...")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    successful_computations = 0

    for topo_file_path in tqdm(topo_list, desc="Processing"):
        topo_name = topo_file_path.stem
        save_path_topo = save_dir / topo_name
        save_path_topo.mkdir(parents=True, exist_ok=True)

        try:
            walk_mats_df, coh_df, walk_frac_df, groups_df = process_topology(
                topo_file_path,
                upto_length=upto_length,
                fast_mode=fast_mode,
                extract_teams=extract_teams,
                final_split=final_split,
            )

            # 2. Stringify all indices for Parquet serialization safety
            if isinstance(coh_df.index, pd.MultiIndex):
                coh_df.index = pd.MultiIndex.from_tuples(
                    [(str(g), str(n)) for g, n in coh_df.index],
                    names=coh_df.index.names,
                )
                coh_df.columns = pd.MultiIndex.from_tuples(
                    [(str(g), str(n)) for g, n in coh_df.columns],
                    names=coh_df.columns.names,
                )

                walk_frac_df.index = pd.MultiIndex.from_tuples(
                    [(str(g), str(n)) for g, n in walk_frac_df.index],
                    names=walk_frac_df.index.names,
                )
                walk_frac_df.columns = pd.MultiIndex.from_tuples(
                    [(str(g), str(n)) for g, n in walk_frac_df.columns],
                    names=walk_frac_df.columns.names,
                )
            else:
                coh_df.index = coh_df.index.astype(str)
                coh_df.columns = coh_df.columns.astype(str)
                walk_frac_df.index = walk_frac_df.index.astype(str)
                walk_frac_df.columns = walk_frac_df.columns.astype(str)

            # 3. Save Outputs
            coh_df.to_parquet(save_path_topo / f"{topo_name}_CohMat.parquet")
            walk_frac_df.to_parquet(save_path_topo / f"{topo_name}_WalkFracMat.parquet")
            # Only save the Teams file if extraction was run
            if groups_df is not None:
                save_path_groups = save_path_topo / f"{topo_name}_Teams.csv"
                groups_df.to_csv(save_path_groups, index=False)

            # 4. Handle detailed tracking matrices if fast_mode is False
            if not fast_mode and walk_mats_df is not None:
                if isinstance(walk_mats_df.index, pd.MultiIndex):
                    walk_mats_df.index = pd.MultiIndex.from_tuples(
                        [(str(ln), str(n)) for ln, n in walk_mats_df.index],
                        names=walk_mats_df.index.names,
                    )
                    walk_mats_df.columns = pd.MultiIndex.from_tuples(
                        [(str(t), str(n)) for t, n in walk_mats_df.columns],
                        names=walk_mats_df.columns.names,
                    )
                else:
                    walk_mats_df.index = walk_mats_df.index.astype(str)
                    walk_mats_df.columns = walk_mats_df.columns.astype(str)

                walk_mats_df.to_parquet(
                    save_path_topo / f"{topo_name}_WalkMats.parquet"
                )

            successful_computations += 1

        except Exception as e:
            print(f"Failed to process {topo_name}: {e}")
            continue

    print(
        f"Success! Processed and saved {successful_computations} out of {len(topo_list)} networks."
    )


def _extract_teams_worker(topo_folder: Path, final_split: bool) -> bool:
    """
    Isolated worker function for parallel processing.
    Reads the CohMat, runs the team ranking algorithm, and saves Teams.csv.
    """
    try:
        if not topo_folder.is_dir():
            return False

        topo_name = topo_folder.name
        coh_path = topo_folder / f"{topo_name}_CohMat.parquet"

        if not coh_path.exists():
            return False

        # Load matrix
        coh_df = pd.read_parquet(coh_path)

        # Flatten MultiIndex if it exists (in case it was run previously)
        if isinstance(coh_df.index, pd.MultiIndex):
            coh_df.index = coh_df.index.get_level_values(-1)
        if isinstance(coh_df.columns, pd.MultiIndex):
            coh_df.columns = coh_df.columns.get_level_values(-1)

        # Find teams using the advanced benefit ranking algorithm
        groups_df = find_teams_benefit_ranking(coh_df, final_split=final_split)

        # Save output
        groups_path = topo_folder / f"{topo_name}_Teams.csv"
        groups_df.to_csv(groups_path, index=False)

        return True
    except Exception as e:
        print(f"\nError extracting teams for {topo_folder.name}: {e}")
        return False


# def find_teams_parallel(
#     results_dir: Union[str, Path], final_split: bool = False, max_workers: int = None
# ):
#     """
#     Scans a directory of processed network folders and extracts teams in parallel.
#
#     Args:
#         results_dir: The directory containing the processed sub-folders (e.g., ./ScaledCohResults/).
#         final_split: Passed to the find_teams algorithm.
#         max_workers: Number of CPU cores to use. Defaults to all available cores.
#     """
#     results_dir = Path(results_dir)
#     folders = [d for d in results_dir.iterdir() if d.is_dir()]
#
#     print(f"\nStarting parallel team extraction for {len(folders)} networks...")
#
#     # Use partial to lock in the final_split argument for the worker mapping
#     worker_func = partial(_extract_teams_worker, final_split=final_split)
#
#     # Execute in parallel using a Process Pool
#     with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
#         results = list(
#             tqdm(
#                 executor.map(worker_func, folders),
#                 total=len(folders),
#                 desc="Extracting Teams",
#             )
#         )
#
#     successful = sum(results)
#     print(
#         f"Finished! Successfully extracted teams for {successful} / {len(folders)} networks."
#     )


def find_teams_parallel(
    results_dir: Union[str, Path], final_split: bool = False, max_workers: int = None
):
    """
    Scans a directory of processed network folders and extracts teams in parallel.
    Uses as_completed for real-time progress tracking.
    """
    results_dir = Path(results_dir)
    folders = [d for d in results_dir.iterdir() if d.is_dir()]

    print(f"\nStarting parallel team extraction for {len(folders)} networks...")

    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 1. Submit all tasks independently to the worker pool
        futures = [
            executor.submit(_extract_teams_worker, folder, final_split)
            for folder in folders
        ]

        # 2. Yield results exactly as they finish, regardless of the input order
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(folders),
            desc="Extracting Teams",
        ):
            results.append(future.result())

    successful = sum(results)
    print(
        f"Finished! Successfully extracted teams for {successful} / {len(folders)} networks."
    )


#######################################################
### Cyclycity Testing
#######################################################


@jit
def calc_cyclicity(matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the Cyclicity (Normality, ν) of any given matrix.
    Formula: sum(|eigenvalues|^2) / Frobenius_Norm_Squared
    """
    eigenvalues = jnp.linalg.eigvals(matrix)
    sum_sq_eig = jnp.sum(jnp.abs(eigenvalues) ** 2)
    frobenius_sq = jnp.sum(jnp.abs(matrix) ** 2)

    # Safely avoid division by zero
    cyclicity = jnp.where(frobenius_sq == 0.0, 0.0, sum_sq_eig / frobenius_sq)
    return cyclicity


@jit
def calc_communicability_metrics(
    adj: jnp.ndarray, mod_adj: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calculates the continuous Communicability matrix and returns it along
    with structural flow metrics.
    """
    # 1. Scale to prevent float64 overflow in dense networks
    N = mod_adj.shape[0]
    max_degree = jnp.max(jnp.sum(mod_adj, axis=1))

    scale_factor = jnp.where(
        jnp.maximum(max_degree, N) > 700.0,
        700.0 / (jnp.maximum(max_degree, N) + 1e-8),
        1.0,
    )

    safe_mod_adj = mod_adj * scale_factor

    # 2. Communicability Matrix (Estrada's G = e^|A|)
    comm_mat = expm(safe_mod_adj)

    # 3. Calculate Cyclicities
    comm_cyclicity = calc_cyclicity(comm_mat)
    adj_cyclicity = calc_cyclicity(mod_adj)

    # 4. Calculate Mean Communicability
    mean_comm = jnp.mean(comm_mat)

    return comm_mat, comm_cyclicity, adj_cyclicity, mean_comm


def compile_communicability_summary(
    topo_pattern: Union[str, Path], save_dir: Union[str, Path]
) -> pd.DataFrame:
    """
    Scans a directory of .topo files, computes communicability metrics for each,
    and compiles a summary dataframe containing TopoName, Cyclicity, and Mean Communicability.
    """
    topo_dir = Path(topo_pattern)
    output_filepath = Path(save_dir)

    # Allow for subfolder wildcard matching or direct directory iteration
    if "*" in topo_dir.name:
        topo_files = sorted(topo_dir.parent.glob(topo_dir.name))
    else:
        topo_files = sorted(topo_dir.glob("*.topo"))

    print(f"\nStarting Communicability Compilation for {len(topo_files)} topologies...")

    all_records = []

    for topo_path in tqdm(topo_files, desc="Compiling Communicability"):
        topo_name = topo_path.stem

        try:
            # Parse network using your existing parser
            adj, node_list = parse_topodf(topo_path)
            mod_adj = jnp.absolute(adj).astype(jnp.float64)

            # JAX Calculation
            comm_mat, comm_cyclicity, adj_cyclicity, mean_comm = (
                calc_communicability_metrics(adj, mod_adj)
            )

            # Convert JAX arrays to Python floats for storage
            record = {
                "TopoName": topo_name,
                "NumNodes": len(node_list),
                "AdjCyclicity": float(adj_cyclicity),
                "CommCyclicity": float(comm_cyclicity),
                "MeanCommunicability": float(mean_comm),
            }

            all_records.append(record)

        except Exception as e:
            print(f"\nError processing {topo_name}: {e}")

    # Build final DataFrame
    final_df = pd.DataFrame(all_records)

    if final_df.empty:
        print("No valid results found. Empty DataFrame returned.")
        return final_df

    # Save outputs
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_filepath.with_suffix(".parquet"), engine="pyarrow")
    final_df.to_csv(output_filepath.with_suffix(".csv"), index=False)

    print(f"\nCompilation complete! Saved {len(final_df)} records to:")
    print(f" -> {output_filepath.with_suffix('.csv')}")

    return final_df


if __name__ == "__main__":
    # Force Python to spawn fresh processes to avoid inheriting JAX's background threads
    multiprocessing.set_start_method("spawn", force=True)

    # ###########################################################################################
    # ### Run on the base circuits
    # ###########################################################################################
    #
    # topo_file_pattern = Path("./AllUniqueNets/Topologies/")
    # result_dir = Path("./MotifCohResults")
    #
    # process_and_save_serial(
    #     topo_pattern="./AllUniqueNets/Topologies/",
    #     save_dir="./MotifCohResults/",
    #     upto_length=10,
    #     fast_mode=True,
    #     final_split=True,
    #     extract_teams=False,
    # )
    #
    # find_teams_parallel(
    #     results_dir="./MotifCohResults/",
    #     final_split=True,  # Toggle your split logic here
    #     max_workers=(multiprocessing.cpu_count() - 5),
    # )
    #
    # compile_coherence_results(
    #     results_dir=result_dir,
    #     output_filepath=result_dir / "CompiledMotifSummary",
    #     parse_artinet=True,
    # )

    ############################################################################################
    #### Run on the scaled topos
    ############################################################################################
    #
    # scaled_topo_dir = Path("./ScaledTopos/")
    # scaled_result_dir = Path("./ScaledCohResults/")
    #
    ## 1. Math Phase (Serial/JAX) - Turn team extraction OFF
    # process_and_save_serial(
    #    topo_pattern=scaled_topo_dir,
    #    save_dir=scaled_result_dir,
    #    upto_length=10,
    #    fast_mode=True,
    #    extract_teams=False,
    # )
    #
    ## 2. Team Phase (Parallel/CPU)
    # find_teams_parallel(
    #    results_dir=scaled_result_dir,
    #    final_split=True,
    #    max_workers=(multiprocessing.cpu_count() - 5),
    # )
    #
    ## 3. Compilation Phase
    # compile_coherence_results(
    #    results_dir=scaled_result_dir,
    #    output_filepath=scaled_result_dir / "CompiledScaledSummary",
    #    parse_scaled=True,
    # )
    #
    ##########################################################################################
    ## Run on the Abasy Networks
    ##########################################################################################

    NETWORK_TO_ORGANISM = {
        "196627_v2020_s21_regNetwork_Strong": "Corynebacterium glutamicum",
        "83332_v2018_s15-16_regNetwork": "Mycobacterium tuberculosis",
        "224308_v2022_sSW22_regNetwork": "Bacillus subtilis",
        "511145_v2022_sRDB22_eStrong_regNetwork_Strong": "Escherichia coli",
        "208964_v2020_sRPA20_regNetwork_Strong": "Pseudomonas aeruginosa",
        "100226_v2019_sA22-DBSCR15_eStrong_regNetwork": "Streptomyces coelicolor",
    }

    # target_networks = list(NETWORK_TO_ORGANISM.keys())
    #
    # # biowg_topo_dir = Path("./AbasyTOPOS/")
    # biowg_targeted_topo_dir = Path("./AbasyTOPOS_Targeted/")
    # biowg_result_dir = Path("./AbasyCohResults_Targeted/")
    #
    # ## Only run when automatically slecting from all the Abasy Topos and _targeted file not present
    # # # Isolate target base networks into a staging directory to keep the serial function unchanged
    # # biowg_targeted_topo_dir.mkdir(parents=True, exist_ok=True)
    # # for net in target_networks:
    # #     src_file = biowg_topo_dir / f"{net}.topo"
    # #     if src_file.exists():
    # #         shutil.copy(src_file, biowg_targeted_topo_dir / f"{net}.topo")
    # #     else:
    # #         print(f"Warning: Base network file not found -> {src_file}")
    #
    # print("\n--- Processing Target Biological Networks ---")
    #
    # process_and_save_serial(
    #     topo_pattern=biowg_targeted_topo_dir,
    #     save_dir=biowg_result_dir,
    #     upto_length=10,
    #     fast_mode=True,
    #     extract_teams=False,
    # )
    #
    # find_teams_parallel(
    #     results_dir=biowg_result_dir,
    #     final_split=True,
    #     max_workers=(multiprocessing.cpu_count() - 5),
    # )
    #
    # compile_coherence_results(
    #     results_dir=biowg_result_dir,
    #     output_filepath=biowg_result_dir / "CompiledTargetSummary",
    #     parse_scaled=False,
    #     parse_artinet=False,
    # )

    ##########################################################################################
    ## Run on the Shuffled Abasy Networks
    ##########################################################################################

    # Getting all the folders with shuffled Newtorks
    shuffled_dirs = list(
        Path("./WTvsShuffledAnalysis_AbasyNets_Targeted/").glob("*/Shuffled_Networks/")
    )
    print(shuffled_dirs)

    for net_name, org_name in NETWORK_TO_ORGANISM.items():
        # Processing the WT topo
        wt_cohdir = Path("./WTvsShuffledAnalysis_AbasyNets_Targeted/WT_TOPO")

        process_and_save_serial(
            topo_pattern=wt_cohdir,
            save_dir=wt_cohdir,
            upto_length=10,
            fast_mode=True,
            extract_teams=False,
        )

        find_teams_parallel(
            results_dir=wt_cohdir,
            final_split=True,
            max_workers=(multiprocessing.cpu_count() - 3),
        )

        compile_coherence_results(
            results_dir=wt_cohdir,
            output_filepath=wt_cohdir / f"CompiledSummary_{net_name}",
            parse_scaled=False,
            parse_artinet=False,
        )

        # Processing the shuffled topos
        sd = (
            Path("./WTvsShuffledAnalysis_AbasyNets_Targeted/")
            / net_name
            / "Shuffled_Networks"
        )

        if not sd.exists() or not sd.is_dir():
            print(f"Skipping {org_name} ({net_name}): Directory not found at {sd}")
            continue

        print(f"\n>> Shuffled Networks for: {org_name} ({net_name})")

        result_dir = sd.parent / "Shuffled_CohMats"

        # Pass the specific target directory
        process_and_save_serial(
            topo_pattern=sd,
            save_dir=result_dir,
            upto_length=10,
            fast_mode=True,
            extract_teams=False,
        )

        find_teams_parallel(
            results_dir=result_dir,
            final_split=True,
            max_workers=(multiprocessing.cpu_count() - 3),
        )

        compile_coherence_results(
            results_dir=result_dir,
            output_filepath=result_dir / f"CompiledShuffledSummary_{net_name}",
            parse_scaled=False,
            parse_artinet=False,
        )
