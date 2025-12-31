import asyncio
import glob
import itertools as it
import os
from functools import partial
from itertools import chain
from tqdm import tqdm
import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
import re
from jax import jit, lax
from jax.debug import print as jprint
from jax.scipy.linalg import expm
from jax.scipy.special import factorial

# Force CPU as the default backend
# jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

##################################################################
##################################################################

# Nord Colors
nord_colors = [
    "#D08770",  # orange
    "#8FBCBB",  # light greenish-blue
    "#B48EAD",  # light purple
    "#A3BE8C",  # green
    "#5E81AC",  # blue
    "#BF616A",  # red
    "#88C0D0",  # light blue
    "#EBCB8B",  # yellow
    "#81A1C1",  # muted blue
]

# Set font and plot styles in Matplotlib
plt.rcParams.update(
    {
        "axes.prop_cycle": plt.cycler(color=nord_colors),
        "font.family": "Roboto",
        "legend.fontsize": 14,
        "legend.title_fontsize": 14,
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.edgecolor": "black",  # For black tick borders
        "axes.linewidth": 1.25,  # Make axis lines more visible
        # "axes.spines.left": True,  # Display the left spine
        # "axes.spines.bottom": True,  # Display the bottom spine
        # "axes.spines.right": True,  # Display the right spine
        # "axes.spines.top": True,  # Display the top spine
        # "lines.linewidth": 0.8,  # Set the default line width for plots
    }
)

# Apply the "ticks" style manually by removing the grid
plt.style.use("seaborn-v0_8-deep")
sns.set_palette(sns.color_palette(nord_colors))
# plt.rcParams["axes.grid"] = False  # Disable gridlines to match Seaborn's 'ticks' style


def create_diverging_palette(color_low, color_mid, color_high, color_bad, n_colors=256):
    """
    Creates a custom diverging color palette for Matplotlib and Seaborn.

    Parameters:
    - color_low (str): Hex code or color name for the low end of the palette.
    - color_mid (str): Hex code or color name for the midpoint of the palette.
    - color_high (str): Hex code or color name for the high end of the palette.
    - n_colors (int): Number of colors in the palette. Default is 256.

    Returns:
    - colormap: A Matplotlib colormap object.
    - palette: A list of colors for use with Seaborn.
    """
    # Create the colormap using LinearSegmentedColormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_diverging_palette", [color_low, color_mid, color_high], N=n_colors
    )

    # Generate the color palette for Seaborn
    palette = [cmap(i) for i in range(cmap.N)]

    # Change the bad color to bad_color
    cmap.set_bad(color_bad)

    return cmap, palette


# Creating a palette for the coherence matrix
cmap, palette = create_diverging_palette("#bf616a", "#eceff4", "#5e81ac", "#4c566a")


def create_sequential_palette(color_mid, color_high, color_bad, n_colors=256):
    """
    Creates a custom sequential color palette for Matplotlib and Seaborn, going from the middle color
    to the high end of the palette.

    Parameters:
    - color_mid (str): Hex code or color name for the starting color of the palette.
    - color_high (str): Hex code or color name for the ending color of the palette.
    - color_bad (str): Color for invalid values (NaNs).
    - n_colors (int): Number of colors in the palette. Default is 256.

    Returns:
    - colormap: A Matplotlib colormap object.
    - palette: A list of colors for use with Seaborn.
    """
    # Create the colormap using LinearSegmentedColormap, transitioning from color_mid to color_high
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_sequential_palette", [color_mid, color_high], N=n_colors
    )

    # Generate the color palette for Seaborn
    palette = [cmap(i) for i in range(cmap.N)]

    # Change the bad color to color_bad
    cmap.set_bad(color_bad)

    return cmap, palette


# Creating a sequential palette for the stripplot
cmap_seq, palette_seq = create_sequential_palette("#2e3440", "#bf616a", "#4c566a")

##################################################################
##################################################################

np.seterr(divide="ignore", invalid="ignore")

#################################################################
# Parsing the topo file
#################################################################


# Function to parse the topology file into a numpy array and return the matrix and its list of nodes in order
def parse_topodf(topo_file_path, sparse=False):
    topo_df = pd.read_csv(topo_file_path, sep=r"\s+")
    topo_df = topo_df.replace({2: -1})
    # node_list = list(set(topo_df["Source"].to_list() + topo_df["Target"].to_list()))
    topo_net = nx.from_pandas_edgelist(
        topo_df,
        source="Source",
        target="Target",
        edge_attr="Type",
        create_using=nx.DiGraph,
    )
    node_list = list(topo_net.nodes())
    # print(len(node_list))
    if not sparse:
        topo_net = nx.to_numpy_array(topo_net, weight="Type")
        topo_net = jnp.array(topo_net).astype(jnp.float64)
    else:
        topo_net = nx.to_scipy_sparse_array(topo_net, weight="Type")
    return topo_net, node_list


#################################################################

#################################################################
# Getting walk matrices
#################################################################


# Internal function to take matrix to a power and then get the relevant walk change matrices
# use @njit(error_model='numpy') for dealing around the divide by zero error
# @njit(nopython=True, cache=True)
@jit
def next_walk_matrices(prev_num_walks, prev_num_poswalks, adj, modadj):
    # Number of negative walks previously
    prev_num_negwalk = prev_num_walks - prev_num_poswalks
    # print(f"Prev Negwalk:\n{prev_num_negwalk}")
    # Number if total possible walk
    num_walks = jnp.dot(modadj, prev_num_walks)
    # print(f"Num of Walks:\n{num_walks}")
    # Max number of pos walks if none convert to neg
    max_cnsrvd_poswalks = jnp.dot(modadj, prev_num_poswalks)
    # print(f"Max Num Pos Walks:\n{max_cnsrvd_poswalks}")
    # Max number of neg walks if none convert to pos
    max_cnsrvd_negwalks = jnp.dot(modadj, prev_num_negwalk)
    # print(f"Max Num Neg Walks:\n{max_cnsrvd_negwalks}")
    # Net positive walks
    net_num_poswalks = jnp.dot(adj, prev_num_poswalks)
    # print(f"Net Num Pos Walks:\n{net_num_poswalks}")
    # Net negative walks
    net_num_negwalks = jnp.dot(adj, prev_num_negwalk)
    # print(f"Net Num Neg Walks:\n{net_num_negwalks}")
    # Both the actual number are converted to int64 back again as they will always be int
    # GIves more accuracy than float32
    # Actual number of pos walks
    cnsrvd_num_poswalks = ((net_num_poswalks + max_cnsrvd_poswalks) / 2).astype(
        jnp.float64
    )
    # print(f"Conserved Num Pos Walks:\n{cnsrvd_num_poswalks}")
    # Actual number of neg walks
    cnsrvd_num_negwalks = ((net_num_negwalks + max_cnsrvd_negwalks) / 2).astype(
        jnp.float64
    )
    # print(f"Conserved Num Neg Walks:\n{cnsrvd_num_negwalks}")
    # Pos walks which became negative
    flip_num_posneg_walks = max_cnsrvd_poswalks - cnsrvd_num_poswalks
    # print(f"Flipped Num PosNeg Walks:\n{flip_num_posneg_walks}")
    # Neg walks which became positive
    flip_num_negpos_walks = max_cnsrvd_negwalks - cnsrvd_num_negwalks
    # print(f"Flipped Num NegPos Walks:\n{flip_num_negpos_walks}")
    # Total positive walks
    tot_num_poswalks = cnsrvd_num_poswalks + flip_num_negpos_walks
    # print(f"Total Num Pos Walks:\n{tot_num_poswalks}")
    # Total negative walks
    tot_num_negwalks = cnsrvd_num_negwalks + flip_num_posneg_walks
    # print(f"Total Num Neg Walks:\n{tot_num_negwalks}")

    # Fraction of positive walks
    frac_poswalks = tot_num_poswalks / num_walks

    # Stack the matrices in the Order num_walks, num_P, num_N, num_PP, num_PN, num_NN, num_NP
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


# Function to wight the matrices fo the fraction of positive walks
@jit
def weight_frac_poswalks(frac_poswalks):
    # Centering all the frac walks
    frac_poswalks = jnp.divide(frac_poswalks - 0.5, 0.5)
    # Getting the lengths of the walks in order
    scaling_factors = jnp.arange(1, frac_poswalks.shape[0] + 1)
    # Reshape to boracast matrix-wise division
    scaling_factors = scaling_factors.reshape(-1, 1, 1)
    # Calcualting the reciporcal
    scaling_factors = jnp.reciprocal(scaling_factors)
    # Weighing and summing the matrices
    sum_frac_poswalks = jnp.nansum(frac_poswalks * scaling_factors, axis=0)
    # Dividing by the sum of the reciprocals
    sum_frac_poswalks = jnp.divide(sum_frac_poswalks, jnp.sum(scaling_factors))
    return sum_frac_poswalks


# Function to get the cohernece matrix from the walk and frac pos matrices
# Ignores the normalisation term for values where there are no walks
# Use factorial on arange if factorial scaling is needed
@jit
def get_weightnorm_coh(walk_mats):
    # Getting the number of nodes
    num_nodes = walk_mats.shape[1] // 8
    # Getting the upto length
    upto_length = walk_mats.shape[0] // num_nodes
    # Getting the num walk matrices
    num_walk_mat = walk_mats[:, :num_nodes].reshape(upto_length, num_nodes, num_nodes)
    frac_poswalk_mat = walk_mats[:, -(num_nodes):].reshape(
        upto_length, num_nodes, num_nodes
    )
    # Making the corresponding values where no walks are present in walkmats as nan
    nowalk_mask = jnp.where(num_walk_mat == 0, jnp.nan, 1)
    # Centering the fraction of poswalks to 0
    # frac_poswalk_mat = jnp.divide((frac_poswalk_mat - 0.5), 0.5)
    frac_poswalk_mat = (2 * frac_poswalk_mat) - 1
    # Making sure nans are present where they should be
    frac_poswalk_mat = jnp.where(num_walk_mat == 0, jnp.nan, frac_poswalk_mat)
    # Getting the lengths of the walks in order
    scaling_factors = jnp.arange(1, frac_poswalk_mat.shape[0] + 1).reshape(
        frac_poswalk_mat.shape[0], 1, 1
    )
    # Getting the lengths of the walks as factorials
    scaling_factors = factorial(jnp.arange(1, frac_poswalk_mat.shape[0] + 1)).reshape(
        frac_poswalk_mat.shape[0], 1, 1
    )
    # Dividing the centered fraction of postivie walks
    frac_poswalk_mat = jnp.nansum(jnp.divide(frac_poswalk_mat, scaling_factors), axis=0)
    # Dividing the walk mats to get the max possible normalising factor
    scaling_factors = jnp.nansum(jnp.divide(nowalk_mask, scaling_factors), axis=0)
    # Getting the new no walks mask
    nowalk_mask = jnp.sum(nowalk_mask, axis=0)
    # Dividing the frac_poswalk_mat sum with the scaling factor sum
    frac_poswalk_mat = jnp.divide(frac_poswalk_mat, scaling_factors)
    # Applying the NaN mask
    frac_poswalk_mat = jnp.where(nowalk_mask == 0, jnp.nan, frac_poswalk_mat)
    # print(frac_poswalk_mat)
    # Dividing the scaling_factors by the max possible value from scaling
    scaling_factors = jnp.divide(
        scaling_factors,
        jnp.sum(jnp.divide(1, factorial(jnp.arange(1, upto_length + 1)))),
    )
    return frac_poswalk_mat, scaling_factors


# Function to calcualte the walk matrices for different walk lengths
@partial(jit, static_argnums=0)
def calc_walk_matrices(upto_length, adj, mod_adj):
    # Number of positive walks
    prev_num_poswalks = (adj == 1).astype(jnp.float64)

    # Gettting the adj shape
    num_nodes = adj.shape[0]

    # Main loop to calcualte the walks of increasing lengths
    def step(carry, _):
        prev_num_walks, prev_num_poswalks = carry
        req_mats = next_walk_matrices(prev_num_walks, prev_num_poswalks, adj, mod_adj)
        # new_num_walks = req_mats[:, :, 0]
        new_num_walks = req_mats[:, :num_nodes]
        # jprint("new_num_walks:\n{}", new_num_walks)
        new_num_poswalks = req_mats[:, num_nodes : 2 * num_nodes]
        # jprint("new_num_poswalks:\n{}", new_num_poswalks)
        return (new_num_walks, new_num_poswalks), req_mats

    # Run the scan function over the walks
    (final_walks, final_poswalks), stacked_results = lax.scan(
        step,
        (mod_adj, prev_num_poswalks),
        xs=None,
        length=upto_length - 1,  # length 1 is initial, so we compute walks for 2..L
    )

    return stacked_results


# Function to get the coherence values related stats
@jit
def mat_stats(coh_mat):
    abs_coh_mat = jnp.abs(coh_mat)
    abs_mean_val = jnp.nanmean(abs_coh_mat)
    abs_median_val = jnp.nanmedian(abs_coh_mat)
    mean_val = jnp.nanmean(coh_mat)
    median_val = jnp.nanmedian(coh_mat)
    return abs_mean_val, abs_median_val, mean_val, median_val


# # Function to get the coherence values related stats
# @jit
# def walk_stats(walk_mat):
#     abs_walk_mat = jnp.round(walk_mat, 4)
#     mean_val = jnp.round(jnp.nanmean(abs_walk_mat), 4)
#     median_val = jnp.round(jnp.nanmedian(abs_walk_mat), 4)
#     min_val = jnp.round(jnp.nanmin(walk_mat), 4)
#     max_val = jnp.round(jnp.nanmax(walk_mat), 4)
#     return mean_val, median_val, min_val, max_val


# Function to get the coherence matrices for the given adjacency matrix of the network till the given network path
def calc_walkmats(topo_file_path, save_path, upto_length=None):
    adj, node_list = parse_topodf(topo_file_path)

    if upto_length is None:
        upto_length = 15
        # print(upto_length)
    # print(topo_file_path, upto_length)

    os.makedirs(save_path, exist_ok=True)

    # Mod version of the adjacency matrix
    mod_adj = jnp.absolute(adj).astype(jnp.float64)

    # Walk mats for the path length 1
    one_walk_mats = jnp.concatenate(
        (
            mod_adj,
            (adj == 1).astype(jnp.float64),
            (adj == -1).astype(jnp.float64),
            jnp.zeros((adj.shape[0], adj.shape[0] * 4)).astype(jnp.float64),
            (adj == 1).astype(jnp.float64) / mod_adj,
        ),
        axis=1,
    )

    # Calculcating the walk matrices for all the lengths
    walk_mats = calc_walk_matrices(upto_length, adj, mod_adj)

    # Stack the one walk mat with the other walk length walk mats
    walk_mats = jnp.concatenate(
        (one_walk_mats, jnp.concatenate(walk_mats, axis=0)), axis=0
    )

    coh_mat, scale_mat = get_weightnorm_coh(walk_mats)

    # Getting the fraction of walks wrt complete graph
    # walk_mat = jnp.divide(
    #     expm(mod_adj) - jnp.eye(mod_adj.shape[0]),
    #     expm(jnp.ones(mod_adj.shape)) - jnp.eye(mod_adj.shape[0]),
    # )
    walk_mat = jnp.divide(
        expm(mod_adj),
        expm(jnp.ones(mod_adj.shape)),
    )

    # List of the walk types in order
    walk_name_cols = [
        "NumWalks",
        "Total_Num_Pos_Walks",
        "Total_Num_Neg_Walks",
        "Conserved_Num_Pos_Walks",
        "Conserved_Num_Neg_Walks",
        "Flipped_Num_PosNeg_Walks",
        "Flipped_Num_NegPos_Walks",
        "Fraction_PosWalks",
    ]

    # Converting to a dataframe with multiple indices
    walk_mats = pd.DataFrame(
        walk_mats,
        index=pd.MultiIndex.from_arrays(
            [
                [
                    wlen
                    for wlen in range(1, upto_length + 1)
                    for _ in range(len(node_list))
                ],
                node_list * upto_length,
            ],
            names=["WalkLength", "SourceNode"],
        ),
        columns=pd.MultiIndex.from_arrays(
            [
                [wlk for wlk in walk_name_cols for _ in range(len(node_list))],
                node_list * len(walk_name_cols),
            ],
            names=["WalkType", "TargetNode"],
        ),
    )

    # # Calculating the coherence matrix (exponential version)
    # cohmat = calc_exp_cohmat(adj)

    # Calucating the absolute mean and median value of the pairwise coherence values
    abs_mean_val, abs_median_val, mean_val, median_val = mat_stats(coh_mat)
    cohinfo_dict = {
        "AbsMeanCohVal": abs_mean_val.item(),
        "AbsMedianCohVal": abs_median_val.item(),
        "CohMatMean": mean_val.item(),
        "CohMatMedian": median_val.item(),
    }
    # Calcualting the walk relatd values
    abs_mean_val, abs_median_val, mean_val, median_val = mat_stats(walk_mat)
    # Calcualting the exponential of themod adj matrix
    mod_exp_adj = expm(mod_adj)
    non_zero_vals = mod_exp_adj[mod_exp_adj > 0]
    cohinfo_dict.update(
        {
            "AbsMeanWalkVal": jnp.log10(abs_mean_val).item(),
            "AbsMedianWalkVal": jnp.log10(abs_median_val).item(),
            "MeanComm": jnp.log10(jnp.nanmean(mod_exp_adj)).item(),
            "MedianComm": jnp.log10(jnp.nanmedian(mod_exp_adj)).item(),
            "MeanCommNZ": jnp.log10(jnp.mean(non_zero_vals)).item(),
            "MedianCommNZ": jnp.log10(jnp.median(non_zero_vals)).item(),
        }
    )
    # Calculcating the scale related values
    abs_mean_val, abs_median_val, mean_val, median_val = mat_stats(
        jnp.where(scale_mat == 0.0, jnp.nan, scale_mat)
    )
    cohinfo_dict.update(
        {
            "AbsMeanScaleVal": jnp.log10(abs_mean_val).item(),
            "AbsMedianScaleVal": jnp.log10(abs_median_val).item(),
        }
    )

    # COnverting to a dataframe
    coh_mat = pd.DataFrame(coh_mat, index=node_list, columns=node_list)
    coh_mat.index.name = "SourceNode"
    coh_mat.columns.name = "TargetNode"

    # Converting the walks ratios to a dataframe
    walk_mat = pd.DataFrame(walk_mat, index=node_list, columns=node_list)

    # COnverting the scale mat to a dataframe
    scale_mat = pd.DataFrame(scale_mat, index=node_list, columns=node_list)

    # Getting the groups from the cohmat
    groups_df, median_coh, mean_coh = find_teams(coh_mat)
    # print(groups_df)

    # Updating the cohinfo with the mean and median coherence values
    cohinfo_dict.update({"MeanCoh": mean_coh, "MedianCoh": median_coh})

    # Checking if all nodes in coh_mat have a group in groups_df
    group_lookup = groups_df.set_index("Node")["Group"]

    # Checking for missing nodes
    missing_nodes = set(coh_mat.index) - set(groups_df["Node"])
    # print(set(groups_df["Node"]))
    # print(set(coh_mat.index))
    if missing_nodes:
        raise ValueError(f"Missing group information for nodes: {missing_nodes}")

    # Creating the multi-index
    row_multiindex = pd.MultiIndex.from_tuples(
        [(group_lookup[node], node) for node in coh_mat.index],
        names=["Group", "SourceNode"],
    )
    col_multiindex = pd.MultiIndex.from_tuples(
        [(group_lookup[node], node) for node in coh_mat.columns],
        names=["Group", "TargetNode"],
    )

    # Set MultiIndex for coh_mat
    coh_mat.index = row_multiindex.copy()
    coh_mat.columns = col_multiindex.copy()

    # Set MultiIndex for walk_mat
    walk_mat.index = row_multiindex.copy()
    walk_mat.columns = col_multiindex.copy()

    return walk_mats, coh_mat, scale_mat, walk_mat, groups_df, cohinfo_dict


#################################################################

####################################################################################
### Getting the Coherence matrix - Matrix exponential version
####################################################################################


# Function to calulate the coherence matrix using the exponential of the matrix
# This version weighs each of the walk cohmats by the factorial of the walk length
@jit
def calc_exp_cohmat(topo_adj):
    # Get the matrix exponent of the adjacency matrix
    adjmat_exp = expm(topo_adj)
    # Getting the max version of the exponent of the adjacency matrix
    max_adjmat_exp = expm(jnp.absolute(topo_adj))
    # Getting the normalised coherence matrix
    coh_mat = jnp.divide((adjmat_exp + max_adjmat_exp) / 2, max_adjmat_exp)
    coh_mat = (coh_mat - 0.5) / 0.5
    return coh_mat


####################################################################################


#################################################################
# For Finding Groups/teams
#################################################################


# Funtion to find teams of nodes which are strictly activating each other
# This algorithm makes sure that that team nodes are weakly connected and
# only have net activation connections bewteen them
# Self inhibitory nodes automatically beome a different team
# For sprase newtorks - even nodes which are not weakly connected to the other members
# are converted into speparte teams based on connected components of teams subgraph
def find_teams(df, max_passes=5):
    # Fill NaNs and convert to NumPy for speed
    # df = df.fillna(1e-10)
    mat = np.sign(df.values)
    # Dropiing the group multi index if present - will happen if cohmat is geven again to this funct
    if "Group" in df.index.names:
        df = df.droplevel("Group", axis=0)
    if "Group" in df.columns.names:
        df = df.droplevel("Group", axis=1)
    nodes = list(df.columns)
    idx_map = {node: i for i, node in enumerate(nodes)}

    # Create an undirected version of the cohamt where
    # if negetive is preesent, make teh two values negetive
    # And even is one positive is present, make bith positive
    for i, j in it.combinations_with_replacement(nodes, 2):
        ij = mat[idx_map[i], idx_map[j]]
        ji = mat[idx_map[j], idx_map[i]]
        if np.isnan(ij) and np.isnan(ji):
            continue  # skip if both entries are NaN
        elif ij <= 0 or ji <= 0:
            mat[idx_map[i], idx_map[j]] = -1
            mat[idx_map[j], idx_map[i]] = -1
        elif ij > 0 or ji > 0:
            mat[idx_map[i], idx_map[j]] = 1
            mat[idx_map[j], idx_map[i]] = 1

    # print(mat)
    # sns.heatmap(mat, xticklabels=nodes, yticklabels=nodes)
    # plt.show()

    groups = []

    for i, node in enumerate(nodes):
        found_group = False
        # If its self inhibitory - never going to go to any other group
        if mat[idx_map[node], idx_map[node]] == -1:
            groups.append({node})
            continue
        # Looping thorugh the groups to see if it can fit in any
        for group in groups:
            # print(f"Grp:{group}")
            # print(f"Node:{node}")
            # Checking if the group is single node one with self-inhibiton
            # If found skip to the next one
            if len(group) == 1:
                grp_nd = next(iter(group))
                if mat[idx_map[grp_nd], idx_map[grp_nd]] == -1:
                    continue
            group_indices = [idx_map[nd] for nd in group]
            i_idx = idx_map[node]
            # Cheking if the node has any connection at all to the group
            # Could be empty row and column for the node's connection to the group
            if 1 in mat[i_idx, group_indices]:
                # Get the slice of the matrix
                test_indices = [i_idx] + group_indices
                # print(test_indices)
                # Only one slice takes as its made symmetrical
                value_slice = mat[np.ix_(test_indices, test_indices)]
                # print(value_slice)
                # Both nans and diagonal values are not taken.
                # Nans mess up the logic
                # DIagonals - self activatory ones with parse outgoing connections
                # can go to any group - avoiding this is necessary
                value_slice = value_slice[
                    ~np.isnan(value_slice) & ~np.eye(len(test_indices), dtype=bool)
                ]
                # print(f"ValSlice:{value_slice}")
                # Check if node is activating with the group and does not bring inhibtion
                if (-1 not in value_slice) and (1 in value_slice):
                    group.add(node)
                    # print(f"UpdatedGrp:{group}")
                    found_group = True
                    break
        if not found_group:
            groups.append({node})

    groups = sorted(groups, key=len, reverse=True)
    # print(groups)

    for _ in range(max_passes):
        moved = False
        for r in reversed(groups):
            for n in list(r):
                i_idx = idx_map[n]
                sorted_groups = sorted(groups, key=len, reverse=True)
                for g in sorted_groups:
                    if g is r:
                        continue
                    group_indices = [idx_map[x] for x in g]
                    # Get the slice of the matrix
                    test_indices = [i_idx] + group_indices
                    # Only one slice taken as its made symmetrical
                    value_slice = mat[np.ix_(test_indices, test_indices)]
                    value_slice = value_slice[~np.isnan(value_slice)]
                    # Check if node is activating with the group and does not bring inhibtion
                    if (
                        (-1 not in value_slice)
                        and (1 in value_slice)
                        and (len(g) > len(r) + 1)
                    ):
                        g.add(n)
                        r.remove(n)
                        moved = True
                        break
        if not moved:
            break

    # groups_dict = {
    #     str(i + 1): sorted(group)
    #     for i, group in enumerate(sorted(groups, key=len, reverse=True))
    # }

    # New list to store groups which are wakly connected compnentes
    groups_concomps = []
    # Looping though the groups and checking if the subnewtork is connected
    for grp in groups:
        # COnverting the set to a list
        grp = list(grp)
        # Geting indices from the matrix
        idxs = [idx_map[n] for n in grp]
        submat = mat[np.ix_(idxs, idxs)]
        # Get valid upper-triangular non-NaN connections
        i_triup, j_triup = np.triu_indices(len(grp), k=1)
        valid_mask = ~np.isnan(submat[i_triup, j_triup])
        i_valid = i_triup[valid_mask]
        j_valid = j_triup[valid_mask]
        # Building the graph from edges
        edges = [(grp[i], grp[j]) for i, j in zip(i_valid, j_valid)]
        subcoh_grp = nx.Graph()
        subcoh_grp.add_nodes_from(grp)
        subcoh_grp.add_edges_from(edges)
        # Getting the list of connected components
        con_comps = nx.connected_components(subcoh_grp)
        # Appending the connected component list as groups to the list
        for comp in con_comps:
            groups_concomps.append(comp)

    # groups_concomps = groups.copy()
    # print(groups_concomps)
    # Redefine the mat to be the values of the coherence matrix
    mat = df.values
    mean_coh = 0
    median_coh = 0
    # Looping though the groups and getting the coherence values
    for grp in groups_concomps:
        # COnverting the set to a list
        grp = list(grp)
        # Geting indices from the matrix
        idxs = [idx_map[n] for n in grp]
        submat = mat[np.ix_(idxs, idxs)]
        # print(submat)
        if not np.all(np.isnan(submat)):
            # Getting the mean and median values
            mean_coh += np.nanmean(submat)
            # print(mean_coh)
            median_coh += np.nanmedian(submat)
            # print(median_coh)

    # print(median_coh / len(groups_concomps), mean_coh / len(groups_concomps))

    groups_dict = {
        str(i + 1): sorted(group)
        for i, group in enumerate(sorted(groups_concomps, key=len, reverse=True))
    }
    # print(groups_dict)
    # print(len(groups_dict))

    # Ensuring no emtpy sets are present
    groups_dict = {k: v for k, v in groups_dict.items() if v}

    groups_dict = pd.DataFrame(
        [
            (group, node)
            for group, node_list in groups_dict.items()
            for node in node_list
        ],
        columns=["Group", "Node"],
    )

    groups_dict["Group"] = groups_dict["Group"].astype(int)

    return (
        groups_dict,
        median_coh / len(groups_concomps),
        mean_coh / len(groups_concomps),
    )


# Function to read, find and write the groups
def save_groups(cohmat_path=None, cohmat=None, save_path=None):
    if cohmat_path is not None:
        cohmat = pd.read_parquet(cohmat_path)
    if cohmat_path is None and save_path is None:
        print("Provide the complete path to save the groups dataframe.")
    groups_dict = find_teams(cohmat)
    # Reorder the cohmat according to the team list
    node_order = list(chain.from_iterable(groups_dict.values()))
    cohmat = cohmat.loc[node_order, node_order]
    cohmat.to_parquet(cohmat_path)
    groups_dict = pd.DataFrame(
        {"Group": group, "Node": node}
        for group, node_list in groups_dict.items()
        for node in node_list
    )
    groups_dict.to_csv(cohmat_path.replace("_CohMat", "_Groups"), index=False)
    return None


#################################################################


#################################################################
# Plotting function of the coherence matrix
#################################################################
def plot_cohmat(cohmat_path):
    cohmat = pd.read_parquet(cohmat_path)

    # 1. FLATTEN INDEX (Fixing the MultiIndex issue)
    if isinstance(cohmat.index, pd.MultiIndex):
        cohmat.index = cohmat.index.get_level_values(1)
        cohmat.index.name = "SourceNode"
    if isinstance(cohmat.columns, pd.MultiIndex):
        cohmat.columns = cohmat.columns.get_level_values(1)
        cohmat.columns.name = "TargetNode"

    # 2. LOAD GROUPS
    # Logic to find the groups file (adjust path replacement as needed)
    groups_path = cohmat_path.replace("_CohMat.parquet", "_Groups.parquet")

    groups_df = pd.read_parquet(groups_path)
    print(groups_df["Group"].value_counts())
    groups_dict = groups_df.groupby("Group")["Node"].apply(list).to_dict()

    # 3. REORDER MATRIX (The fix for your "impossible" boxes)
    ordered_nodes = []
    # Sort groups by ID (1, 2, 3...) so they appear in order on the plot
    for group_id in sorted(groups_dict.keys()):
        # Get nodes for this group
        nodes = groups_dict[group_id]
        # Only add nodes that actually exist in the current coherence matrix
        ordered_nodes.extend([n for n in nodes if n in cohmat.index])

    # Append any nodes in the matrix that might not be in the groups file (orphans)
    # This prevents data loss if the groups file is incomplete
    existing_nodes = set(ordered_nodes)
    orphans = [n for n in cohmat.index if n not in existing_nodes]
    # final_order = ordered_nodes + orphans

    # Apply the new order to the DataFrame
    # Ordered Nodes
    # ordered_nodes = sorted(ordered_nodes)
    cohmat = cohmat.reindex(index=ordered_nodes, columns=ordered_nodes)

    # # 4. PLOTTING
    # cmap = plt.get_cmap("RdBu_r")
    # cmap.set_bad(color="lightgray")

    plt.figure(figsize=(20, 18))
    ax = sns.heatmap(
        cohmat,
        cmap=cmap,
        cbar=True,
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="#2e3440",
        square=True,
        cbar_kws={"shrink": 0.75},
        annot=True,
    )

    # Force X-axis labels to be vertical (90 degrees)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        ha="right",  # Aligns the label correctly so it doesn't overlap the tick
        rotation_mode="anchor",
    )

    # Force Y-axis labels to be horizontal (0 degrees)
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        va="center",  # Vertically centers the text on the tick
    )

    # 5. DRAW BOXES
    # Now that the matrix is sorted, your original simple logic works perfect
    for group_id, group_nodes in groups_dict.items():
        try:
            # Filter for nodes present in this matrix
            valid_nodes = [n for n in group_nodes if n in cohmat.index]
            if not valid_nodes:
                continue

            # Get indices (now they will be contiguous integers, e.g., 5,6,7)
            indices = [cohmat.index.get_loc(node) for node in valid_nodes]

            min_idx = min(indices)
            max_idx = max(indices)
            size = max_idx - min_idx + 1

            # Draw the box
            ax.add_patch(
                patches.Rectangle(
                    (min_idx, min_idx),  # x, y (top-left of the diagonal block)
                    size,  # width
                    size,  # height
                    edgecolor="black",
                    facecolor="none",
                    lw=4,
                )
            )
        except Exception as e:
            print(f"Skipping group {group_id}: {e}")

    # Borders and Labels
    ax.axhline(y=0, color="k", linewidth=1)
    ax.axhline(y=cohmat.shape[0], color="k", linewidth=1)
    ax.axvline(x=0, color="k", linewidth=1)
    ax.axvline(x=cohmat.shape[1], color="k", linewidth=1)

    cbar = ax.collections[0].colorbar
    for spine in cbar.ax.spines.values():
        spine.set(visible=True, linewidth=1, edgecolor="black")

    plt.title(
        os.path.basename(cohmat_path).replace("_CohMat.parquet", " Coherence Matrix")
    )
    plt.tight_layout()
    plt.savefig(os.path.basename(cohmat_path).replace("_CohMat.parquet", ".png"))
    plt.savefig(os.path.basename(cohmat_path).replace("_CohMat.parquet", ".svg"))
    plt.show()


def plot_cohmat_sorted_by_role(cohmat_path):
    """
    Plots a Coherence Matrix heatmap sorted by Functional Role:
    Input -> Middle -> Output -> Isolated.
    Within each role, nodes are sorted alphabetically.
    """
    # 1. Read and Flatten
    cohmat = pd.read_parquet(cohmat_path)

    # Handle MultiIndex (Flattening to simple node names)
    if isinstance(cohmat.index, pd.MultiIndex):
        cohmat.index = cohmat.index.get_level_values(1)
        cohmat.index.name = "SourceNode"
    if isinstance(cohmat.columns, pd.MultiIndex):
        cohmat.columns = cohmat.columns.get_level_values(1)
        cohmat.columns.name = "TargetNode"

    # 2. Determine Roles based on Matrix Connectivity
    # Create a boolean mask of valid connections (True if connection exists)
    # We treat NaN and 0.0 as "No Connection"
    has_conn = cohmat.notna() & (cohmat != 0.0)

    # IMPORTANT: We must ignore self-loops (the diagonal) for role classification.
    # Otherwise, every node looks like a "Middle" node because it affects itself.
    np.fill_diagonal(has_conn.values, False)

    # Calculate status
    # Axis 1 (Rows) = Sources. If True anywhere in row, it's an active Source.
    is_source = has_conn.any(axis=1)
    # Axis 0 (Cols) = Targets. If True anywhere in col, it's an active Target.
    is_target = has_conn.any(axis=0)

    # 3. Classify Nodes
    node_data = []

    # We iterate through the index to categorize every node
    for node in cohmat.index:
        src_status = is_source.get(node, False)
        tgt_status = is_target.get(node, False)

        # Logic matches your adjacency example:
        if not src_status and not tgt_status:
            role = 3  # Isolated
            lbl = "(Iso)"
        elif src_status and not tgt_status:
            role = 0  # Input (Source only)
            lbl = "(I)"
        elif not src_status and tgt_status:
            role = 2  # Output (Target only)
            lbl = "(O)"
        else:
            role = 1  # Middle (Both)
            lbl = "(M)"

        node_data.append({"node": node, "role": role, "label": f"{lbl} {node}"})

    # 4. Sort: Role first, then Alphabetical Name
    sorted_node_data = sorted(node_data, key=lambda x: (x["role"], x["node"]))

    sorted_nodes = [d["node"] for d in sorted_node_data]

    # 5. Reorder the Matrix
    # We use reindex to shuffle rows and columns to the new order
    cohmat_sorted = cohmat.reindex(index=sorted_nodes, columns=sorted_nodes)

    # 6. Plotting
    # Setup Colormap (Gray for NaNs)
    # cmap = plt.get_cmap("RdBu_r")
    # cmap.set_bad(color="lightgray")

    plt.figure(figsize=(20, 18))

    ax = sns.heatmap(
        cohmat_sorted,
        cmap=cmap,
        cbar=True,
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="#2e3440",  # Dark grey grid lines
        square=True,
        cbar_kws={"shrink": 0.75},
        xticklabels=sorted_nodes,
        yticklabels=sorted_nodes,
    )

    # Force Axis Orientation (X=90, Y=0) for readability
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor"
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")

    # Add borders around the whole plot
    ax.axhline(y=0, color="k", linewidth=2)
    ax.axhline(y=cohmat_sorted.shape[0], color="k", linewidth=2)
    ax.axvline(x=0, color="k", linewidth=2)
    ax.axvline(x=cohmat_sorted.shape[1], color="k", linewidth=2)

    # Add border to colorbar
    cbar = ax.collections[0].colorbar
    for spine in cbar.ax.spines.values():
        spine.set(visible=True, linewidth=1, edgecolor="black")

    # Title
    base_name = os.path.basename(cohmat_path).replace("_CohMat.parquet", "")
    plt.title(f"{base_name}\nSorted by Role: Input -> Middle -> Output")

    plt.tight_layout()
    plt.show()


def parse_toponame(topo_name):
    pattern = re.compile(
        r"^(?P<num_nodes>\d+)N_"
        r"(?P<basenet>.+?TN)_"
        r"(?P<density>\d+)D_"
        r"(?:(?P<nettype>[A-Z]+)_)?"
        r"(?P<replicate>\d+)R"
    )

    m = pattern.match(topo_name)
    if not m:
        raise ValueError(f"Invalid topo name: {topo_name}")

    return {
        "NumNodesPerGroup": int(m.group("num_nodes")),
        "BaseNet": m.group("basenet"),
        "Density": int(m.group("density")),
        "Replicate": int(m.group("replicate")),
        "NetType": m.group("nettype"),
    }


#################################################################
# Async Saving Functions
#################################################################

# Semaphore to limit the number of concurrent writes
WRITE_SEMAPHORE = asyncio.Semaphore(10)


async def save_dataframe_async(df, filepath):
    """Asynchronously writes a Pandas DataFrame to a Parquet file with explicit flushing."""

    async with WRITE_SEMAPHORE:
        try:
            await asyncio.to_thread(df.to_parquet, filepath, engine="pyarrow")

            # Ensure the file is flushed to disk
            with open(filepath, "rb") as f:
                os.fsync(f.fileno())

            # print(f"Saved: {filepath}")
        except Exception as e:
            print(f"Error saving {filepath}: {e}")


async def calc_walkmats_all_topos(topo_file_dir, save_dir, artinet_info=True):
    # print(os.path.join(topo_file_dir, "*.topo"))
    # Get the list of topos
    # topo_list = sorted(glob.glob(os.path.join(topo_file_dir, "*.topo")))
    topo_list = sorted(glob.glob(topo_file_dir))
    print(f"Number of topo files: {len(topo_list)}")

    # Mkaing the result directory if not already present
    os.makedirs(save_dir, exist_ok=True)

    # Create the tasks list
    tasks = []

    # Schema for parquet file
    cohres_schema = pa.schema(
        [
            ("NumNodesPerGroup", pa.int32()),
            ("BaseNet", pa.string()),
            ("Density", pa.int32()),
            ("Replicate", pa.int32()),
            ("NetType", pa.string()),
            ("NumGroups", pa.int32()),
            ("NumNodes", pa.int32()),
            ("AbsMeanCohVal", pa.float64()),
            ("AbsMedianCohVal", pa.float64()),
            ("CohMatMean", pa.float64()),
            ("CohMatMedian", pa.float64()),
            ("AbsMeanWalkVal", pa.float64()),
            ("AbsMedianWalkVal", pa.float64()),
            ("MeanComm", pa.float64()),
            ("MedianComm", pa.float64()),
            ("MeanCommNZ", pa.float64()),
            ("MedianCommNZ", pa.float64()),
            ("AbsMeanScaleVal", pa.float64()),
            ("AbsMedianScaleVal", pa.float64()),
            ("MeanCoh", pa.float64()),
            ("MedianCoh", pa.float64()),
        ]
    )

    # Setting up the cohres file buffer
    cohres_buffer = []

    # OPening an output file
    cohres_name = os.path.join(save_dir, "ArtiNetCohResults.parquet")
    cohres_writer = None

    # for topo_file_path in topo_list:
    for topo_file_path in tqdm(topo_list, desc="Processing "):
        # Get the topo_name
        topo_name = os.path.basename(topo_file_path).replace(".topo", "")
        # print(topo_name)
        # Getting the net info from arti net name
        if artinet_info:
            net_info = parse_toponame(topo_name)
            # print(net_info)
        # Making the results folder to save all topo related results
        save_path_topo = os.path.join(result_dir, topo_name)
        # print(save_path_topo)
        os.makedirs(save_path_topo, exist_ok=True)

        # Getting walk matrices
        walk_mats_df, coh_mat, scale_mat, walk_mat, groups_df, cohinfo_dict = (
            calc_walkmats(topo_file_path, save_path_topo, upto_length=None)
        )
        # print(coh_mat.shape)
        # print(coh_mat)
        # print(groups_df.shape)
        # print(groups_df["Node"].nunique())
        # print(f"NumGroups: {groups_df['Group'].nunique()}")
        # print(groups_df["Group"].value_counts())
        # print(coh_mat.isna().sum().sum())

        # Adding the the number of groups info to the network info
        if artinet_info:
            net_info["NumGroups"] = groups_df["Group"].nunique()
            net_info["NumNodes"] = coh_mat.shape[0]
            net_info.update(cohinfo_dict)
            # Appending to the cohres buffer
            cohres_buffer.append(net_info)
            # Writing to the cohres file if the buffer is full
            if len(cohres_buffer) == 10000:
                # Conversting to pyarrow table
                cohres_table = pa.Table.from_pylist(cohres_buffer, schema=cohres_schema)
                # Intialising the writer - for the first write
                if cohres_writer is None:
                    cohres_writer = pq.ParquetWriter(cohres_name, cohres_schema)
                # Writing to the table
                cohres_writer.write_table(cohres_table)
                # Clearing the buffer
                cohres_buffer = []

        # Saving the walk mats ascyncronously
        # Copy to avoid race and duplicate column errors while saving sometimes.
        tasks.append(
            asyncio.create_task(
                save_dataframe_async(
                    walk_mats_df,
                    os.path.join(save_dir, topo_name, f"{topo_name}_WalkMats.parquet"),
                )
            )
        )
        # Saving the Coherence matrix
        tasks.append(
            asyncio.create_task(
                save_dataframe_async(
                    coh_mat,
                    os.path.join(save_dir, topo_name, f"{topo_name}_CohMat.parquet"),
                )
            )
        )
        # Saving the walk ratio matrix
        tasks.append(
            asyncio.create_task(
                save_dataframe_async(
                    walk_mat,
                    os.path.join(
                        save_dir, topo_name, f"{topo_name}_WalkFracMat.parquet"
                    ),
                )
            )
        )
        # Saving the walk ratio matrix
        tasks.append(
            asyncio.create_task(
                save_dataframe_async(
                    scale_mat,
                    os.path.join(save_dir, topo_name, f"{topo_name}_ScaleMat.parquet"),
                )
            )
        )
        # Saving the Groups df
        tasks.append(
            asyncio.create_task(
                save_dataframe_async(
                    groups_df,
                    os.path.join(save_dir, topo_name, f"{topo_name}_Groups.parquet"),
                )
            )
        )
        await asyncio.sleep(0)
        # Process tasks in batches to free memory
        if len(tasks) >= 30:
            await asyncio.gather(*tasks)
            tasks.clear()  # Free memory by clearing completed tasks

    if artinet_info:
        # Writing the remaning rows to the cohres file
        if cohres_buffer:
            # Conversting to pyarrow table
            cohres_table = pa.Table.from_pylist(cohres_buffer, schema=cohres_schema)
            # Intialising the writer - for the first write
            if cohres_writer is None:
                cohres_writer = pq.ParquetWriter(cohres_name, cohres_schema)
            # Writing to the table
            cohres_writer.write_table(cohres_table)
        # Closing the writer
        if cohres_writer:
            cohres_writer.close()
    # Completing the remaining tasks
    if tasks:
        await asyncio.gather(*tasks)


####################################################################################


if __name__ == "__main__":
    print("Calcualte Coherence Matrix")
    num_cores = 10
    print(f"Using {num_cores} number of cores.")

    # Speci
    # Topo file directory
    # result_dir = "../../AbasyTOPOS/AbasyCohResults/"
    # topo_file_dir = "../../AbasyTOPOS/*.topo"
    # result_dir = "../../CohResultsV2/"
    # topo_file_dir = "../../ArtiToposV2/*.topo"
    result_dir = "../../ArtiTopos_1N/CohResults/"
    topo_file_dir = "../../ArtiTopos_1N//*.topo"
    # result_dir = "./CohResults/"
    # topo_file_dir = "/mnt/4TB/Pradyumna/CohModTroph/ArtiNets/*.topo"

    # Getting the coherence matrices in async
    # asyncio.run(calc_walkmats_all_topos(topo_file_dir, result_dir, artinet_info=True))

    # # # Reading the artinet_info file
    # anres_df = pd.read_parquet("../../CohResultsV2/ArtiNetCohResults.parquet")
    # print(anres_df)
    # print(anres_df.dtypes)
    # # print(
    # #     anres_df[
    # #         [
    # #             "MinCohVal",
    # #             "MaxCohVal",
    # #             "MinWalkVal",
    # #             "MaxWalkVal",
    # #             "MinScaleVal",
    # #             "MaxScaleVal",
    # #         ]
    # #     ]
    # # )
    # ##
    # Pot all the cohmats
    # for cm in glob.glob(result_dir + "/010N*2_3_0*090D*HI*/*CohMat.parquet"):
    # for cm in glob.glob("../../../testNetsTeams/" + "/010N*2_2_0*/*CohMat.parquet"):
    for cm in glob.glob("../../ArtiTopos_1N/CohResults/" + "/001N*/*CohMat.parquet"):
        print(cm)
        print(pd.read_parquet(cm).shape)
        print(pd.read_parquet(cm.replace("_CohMat", "_Groups")).shape)
        cmat = pd.read_parquet(cm)
        print(cmat)
        print(find_teams(cmat))
        # plot_cohmat_sorted_by_role(cm)
        plot_cohmat(cm)
        # break
