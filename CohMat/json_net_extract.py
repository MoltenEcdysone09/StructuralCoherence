import os
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# --- Configuration ---
# IMPORTANT: This is the path to your main folder
ROOT_DIRECTORY = "../../../AbasyNets/"
# IMPORTANT: This is the new base folder where all outputs will be saved.
# The script will re-create the sub-folder structure from ROOT_DIRECTORY inside this folder.
OUTPUT_DIRECTORY = "../../../AbasyNets/RAW/"
# New folder for .topo files
TOPO_OUTPUT_DIRECTORY = "../../AbasyTOPOS/"
# ---------------------


def process_all_json_to_csv(root_dir, output_dir_base):
    """
    Walks through a directory, finds all .json files, extracts
    edge and node data from them, and saves the data to corresponding .csv
    files in a new output directory, mirroring the source structure.

    Skips any JSON file where one or more edges are missing the 'Effect' key.

    Also creates a .topo file for each valid network.

    Args:
        root_dir (str): The path to the root directory to start searching.
        output_dir_base (str): The base path for all output files.
    """
    file_count = 0
    processed_count = 0
    error_count = 0
    skipped_count = 0

    print(f"Starting search in directory: {root_dir}")
    print(f"Saving all output to base directory: {output_dir_base}")
    print(f"Saving all topo files to directory: {TOPO_OUTPUT_DIRECTORY}")

    if not os.path.exists(root_dir):
        print(f"Error: Directory not found at path: {root_dir}")
        print(
            "Please create this directory and add your JSON files, or update the ROOT_DIRECTORY variable."
        )
        return

    # Create the base output directory if it doesn't exist
    os.makedirs(output_dir_base, exist_ok=True)
    os.makedirs(TOPO_OUTPUT_DIRECTORY, exist_ok=True)  # Create the topo folder

    # Recursively walk through the directory tree
    for dirpath, dirnames, files in os.walk(root_dir):
        # Skip the output directory itself to avoid processing files it just created
        if (
            os.path.commonpath([dirpath, output_dir_base]) == output_dir_base
            or os.path.commonpath([dirpath, TOPO_OUTPUT_DIRECTORY])
            == TOPO_OUTPUT_DIRECTORY
        ):
            continue

        for filename in files:
            if filename.endswith(".json"):
                file_count += 1
                file_path = os.path.join(dirpath, filename)

                # --- New Output Path Logic ---
                # Get the relative path from the root_dir (e.g., "folder1/subfolderA")
                relative_dir = os.path.relpath(dirpath, root_dir)

                # Create the target directory in the output path
                # e.g., "output_data/folder1/subfolderA"
                if relative_dir == ".":
                    target_dir = output_dir_base
                else:
                    target_dir = os.path.join(output_dir_base, relative_dir)

                os.makedirs(target_dir, exist_ok=True)

                # Get the JSON filename without extension (e.g., "my_graph")
                json_filename_no_ext = os.path.splitext(filename)[0]

                # Define the final output paths
                output_edge_csv_path = os.path.join(
                    target_dir, json_filename_no_ext + ".csv"
                )
                output_node_csv_path = os.path.join(
                    target_dir, json_filename_no_ext + "_NodeData.csv"
                )
                # --- End New Output Path Logic ---

                print(f"\nProcessing file: {file_path}")

                processed_this_file = False

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Navigate the JSON structure to find 'elements'
                    elements = data.get("elements", {})
                    if not elements:
                        print("  -> Warning: No 'elements' key found. Skipping file.")
                        continue  # Skip to the next file

                    # --- 1. Process Edges ---
                    edges = elements.get("edges", [])
                    current_file_edges_data = []  # Reset for each file

                    if edges:
                        # --- New 'Effect' Check ---
                        file_is_valid = True
                        for edge in edges:
                            # Check if 'data' exists AND 'Effect' exists within 'data'
                            if "data" not in edge or "Effect" not in edge["data"]:
                                file_is_valid = False
                                break  # Found an edge missing 'data' or 'Effect'

                        if not file_is_valid:
                            print(
                                f"  -> Skipping: File is missing 'Effect' key in one or more edges."
                            )
                            skipped_count += 1
                            continue  # Skip to the next JSON file
                        # --- End 'Effect' Check ---

                        # If we are here, all edges are valid. Build the list.
                        for edge in edges:
                            if "data" in edge:
                                current_file_edges_data.append(edge["data"])

                        if current_file_edges_data:
                            # Convert the list of dictionaries into a pandas DataFrame
                            df_edges = pd.DataFrame(current_file_edges_data)

                            # Save the DataFrame to its corresponding CSV file
                            df_edges.to_csv(output_edge_csv_path, index=False)
                            processed_this_file = True

                            # --- New NetworkX and Plotting Logic ---
                            # Load the dataframe we just saved
                            try:
                                df_for_network = pd.read_csv(output_edge_csv_path)

                                # Check the weak vs strong percentage
                                print(output_edge_csv_path)
                                print(
                                    df_for_network["Evidence"].value_counts(
                                        normalize=True
                                    )
                                )

                                # --- New .topo File Logic (Added) ---
                                try:
                                    # 1. Define path (goes to the flat TOPO_OUTPUT_DIRECTORY)
                                    output_topo_path = os.path.join(
                                        TOPO_OUTPUT_DIRECTORY,
                                        json_filename_no_ext + ".topo",
                                    )

                                    # 2. Select columns from the loaded dataframe
                                    df_topo = df_for_network[
                                        ["source", "target", "Effect"]
                                    ].copy()

                                    # 3. Map values (+ -> 1, - -> 2, others -> NaN)
                                    # This handles '?', '-/+', etc., by mapping them to NaN
                                    df_topo["Effect"] = df_topo["Effect"].map(
                                        {"+": 1, "-": 2}
                                    )

                                    # 4. Drop rows that weren't '+' or '-'
                                    df_topo = df_topo.dropna(subset=["Effect"])

                                    # 5. Convert type to integer
                                    if not df_topo.empty:
                                        df_topo["Effect"] = df_topo["Effect"].astype(
                                            int
                                        )

                                    # 6. Rename columns
                                    df_topo.rename(
                                        columns={
                                            "source": "Source",
                                            "target": "Target",
                                            "Effect": "Type",
                                        },
                                        inplace=True,
                                    )

                                    # 7. Save as space-separated file without index
                                    if not df_topo.empty:
                                        # Use sep=' ' for space-separated columns
                                        df_topo.to_csv(
                                            output_topo_path, sep=" ", index=False
                                        )
                                        print(
                                            f"  -> Success: Saved topo data to {output_topo_path}"
                                        )
                                    else:
                                        # This case will be hit if the file had 'Effect'
                                        # but none were '+' or '-'
                                        print(
                                            "  -> Info: No '+' or '-' edges found. Skipping .topo file."
                                        )

                                    # 1. Define path (goes to the flat TOPO_OUTPUT_DIRECTORY)
                                    output_topo_path = os.path.join(
                                        TOPO_OUTPUT_DIRECTORY,
                                        json_filename_no_ext + "_Strong.topo",
                                    )

                                    # 2. Select columns from the loaded dataframe
                                    df_topo_strong = df_for_network[
                                        df_for_network["Evidence"] == "Strong"
                                    ][["source", "target", "Effect"]].copy()

                                    if (not df_topo_strong.empty) and (
                                        len(df_topo) / len(df_for_network) >= 0.20
                                    ):
                                        # 3. Map values (+ -> 1, - -> 2, others -> NaN)
                                        # This handles '?', '-/+', etc., by mapping them to NaN
                                        df_topo_strong["Effect"] = df_topo_strong[
                                            "Effect"
                                        ].map({"+": 1, "-": 2})

                                        # 4. Drop rows that weren't '+' or '-'
                                        df_topo_strong = df_topo_strong.dropna(
                                            subset=["Effect"]
                                        )

                                        # 5. Convert type to integer
                                        df_topo_strong["Effect"] = df_topo_strong[
                                            "Effect"
                                        ].astype(int)

                                        # 6. Rename columns
                                        df_topo_strong.rename(
                                            columns={
                                                "source": "Source",
                                                "target": "Target",
                                                "Effect": "Type",
                                            },
                                            inplace=True,
                                        )

                                        # Use sep=' ' for space-separated columns
                                        df_topo_strong.to_csv(
                                            output_topo_path, sep=" ", index=False
                                        )
                                        print(
                                            f"  -> Success: Saved topo data to {output_topo_path}"
                                        )
                                    else:
                                        # This case will be hit if the file had 'Effect'
                                        # but none were '+' or '-'
                                        print(
                                            "  -> Info: No '+' or '-' edges found. Skipping .topo file."
                                        )

                                except Exception as e:
                                    print(f"  -> Error during .topo file creation: {e}")
                                # --- End .topo File Logic ---

                                # --- Existing Plotting Logic ---
                                # Filter for only '+' or '-' effects
                                df_filtered = df_for_network[
                                    df_for_network["Effect"].isin(["+", "-"])
                                ].copy()

                                if df_filtered.empty:
                                    print(
                                        "  -> Info: No '+' or '-' edges found. Skipping plot."
                                    )
                                else:
                                    # Create a directed graph
                                    G = nx.from_pandas_edgelist(
                                        df_filtered,
                                        "source",
                                        "target",
                                        edge_attr=["Effect"],
                                        create_using=nx.DiGraph(),
                                    )

                                    # Calculate degree distributions
                                    in_degree_vals = list(dict(G.in_degree()).values())
                                    out_degree_vals = list(
                                        dict(G.out_degree()).values()
                                    )

                                    # Note: We no longer need in_dist and out_dist for histograms

                                    # Create the plot
                                    fig, (ax1, ax2) = plt.subplots(
                                        1, 2, figsize=(14, 6)
                                    )
                                    fig.suptitle(
                                        f"Degree Distributions for {json_filename_no_ext}",
                                        fontsize=16,
                                    )

                                    # --- In-Degree Histogram ---
                                    max_in = (
                                        max(in_degree_vals) if in_degree_vals else 0
                                    )

                                    if (
                                        max_in < 50
                                    ):  # For low degrees, use integer bins (like a bar plot)
                                        # Bins from 0 up to max_in + 2
                                        in_bins = range(max_in + 3)
                                        ax1.hist(
                                            in_degree_vals,
                                            bins=in_bins,
                                            color="skyblue",
                                            edgecolor="black",
                                            align="left",
                                        )
                                        ax1.set_xticks(
                                            range(max_in + 2)
                                        )  # Set ticks at integer values
                                    else:  # For high degrees, let matplotlib decide the bins
                                        ax1.hist(
                                            in_degree_vals,
                                            bins="auto",
                                            color="skyblue",
                                            edgecolor="black",
                                        )

                                    ax1.set_title("In-Degree Distribution")
                                    ax1.set_xlabel("In-Degree")
                                    ax1.set_ylabel("Frequency (Node Count)")

                                    # --- Out-Degree Histogram ---
                                    # Use 20 bins as requested by the user
                                    out_bins = 20

                                    ax2.hist(
                                        out_degree_vals,
                                        bins=out_bins,
                                        color="salmon",
                                        edgecolor="black",
                                    )
                                    ax2.set_title("Out-Degree Distribution")
                                    ax2.set_xlabel("Out-Degree")
                                    ax2.set_ylabel("Frequency (Node Count)")
                                    # We don't set x-ticks here to let the histogram define its own bin edges

                                    fig.tight_layout(
                                        rect=[0, 0.03, 1, 0.95]
                                    )  # Adjust for suptitle

                                    # Define plot save path
                                    plot_path = os.path.join(
                                        target_dir,
                                        json_filename_no_ext
                                        + "_degree_distribution.png",
                                    )

                                    # Save and close the figure
                                    fig.savefig(plot_path)
                                    plt.close(fig)  # Close to free up memory

                            except Exception as e:
                                print(
                                    f"  -> Error during plotting/network analysis: {e}"
                                )

                    # --- 2. Process Nodes (New) ---
                    nodes = elements.get("nodes", [])
                    current_file_nodes_data = []  # Reset for each file

                    if nodes:
                        # Extract the 'data' dictionary from each node
                        for node in nodes:
                            if "data" in node:
                                current_file_nodes_data.append(node["data"])

                        if current_file_nodes_data:
                            # Convert the list of dictionaries into a pandas DataFrame
                            df_nodes = pd.DataFrame(current_file_nodes_data)

                            # Save the DataFrame to its corresponding CSV file
                            df_nodes.to_csv(output_node_csv_path, index=False)
                            processed_this_file = True

                    if processed_this_file:
                        processed_count += 1
                    else:
                        print(f"  -> Info: No node or edge data found to save.")

                except json.JSONDecodeError:
                    print("  -> Error: Could not decode JSON. File might be corrupt.")
                    error_count += 1
                except KeyError as e:
                    print(f"  -> Error: Missing expected key {e}.")
                    error_count += 1
                except Exception as e:
                    print(f"  -> An unexpected error occurred: {e}")
                    error_count += 1

    print("-" * 30)
    print("Search complete.")
    print(f"Found {file_count} JSON file(s).")
    print(
        f"Successfully processed {processed_count} file(s) (creating one or more CSVs for each)."
    )
    if skipped_count > 0:
        print(f"Skipped {skipped_count} file(s) due to missing 'Effect' key.")
    if error_count > 0:
        print(f"Encountered errors on {error_count} file(s).")


if __name__ == "__main__":
    # This function now handles all processing and saving.
    process_all_json_to_csv(ROOT_DIRECTORY, OUTPUT_DIRECTORY)
