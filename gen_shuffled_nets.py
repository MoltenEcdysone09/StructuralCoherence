import pandas as pd
from pathlib import Path


def process_networks(input_dir, output_dir, num_shuffles):
    # Ensure the base output directory exists
    output_base_dir.mkdir(exist_ok=True, parents=True)

    # Get all .topo files in the input directory
    topo_files = list(input_dir.glob("*.topo"))

    for network_path in topo_files:
        print(f"Processing: {network_path.name}")

        # Create folder named after the topo file
        network_folder = output_base_dir / network_path.name.replace(".topo", "")
        network_folder.mkdir(exist_ok=True, parents=True)

        # Load and clean the network
        net_df = pd.read_csv(network_path, sep=r"\s+")

        # Apply the specified type transformation: 2 -> -1
        net_df["Type"] = net_df["Type"].replace(2, -1)
        net_df = net_df.rename(columns={"Type": "sign"})

        # Clean whitespace from Source and Target
        net_df["Source"] = net_df["Source"].astype(str).str.strip()
        net_df["Target"] = net_df["Target"].astype(str).str.strip()

        # print(net_df[net_df["Source"] == net_df["Target"]]["sign"].value_counts())
        # print(len(net_df))

        # Prepare Shuffled_Networks directory
        shuffle_dir = network_folder / "Shuffled_Networks"
        shuffle_dir.mkdir(exist_ok=True, parents=True)

        # net_df.rename(columns={"sign": "Type"}).to_csv(
        #     shuffle_dir / network_path.name, index=False
        # )

        # Perform shuffling
        for seed in range(1, num_shuffles + 1):
            # Create a copy and shuffle the 'Target' column
            shuffled_df = net_df.copy()
            shuffled_df["Target"] = (
                shuffled_df["Target"].sample(frac=1, random_state=seed).values
            )

            # Prepare filename and save
            output_filename = (
                f"{network_path.name.replace('.topo', '')}_Random{seed:03d}.topo"
            )

            # Rename column back to 'Type' for output
            shuffled_df.rename(columns={"sign": "Type"}).to_csv(
                shuffle_dir / output_filename, sep=" ", index=False
            )

    print("Processing complete.")


if __name__ == "__main__":
    # --- Configuration ---
    # Folder containing your .topo files
    input_dir = Path("./AbasyTOPOS_Targeted")
    # Base output folder
    output_base_dir = Path("./WTvsShuffledAnalysis_AbasyNets_Targeted")
    # Number of shuffles to perform per network
    num_shuffles = 50

    process_networks(input_dir, output_base_dir, num_shuffles)
