# Project Workflow and Script Pipeline

## 1. Theoretical Topologies and Artificial Networks (ER & HI)

**Relevant Directory Structures:**

```text
AllUniqueNets/
├── Topologies/
├── get_allunquienets.r
└── signnet_138_codes.csv

AE_Plots/
├── F2/
├── F3/
├── F4/
├── F5/
└── F9/
```

**Execution Order & Descriptions:**

1. **`get_allunquienets.r`**: Located in `AllUniqueNets/`, this script uses the `signnet` package and a dummy MAN-<PN> string to extract unique MAN codes for triads (unique, non-isomorphic motif codes). Outputs results to `AllUniqueNets/signnet_138_codes.csv`.
2. **`make_uniqueNets.py`**: Processes the generated CSV to create `.topo` files in the `AllUniqueNets/Topologies/` directory. It incorporates hardcoded dyad topologies. The output format is `source target Type` (including 0-links for induced subgraphs, along with 1 and -1).
3. **`scale_man_nets.py`**: Utilizes the topology files to artificially generate Erdős-Rényi (ER) and Hierarchical (HI) networks across different input scales, densities, and multiple replicates per motif combination.
4. **`cohcalc.py`**: The core calculation script. It computes coherence matrices (featuring two fast calculation versions and a path-tracking version for positive/negative steps) and executes team calculation algorithms on the networks within the `AllUniqueNets/` directory.
5. **`abserror_analysis.py`**: Analyzes the ER and HI artificial network data. Generates main and supplementary figure panels saved in `AE_Plots/` (specifically `F2`, `F3`, `F4`, `F5`, and `F9`).

## 2. Biological Gene Regulatory Networks (Abasy Atlas)

**Relevant Directory Structures:**

```text
WTvsShuffledAnalysis_AbasyNets_Targeted/
├── 83332_v2018_s15-16_regNetwork/
├── 100226_v2019_sA22-DBSCR15_eStrong_regNetwork/
│   ├── MotifsCounts/
│   ├── ProcessedMotifCounts/
│   ├── Shuffled_CohMats/
│   └── Shuffled_Networks/
├── 196627_v2020_s21_regNetwork_Strong/
├── 208964_v2020_sRPA20_regNetwork_Strong/
├── 224308_v2022_sSW22_regNetwork/
└── 511145_v2022_sRDB22_eStrong_regNetwork_Strong/

GRN_Plots/
├── Fig6/
├── Fig7/
└── Fig8/
```

*(Note: Each targeted network folder follows a similar subdirectory structure to `100226_...`)*

**Execution Order & Descriptions:**

1. **`json_net_extract.py`**: Processes raw whole-genome GRN data downloaded from the Abasy Atlas (located in `AbasyNets/`) and extracts the network topologies, saving them as `.topo` files.
2. **`gen_shuffled_nets.py`**: Generates shuffled networks from the wild-type (WT) GRNs. Outputs are saved in corresponding subdirectories under `WTvsShuffledAnalysis_AbasyNets_Targeted/` (e.g., inside `Shuffled_Networks/`).
3. **`lean_find_motifs.py`**: Computes motif counts, generating data stored in `MotifsCounts/` and `ProcessedMotifCounts/` within the target network directories.
4. **`cohcalc.py`**: Executed on both WT and shuffled networks to calculate coherence matrices and identify teams. Results are saved in the respective WT or shuffled subdirectories (e.g., `Shuffled_CohMats/`).
5. **`go_annot.py`**: Downloads Gene Ontology metadata corresponding to the targeted networks for use in downstream module analysis.
6. **Analysis Scripts (`wtshuffled_teams_analysis.py`, `motifenrichment_analysis.py`, `module_size_analysis.py`)**: Process the generated GRN data, motif counts, and GO annotations. Outputs and figure panels are saved in the `GRN_Plots/` directory (specifically `Fig6`, `Fig7`, and `Fig8`).

## 3. iModulon and RNA Polymerase Analysis

**Relevant Directory Structures:**

```text
iModulon_Plots/
└── Fig10/

lf2c_rnap/  # Contains RNA polymerase perturbation data
precise1k-v1/ # Contains iModulon matrices
```

**Execution Order & Descriptions:**

1. **`prec1k_analysis.py`**: Performs iModulon analysis using Reconstructed GRN (RGRN) data and iModulon matrices loaded from the `precise1k-v1/` directory. Generates figure panels saved in `iModulon_Plots/` (specifically `Fig10`).
2. **`rnap_analyse.py`**: Processes RNA polymerase perturbation data located in the `lf2c_rnap/` directory and generates corresponding figure panels.
