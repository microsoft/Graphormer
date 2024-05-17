# DiG Protein-ligand System

Welcome to the DiG Protein-ligand System repository! This README provides essential information for using the dataset, running the provided scripts, and conducting evaluations.

## Dataset

The DiG protein-ligand System dataset contains 16 different protein-ligand systems, with a total size of approximately 79 MB.

Dataset should be extracted to the `dig/protein-ligand/src/dataset` directory.

   ```bash
   cd dig/protein-ligand/src
   # download the dataset.tar file
   wget "https://ai4scienceasiaedp.blob.core.windows.net/dig/dig/protein-ligand/dataset.tar$SAS" -O dataset.tar
   tar -xvf dataset.tar
   ```


## Checkpoint

The DiG Protein-ligand System checkpoint contains the model weights for the protein-ligand system.
Checkpoint should be placed in the `dig/protein-ligand/src/saved_checkpoints` directory.

   ```bash
   cd dig/protein-ligand/src
   # download the saved_checkpoints.tar file
   wget "https://ai4scienceasiaedp.blob.core.windows.net/dig/dig/protein-ligand/saved_checkpoints.tar$SAS" -O saved_checkpoints.tar
   tar -xvf saved_checkpoints.tar
   ```


## Usage

### Environment Setup

To set up on your own environment, follow these steps:

```bash
conda env create -f env.yml
./install.sh
```


### Single Datapoint Sampling

To perform single datapoint sampling using this dataset, follow these steps:

1. **Download the Dataset**: First, download the dataset using the link provided above.

2. **Run the Script**:

   ```bash
   bash evaluation/single_datapoint_sampling.sh --pdbid <selected pdbid> --number 50
   ```

   This command will generate 50 conformations for the selected pdbid. The conformations will be saved in the `src/output` directory. Available pdbids are listed in the `src/dataset/all_md.list` file.

### Evaluation

To evaluate the results obtained from your single datapoint sampling or any other experiments, follow these steps:

1. **Run the Evaluation Script**:

   ```bash
   bash evaluation/full_evaluation.sh
   ```

   This will take about 10 hours to generate conformations on a single A100/A40 GPU. This script will analyze the output from your experiments and provide relevant metrics or results.

### Docker

We provide a Docker image with installed dependencies for running the scripts.


To use the Docker image, follow these steps:

   ```bash
   wget "https://ai4scienceasiaedp.blob.core.windows.net/dig/dig/protein-ligand/dig-prolig-docker.tar$SAS" -O dig-prolig-docker.tar
   docker load < dig-prolig-docker.tar
   cd dig/protein-ligand/src
   docker run -it --gpus all --ipc=host -v $(pwd):/workspace dig/dig-prolig
   conda activate dig_dock
   ```

   then you can run the scripts as described above.
