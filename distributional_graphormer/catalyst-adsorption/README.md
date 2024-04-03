# DiG Catalyst-Adsorption System

Welcome to the DiG Catalyst-Adsorption System repository! This README provides essential information for using the dataset, running the provided scripts, and conducting evaluations.

## Dataset

### Training
To training of DiG catalyst-adsorption model uses the MD partition of OC20 dataset and the initial structures from IS2RE partition of OC20. The dataset is preprocessed into binary format that can be directly utilized by the code and stored in LMDB. The LMDB files for training is available at `https://ai4scienceasiaedp.blob.core.windows.net/dig/dig/catalyst-adsorption/train.zip`. The training data should be extracted under tha path `dig/catalyst-adsorption/dataset/`.
```bash
cd dig/catalyst-adsorption/dataset/
# download the train.zip file
wget "https://ai4scienceasiaedp.blob.core.windows.net/dig/dig/catalyst-adsorption/train.zip$SAS" -O train.zip
unzip train.zip
mv train/* .
```


### Evaluation
Evaluation of catalyst-adsorption model uses 4 systems. For the catalyst-adsorption structure sampling, we use the system with ID 1681620 in OC20. For the density map calculation, we use systems 39254, 42728, and 209642. The preprocessed LMDB files is available at `https://ai4scienceasiaedp.blob.core.windows.net/dig/dig/catalyst-adsorption/sample.zip`. The compressed file contains initial structures used for evaluation of these 4 systems. The dataset for evaluation should be extracted under the path `dig/catalyst-adsorption/dataset/`.
```bash
cd dig/catalyst-adsorption/dataset/
# download the sample.zip file
wget "https://ai4scienceasiaedp.blob.core.windows.net/dig/dig/catalyst-adsorption/sample.zip$SAS" -O sample.zip
unzip sample.zip
mv sample/* .
```


### Initial Structures

For both training and evaluation, we need the initial structures of systems. The preprocessed initial structures are available at `https://ai4scienceasiaedp.blob.core.windows.net/dig/dig/catalyst-adsorption/is2re-with-bonds.zip`. The initial structures should be extracted under the path `dig/catalyst-adsorption/dataset/is2re-with-bonds`
```bash
cd dig/catalyst-adsorption/dataset/
# download the is2re-with-bonds.zip file
wget "https://ai4scienceasiaedp.blob.core.windows.net/dig/dig/catalyst-adsorption/is2re-with-bonds.zip$SAS" -O is2re-with-bonds.zip
unzip is2re-with-bonds.zip
```



## Checkpoint

We provide the trained DiG Catalyst-adsorption model checkpoint with the model weights.
Checkpoint should be placed in the `dig/catalyst-adsorption/checkpoints` directory.
```bash
cd dig/catalyst-adsorption/checkpoints/
# download the checkpoint
wget "https://ai4scienceasiaedp.blob.core.windows.net/dig/dig/catalyst-adsorption/checkpoint.pt$SAS" .
```


## Usage

### Environment Setup

To set up on your own environment, follow these steps:

```bash
conda create -n dig-catalyst python=3.9
conda activate dig-catalyst
bash install.sh
```


### Training

To perform training of the catalyst-adsorption model, follow these steps:

1. **Download the Dataset**: First, download the training data by following the previous instructions in the `Dataset` section.

2. **Run the Script**:

   ```bash
   bash scripts/train.sh <num_gpus> <batch_size_per_gpu> <save_dir>
   ```
   In the command, `<num_gpus>` is the number of GPUs to use, `<batch_size_per_gpu>` is the batch size per GPU for evaluation, and `<save_dir>` is the path to save the results.



### Sample Catalyst-Adsorption Structures

To sample catalyst-adsorption structures, follow these steps:

1. **Download the Initial Structures**: Download the initial structures by following the instructions in the `Dataset` section.

2. **Run the Script for Sampling**:

   ```bash
   bash scripts/sample.sh <num_gpus> <batch_size_per_gpu> <save_dir>
   ```
   In the command, `<num_gpus>` is the number of GPUs to use, `<batch_size_per_gpu>` is the batch size per GPU for evaluation, and `<save_dir>` is the path to save the results.



### Density Caculation

The density calculation follows these steps:

1. **Download the Initial Structures**: Download the initial structures by following the instructions in the `Dataset` section.

2. **Run the Script for Density Calculation**:

   ```bash
   for i in {0..10}; do bash scripts/density.sh <num_gpus> <batch_size_per_gpu> <save_dir> $i; done
   ```
   In the command, `<num_gpus>` is the number of GPUs to use, `<batch_size_per_gpu>` is the batch size per GPU for evaluation, and `<save_dir>` is the path to save the results. We loop over 0 to 10 to calculate density maps at different heights (ranging from 0 to 1 Angstrom) from the surface of the catalyst.
