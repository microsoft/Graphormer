# DiG for Property-Guided Structure Generation

Welcome to the DiG for Property-Guided Structure Generation repository! This README provides essential information for using the dataset, running the provided scripts, and conducting evaluations.

## Dataset

### Training
The training of DiG for property-guided structure generation uses a set of carbon structures calculated by random structure search (RSS). The dataset is preprocessed into binary format that can be directly utilized by the code and stored in LMDB. The LMDB files for training is available at `https://ai4scienceasiaedp.blob.core.windows.net/dig/dig/property-guided/rss_carbon.zip`. The data should be extracted under tha path `dig/property-guided/dataset/rss_carbon`.
```bash
cd dig/property-guided/dataset/
# download the rss_carbon.zip file
wget "https://ai4scienceasiaedp.blob.core.windows.net/dig/dig/property-guided/rss_carbon.zip$SAS" -O rss_carbon.zip
unzip rss_carbon.zip
```


## Checkpoint

We provide the trained DiG for property-guided structure generation model checkpoint with the model weights.
Checkpoint should be placed in the `dig/property-guided/checkpoints` directory.
```bash
cd dig/property-guided/checkpoints/
# download the checkpoint
wget "https://ai4scienceasiaedp.blob.core.windows.net/dig/dig/property-guided/checkpoint.pt$SAS" .
```


## Usage

### Environment Setup

To set up on your own environment, follow these steps:

```bash
conda create -n dig-property-guided python=3.9
conda activate dig-property-guided
bash install.sh
```


### Training

To perform training of the property-guided structure generation model, follow these steps:

1. **Download the Dataset**: First, download the training data by following the previous instructions in the `Dataset` section.

2. **Run the Script**:

   ```bash
   bash scripts/train.sh <num_gpus> <batch_size_per_gpu> <save_dir>
   ```
   In the command, `<num_gpus>` is the number of GPUs to use, `<batch_size_per_gpu>` is the batch size per GPU for evaluation, and `<save_dir>` is the path to save the results.



### Property-guided Structure Sampling

To sample property-guided structures, follow these steps:

1. **Download the Dataset**: First, download the training data by following the previous instructions in the `Dataset` section.

2. **Download the checkpoint for sampling**: Download the checkpoint by following the instructions in the `Checkpoints` section.

3. **Run the Script for Sampling**:

   ```bash
   bash scripts/sample.sh <num_gpus> <batch_size_per_gpu> <save_dir> <num_atoms> <target_bandgap>
   ```
   In the command, `<num_gpus>` is the number of GPUs to use, `<batch_size_per_gpu>` is the batch size per GPU for evaluation, `<save_dir>` is the path to save the results, `<num_atoms>` is the number of carbon atoms per unit cell, and `<target_bandgap>` is the value of target bandgap for the generated structures (negative for unconditional generation).
