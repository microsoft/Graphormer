#  DiG Protein System

## Dataset

We have prepared a dataset of descriptors for proteins in the DiG paper using `https://github.com/google-deepmind/alphafold`. You can download the dataset and use them for the next step.

```bash
# cd /path/to/dig/protein
mkdir dataset
cd dataset
wget https://ai4scienceasiaedp.blob.core.windows.net/dig/dig/protein/dataset/protein-dataset.tar.gz$SAS -O protein-dataset.tar.gz
# including pickle format single and pair representations, FASTA format sequence files
tar zxvf protein-dataset.tar.gz
```

## Trained Parameters

Download the trained parameters for the DiG protein system.

```bash
# cd /path/to/dig/protein
mkdir checkpoints
cd checkpoints
wget https://ai4scienceasiaedp.blob.core.windows.net/dig/dig/protein/checkpoints/checkpoint-520k.pth$SAS -O checkpoint-520k.pth
```

## Environment

Install the environment for the DiG protein system.

```bash
conda create -n dig-pro python=3.10
conda activate dig-pro
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

# or only use pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Inference (Sampling)

```bash
# cd /path/to/dig/protein
PDBID="6lu7"
CKPT_PATH=./checkpoints/checkpoint-520k.pth
FEATURE_PATH=./dataset/${PDBID}.pkl
FASTA_PATH=./dataset/${PDBID}.fasta
OUTDIR=./output/
mkdir -p ${OUTDIR}
python run_inference.py -c ${CKPT_PATH} -i ${FEATURE_PATH} -s ${FASTA_PATH} -o ${PDBID} --output-prefix ${OUTDIR}  -n 1 --use-gpu --use-tqdm
```

After running the above sampling command (~30 seconds on GPU), you will get the PDB format files in the `${OUTDIR}` directory.

NOTE: It will take up to 10 minutes to build the SO(3) helper array for the first time, which would not be repeated in the next inference no matter what the input is.
