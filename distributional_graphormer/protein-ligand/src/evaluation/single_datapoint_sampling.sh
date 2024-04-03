#!/bin/bash

# Define the default values
selected_pdbid=""
number=50

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --pdbid)
            selected_pdbid="$2"
            shift # past argument
            shift # past value
            ;;
        --number)
            number="$2"
            shift # past argument
            shift # past value
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if the selected_pdbid is provided
if [ -z "$selected_pdbid" ]; then
    echo "Error: You must provide a PDB ID using the --pdbid option."
    exit 1
fi

python evaluation/genlist.py --selected_pdbid "$selected_pdbid" --num_samples "$number"



echo "Generating $number conformations for PDB ID: $selected_pdbid"

rm position_pt/*

CUDA_VISIBLE_DEVICES=0 ./examples/diffusion/train_cli.sh

python samplepdb/cordinate2pdb.py
rm position_pt/*

echo "Conformations generated and saved in the EDP/output directory."
