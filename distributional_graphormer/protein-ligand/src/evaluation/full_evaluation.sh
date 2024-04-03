#!/bin/bash

number=50

cp dataset/16sys_db/all50_md.list dataset/16sys_db/test_md.list
rm position_pt/*

CUDA_VISIBLE_DEVICES=0 ./examples/diffusion/train_cli.sh

python samplepdb/cordinate2pdb.py
rm position_pt/*

echo "Conformations generated and saved in the EDP/output directory."

python evaluation/calc_rmsd.py
echo "RMSD calculated and saved in the EDP/output directory."
