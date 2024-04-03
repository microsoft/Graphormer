import pickle
import torch
from tqdm import tqdm
import glob
import json
import pandas as pd
import shutil
import re
import os
import argparse

##### CHANGE THIS ACCRODING YOUR NEEDS #####
SYS_CSV_MD = './samplepdb/16sys_md.csv'
PT_SOURCE_MD = './position_pt'
VALIDATION_DATALIST_MD = './dataset/16sys_db/test_md.list'
############################################

def transform_pdb_file(trj_pdb, atom_pos,trj_out_pdb):
    trj_file = open(trj_pdb, "r")
    trj_output_file = open(trj_out_pdb, "w")
    # read from pdb file
    count = 0
    new_trj_file = []
    for line in trj_file:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_cordinate = atom_pos[count]
            org_x = line[30:38].strip()
            org_y = line[38:46].strip()
            org_z = line[46:54].strip()
            x = "{:{}.3f}".format(atom_cordinate[0],len(org_x))
            y = "{:{}.3f}".format(atom_cordinate[1],len(org_y))
            z = "{:{}.3f}".format(atom_cordinate[2],len(org_z))
            newline = line[0:38-len(x)] + x + \
                        line[38:46-len(y)] + y + \
                        line[46:54-len(z)] + z + line[54:]
            assert len(newline) == len(line)
            count += 1
            new_trj_file.append(newline)
        else:
            new_trj_file.append(line)
    for line in new_trj_file:
        trj_output_file.write(line)
    trj_file.close()
    trj_output_file.close()


def main(example_folder, output_folder, batch_size):
    # loop through all the pdb files
    pdb_df = pd.read_csv(SYS_CSV_MD)

    # read validation list
    with open(VALIDATION_DATALIST_MD, 'r') as f:
        validation_list = f.readlines()
    validation_list = [x.strip() for x in validation_list]

    # iterate over the dataframe with i as index
    for i in tqdm(range(len(validation_list))):
        current_pdb = validation_list[i]
        # select the row with the current pdb
        cur_pdb_df = pdb_df[pdb_df['key'] == current_pdb]
        sdf_file = example_folder +'/'+ cur_pdb_df['ligand_fname'].values[0]
        pdb_file = sdf_file.rsplit('.',1)[0] + '.pdb'
        pro_file = example_folder +'/'+ cur_pdb_df['receptor_fname'].values[0]

        lout_file = output_folder +'/'+ cur_pdb_df['ligand_fname'].values[0].rsplit('.',1)[0] + f'/ligand_{i%50}_mdpred.pdb'
        pout_file = output_folder +'/'+ cur_pdb_df['receptor_fname'].values[0].rsplit('.',1)[0] + f'/protein_{i%50}_mdpred.pdb'

        # check if folder for pout_file exists
        if not os.path.exists(os.path.dirname(lout_file)):
            os.makedirs(os.path.dirname(lout_file))
        if not os.path.exists(os.path.dirname(pout_file)):
            os.makedirs(os.path.dirname(pout_file))
        if i % batch_size == 0:
            a = torch.load(PT_SOURCE_MD+'/pred_pos_'+str(i//batch_size)+'.pt',map_location=torch.device('cpu'))
            ln = torch.load(PT_SOURCE_MD+'/lnode_'+str(i//batch_size)+'.pt',map_location=torch.device('cpu'))
            pn = torch.load(PT_SOURCE_MD+'/pnode_'+str(i//batch_size)+'.pt',map_location=torch.device('cpu'))

        pos = a[i % batch_size]

        transform_pdb_file(pdb_file, pos[:ln[i % batch_size]], lout_file)
        transform_pdb_file(pro_file, pos[ln[i % batch_size]:ln[i % batch_size] + pn[i % batch_size]], pout_file)

if __name__ == "__main__":
    # add a output folder argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--example_folder', type=str, default='./dataset/Pack_16_systems')
    parser.add_argument('--output_folder', type=str, default='./output')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    main(args.example_folder, args.output_folder, args.batch_size)