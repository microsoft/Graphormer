import os
import numpy as np
from tqdm import tqdm
import glob
import pandas as pd
from conform_rmsd import get_full_rmsd, get_conf_rmsd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    crystal_lig = glob.glob('./dataset/Pack_16_systems/*/*/ligands/ligand*.pdb')
    sampled_lig = glob.glob('./output/*/*/ligands/ligand*/ligand_*_mdpred.pdb')

    # create dataframe
    crystal_lig_df = pd.DataFrame(crystal_lig, columns=['crystal_lig_path'])
    crystal_lig_df['system'] = crystal_lig_df['crystal_lig_path'].apply(lambda x: x.split('/')[-3])
    crystal_lig_df['ligand'] = crystal_lig_df['crystal_lig_path'].apply(lambda x: x.split('/')[-1].split('.')[0])
    crystal_lig_df['system_ligand'] = crystal_lig_df['system'] + '_' + crystal_lig_df['ligand']

    # create dataframe
    sampled_lig_df = pd.DataFrame(sampled_lig, columns=['sampled_lig_path'])
    sampled_lig_df['system'] = sampled_lig_df['sampled_lig_path'].apply(lambda x: x.split('/')[-4])
    sampled_lig_df['ligand'] = sampled_lig_df['sampled_lig_path'].apply(lambda x: x.split('/')[-2])
    sampled_lig_df['system_ligand'] = sampled_lig_df['system'] + '_' + sampled_lig_df['ligand']

    # merge two dataframes
    df = pd.merge(crystal_lig_df, sampled_lig_df, on='system_ligand', how='inner')

    # iterate over rows
    all_rmsd_list = []
    conf_rmsd_list = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            rms = get_full_rmsd(row['crystal_lig_path'], row['sampled_lig_path'])
            all_rmsd_list.append(rms)
            rms = get_conf_rmsd(row['crystal_lig_path'], row['sampled_lig_path'])
            conf_rmsd_list.append(rms)
        except KeyboardInterrupt:
            raise
        except:
            all_rmsd_list.append(np.nan)
            conf_rmsd_list.append(np.nan)
            print('error', row['crystal_lig_path'], row['sampled_lig_path'])
            pass

    df['rmsd'] = all_rmsd_list
    df['conf_rmsd'] = conf_rmsd_list
    df = df.drop(columns=['system_x', 'ligand_x', 'system_y', 'ligand_y'])
    df.to_csv('./output/rmsd.csv', index=False)


def statistics():
    df = pd.read_csv('./output/rmsd.csv')
    # drop rows with missing values
    df = df.dropna()
    sns.set_style("whitegrid")
    # sns.distplot(df['rmsd'], bins=50, kde=False)
    # normalize distribution, make sure the sum of the area under the curve is 1
    sns.distplot(df['rmsd'], bins=50, kde=False, norm_hist=True)
    sns.distplot(df['conf_rmsd'], bins=50, kde=False, norm_hist=True)


    # select the best rmsd for each system_ligand
    df = df.sort_values(by=['rmsd'])
    df_best = df.drop_duplicates(subset=['system_ligand'], keep='first')
    sns.distplot(df_best['rmsd'], bins=50, kde=False, norm_hist=True)

    # select medium rmsd for each system_ligand, sort and select the middle one
    group = df.groupby('system_ligand')
    df_medium = group.apply(lambda x: x.sort_values(by=['rmsd']).iloc[len(x) // 4])
    print(df_medium)
    sns.distplot(df_medium['rmsd'], bins=50, kde=False, norm_hist=True)

    plt.xlabel('RMSD')
    plt.ylabel('Density')
    plt.title('RMSD Distribution')
    #save figure
    plt.savefig('./output/rmsd.png', dpi=300)


if __name__ == '__main__':
    main()
    statistics()