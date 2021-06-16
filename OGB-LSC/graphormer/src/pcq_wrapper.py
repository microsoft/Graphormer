# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset

from rdkit import Chem
import rdkit.Chem.AllChem as AllChem

import joblib
import numpy as np
import math
from scipy.spatial.distance import cdist

# ===================== NODE START =====================
atomic_num_list = list(range(119))
chiral_tag_list = list(range(4))
degree_list = list(range(11))
possible_formal_charge_list = list(range(16))
possible_numH_list = list(range(9))
possible_number_radical_e_list = list(range(5))
possible_hybridization_list = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'S']
possible_is_aromatic_list = [False, True]
possible_is_in_ring_list = [False, True]
explicit_valence_list = list(range(13))
implicit_valence_list = list(range(13))
total_valence_list = list(range(26))
total_degree_list = list(range(32))

def simple_atom_feature(atom):
    atomic_num = atom.GetAtomicNum()
    assert atomic_num in atomic_num_list

    chiral_tag = int(atom.GetChiralTag())
    assert chiral_tag in chiral_tag_list

    degree = atom.GetTotalDegree()
    assert degree in degree_list

    possible_formal_charge = atom.GetFormalCharge()
    possible_formal_charge_transformed = possible_formal_charge + 5
    assert possible_formal_charge_transformed in possible_formal_charge_list
    
    possible_numH = atom.GetTotalNumHs()
    assert possible_numH in possible_numH_list
    # 5
    possible_number_radical_e = atom.GetNumRadicalElectrons()
    assert possible_number_radical_e in possible_number_radical_e_list

    possible_hybridization = str(atom.GetHybridization())
    assert possible_hybridization in possible_hybridization_list
    possible_hybridization = possible_hybridization_list.index(possible_hybridization)

    possible_is_aromatic = atom.GetIsAromatic()
    assert possible_is_aromatic in possible_is_aromatic_list
    possible_is_aromatic = possible_is_aromatic_list.index(possible_is_aromatic)

    possible_is_in_ring = atom.IsInRing()
    assert possible_is_in_ring in possible_is_in_ring_list
    possible_is_in_ring = possible_is_in_ring_list.index(possible_is_in_ring)

    explicit_valence = atom.GetExplicitValence()
    assert explicit_valence in explicit_valence_list
    # 10
    implicit_valence = atom.GetImplicitValence()
    assert implicit_valence in implicit_valence_list

    total_valence = atom.GetTotalValence()
    assert total_valence in total_valence_list
    
    total_degree = atom.GetTotalDegree()
    assert total_degree in total_degree_list

    sparse_features = [
        atomic_num, chiral_tag, degree, possible_formal_charge_transformed, possible_numH,
        possible_number_radical_e, possible_hybridization, possible_is_aromatic, possible_is_in_ring, explicit_valence,
        implicit_valence, total_valence, total_degree,
    ]
    return sparse_features

def easy_bin(x, bin):
    x = float(x)
    cnt = 0
    if math.isinf(x):
        return 120
    if math.isnan(x):
        return 121

    while True:
        if cnt == len(bin):
            return cnt
        if x > bin[cnt]:
            cnt += 1
        else:
            return cnt


def peri_features(atom, peri):
    rvdw = peri.GetRvdw(atom.GetAtomicNum())
    default_valence = peri.GetDefaultValence(atom.GetAtomicNum())
    n_outer_elecs = peri.GetNOuterElecs(atom.GetAtomicNum())
    rb0 = peri.GetRb0(atom.GetAtomicNum())
    sparse_features = [
        default_valence,
        n_outer_elecs,
        easy_bin(rvdw, [1.2 , 1.5 , 1.55, 1.6 , 1.7 , 1.8 , 2.4]),
        easy_bin(rb0, [0.33 , 0.611, 0.66 , 0.7  , 0.77 , 0.997, 1.04 , 1.54])
    ]
    return sparse_features

def envatom_feature(mol, radius, atom_idx):  
    env= Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx, useHs=True)
    submol=Chem.PathToSubmol(mol, env, atomMap={})
    return submol.GetNumAtoms()

def envatom_features(mol, atom):
    return [
        envatom_feature(mol, r, atom.GetIdx()) for r in range(2, 9)
    ]

def atom_to_feature_vector(atom, peri, mol):
    sparse_features = []
    sparse_features.extend(simple_atom_feature(atom))
    sparse_features.extend(peri_features(atom, peri))
    sparse_features.extend(envatom_features(mol, atom))
    sparse_features.append(easy_bin(atom.GetProp('_GasteigerCharge'),
      [-0.87431233, -0.47758285, -0.38806704, -0.32606976, -0.28913129,
       -0.25853269, -0.24494531, -0.20136365, -0.12197541, -0.08234462,
       -0.06248558, -0.06079668, -0.05704827, -0.05296379, -0.04884997,
       -0.04390136, -0.03881107, -0.03328515, -0.02582824, -0.01916618,
       -0.01005982,  0.0013529 ,  0.01490858,  0.0276433 ,  0.04070013,
        0.05610381,  0.07337645,  0.08998278,  0.11564625,  0.14390777,
        0.18754518,  0.27317209,  1.        ]))
    return sparse_features


import os.path as osp
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

def donor_acceptor_feature(x_num, mol):
    chem_feature_factory_feats = chem_feature_factory.GetFeaturesForMol(mol)
    features = np.zeros([x_num, 2], dtype = np.int64)
    for i in range(len(chem_feature_factory_feats)):
        if chem_feature_factory_feats[i].GetFamily() == 'Donor':
            node_list = chem_feature_factory_feats[i].GetAtomIds()
            for j in node_list:
                features[j, 0] = 1
        elif chem_feature_factory_feats[i].GetFamily() == 'Acceptor':
            node_list = chem_feature_factory_feats[i].GetAtomIds()
            for j in node_list:
                features[j, 1] = 1
    return features

chiral_centers_list = ['R', 'S']
def chiral_centers_feature(x_num, mol):
    features = np.zeros([x_num, 1], dtype = np.int64)
    t = Chem.FindMolChiralCenters(mol)
    for i in t:
        idx, type = i
        features[idx] = chiral_centers_list.index(type) + 1 # 0 for not center
    return features
# ===================== NODE END =====================

# ===================== BOND START =====================
possible_bond_type_list = list(range(32))
possible_bond_stereo_list = list(range(16))
possible_is_conjugated_list = [False, True]
possible_is_in_ring_list = [False, True]
possible_bond_dir_list = list(range(16))

def bond_to_feature_vector(bond):
    # 0
    bond_type = int(bond.GetBondType())
    assert bond_type in possible_bond_type_list

    bond_stereo = int(bond.GetStereo())
    assert bond_stereo in possible_bond_stereo_list

    is_conjugated = bond.GetIsConjugated()
    assert is_conjugated in possible_is_conjugated_list
    is_conjugated = possible_is_conjugated_list.index(is_conjugated)

    is_in_ring = bond.IsInRing()
    assert is_in_ring in possible_is_in_ring_list
    is_in_ring = possible_is_in_ring_list.index(is_in_ring)

    bond_dir = int(bond.GetBondDir())
    assert bond_dir in possible_bond_dir_list

    bond_feature = [
        bond_type,
        bond_stereo,
        is_conjugated,
        is_in_ring,
        bond_dir,
    ]
    return bond_feature
# ===================== BOND END =====================

# ===================== ATTN START =====================
def get_rel_pos(mol):
    try:
        new_mol = Chem.AddHs(mol)
        res = AllChem.EmbedMultipleConfs(new_mol, numConfs=10)
        ### MMFF generates multiple conformations
        res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        new_mol = Chem.RemoveHs(new_mol)
        index = np.argmin([x[1] for x in res])
        energy = res[index][1]
        conf = new_mol.GetConformer(id=int(index))
    except:
        new_mol = mol
        AllChem.Compute2DCoords(new_mol)
        energy = 0
        conf = new_mol.GetConformer()

    atom_poses = []
    for i, atom in enumerate(new_mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return [[0.0, 0.0, 0.0]] * len(new_mol.GetAtoms())
        pos = conf.GetAtomPosition(i)
        atom_poses.append([pos.x, pos.y, pos.z])
    atom_poses = np.array(atom_poses, dtype=float)
    rel_pos_3d = cdist(atom_poses, atom_poses)
    return rel_pos_3d
# ===================== ATTN START =====================


def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)
    AllChem.ComputeGasteigerCharges(mol)
    peri=Chem.rdchem.GetPeriodicTable()

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom, peri, mol))
    x = np.array(atom_features_list, dtype = np.int64)
    x = np.concatenate([x, donor_acceptor_feature(x.shape[0], mol)], axis=1)
    x = np.concatenate([x, chiral_centers_feature(x.shape[0], mol)], axis=1)

    # bonds
    num_bond_features = 5
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    # attn
    rel_pos_3d = get_rel_pos(mol)
    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)
    graph['rel_pos_3d'] = rel_pos_3d
    return graph 


import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
from multiprocessing import Pool

class MyPygPCQM4MDataset(PygPCQM4MDataset):
    def __init__(self, root = 'dataset/mypcq_v4', smiles2graph=smiles2graph):
        super().__init__(root=root, smiles2graph=smiles2graph)

    def download(self):
        super(MyPygPCQM4MDataset, self).download()

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']

        print('Converting SMILES strings into graphs...')
        np_data_list = []
        all_rel_pos_3d = []
        with Pool(processes=90) as pool:
            iter = pool.imap(self.smiles2graph, smiles_list)
            for idx, graph in tqdm(enumerate(iter), total=len(homolumogap_list)):
                data = Data()
                homolumogap = homolumogap_list[idx]
                
                assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
                assert(len(graph['node_feat']) == graph['num_nodes'])

                data.__num_nodes__ = int(graph['num_nodes'])
                data.edge_index = graph['edge_index']
                data.edge_attr = graph['edge_feat']
                data.x = graph['node_feat']
                data.y = float(homolumogap)

                np_data_list.append(data)
                all_rel_pos_3d.append(graph['rel_pos_3d'])
        
        print('Converting numpy to torch tensor...')
        data_list = []
        for i in tqdm(range(len(np_data_list))):
            data = np_data_list[i]
            data.edge_index = torch.from_numpy(data.edge_index).to(torch.int64)
            data.edge_attr = torch.from_numpy(data.edge_attr).to(torch.int64)
            data.x = torch.from_numpy(data.x).to(torch.int64)
            data.y = torch.Tensor([data.y])
            data_list.append(data)

        print('Collating...')
        # double-check prediction target
        split_dict = self.get_idx_split()
        assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        assert(all([torch.isnan(data_list[i].y)[0] for i in split_dict['test']]))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

        joblib.dump(all_rel_pos_3d, 'dataset/all_rel_pos_3d.pkl')

if __name__ == '__main__':
    graph = smiles2graph('O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5')
    print(graph)

    dataset = MyPygPCQM4MDataset()
