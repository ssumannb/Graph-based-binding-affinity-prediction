import os
import warnings
import glob
import pickle
import torch
import dgl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from Bio.PDB import *
from scipy.spatial import distance_matrix

from preprocess import main as pp_main
from HD011_DB.libs.db_utils import connect2pgSQL as pgDB

# 원소기호 40개가 정의된 변수
ATOM_VOCAB = [
    'C', 'N', 'O', 'S', 'F',  # 탄소, 질소, 산소, 황, 플루오린
    'H', 'Si', 'P', 'Cl', 'Br',  # 수소, 규소, 인, 염소, 브로민
    'Li', 'Na', 'K', 'Mg', 'Ca',  # 리튬, 나트륨, 칼륨, 마그네슘, 칼슘
    'Fe', 'As', 'Al', 'I', 'B',  # 철, 비소, 알루미늄, 아이오딘, 붕소
    'V', 'Tl', 'Sb', 'Sn', 'Ag',  # 비나듐, 탈륨, 안티모니, 주석, 은
    'Pd', 'Co', 'Se', 'Ti', 'Zn',  # 팔라듐, 코발트, 셀레늄, 타이타늄, 아연
    'Ge', 'Cu', 'Au', 'Ni', 'Cd',  # 저마늄, 구리, 금, 니켈, 카드뮴
    'Mn', 'Cr', 'Pt', 'Hg', 'Pb'  # 망간, 크로뮴, 백금(플래티나), 수은, 납
]
warnings.filterwarnings(action='ignore')


def one_of_k_encoding(x, vocab):
    if x not in vocab:
        x = vocab[-1]
    return list(map(lambda s: float(x == s), vocab))


def get_atom_feature(atom):
    atom_feature = one_of_k_encoding(atom.GetSymbol(), ATOM_VOCAB)
    atom_feature += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    atom_feature += one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    atom_feature += one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
    atom_feature += [atom.GetIsAromatic()]
    # atom_feature length = 58 (50+6+5+6)
    return atom_feature


def get_bond_feature(bond):
    bt = bond.GetBondType()
    bond_feature = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return bond_feature


def get_molecular_graph(mol):
    '''
    convert the molecular into graph
    :param mol: chemical object with 'mol' format
        (1) coordinate {x,y,z}
        (2) bond information
        (3) name of molecular
    :return: graph
    ** mol = Chem.MolFromSmiles(smi)  : graph from 'compound' with smiles format
    '''

    graph = dgl.DGLGraph()

    # graph node generation
    atom_list = mol.GetAtoms()
    num_atoms = len(atom_list)
    graph.add_nodes(num_atoms)

    atom_feature_list = [get_atom_feature(atom) for atom in atom_list]  # 3D-list로 분자를 구성하는 원자의 특징 표현
    atom_feature_list = torch.tensor(atom_feature_list, dtype=torch.float64)  # atom_feature_list -> tensor 초기화

    '''
    assigning the features of node and edge respectively
    shape of feature tensor : (number of node/edge, number of features)
    dtype of feature        : dictionary
                              i.e. graph.ndata['feature_name'], g.edata['feature_name']
    '''
    graph.ndata['h'] = atom_feature_list  # 'h': feature name

    bond_list = mol.GetBonds()
    bond_feature_list = []
    for bond in bond_list:
        bond_feature = get_bond_feature(bond)

        ## 단백질 원자와 리간드 원자 간의 edge의 경우 원자간 거리를 측정하여 그걸 특징으로 넣는다.?
        src = bond.GetBeginAtom().GetIdx()
        dst = bond.GetEndAtom().GetIdx()

        # bond direction : i-->j
        graph.add_edges(src, dst)
        bond_feature_list.append(bond_feature)

        # bond direction : j-->i
        graph.add_edges(dst, src)
        bond_feature_list.append(bond_feature)

    bond_feature_list = torch.tensor(bond_feature_list, dtype=torch.float64)
    graph.edata['e_ij'] = bond_feature_list

    # graph.to('cuda:0')

    return graph


def visualize_graph(data):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    pos_p = np.array(data[0].GetConformer().GetPositions())
    pos_l = np.array(data[1].GetConformer().GetPositions())
    pos_r = np.array(data[4].GetConformer().GetPositions())

    for dot in pos_p:
        ax.scatter(dot[0], dot[1], dot[2], c=dot[1], marker='o', s=15, cmap=plt.cm.winter)  # blue
    for dot_l in pos_l:
        ax.scatter(dot_l[0], dot_l[1], dot_l[2], c=dot_l[1], marker='o', s=15, cmap=plt.cm.autumn)  # red
    for dot_r in pos_r:
        ax.scatter(dot_r[0], dot_r[1], dot_r[2], c=dot_r[1], marker='o', s=15, cmap=plt.cm.summer)  # green

    plt.savefig(f'D:/Project/HD101/HD101_DB/Report/IMG_atom_coordinate/{data[3]}.png', dpi=300)
    # plt.show()


def get_atom_coord(data):
    return data.GetConformer().GetPositions().tolist()


def collate_func(batch):
    graph_list_ligand = []
    graph_list_residue = []
    label_list = []

    for data in batch:
        if os.path.isfile(f'D:/Project/HD101/HD101_DB/Source/graph/{data[2]}.pkl'):
            with open(f'D:/Project/HD101/HD101_DB/Source/graph/{data[2]}.pkl', 'rb') as f:
                graphs = pickle.load(f)
                graph_l, graph_p = graphs[0], graphs[1]
        else:
            graph_l = get_molecular_graph(data[0])
            graph_p = get_molecular_graph(data[3])
            with open(f'D:/Project/HD101/HD101_DB/Source/graph/{data[3]}.pkl', 'wb') as f:
                pickle.dump((graph_l, graph_p), f, pickle.HIGHEST_PROTOCOL)

        graph_list_ligand.append(graph_l)
        graph_list_residue.append(graph_p)
        label_list.append(float(data[1]))

    # Batch a collection of DGLGraph s into one graph for more efficient graph computation.
    graph_list_ligand = dgl.batch(graph_list_ligand)
    graph_list_residue = dgl.batch(graph_list_residue)
    label_list = torch.tensor(label_list, dtype=torch.float64)

    return graph_list_residue, graph_list_ligand, label_list

# def collate_func_eTest(batch):
#     graph_list_ligand = []
#     graph_list_residue = []
#     label_list = []
#
#     for data in batch:
#         if os.path.isfile(f'../BA_prediction_ignore/data/graph/{data[2]}.pkl'):
#             with open(f'../BA_prediction_ignore/data/graph/{data[2]}.pkl', 'rb') as f:
#                 graphs = pickle.load(f)
#                 graph_l, graph_p = graphs[0], graphs[1]
#         else:
#             graph_l = get_molecular_graph(data[0])
#             graph_p = get_molecular_graph(data[3])
#             with open(f'../BA_prediction_ignore/data/graph/{data[2]}.pkl', 'wb') as f:
#                 pickle.dump((graph_l, graph_p), f, pickle.HIGHEST_PROTOCOL)
#
#         graph_list_ligand.append(graph_l)
#         graph_list_residue.append(graph_p)
#         label_list.append(float(data[1]))
#
#     # Batch a collection of DGLGraph s into one graph for more efficient graph computation.
#     graph_list_ligand = dgl.batch(graph_list_ligand)
#     graph_list_residue = dgl.batch(graph_list_residue)
#     label_list = torch.tensor(label_list, dtype=torch.float64)
#
#     return graph_list_residue, graph_list_ligand, label_list


def select_residue(ligand, pdb):
    parser = PDBParser()
    # if not os.path.exists(pdb):
    #     return None
    structure = parser.get_structure('protein', pdb)
    ligand_positions = ligand.GetConformer().GetPositions()

    # Get distance between ligand positions (N_ligand, 3) and
    # residue positions (N_residue, 3) for each residue
    # only select residue with minimum distance of it is smaller than 5A
    # → 각각의 잔기에 대해 리간드 위치, 잔기 위치간의 거리를 측정하여 최소 거리가 5A보다 짧은 잔기만을 선택

    class ResidueSelect(Select):
        # 단백질을 구성하는 아미노산 잔기들 각각에 대한 리간드와의 거리 측정 후 선택/제외 결정
        def accept_residue(self, residue):
            residue_positions = np.array([np.array(list(atom.get_vector())) \
                                          for atom in residue.get_atoms() if 'H' not in atom.get_id()])
            if len(residue_positions.shape) < 2:
                print(residue)
                return 0
            min_dis = np.min(distance_matrix(residue_positions, ligand_positions))

            if min_dis < 5.0:
                return 1
            else:
                return 0

    io = PDBIO()
    io.set_structure(structure)
    pdbid = pdb.split('/')[-1].split('_')[0]
    fn = f'D:/Project/HD101/HD101_DB/Source/residues/{pdbid}_residues.pdb'
    fn_infold = pdb.replace('pocket','residues')
    # fn = f'BS_tmp_{str(np.random.randint(0, 1000000, 1)[0])}.pdb'
    io.save(fn, ResidueSelect())
    io.save(fn_infold, ResidueSelect())

    m2 = Chem.MolFromPDBFile(fn)
    # os.system('del ' + fn)

    return m2


def get_complex_list(
        data_seed=999,
        frac=[0.8, 0.2],
        cv=True,
        af_type = 'kdki'
    ):
    data_info = pd.DataFrame()
    db = pgDB.CRUD()

    try:
        query = "SELECT pdb_code, pKd_pKi_pIC50 " \
                "FROM pdbbind.binding " \
                "WHERE affinity_data " \
                "NOT LIKE 'IC50%'"

        return_val = db.run_query(query)
        data_info = pd.DataFrame(return_val, columns=['PDB_id', 'BA_log'])

    except:
        warnings.warn('Error raised by importing database for DB')

    complex_dataset_list = []
    for i, _ in enumerate(frac):
        complex_dataset_list.append(data_info.sample(
            frac=frac[i],random_state=data_seed).reset_index(drop=True))

    return complex_dataset_list


class Complex_Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            splitted_set,
            is_coreset=False
    ):
        '''
        :param splitted_set: [PDB_id, binding affinity value]
        '''
        self.PDB_id = list(splitted_set['PDB_id'])
        self.label = list(splitted_set['BA_log'])
        self.is_coreset = is_coreset

    def __len__(
            self
    ):
        return len(self.PDB_id)

    def __getitem__(
            self,
            item
    ):
        if self.is_coreset == False:
            PATH = 'D:/Data/PDBbind/raw_data/PDBbind_2020/'
        else:
            # PATH = '../BA_prediction_ignore/data/CASF-2016/coreset/'
            PATH = 'D:/Data/PDBbind/raw_data/CASF-2016/'

        # PATH_residue = '../BA_prediction_ignore/data/residues/'
        rPATH = 'D:/Project/HD101/HD101_DB/Source/residues/'

        path_pocket = f"{self.PDB_id[item]}/{self.PDB_id[item]}_pocket"
        path_ligand = f"{self.PDB_id[item]}/{self.PDB_id[item]}_ligand"
        path_residue = f"{self.PDB_id[item]}_residues"

        mol_ligand = Chem.SDMolSupplier(f"{PATH}{path_ligand}.sdf")[0]

        if not mol_ligand:
            mol_ligand = Chem.MolFromMol2File(f'{PATH}{path_ligand}.mol2')
        try:
            # mol_pocket = Chem.MolFromPDBFile(f'{PATH}{path_pocket}.pdb')
            if os.path.isfile(f'{rPATH}{path_residue}.pdb'):
                residue_structure = Chem.MolFromPDBFile(f'{rPATH}{path_residue}.pdb')
            else:
                residue_structure = select_residue(mol_ligand, f'{PATH}{path_pocket}.pdb')

        except:
            raise NotImplementedError

        return mol_ligand, self.label[item], self.PDB_id[item], residue_structure


def debugging():
    pass


if __name__ == '__main__':
    debugging()
