import dgl
import torch
import numpy as np

from rdkit import Chem
from Bio.PDB import PDBParser, Select, PDBIO
from scipy.spatial import distance_matrix


import warnings
warnings.filterwarnings(action='ignore')

class Graph():
    def __init__(self, molecule):
        # 40 atoms
        self.ATOM_VOCAB = ['C', 'N', 'O', 'S', 'F',  # 탄소, 질소, 산소, 황, 플루오린
                            'H', 'Si', 'P', 'Cl', 'Br',  # 수소, 규소, 인, 염소, 브로민
                            'Li', 'Na', 'K', 'Mg', 'Ca',  # 리튬, 나트륨, 칼륨, 마그네슘, 칼슘
                            'Fe', 'As', 'Al', 'I', 'B',  # 철, 비소, 알루미늄, 아이오딘, 붕소
                            'V', 'Tl', 'Sb', 'Sn', 'Ag',  # 비나듐, 탈륨, 안티모니, 주석, 은
                            'Pd', 'Co', 'Se', 'Ti', 'Zn',  # 팔라듐, 코발트, 셀레늄, 타이타늄, 아연
                            'Ge', 'Cu', 'Au', 'Ni', 'Cd',  # 저마늄, 구리, 금, 니켈, 카드뮴
                            'Mn', 'Cr', 'Pt', 'Hg', 'Pb']  # 망간, 크로뮴, 백금(플래티나), 수은, 납
        self.BOND_VOCAB = [Chem.rdchem.BondType.SINGLE,
                           Chem.rdchem.BondType.DOUBLE,
                           Chem.rdchem.BondType.TRIPLE,
                           Chem.rdchem.BondType.AROMATIC]

        self.molecule = molecule
        self.atoms = self.molecule.GetAtoms()
        self.bonds = self.molecule.GetBonds()

        self.graph = dgl.DGLGraph()

    def _onehot_encoding(self, x, vocab):
        if x not in vocab:
            x = vocab[-1]
        return list(map(lambda s: float(x == s), vocab))

    def _tf_encoding(self, x, vocab):
        return list(map(lambda s: bool(x == s), vocab))

    def _extract_atom_feature(self, atom):
        # atom_feature length = 58 (50+6+5+6)
        atom_feature = self._onehot_encoding(atom.GetSymbol(), self.ATOM_VOCAB)
        atom_feature += self._onehot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        atom_feature += self._onehot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        atom_feature += self._onehot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
        atom_feature += [atom.GetIsAromatic()]

        return atom_feature

    def _extract_bond_feature(self, bond):
        # bt = bond.GetBondType()
        # bond_feature_old = [
        #     bt == Chem.rdchem.BondType.SINGLE,
        #     bt == Chem.rdchem.BondType.DOUBLE,
        #     bt == Chem.rdchem.BondType.TRIPLE,
        #     bt == Chem.rdchem.BondType.AROMATIC,
        #     bond.GetIsConjugated(),
        #     bond.IsInRing()
        # ]
        bond_feature = self._tf_encoding(bond.GetBondType(), self.BOND_VOCAB)
        bond_feature.append(bond.GetIsConjugated())
        bond_feature.append(bond.IsInRing())

        return bond_feature

    def generate(self):
        # generate empty graph
        tmp_graph = dgl.DGLGraph()
        tmp_graph.add_nodes(len(self.atoms))

        # fill atom features to graph nodes
        atom_feature_list = [self._extract_atom_feature(atom) for atom in self.atoms]
        tensor_atom_feature = torch.tensor(atom_feature_list, dtype=torch.float64)

        tmp_graph.ndata['h'] = tensor_atom_feature

        # fill node features to graph edges
        bond_feature_list = []
        for bond in self.bonds:
            bond_feature = self._extract_bond_feature(bond)
            src = bond.GetBeginAtom().GetIdx()
            dst = bond.GetEndAtom().GetIdx()

            # bond direction : i-->j
            tmp_graph.add_edges(src, dst)
            bond_feature_list.append(bond_feature)

            # bond direction : j-->i
            tmp_graph.add_edges(dst, src)
            bond_feature_list.append(bond_feature)

        torch_bond_feature = torch.tensor(bond_feature_list, dtype=torch.float64)
        tmp_graph.edata['e_ij'] = torch_bond_feature

        self.graph = tmp_graph
        return self.graph


def select_residue(ligand, pdb, save_path):
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
    fn = f'{save_path}/{pdbid}_residues.pdb'
    fn_infold = pdb.replace('pocket','residues')
    # fn = f'BS_tmp_{str(np.random.randint(0, 1000000, 1)[0])}.pdb'
    io.save(fn, ResidueSelect())
    io.save(fn_infold, ResidueSelect())

    m2 = Chem.MolFromPDBFile(fn)
    # os.system('del ' + fn)

    return m2


if __name__ == '__main__':
    mol_ligand = Chem.MolFromMol2File('D:/Data/PDBbind/PDBbind_v2020_refined/1a1e/1a1e_ligand.mol2')
    mol_ligand_sdf = Chem.SDMolSupplier('D:/Data/PDBbind/PDBbind_v2020_refined/1a1e/1a1e_ligand.sdf')[0]
    graph = Graph(molecule=mol_ligand_sdf).generate()
