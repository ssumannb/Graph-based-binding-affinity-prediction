import dgl
import torch

from rdkit import Chem

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


if __name__ == '__main__':
    mol_ligand = Chem.MolFromMol2File('D:/Data/PDBbind/PDBbind_v2020_refined/1a1e/1a1e_ligand.mol2')
    mol_ligand_sdf = Chem.SDMolSupplier('D:/Data/PDBbind/PDBbind_v2020_refined/1a1e/1a1e_ligand.sdf')[0]
    graph = Graph(molecule=mol_ligand_sdf).generate()
