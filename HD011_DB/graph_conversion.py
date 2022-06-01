import os
import pickle
from rdkit import Chem
from tqdm import tqdm

from libs.preprocess_utils import Graph, select_residue
from libs.db_utils import connect2pgSQL as pgDB

def convert_trainingset(db:pgDB, query):
    result = db.execute(query)
    root_PATH = 'D:/Data/PDBbind/raw_data'

    pdb_code_list = []
    mol_ligand_list = []
    mol_residues_list = []

    with tqdm(total=len(result)) as pbar:
        pbar.set_description('load molecule data')
        for _, data in enumerate(result):
            pbar.update(1)
            pdb_code = data[0]
            subset = data[1]

            if subset == 'general':
                PATH = f'{root_PATH}/PDBbind_v2020/{pdb_code}'
            elif subset == 'refined':
                PATH = f'{root_PATH}/refined-set_v2020/{pdb_code}'
            elif subset == 'coreset':
                PATH = f'{root_PATH}/CASF-2016/coreset/{pdb_code}'

            lPATH_sdf = f'{PATH}/{pdb_code}_ligand.sdf'
            lPATH_mol2 = f'{PATH}/{pdb_code}_ligand.mol2'
            pPATH_pck = f'{PATH}/{pdb_code}_pocket.pdb'
            pPATH_rsd = f'D:/Data/PDBbind/preprocessed/HD011-training/residues/{pdb_code}_residues.pdb'

            mol_ligand = Chem.SDMolSupplier(lPATH_sdf)[0]
            if not mol_ligand:
                mol_ligand = Chem.MolFromMol2File(lPATH_mol2)
            try:
                if os.path.isfile(pPATH_rsd):
                    mol_residues = Chem.MolFromPDBFile(pPATH_rsd)
                else:
                    mol_residues = select_residue(mol_ligand, pPATH_pck)
            except:
                pass

            pdb_code_list.append(pdb_code)
            mol_ligand_list.append(mol_ligand)
            mol_residues_list.append(mol_residues)

    with tqdm(total=len(result)) as pbar2:
        pbar2.set_description('convert to graph')
        for id, lig, rsd in zip(pdb_code_list, mol_ligand_list, mol_residues_list):
            pbar2.update(1)
            graph_l = Graph(molecule=lig).generate()
            graph_p = Graph(molecule=rsd).generate()

            save_path = 'D:/Data/PDBbind/preprocessed/HD011-training/graph'

            with open(f'{save_path}/{id}.pkl', 'wb') as f:
                pickle.dump((graph_l, graph_p), f, pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    db = pgDB()
    query = 'SELECT binding.pdb_code, binding.subset FROM pdbbind.binding ' \
            'INNER JOIN pdbbind.available ON binding.pdb_code = available.pdb_code ' \
            'WHERE available.available = TRUE AND binding.pdb_code NOT IN (SELECT pdb_code FROM pdbbind.coreset)'

    convert_trainingset(db, query)