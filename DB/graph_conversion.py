import os
import pickle
import glob
from rdkit import Chem
from tqdm import tqdm

from libs.preprocess_utils import Graph, select_residue
from libs.db_utils import Connect2pgSQL as pgDB

def convert_dataset(db:pgDB, query, experiment_type='training'):

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
            pPATH_rsd = f'D:/Data/PDBbind/HD011/model_input/{experiment_type}/residues/{pdb_code}_residues.pdb'

            mol_ligand = Chem.SDMolSupplier(lPATH_sdf)[0]
            if not mol_ligand:
                mol_ligand = Chem.MolFromMol2File(lPATH_mol2)
            # mol_ligand_addHs = Chem.AddHs(mol_ligand)
            # mol_ligand_rmHs = Chem.RemoveHs(mol_ligand)
            # mol_ligand_neutralize = Chem.neutralize_atoms(mol_ligand)


            try:
                if os.path.isfile(pPATH_rsd):
                    mol_residues = Chem.MolFromPDBFile(pPATH_rsd)
                else:
                    # D:/Data/PDBbind/preprocessed/HD011-training/residues
                    mol_residues = select_residue(mol_ligand, pPATH_pck,
                                                  save_path=pPATH_rsd)
            except:
                pass

            pdb_code_list.append(pdb_code)
            mol_ligand_list.append(mol_ligand)
            mol_residues_list.append(mol_residues)

    with tqdm(total=len(result)) as pbar2:
        pbar2.set_description('convert to graph')

        exstGrph = os.listdir(f'D:/Data/PDBbind/HD011/model_input/{experiment_type}/graph/')
        exstGrphs = [graph.split('.')[0] for graph in exstGrph if graph.endswith('.pkl')]

        for id, lig, rsd in zip(pdb_code_list, mol_ligand_list, mol_residues_list):
            pbar2.update(1)
            if id in exstGrphs:
                continue

            if (lig is None) | (rsd is None):
                print(f'\n<!> {id} is NoneType')
                # updateDB(self, schema, table, column, value, c_column, condition):
                # sql = " UPDATE {schema}.{table} SET {column}=\'{value}\' WHERE {c_column}=\'{condition}\'"\
                inav_type = f"SELECT inavailability_type FROM pdbbind.available WHERE pdb_code='{id}'"
                inav_type = db.execute(inav_type)[0][0]
                db.commit()
                if inav_type == 'None':
                    inav_type = 'empty_rsd_file'
                else:
                    inav_type = f"{inav_type}, empty_rsd_file"

                try:
                    query_up = f"UPDATE pdbbind.available SET available=False, " \
                               f"inavailability_type = '{inav_type}' WHERE pdb_code='{id}'"
                    result = db.execute(query_up)

                    db.commit()
                except:

                    print(query_up)
                    continue

                continue

            graph_l = Graph(molecule=lig).generate()
            graph_p = Graph(molecule=rsd).generate()

            save_path = f'D:/Data/PDBbind/HD011/model_input/{experiment_type}/graph/'

            with open(f'{save_path}/{id}.pkl', 'wb') as f:
                pickle.dump((graph_l, graph_p), f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    db = pgDB()
    query = 'SELECT binding.pdb_code, binding.subset FROM pdbbind.binding ' \
            'INNER JOIN pdbbind.available ON binding.pdb_code = available.pdb_code ' \
            'WHERE available.available = TRUE AND binding.pdb_code NOT IN (SELECT pdb_code FROM pdbbind.coreset) AND available.pdb_code=\'2r1w\''
    # 2r1w는 뭐야

    query_coreset = 'SELECT coreset.pdb_code, available.subset FROM pdbbind.coreset ' \
                    'INNER JOIN pdbbind.available ON coreset.pdb_code = available.pdb_code ' \
                    'WHERE available.available=TRUE AND available.subset=\'coreset\''

    query_tmp = 'SELECT binding.pdb_code, binding.subset FROM pdbbind.binding ' \
                'INNER JOIN pdbbind.available ON binding.pdb_code = available.pdb_code ' \
                'WHERE available.available = TRUE '

    convert_dataset(db, query_coreset, experiment_type='external_validation_removeHs')