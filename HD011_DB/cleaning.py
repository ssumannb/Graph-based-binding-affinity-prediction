import os
import time
import pandas as pd

from rdkit import Chem
from Bio.PDB import *
from tqdm import tqdm

import DataBase.connection_postgresql as pgDB

gPATH = 'D:/Data/PDBbind/raw_data/PDBbind_v2020/'
rPATH = 'D:/Data/PDBbind/raw_data/PDBbind_v2020_refined/'
cPATH = 'D:/Data/PDBbind/raw_data/CASF-2016/coreset/'

'''
<postgresql available table>
-- CREATE TABLE pdbbind.available (
-- 	id integer generated by default as identity,
-- 	pdb_code char(4) unique NOT NULL,
-- 	available boolean NOT NULL,
-- 	inavailable_type character varying,
-- 	constraint fk_pdb_code foreign key(pdb_code) references
-- 	pdbbind.binding(pdb_code) on delete cascade on update cascade
-- );
'''

def extract_list(db:pgDB):
    query = 'SELECT pdb_code, subset FROM pdbbind.binding;'
    return db.run_query(query)


def no_Folder(org_data_list:pd.DataFrame):
    result = []

    with tqdm(total=len(org_data_list)) as pbar:
        pbar.set_description('Check folder ')
        for i, row in org_data_list.iterrows():
            pbar.update(1)
            path_tmp = f"{row['path']}{row['pdb_code']}/"
            result.append(os.path.isdir(path_tmp))

    return result


def no_lig_file(org_data_list:pd.DataFrame):
    result_sdf = []
    result_mol2 = []

    with tqdm(total=len(org_data_list)) as pbar:
        pbar.set_description('Check ligand file ')
        for i, row in org_data_list.iterrows():
            pbar.update(1)
            id = row['pdb_code']
            path_tmp = f"{row['path']}{id}/"
            if not os.path.isdir(path_tmp):
                result_sdf.append('False')
                result_mol2.append('False')
            else:
                f_list = os.listdir(path_tmp)
                result_sdf.append(f'{id}_ligand.sdf' in f_list)
                result_mol2.append(f'{id}_ligand.mol2' in f_list)

    return result_sdf, result_mol2


def no_ptn_file(org_data_list:pd.DataFrame):
    result_protein = []
    result_pocket = []

    with tqdm(total=len(org_data_list)) as pbar:
        pbar.set_description('Check protein file ')
        for i, row in org_data_list.iterrows():
            pbar.update(1)
            id = row['pdb_code']
            path_tmp = f"{row['path']}{id}/"
            if not os.path.isdir(path_tmp):
                result_protein.append('False')
                result_pocket.append('False')
            else:
                f_list = os.listdir(path_tmp)
                result_protein.append(f'{id}_protein.pdb' in f_list)
                result_pocket.append(f'{id}_pocket.pdb' in f_list)

    return result_protein, result_pocket


def empty_lig(org_data_list:pd.DataFrame):
    result_sdf = []
    result_mol2 = []

    with tqdm(total=len(org_data_list)) as pbar:
        pbar.set_description('Check ligand file contents ')
        for i, row in org_data_list.iterrows():
            pbar.update(1)
            id = row['pdb_code']
            path_sdf = f"{row['path']}{id}/{id}_ligand.sdf"
            path_mol2 = f"{row['path']}{id}/{id}_ligand.mol2"

            if not os.path.isfile(path_sdf):
                result_sdf.append('False')
            else:
                if Chem.SDMolSupplier(path_sdf)[0]:
                    result_sdf.append('True')
                else:
                    result_sdf.append('False')

            if not os.path.isfile(path_mol2):
                result_mol2.append('False')
            else:
                if Chem.MolFromMol2File(path_mol2):
                    result_mol2.append('True')
                else:
                    result_mol2.append('False')

    return result_sdf, result_mol2


def empty_ptn(org_data_list:pd.DataFrame):
    result_protein = []
    result_pocket = []

    with tqdm(total=len(org_data_list)) as pbar:
        pbar.set_description('Check protein file contents ')
        for i, row in org_data_list.iterrows():
            pbar.update(1)
            id = row['pdb_code']
            path_protein = f"{row['path']}{id}/{id}_protein.pdb"
            path_pocket = f"{row['path']}{id}/{id}_pocket.pdb"

            if not os.path.isfile(path_protein):
                result_protein.append('False')
            else:
                if Chem.MolFromPDBFile(path_protein):
                    result_protein.append('True')
                else:
                    result_protein.append('False')

            if not os.path.isfile(path_pocket):
                result_pocket.append('False')
            else:
                if Chem.MolFromPDBFile(path_pocket):
                    result_pocket.append('True')
                else:
                    result_pocket.append('False')

    return result_protein, result_pocket


if __name__ == '__main__':
    db = pgDB.CRUD()
    org_data = pd.DataFrame()

    extracted_list = extract_list(db)[:10]

    org_data['pdb_code'] = [x[0] for i, x in enumerate(extracted_list)]
    org_data['subset'] = [x[1] for i, x in enumerate(extracted_list)]
    path_list = []
    for i, s in enumerate(org_data['subset']):
        if s == 'general':
            path_list.append(gPATH)
        elif s == 'refined':
            path_list.append(rPATH)
        elif s == 'coreset':
            path_list.append(cPATH)

    org_data['path'] = path_list

    # filtering
    filtered_data = pd.DataFrame()
    filtered_data['pdb_code'] = org_data['pdb_code']
    filtered_data['no_folder'] = no_Folder(org_data)
    filtered_data['no_ptn_file'], filtered_data['no_pck_file'] = no_ptn_file(org_data)
    filtered_data['no_lig_file_sdf'], filtered_data['no_lig_file_mol2'] = no_lig_file(org_data)
    filtered_data['empty_ptn_file'], filtered_data['empty_pck_file'] = empty_ptn(org_data)
    filtered_data['empty_lig_file_sdf'], filtered_data['empty_lig_file_mol2'] = empty_lig(org_data)

    filtered_data['available'] = True

    filtered_data.loc[filtered_data['no_folder'] == False, 'available'] = False
    filtered_data.loc[filtered_data['no_pck_file'] == False, 'available'] = False

    # true는 fillna 사용
    filtered_data.log[(filtered_data['no_lig_file_sdf'] == False) | (filtered_data['no_lig_file_mol2'] == False), 'available'] = False
    filtered_data.log[(filtered_data['empty_lig_file_sdf'] == False) | (filtered_data['empty_lig_file_sdf'] == False), 'available'] = False
    filtered_data.log[(filtered_data['empty_pck_file'] == False) | (filtered_data['empty_pck_file'] == False), 'available'] = False

    filtered_data

'''
    filtered_data[filtered_data['no_folder']==False]['available'] = False
    filtered_data[filtered_data['no_pck_file']==False]['available'] = False
    
    # true는 fillna 사용
    filtered_data[filtered_data['no_lig_file_sdf']==False]or filtered_data['no_lig_file_mol2']==False]['available'] = False
    filtered_data[filtered_data['empty_lig_file_sdf']==False or filtered_data['empty_lig_file_mol2']==False]['available'] = False
    filtered_data[filtered_data['empty_pck_file']==False]['available'] = False
    
    filtered_data['available'] = filtered_data['available'].fillna('True')
'''