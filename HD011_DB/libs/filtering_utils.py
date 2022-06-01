import os
import time
import sys
import pandas as pd
import numpy as np

from rdkit import Chem
from Bio.PDB import *
from tqdm import tqdm

from db_utils import connect2pgSQL as pgDB

def extract_list(db:pgDB, columns:list, schema:str, table:str):
    columns = ', '.join(columns)

    query = f'SELECT {columns} FROM {schema}.{table};'
    return db.run_query(query)


def _no_Folder(org_data_list:pd.DataFrame):
    result = []

    with tqdm(total=len(org_data_list)) as pbar:
        pbar.set_description('Check folder ')
        for i, row in org_data_list.iterrows():
            pbar.update(1)
            path_tmp = f"{row['path']}{row['pdb_code']}/"
            result.append(not os.path.isdir(path_tmp))

    return result


def _no_lig_file(org_data_list:pd.DataFrame):
    result_sdf = []
    result_mol2 = []

    with tqdm(total=len(org_data_list)) as pbar:
        pbar.set_description('Check ligand file ')
        for i, row in org_data_list.iterrows():
            pbar.update(1)
            id = row['pdb_code']
            path_tmp = f"{row['path']}{id}/"
            if not os.path.isdir(path_tmp):
                result_sdf.append(True)
                result_mol2.append(True)
            else:
                f_list = os.listdir(path_tmp)
                result_sdf.append(not f'{id}_ligand.sdf' in f_list)
                result_mol2.append(not f'{id}_ligand.mol2' in f_list)

    return result_sdf, result_mol2


def _no_ptn_file(org_data_list:pd.DataFrame):
    result_protein = []
    result_pocket = []

    with tqdm(total=len(org_data_list)) as pbar:
        pbar.set_description('Check protein file ')
        for i, row in org_data_list.iterrows():
            pbar.update(1)
            id = row['pdb_code']
            path_tmp = f"{row['path']}{id}/"
            if not os.path.isdir(path_tmp):
                result_protein.append(True)
                result_pocket.append(True)
            else:
                f_list = os.listdir(path_tmp)
                result_protein.append(not f'{id}_protein.pdb' in f_list)
                result_pocket.append(not f'{id}_pocket.pdb' in f_list)

    return result_protein, result_pocket


def _empty_lig(org_data_list:pd.DataFrame):
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
                result_sdf.append(True)
            else:
                if Chem.SDMolSupplier(path_sdf)[0]:
                    result_sdf.append(False)
                else:
                    result_sdf.append(True)

            if not os.path.isfile(path_mol2):
                result_mol2.append(True)
            else:
                if Chem.MolFromMol2File(path_mol2):
                    result_mol2.append(False)
                else:
                    result_mol2.append(True)

    return result_sdf, result_mol2


def _empty_ptn(org_data_list:pd.DataFrame):
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
                result_protein.append(True)
            else:
                if Chem.MolFromPDBFile(path_protein):
                    result_protein.append(False)
                else:
                    result_protein.append(True)

            if not os.path.isfile(path_pocket):
                result_pocket.append(True)
            else:
                if Chem.MolFromPDBFile(path_pocket):
                    result_pocket.append(False)
                else:
                    result_pocket.append(True)

    return result_protein, result_pocket


def filtering(src:pd.DataFrame):
    # filtering
    filtered_data = pd.DataFrame()
    filtered_data['pdb_code'] = src['pdb_code']
    filtered_data['no_folder'] = _no_Folder(src)
    filtered_data['no_ptn_file'], filtered_data['no_pck_file'] = _no_ptn_file(src)
    filtered_data['no_lig_file_sdf'], filtered_data['no_lig_file_mol2'] = _no_lig_file(src)
    filtered_data['empty_ptn_file'], filtered_data['empty_pck_file'] = _empty_ptn(src)
    filtered_data['empty_lig_file_sdf'], filtered_data['empty_lig_file_mol2'] = _empty_lig(src)

    filtered_data['available'] = True

    filtered_data.loc[filtered_data['no_folder'] == True, 'available'] = False
    filtered_data.loc[filtered_data['no_ptn_file'] == True, 'available'] = False

    # true는 fillna 사용
    filtered_data.loc[(filtered_data['no_lig_file_sdf'] == True) &
                      (filtered_data['no_lig_file_mol2'] == True), 'available'] = False
    filtered_data.loc[(filtered_data['empty_lig_file_sdf'] == True) &
                      (filtered_data['empty_lig_file_mol2'] == True), 'available'] = False
    filtered_data.loc[(filtered_data['empty_pck_file'] == True) &
                      (filtered_data['empty_pck_file'] == True), 'available'] = False
    filtered_data['subset'] = src['subset']

    caller = sys._getframe(1).f_code.co_name
    f_name = f"./Report/check_inavailability_({caller})-{time.strftime('%Y-%m-%d', time.localtime(time.time()))}.csv"
    if os.path.isfile(f_name):
        filtered_data.to_csv(f_name, mode='a', header=False, index=False)
    else:
        filtered_data.to_csv(f_name)

    return filtered_data


if __name__ == '__main__':
    pass
