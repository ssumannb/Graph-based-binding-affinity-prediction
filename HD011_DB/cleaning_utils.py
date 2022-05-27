"""
Definition of 'cleaning_utils' module
■ Input : Information list of the complex data
    - complex id (indicate the ligand/protein structure files)
    - logKa value
■ Output : complex information list applied the exception constraint
    - csv files : complex information list
    - pkl files : complex information saved as 'DataFrame' format
    - residues : residue structure files, which is selected with the constraint
□ Class
    - ResidueSelect : class for residue selection with overrided 'accept_residue' function of 'Select' class in PDBIO
□ Function
    - extract_residue : function for extract the structure of selected residues
    - cleaning        : function for cleaning complex data list
    - print_result    : function for printing the cleaning result
"""
import pandas as pd
import numpy as np
import warnings
import os
import re
import time
import glob

from tqdm import tqdm
from rdkit import Chem
from tabulate import tabulate
from Bio.PDB import *
from scipy.spatial import distance_matrix

warnings.filterwarnings(action='ignore')

PATH = {'in': './cleaning/Input',
        'out': './cleaning/Output',
        'src': './DATABASE',}

class ResidueSelect(Select):
    """
    class for residue selection with overrided 'accept_residue' function of 'Select' class in PDBIO
    - PDBIO class inheritance, accept_residue method overriding
    - use with PDBIO object
    - Get distance between ligand positions (N_ligand, 3) and
    - residue positions (N_residue, 3) for each residue
    - only select residues with minimum distance of it is smaller than 5A
    → 각각의 잔기에 대해 리간드 위치, 잔기 위치간의 거리를 측정하여 최소 거리가 5A보다 짧은 잔기만을 선택
    - 단백질을 구성하는 아미노산 잔기들 각각에 대한 리간드와의 거리 측정 후 선택/제외 결정
    """
    def accept_residue(self,
                       ligand_positions,
                       residue):

        residue_positions = np.array([np.array(list(atom.get_vector())) \
                                      for atom in residue.get_atoms() if 'H' not in atom.get_id()])
        residue_position_loop = []

        for atom in residue.get_atoms():
            if 'H' not in atom.get_id():
                residue_position_loop.append(np.array(list(atom.get_vector())))

        residue_position_loop = np.array(residue_position_loop)

        if len(residue_positions.shape) < 2:
            return 0

        min_dis = np.min(distance_matrix(residue_positions, ligand_positions))

        if min_dis < 5.0:
            return 1
        else:
            return 0


def extract_residue(ligand: Chem.rdmolfiles,
                    protein_file: str):
    """
    function for extract the structure of selected residues
    -
    """
    pdbID = protein_file.split('/')[-1].split('_')[0]
    file_name = f"{PATH['out']}/residues/{pdbID}_residues.pdb"
    # file_name = f'../../BA_prediction_ignore/data/residues/{pdbid}_residues.pdb'

    parser = PDBParser()
    io = PDBIO()

    # structure preparation for extract the residue structure
    protein_structure = parser.get_structure('protein', protein_file)
    ligand_positions = ligand.GetConformer().GetPositions()

    # set the protein structure in PDBIO object and save residue structure
    io.set_structure(protein_structure)
    io.save(file_name, ResidueSelect(ligand_positions))

    # load the residue structure from saved file
    residue_structure = Chem.MolFromPDBFile(file_name)

    return residue_structure


def list2df(lines: list,
            column: list):
    data_df = pd.DataFrame()

    for i, line in enumerate(lines):
        splitted_line = pd.Series(line.split(), name=i)
        data_df = data_df.append(splitted_line)

    data_df.columns = column

    return data_df

# def filtering(data_info: pd.DataFrame):

def column_name(col_lines: str):
    if not 'ID' | 'code' in col_lines:
        raise ValueError('There is no column name including code or ID')
    if not 'label' | 'y' | 'log' in col_lines:
        raise ValueError('There is no column name including label, y or log')

    # 컬럼 구분 기호 : .
    splitted_col = col_lines.split('.')

    return splitted_col


def cleaning(file_name: str,
             source_data: str,
             out_file_name: str,
             type,
             save_cycle = 1000):

    log_noFolder = {'cnt': 0, 'list': []}
    log_noFile_l = {'cnt': 0, 'list': []}
    log_noFile_p = {'cnt': 0, 'list': []}
    log_empFile = {'cnt': 0, 'list': []}
    log_dup = {'cnt': 0, 'list': []}

    PDB_id = pd.Series([], dtype='object')
        # if not type == 'test':
        #     data_type = 'gen' if 'general' in data_file else 'rfn'

    with open(file_name, encoding='utf-8') as f:
        lines = f.readlines()

    lines = lines[6:]   # comment delete

    n_lines = len(lines)
    n_split = int(n_lines/save_cycle)+1
    if n_split < 1:
        raise ValueError('The number of data of one save cycle is larger than data lengh!'\
                         '\n please correct the parameter \'save_cycle\'')

    with tqdm(total=len(lines)) as pbar:
        pbar.set_description('complex list reading')

        for j in range(n_split):
            i_start = save_cycle * (j)
            i_end = save_cycle * (j+1)
            if i_end > n_lines: i_end = -1

            splitted_lines = lines[i_start:i_end]
            splitted_lines

            pbar.set_description('reading dataInfo(df) list from csv')
            data_info = pd.DataFrame()
            ### list2df function start
            for k, info in enumerate(splitted_lines):
                pbar.update(1)
                tmp = info.split()
                # if not type == 'test':
                #     tmp.append(data_type)

                data_info = data_info.append(pd.Series(tmp, name=k))
                # data_info.append(pd.Series(tmp, name=j))
                time.sleep(0.1)

            pbar.set_description('column arrange, check duplication')
            if not type == 'test':
                data_info.columns = ['PDB_id', 'res', 'release', 'BA_log', 'Kd/Ki/IC50', '//', 'refer', 'ligand_name']
                del data_info['//']
            else:
                data_info.columns = ['PDB_id', 'res', 'release', 'BA_log', 'Kd/Ki', 'targetNum']
            #### list2df function end

            # 칼럼 삭제, 중복 삭제, 인덱스 초기화
            tmp = data_info['PDB_id'].append(PDB_id)
            dup = tmp.duplicated(keep='first')[:len(data_info)]
            dup_true = dup[dup=='True']
            dup_idx_true = []
            if not dup_true.empty:
                dup_idx_true = dup_true.index[0]
                dup_idx_true
            else:
                log_dup['cnt'] += len(dup_idx_true)
            data_info.drop(dup_idx_true)
            data_info.reset_index(drop=True)

            pbar.set_description('Spiltting binding affinity data')
            # binding affinity data 전처리
            ba_data = pd.DataFrame(columns=['BA_type', 'BA_val', 'BA_range', 'BA_unit'])
            re_compiler = re.compile(r'[a-zA-Z]')

            # BA_data extraction {'Kd/Ki/IC50'} -> {BA_log, BA_type, BA_val, BA_range, BA_unit}
            col_name = 'Kd/Ki/IC50'
            if type == 'test':
                col_name = 'Kd/Ki'
            ba_splitted = data_info[col_name].apply(lambda x : re.split('=|>|<|~', x))           # 등호를 기준으로 type, value split
            ba_data['BA_type'] = ba_splitted.apply(lambda x : x[0])                              # BA_type 할당
            ba_data['BA_val'] = ba_splitted.apply(lambda x : re_compiler.sub('', x[1]))          # BA_val 추출: 컴파일러로 문자열 제거
            ba_data['BA_unit'] = ba_splitted.apply(lambda x : ''.join(re_compiler.findall(x[1])))# BA_unit 추출: 컴파일러로 문자열 서치

            re_compiler = re.compile(r'\W')                                                         # 등호, 부등호 추출을 위한 컴파일러 재정의
            ba_data['BA_range'] = data_info[col_name].apply(lambda x : re_compiler.findall(x)[0])   # .을 제외한 기호 추출
            data_info = pd.concat([data_info, ba_data], axis=1)
            data_info = data_info[data_info['BA_type']!='IC50'].reset_index(drop=True)

            del data_info[col_name], ba_splitted, ba_data

            pbar.set_description('Check not existing file')
            # empty data delete from data_info
            if not type == 'test':
                # data_path = '../../BA_prediction_ignore/data/PDBbind_v2020/'
                data_path = f"{PATH['src']}/PDBbind2020/"
            else:
                # data_path = '../../BA_prediction_ignore/data/CASF-2016/coreset/'
                data_path = f"{PATH['src']}/CASF-2016/coreset/"

            for idx, data in enumerate(data_info['PDB_id']):
                # 데이터 파일 존재 check
                if not os.path.isdir(data_path + data):
                    log_noFolder['cnt'] += 1
                    log_noFolder['list'].append(data)
                    data_info = data_info.drop(idx)

                else:
                    file_list = os.listdir(data_path + data)
                    # 리간드 파일 없으면 drop
                    if not f'{data}_ligand.sdf' in file_list:
                        if not f'{data}_ligand.mol2' in file_list:
                            log_noFile_l['cnt'] += 1
                            log_noFile_l['list'].append(data)
                            data_info = data_info.drop(idx)
                            continue
                    # 단백질 파일 없으면 drop
                    if not f'{data}_pocket.pdb' in file_list:
                        log_noFile_p['cnt'] += 1
                        log_noFile_p['list'].append(data)
                        data_info = data_info.drop(idx)
                        continue

                    # 빈 내용이 읽히는 분자,포켓 파일 처리
                    # if not Chem.SDMolSupplier(data_path + f"{data}/{data}_ligand.sdf")[0]:
                    #     if not Chem.MolFromMol2File(data_path + f"{data}/{data}_ligand.mol2"):
                    #         log_empFile['cnt'] += 1
                    #         log_empFile['list'].append(data)
                    #         data_info = data_info.drop(idx)
                    #         continue
                    # if not Chem.MolFromPDBFile(data_path + f"{data}/{data}_pocket.pdb"):
                    #     log_empFile['cnt'] += 1
                    #     log_empFile['list'].append(data)
                    #     data_info = data_info.drop(idx)
                    #     continue

                    if not Chem.SDMolSupplier(f"{PATH['src']}/Ligand/{data}_ligand.sdf")[0]:
                        if not Chem.MolFromMol2File(f"{PATH['src']}/Ligand/{data}_ligand.mol2"):
                            log_empFile['cnt'] += 1
                            log_empFile['list'].append(data)
                            data_info = data_info.drop(idx)
                            continue
                    if not Chem.MolFromPDBFile(f"{PATH['src']}/Protein/{data}_pocket.pdb"):
                        log_empFile['cnt'] += 1
                        log_empFile['list'].append(data)
                        data_info = data_info.drop(idx)
                        continue



                    mol_ligand = Chem.SDMolSupplier(f"{PATH['src']}/Ligand/{data}_ligand.sdf")[0]
                    if not mol_ligand:
                        mol_ligand = Chem.MolFromMol2File(f"{PATH['src']}/Ligand/{data}_ligand.mol2")

                    residue_structure = extract_residue(mol_ligand,
                                                        f"{PATH['src']}/Protein/{data}_pocket.pdb")

                    if residue_structure == None:
                        log_empFile['cnt'] += 1
                        log_empFile['list'].append(data)
                        data_info = data_info.drop(idx)


            # save data_info
            if os.path.isfile(f"{PATH['out']}/csv/{out_file_name}.csv"):
                data_info.to_csv(f"{PATH['out']}/csv/{out_file_name}.csv",
                                 mode='a', header=False, index=False)
            else:
                data_info.to_csv(f"{PATH['out']}/csv/{out_file_name}.csv", header=False, index=False)

            data_info.to_pickle(f"{PATH['out']}/pkl/{out_file_name}.pkl")
            PDB_id = PDB_id.append(pd.Series(data_info['PDB_id']))

            ### 메모리 삭제
            del data_info, splitted_lines

    return print_result(log_noFolder, log_noFile_l, log_noFile_p, log_empFile, log_dup)


class Clean():
    def __init__(self,
                 input_file: str,
                 source_data: str,
                 output_file: str,
                 save_cycle = 1000):

        self.input_file = input_file
        self.source_data = source_data
        self.output_file = output_file
        self.save_cycle = save_cycle

        self.col_list = []

        self.cleaning_report = {}.setdefault(['noFolder','noFile_l','noFile_p','empFile','dup'], 0)


    def list2df(self,
                lines: list,
                column: list):

        with open(self.input_file, encoding='utf-8') as f:
            lines = f.readlines()




def print_result(noFolder, noFile_l, noFile_p, empFile, dup):
    '''
    noFolder = {'cnt': 0, 'list': []}
    noFile_l = {'cnt': 0, 'list': []}
    noFile_p = {'cnt': 0, 'list': []}
    empFile = {'cnt': 0, 'list': []}
    '''
    col = ['noFolder', 'noFile_l', 'noFile_p', 'empFile','dup']
    cnt = [noFolder['cnt'], noFile_l['cnt'], noFile_p['cnt'], empFile['cnt'], dup['cnt']]

    table = {
        '-' : col,
        'count' : cnt
    }

    print(tabulate(table, headers = 'keys', tablefmt='psql', showindex=True))

    pass


def main():

    # data_files = ['../../BA_prediction_ignore/data/PDBbind_v2020_plain_text_index/index/INDEX_general_PL_data_2020_copy.csv']
    # type = 'kdki'
    # for i, file in enumerate(glob.glob(f'../../BA_prediction_ignore/data/dataInfo_PL-complex_{type}*.*')):
    #     os.remove(file)
    #
    # save_PL_list(data_files, type)

    data_files_test = f"{PATH['in']}/CoreSet.dat"

    type = 'test'
    for i, file in enumerate(glob.glob(f'../../BA_prediction_ignore/data/dataInfo_PL-complex_{type}*.*')):
        os.remove(file)

    cleaning(file_name=data_files_test, source_data='CoreSet', out_file_name='dataInfo_PL-complex_', type=type)


if __name__ == '__main__':
    main()

