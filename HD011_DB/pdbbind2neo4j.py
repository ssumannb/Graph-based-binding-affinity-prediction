import os
import shutil
import pandas as pd
import time
import re

from tqdm import tqdm
from rdkit import Chem

from libs.db_utils import Connect2neo4j

class DataBase:
    def __init__(self, filename, raw_file_path):
        self.dPATH = raw_file_path
        loaded_data = []
        with open(filename, encoding='utf-8') as f:
            loaded_data = f.readlines()

        self.data = loaded_data[6:]
        self.tag = loaded_data[:6]

        self._dList4dupCheck = []
        self.saveCnt = 200

        for i, t in enumerate(self.tag):
            print(t)

        self.column = []
        self._col_setting(loaded_data[7])


    def _col_setting(self, data_sample):
        input_column = input(f"\nset column name, column number:{len(data_sample.split())}"
                             f"(*only text and no sapce) : "
                             f"\n<data sample> \n:{data_sample}\n").split()

        if 'PDBcode' not in input_column:
            print('<!>Please set column name again (add PDBcode)')
            self._col_setting(data_sample)
        elif 'measurement_val' not in input_column:
            print('<!>Please set column name again (add measurement_val)')
            self._col_setting(data_sample)
        else:
            self.column = input_column


    def _loading(self, fold_data: list, pbar:tqdm) -> pd.DataFrame:
        '''
        기능
        1. load_INDEX_file 함수로부터 넘겨 받은 data, column 으로 table 생성
        '''
        data_info = pd.DataFrame()

        for j, data in enumerate(fold_data):
            pbar.update(1)
            data_split = data.split()
            data_split.remove('//')

            data_info = data_info.append(pd.Series(data_split, name=j))
            time.sleep(0.1)

        if len(data_info.columns) == len(self.column):
            data_info.columns = self.column
        else:
            self.column = self._col_setting(' / '.join(data_split))
            data_info.columns = self.column

            # self._dList4dupCheck = self._dList4dupCheck.append(data_info['PDBcode'])

        return data_info


    def _check_exist_fold(self, data_list:list) -> list:
        '''
        파일이 존재하지 않는 데이터 삭제
        **엄밀히 말하면 complexID에 해당하는 폴더가 있는지 확인
        '''
        cnt = 0
        idxs = []
        for i, data in enumerate(data_list):
            if not os.path.isdir(self.dPATH+data):
                cnt += 1
                idxs.append(i)

        print(f'\n<!> {cnt} folders of complexes not exist')
        return idxs

        #example: data_info[~data_info['# PDB code'].isin(idxs)]


    def _check_available_file(self, data_list:list) -> list:
        '''
        파일이 비어있는 (사용할 수 없는) 데이터 삭제
        '''
        cnt1 = 0
        cnt2 = 0
        idxs = []
        with tqdm(total=len(data_list)) as pbar:
            pbar.set_description('Check file available')

            for i, data in enumerate(data_list):
                pbar.update(1)
                file_list = os.listdir(self.dPATH + data)

                if not f'{data}_ligand.sdf' in file_list:
                    if not f'{data}_ligand.mol2' in file_list:
                        cnt1 += 1
                        idxs.append(i)
                        time.sleep(0.1)
                        continue

                if not f'{data}_protein.pdb' in file_list:
                    cnt1 += 1
                    idxs.append(i)
                    time.sleep(0.1)
                    continue

                if not Chem.SDMolSupplier(f'{self.dPATH}/{data}/{data}_ligand.sdf')[0]:
                    if not Chem.MolFromMol2File(f'{self.dPATH}/{data}/{data}_ligand.mol2'):
                        cnt2 += 1
                        idxs.append(i)
                        time.sleep(0.1)
                        continue

                if not Chem.MolFromPDBFile(f'{self.dPATH}/{data}/{data}_protein.pdb'):
                    cnt2 += 1
                    idxs.append(i)
                    time.sleep(0.1)
                    continue

                if not Chem.MolFromPDBFile(f'{self.dPATH}/{data}/{data}_pocket.pdb'):
                    cnt2 += 1
                    idxs.append(i)
                    time.sleep(0.1)
                    continue

                time.sleep(0.1)

        print(f'\n<!> {cnt1} files not exist \n<!> {cnt2} files are not available')
        return idxs


    def _detail_extract(self, data_info:pd.DataFrame) -> pd.DataFrame:
        '''
        binding affinity value 상세 사항 extract
        '''
        data = data_info['measurement_val']
        data_detail = pd.DataFrame(columns=['binding_affinity_type', 'binding_affinity_val', 'binding_affinity_range', 'binding_affinity_unit'])

        re_compiler = re.compile(r'[a-zA-Z]')
        re_compiler2 = re.compile(r'\W')

        data_split = data.apply(lambda x : re.split('=|>|<|~', x))
        data_split = data_split.apply(lambda x : [x[0], x[-1]])
        data_detail['binding_affinity_type'] = data_split.apply(lambda x : x[0])
        data_detail['binding_affinity_val'] = data_split.apply(lambda x : re_compiler.sub('', x[1]))
        data_detail['binding_affinity_unit'] = data_split.apply(lambda x : ''.join(re_compiler.findall(x[1])))
        data_detail['binding_affinity_range'] = data.apply(lambda x : f"{''.join(re_compiler2.findall(x))}".strip('.'))


        data_info = pd.concat([data_info, data_detail], axis=1)
        # data_info = data_info[data_info['type']!='IC50'].reset_index(drop=True)

        return data_info


    def _check_duplicate(self, data_list:list) -> list:
        '''
        데이터 cleaning 맨 마지막에 duplicate 검사
        duplicate 검사를 위해서는 data index를 모두 저장해놓아야 한다.
        '''
        pass


    def _assert_u_constraints_neo4j(self):
        constraint_node = input("Please enter a node name which have to assert constraint")
        constraint_attr = input("Please enter the attribute name of the node which have constraint").split()

        node_name = constraint_node
        query = []

        for i, attr in enumerate(constraint_attr):
            query.append(f"CREATE CONSTRAINT {node_name.lower()}_{attr.lower()} ON (n:{node_name}) ASSERT n.{attr} IS UNIQUE ")

        return query

    def _load2GDB(self, GDBclass:Connect2neo4j, data_info:pd.DataFrame, constraint_list:list):
        attribute = data_info.columns
        command = "CREATE "
        query_head = "(:Complex{"
        query_tail = "})"
        query_body = []

        # attribute data type 지정
        attr_dType = []
        for idx, data in enumerate(data_info.loc[0]):
            try:
                float(data)
                attr_dType.append('float')
            except:
                # raise value error
                attr_dType.append('str')

        for i, data in data_info.iterrows():
            _query = []
            _query.append(query_head)

            for j, attr in enumerate(attribute):
                if attr_dType[j] != 'str':
                    _query.append(f"{attr}:{data[j]}")
                else:
                    _query.append(f"{attr}:\'{data[j]}\'")
                _query.append(",")

            _query.pop()
            _query = ''.join(_query)

            query_body.append(_query)
            query_body.append('}),')

        query_body.pop()
        query_body.append('')
        query = command + ''.join(query_body) + query_tail

        GDBclass.run_query(query)

        # for i, q in enumerate(constraint_list):
        #     GDBclass.run_query(q)


    def iteration(self, GDBclass:Connect2neo4j):
        n_data = len(self.data)
        n_fold = int(n_data/self.saveCnt)
        i_fold_start = 0
        i_fold_end = 0

        with tqdm(total=len(self.data)) as pbar:
            pbar.set_description('Data List reading')

            # cst_flg = input("\nare there constraints? (Y/N)")
            # cst_cnt = int(input("how many constraints (numeric)"))
            cst_list = []
            # if cst_flg == 'Y':
            #     for j in range(cst_cnt):
            #         print("Please type the information for constraint")
            #         cst_list = self._assert_u_constraints_neo4j()


            for i in range(n_fold+1):
                i_fold_start = self.saveCnt * i
                i_fold_end = self.saveCnt * (i+1)

                if i_fold_end > n_data: i_fold_end = -1

                fold_data = self.data[i_fold_start:i_fold_end]
                data_info = self._loading(fold_data, pbar)

                idxs4rm = self._check_exist_fold(data_info['PDBcode'])
                data_info[~data_info['PDBcode'].isin(idxs4rm)].reset_index(drop=True)

                idxs4rm_av = self._check_available_file(data_info['PDBcode'])
                data_info[~data_info['PDBcode'].isin(idxs4rm_av)].reset_index(drop=True)

                data_info = self._detail_extract(data_info)

                self._load2GDB(GDBclass, data_info, cst_list)



if __name__ == "__main__":
    filename = 'D:/DATA/PDBbind/index/INDEX_refined_Data.2020'
    GDB = Connect2neo4j()
    raw_file_path = 'D:/DATA/PDBbind/raw_data/refined-set_v2020/'

    refinedDB = DataBase(filename, raw_file_path)
    refinedDB.iteration(GDB)



