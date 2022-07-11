import os
import glob
import warnings
import pickle
import torch
import dgl
import pandas as pd
import numpy as np

from multiprocessing import Manager

from DB.libs.db_utils import Connect2pgSQL as pgDB


def collate_func(batch):
    graph_list_ligand = []
    graph_list_residue = []
    label_list = []

    for data in batch:
        graph_l = data[0][0]
        graph_p = data[0][1]

        graph_list_ligand.append(graph_l)
        graph_list_residue.append(graph_p)
        label_list.append(float(data[1]))
        label_list

    # Batch a collection of DGLGraph s into one graph for more efficient graph computation.
    graph_list_ligand = dgl.batch(graph_list_ligand)
    graph_list_residue = dgl.batch(graph_list_residue)
    label_list = torch.tensor(label_list, dtype=torch.float64)

    return graph_list_residue, graph_list_ligand, label_list


def get_complex_list(data_seed=999, frac=[0.8, 0.2], dataset_name='None'):
    data_info = pd.DataFrame()
    db = pgDB()

    table_name = ''
    table_scheme = ''

    try:
        type, version = dataset_name.split()
        type = type.lower()
        if type == 'pdbbind':
            table_scheme = 'pdbbind'

        if version == '2020':
            table_name = 'binding_index'
    except:
        raise ValueError('error in <dataset_name> of argument')

    try:
        query = f"SELECT {table_name}.pdb_code, {table_name}.plog_affinity FROM {table_scheme}.binding "
        condition = f"INNER JOIN {table_scheme}.available ON {table_name}.pdb_code = available.pdb_code " \
                    f"WHERE available.available = True AND {table_name}.pdb_code NOT IN (SELECT pdb_code FROM {table_scheme}.coreset) " \
                    f"AND affinity_data NOT LIKE 'IC50%'"
        # condition = f"INNER JOIN {table_scheme}.available ON {table_name}.pdb_code = available.pdb_code " \
        #             f"WHERE available.available = True AND affinity_data NOT LIKE 'IC50%'"

        return_val = db.readDB(schema=table_scheme, table=table_name,
                               column=f'{table_name}.pdb_code, {table_name}.plog_affinity', condition=condition)

        data_info = pd.DataFrame(return_val, columns=['PDB_code', 'plog_affinity'])

    except:
        warnings.warn('Error raised by importing database for DB')

    complex_dataset_list = []
    for i, _ in enumerate(frac):
        complex_dataset_list.append(data_info.sample(
            frac=frac[i],random_state=data_seed).reset_index(drop=True))

    return complex_dataset_list


class DatasetCache(object):
    def __init__(self, manager, use_cache=True):
        self.use_cache = use_cache
        self.manager = manager
        self._dict = manager.dict()

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return str(key) in self._dict

    def reset(self):
        self._dict.clear()

    def get(self, key):
        if not self.use_cache:
            raise AttributeError('Data caching is disabled and get funciton is unavailable! Check your config.')
        return self._dict[str(key)]

    def cache(self, key, graph, lbl):
        # only store if full data in memory is enabled
        if not self.use_cache:
            return
        # only store if not already cached
        if str(key) in self._dict:
            return
        self._dict[str(key)] = (graph, lbl)



class Cached_Complex_Dataset(torch.utils.data.Dataset):
    def __init__(self, cache,  # cached
                 splitted_set,
                 preprocess=None, transform=None, # cached
                 training=True, dataset_name='v2020', use_cache=False):
        self.call_count = 0  # for iteration
        self.use_cache = use_cache
        self.PDB_code = list(splitted_set['PDB_code'])
        self.label = list(splitted_set['plog_affinity'])
        self.path = f'D:/Data/PDBbind/HD011/model_input/training/graph/{dataset_name}'
        if training == False:
            self.path = f'D:/Data/PDBbind/HD011/model_input/external_validation/graph/{dataset_name}'
        if not os.path.isdir(self.path):
            raise FileNotFoundError(f'There is no folder in this location : {self.path}')

        self.cache = cache
        self.preprocess = preprocess
        self.transform = transform


    def set_use_cache(self, use_cache):
        if use_cache:
            self.cache_data = torch.stack(self.cached_data)
        else:
            self.cached_data = []
        self.use_cache = use_cache

    def reset_memory(self):
        self.cache.reset()

    def set_memory(self, cache):
        self.cache = cache

    def __len__(self):
        return len(self.PDB_code)

    def __getitem__(self, item):
        if self.cache.is_cached(item):
            graphs, self.label[item] = self.cache.get(item)
        else:
            fname = self.PDB_code[item]
            graphs = None,
            try:
                os.path.isfile(f'{self.path}/{self.PDB_code[item]}.pkl')
                with open(f'{self.path}/{self.PDB_code[item]}.pkl', 'rb') as f:
                    graphs = pickle.load(f)

                self.cache.cache(item, graphs, self.label[item])
                # self.cached_data.append([graphs, self.label[item]])

            except:
                raise ValueError(f"\n<!> no graph file: {self.path}/{self.PDB_code[item]}.pkl")

        return graphs, self.label[item]

    def __next__(self):
        if len(self.PDB_code) <= self.call_count:
            raise StopIteration

        self.call_count += 1
        return self.__getitem__(item=self.call_count-1)

    def __iter__(self):
        self.call_count = 0
        return self


def debugging():
    pass


if __name__ == '__main__':
    debugging()
