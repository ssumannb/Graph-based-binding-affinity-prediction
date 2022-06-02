'''
input: PDBbind_19444_for_postgresql (including almost information about pdbbind general/refined set
to pdbbind.binding Table
'''
import os
import openpyxl
import pandas as pd

from openpyxl import load_workbook
from openpyxl.utils.cell import range_boundaries
from copy import copy

from libs.db_utils import connect2pgSQL as pg

fPATH = "D:/Project/HD011/HD011_DB/DataBase/PDBbind_19444_for_postgresql.xlsx"

load_wb = load_workbook(fPATH, data_only=True)
load_ws = load_wb['Search Results']

db = pg.CRUD()

# columns = [x for row in load_ws.rows for x in row]

list_data = []

idx = 0
col = []
schema = 'pdbbind'
table = 'binding'

for row in load_ws.rows:
    if idx == 0:
        col = [x.value.replace(' ', '_').replace('.','') for x in row]
        col_str = ', '.join(col)
        ### create table
        # structure = []
        # for i, col_name in enumerate(col):
        #     query_tmp = input(f'Type a query for the column {col_name}')
        #     query = f"{col_name} {query_tmp}"
        #     structure.append(query)
        #
        # db.createDB(schema, table, structure)
        ###
        print('done?!')
    else:
        val = [f"\"{str(x.value)}\"" if type(x.value) is str else str(x.value) for x in row]
        val_str = ', '.join(val)
        val_str = val_str.replace('\'', '\\\'')
        val_str = val_str.replace('None', '\'\'')
        val_str = val_str.replace('\"', '\'')
        val_str = val_str.replace('\\\'', '\"')
        db.insertDB(schema, table, col_str, val_str)

    db.commit()
    idx += 1

df_data = pd.DataFrame(list_data, columns=col)

print(f'NULL info\n {df_data.isnull().sum()}')
print(df_data.columns)
