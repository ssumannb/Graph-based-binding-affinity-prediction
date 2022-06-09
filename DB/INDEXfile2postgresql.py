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

from libs.db_utils import Connect2pgSQL as pg

fPATH = "D:/Data/PDbbind/index/INDEX_general_PL_data_2020.xlsx"

load_wb = load_workbook(fPATH, data_only=True)
load_ws = load_wb['Sheet1']

db = pg()

# columns = [x for row in load_ws.rows for x in row]

list_data = []

col = []
schema = 'pdbbind'
table = 'binding_INDEX'
col_str = ', '.join(['pdb_code', 'resolution', 'release_year', 'pLog', 'affinity_data', 'reference', 'ligand_name'])

flg = True

for row in load_ws.rows:
    if flg == True:
        flg = False
        continue
    val = [f"\"{str(x.value)}\"" if type(x.value) is str else str(x.value) for x in row]
    val_str = ', '.join(val)
    val_str = val_str.replace('\'', '\\\'')
    val_str = val_str.replace('None', '\'\'')
    val_str = val_str.replace('\"', '\'')
    val_str = val_str.replace('\\\'', '\"')
    db.insertDB(schema, table, col_str, val_str, multiple=False)

    db.commit()

df_data = pd.DataFrame(list_data, columns=col)

print(f'NULL info\n {df_data.isnull().sum()}')
print(df_data.columns)
