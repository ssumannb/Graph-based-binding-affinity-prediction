'''
input: coreset_index
to pdbbind.coreset table
'''
import pandas as pd

from libs.db_utils import Connect2pgSQL as pgDB

if __name__ == '__main__':
    db = pgDB()
    # coreset_raw_data = pd.read_csv('D:/Data/PDBbind/index/org/Coreset_index.txt')
    with open('D:/Data/PDBbind/index/Coreset_index.txt') as rb:
        coreset_raw_data = rb.readlines()
    coreset_list = []
    for i, line in enumerate(coreset_raw_data):
        coreset_list.append(line.split())

    coreset_df = pd.DataFrame(coreset_list)
    coreset_df.columns = ['pdb_code','resolution', 'release_year', 'pLog_affinity','affinity_data','target']

    values = []
    for i, row in coreset_df.iterrows():
        value = f"('{row['pdb_code']}', '{row['resolution']}', '{row['release_year']}', " \
                f"{row['pLog_affinity']}, '{row['affinity_data']}', {row['target']})"
        values.append(value)

    values = ', '.join(values)

    db.insertDB(schema='pdbbind', table='coreset', column=', '.join(coreset_df.columns),
                data=values, multiple=True)