-'''
filtering code about pdbbind general, refined, coreset data
and load filtered data to postgresql database
'''
import pandas as pd

from libs.filtering_utils import extract_list
from libs.filtering_utils import filtering
from libs.db_utils import CRUD_pgSQL as pgDB

gPATH = 'D:/Data/PDBbind/raw_data/PDBbind_v2020/'
rPATH = 'D:/Data/PDBbind/raw_data/refined-set_v2020/'
cPATH = 'D:/Data/PDBbind/raw_data/CASF-2016/coreset/'

def general_refine(db:pgDB):
    data_list = extract_list(db, columns=['pdb_code', 'subset'],
                             schema='pdbbind', table='binding')
    data_df = pd.DataFrame()

    data_df['pdb_code'] = [x[0] for i, x in enumerate(data_list)]
    data_df['subset'] = [x[1] for i, x in enumerate(data_list)]

    path_list = []
    for i, type in enumerate(data_df['subset']):
        if type == 'general':
            path_list.append(gPATH)
        elif type == 'refined':
            path_list.append(rPATH)

    data_df['path'] = path_list
    data_df = data_df[data_df['subset']=='refined'] #

    # loop
    unit = 1000
    loop = int(len(data_df) / unit) + 1

    for i in range(loop):
        i_start = i * unit
        i_end = i_start + unit

        if i_end > len(data_df): i_end = -1

        org_data_subset = data_df[i_start:i_end]
        filtered_data = filtering(org_data_subset)

        filtered_data2db = pd.DataFrame(filtered_data[['pdb_code', 'available','subset']], columns=['pdb_code', 'available', 'subset'])
        filtered_data2db['inavailability_type'] = None

        idx_invailability = filtered_data2db[filtered_data2db['available'] == False].index

        for _, idx in enumerate(idx_invailability):
            row_val = filtered_data.loc[idx]
            inav_type = row_val.index[row_val == True].tolist()

            if 'available' in inav_type:
                inav_type.remove('available')
            filtered_data2db.loc[idx, 'inavailability_type'] = ', '.join(inav_type)  # np.array(inav_type).reshape(-1, 1)

        del filtered_data

        query_values = []
        for idx, row in filtered_data2db.iterrows():
            query = f"('{row['pdb_code']}', '{row['available']}', " \
                    f"'{row['inavailability_type']}', '{row['subset']}')"
            query_values.append(query)
        query_values = ', '.join(query_values)

        db.insertDB(schema='pdbbind', table='available',
                    column=', '.join(filtered_data2db), data=query_values, multiple=True)


def coreset(db:pgDB):
    data_list = extract_list(db, columns=['pdb_code'], schema='pdbbind', table='coreset')
    data_df = pd.DataFrame()

    data_df['pdb_code'] = [x[0] for i, x in enumerate(data_list)]
    data_df['path'] = cPATH
    data_df['subset'] = 'coreset'

    filtered_data = filtering(data_df)
    filtered_data2db = pd.DataFrame(filtered_data[['pdb_code', 'available', 'subset']])
    filtered_data2db.columns = ['pdb_code', 'available', 'subset']
    filtered_data2db['inavailability_type'] = None
    # filtered_data2db['subset'] = 'coreset'

    idx_inavailability = filtered_data2db[filtered_data2db['available']==False].index

    for _, idx in enumerate(idx_inavailability):
        row_val = filtered_data.loc[idx]
        inav_type = row_val.index[row_val == True].tolist()

        if 'available' in inav_type:
            inav_type.remove('available')
        filtered_data2db.loc[idx, 'inavailability_type'] = ', '.join(inav_type)

    query_values = []
    for idx, row in filtered_data2db.iterrows():
        query = f"('{row['pdb_code']}', '{row['available']}', " \
                f"'{row['subset']}, '{row['inavailability_type']}')"
        query_values.append(query)
        # set = f"available = {row['available']}, " \
        #         f"subset = '{row['subset']}', " \
        #        f"inavailability_type = '{row['inavailability_type']}' "
        # condition = f"pdb_code = '{row['pdb_code']}' AND subset!='refined' AND subset!='general'"
        #
        # db.updateDB('pdbbind', 'available', set, condition)
    query_values = ', '.join(query_values)

    db.insertDB(schema='pdbbind', table='available',
               column=', '.join(filtered_data2db), data=query_values, multiple=True)


if __name__ == '__main__':
    db = pgDB()
    # general_refine(db)
    coreset(db)