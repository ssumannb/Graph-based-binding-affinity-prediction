import pandas as pd

import DataBase.connection_postgresql as pgDB

from filtering_utils import extract_list
from filtering_utils import filtering

gPATH = 'D:/Data/PDBbind/raw_data/PDBbind_v2020/'
rPATH = 'D:/Data/PDBbind/raw_data/PDBbind_v2020_refined/'
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

    # loop
    unit = 200
    loop = int(len(data_df) / unit) + 1

    for i in range(loop):
        i_start = i * unit
        i_end = i_start + unit

        if i_end > len(data_df): i_end = -1

        org_data_subset = data_df[i_start:i_end]
        filtered_data = filtering(org_data_subset)

        filtered_data2db = pd.DataFrame(filtered_data[['pdb_code', 'available']], columns=['pdb_code', 'available'])
        filtered_data2db['inavailability_type'] = None
        filtered_data2db['subset'] = org_data_subset['subset']

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

        db.insertDB(schema='pdbbind', table='available', column=', '.join(filtered_data2db), data=query_values, multiple=True)


def coreset(db:pgDB):
    data_list = extract_list(db, columns=['pdb_code'], schema='pdbbind', table='coreset')
    data_df = pd.DataFrame()

    data_df['pdb_code'] = [x[0] for i, x in enumerate(data_list)]
    data_df['path'] = cPATH
    data_df['subset'] = 'coreset'

    filtered_data = filtering(data_df)
    filtered_data2db = pd.DataFrame(filtered_data[['pdb_code', 'available']])
    filtered_data2db.columns = ['pdb_code', 'available']
    filtered_data2db['inavailability_type'] = None
    # filtered_data2db['subset'] = 'coreset'

    idx_inavailability = filtered_data2db[filtered_data2db['available']==False].index

    for idx, row in enumerate(idx_inavailability):
        row_val = filtered_data.loc[idx]
        inav_type = row_val.index[row_val == True].tolist()

        if 'available' in inav_type:
            inav_type.remove('available')
        filtered_data2db.loc[idx, 'inavailability_type'] = ', '.join(inav_type)

    del filtered_data

    query_values = []

    for idx, row in filtered_data2db.iterrows():
        query = f"('{row['pdb_code']}', '{row['available']}', " \
                f"'{row['inavailability_type']}', '{row['subset']}')"
        query_values.append(query)

    query_values = ', '.join(query_values)

    db.insertDB(schema='pdbbind', table='available',
                column=', '.join(filtered_data2db), data=query_values, multiple=True)


if __name__ == '__main__':
    db = pgDB.CRUD()
    general_refine(db)
    coreset(db)