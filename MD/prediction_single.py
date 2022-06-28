import torch
import sys
import argparse
import getopt
import warnings
import dgl
import pandas as pd

from torch.utils.data import DataLoader
from tqdm import tqdm

lPATHs = ['D:/Project/HD011/', 'D:/Project/HD011/MD/libs']
for _, path in enumerate(lPATHs):
    sys.path.append(path)

from DB.libs.db_utils import Connect2pgSQL as pgDB
from libs.models import Model
from libs.io_utils import Complex_Dataset
from libs.io_utils import collate_func
from libs.utils import set_device
from libs.utils import evaluate_regression
from train import evaluate, print_metric, prepare_model



if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Please input parameter files or use -h for help")
        sys.exit()

    def usage():
        print("-s or --schema: specify the schema of dataset for prediction in postgreSQL DBMS")
        print("-t or --table: specify the table name of dataset for prediction in postgreSQL DBMS")
        print("\nExample: python prediction_single.py -s pdbbind -t coreset -o prediction.txt")

    try:
        options, args = getopt.getopt(sys.argv[1:], "hs:t:", ["schema=", "table="])
    except getopt.GetoptError:
        print('error in getopt error')
        sys.exit()

    # set query
    schema = ''
    table = ''

    for name, value in options:
        if name in ('-h', '--help'):
            usage()
            sys.exit()
        if name in ('-s', '--schema'):
            schema = value
        if name in ('-t', '--table'):
            table = value

    if (schema == '') & (table == ''):
        print('Please enter the existing schema and table ')
        sys.exit()

    # schema = 'pdbbind'
    # table = 'coreset'

    # condition = f"INNER JOIN {schema}.available ON {table}.pdb_code = available.pdb_code " \
    #             f"WHERE available.available = True AND available.subset=\'{'table'}\'"
    condition = f"INNER JOIN {schema}.available ON {table}.pdb_code = available.pdb_code " \
                f"WHERE available.available = True AND available.subset='refined'"

    db = pgDB()

    # arguments for training
    config = {'num_workers': 8,
                'use_gpu': 1,
                'gpu_idx': 0 }

    # prepare data
    data_list = []
    data_info = pd.DataFrame()
    try:
        data_list = db.readDB(schema=schema, table=table, column=f'{table}.pdb_code, {table}.plog_affinity',
                              condition=condition)
        data_info = pd.DataFrame(data_list, columns=['PDB_code', 'plog_affinity'])
    except:
        warnings.warn('Error raised by importing database for DB\n'
                      'Please check the dataset has been stored in the DB')
        sys.exit()

    dataset = Complex_Dataset(splitted_set=data_info, training=False, dataset_name='coreset')

    # prediction
    device = set_device(use_gpu=config['use_gpu'], gpu_idx=config['gpu_idx'])
    model = torch.load('./model.pth')
    model.to(device)
    model.eval()

    with torch.no_grad():
        # num_batches = len(data_loader)
        num_data = dataset.__len__()
        y_list = []
        pred_list = []

        with tqdm(total=num_data) as pbar:
            pbar.set_description('> Prediction')
            # y_list, pred_list = evaluate(data_loader, pbar, model, device)
            for data in dataset:
                pbar.update(1)
                graphs, y = data
                graph_p = dgl.batch([graphs[0]]).to(device)
                graph_l = dgl.batch([graphs[1]]).to(device)

                output = model(graph_p, graph_l)

                y_list.append(y)
                pred_list.append(output)

                pred = output.squeeze()

                del graph_p, graph_l, output

            pred_metrics = evaluate_regression(y_list=y_list, pred_list=pred_list)

        print_metric(pred_metrics, title='Prediction')

    head = ['#code', 'score']
    body = []
    pdb_code = data_info['PDB_code'].tolist()
    pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy().reshape(-1,)
    for code, pred in zip(pdb_code, pred_list):
        pred = round(pred, 2)
        body.append([code, pred])

    result = pd.DataFrame(body, columns=head).sort_values(by='#code', ascending=True)
    result = result.set_index('#code', drop=True, append=False)

    with open('./prediction_single.txt', 'wb') as f:
        result.to_csv(f, sep=' ')

