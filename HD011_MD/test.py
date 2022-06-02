import time
import argparse
import pickle
import os
import dgl
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from tabulate import tabulate


from rdkit import Chem

from torch.utils.tensorboard import SummaryWriter

from libs.io_utils import select_residue
from libs.io_utils import get_molecular_graph

from libs.utils import set_seed
from libs.utils import set_device
from libs.utils import evaluate_regression
from libs.utils import evaluate_regression_test

from libs.models import Model_vectorization

# 오류 출력 명확하게 하기 위해 환경변수 설정
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
writer = SummaryWriter()


def preparation_y():
    file = '../BA_prediction_ignore/data/CoreSet.dat'
    y_df = pd.read_table(file, sep='\s+')

    y = pd.DataFrame(y_df['#code'])

    y['logKa'] = list(map(lambda x: round(x, 2), y_df['logKa']))

    return y


def preparation():
    # Prepare datasets for test
    PATH = '../BA_prediction_ignore/data/CASF-2016/coreset/'
    PATH_graph = '../BA_prediction_ignore/data/CASF-2016/coreset_graph/'
    complex_list = os.listdir(PATH)
    y_list = preparation_y()

    opt = '_opt'

    graph_l_list = []
    graph_p_list = []

    remove_list = []

    complex_list.sort()
    y_list = y_list.sort_values(by=['#code'], axis=0).reset_index(drop=True)

    # Order check
    for complex, y in zip(complex_list, y_list['#code']):
        if complex == y:
            continue
        else:
            raise ValueError(f'complex list code 순서와 y list code 순서가 다릅니다. {complex}, {y}')

    with tqdm(total=len(complex_list)) as pbar:
        pbar.set_description('> Preparation')
        for idx, complex in enumerate(complex_list):
            pbar.update(1)
            if os.path.isfile(f"{PATH_graph}{complex}.pkl"):
                with open(f"{PATH_graph}{complex}.pkl", 'rb') as f:
                    graphs = pickle.load(f)
                    graph_l, graph_p = graphs[0], graphs[1]
            else:
                # load ligand data
                # mol_ligand = Chem.SDMolSupplier(f'{PATH}{complex}/{complex}_ligand.sdf')[0]
                mol_ligand = Chem.MolFromMol2File(f'{PATH}{complex}/{complex}_ligand{opt}.mol2')

                # load residue data
                if not mol_ligand:
                    mol_ligand = Chem.MolFromMol2File(f'{PATH}{complex}/{complex}_ligand{opt}.mol2')
                try:
                    if os.path.isfile(f'{PATH}{complex}/{complex}_residues{opt}.pdb'):
                        mol_residue = Chem.MolFromPDBFile(f'{PATH}{complex}/{complex}_residues{opt}.pdb')
                    else:
                        mol_residue = select_residue(mol_ligand, f'{PATH}{complex}/{complex}_pocket.pdb')
                except:
                    remove_list.append(complex)
                    continue

                # convert to graph
                graph_l = get_molecular_graph(mol_ligand)
                graph_p = get_molecular_graph(mol_residue)

                with open(f"{PATH_graph}{complex}{opt}.pkl", 'wb') as f:
                    pickle.dump((graph_l, graph_p), f, pickle.HIGHEST_PROTOCOL)

            graph_l_list.append(graph_l)
            graph_p_list.append(graph_p)

    for _, item in enumerate(remove_list):
        complex_list.remove(item)
        y_list = y_list[(y_list['#code']!=item)]
    y_list = y_list['logKa'].to_list()

    graph_list = dgl.batch(graph_l_list), dgl.batch(graph_p_list)

    return complex_list, graph_list, y_list


def test(args, test_loader, mlflow):
    set_seed(seed=args.seed)
    device = set_device(use_gpu=args.use_gpu, gpu_idx=args.gpu_idx)

    loss_fn = nn.MSELoss()

    saved_model = torch.load('./checkpoint.pt')

    model = Model_vectorization(num_g_layers=args.num_g_layers,
                                num_d_layers=args.num_d_layers,
                                hidden_dim_g=args.hidden_dim_g,
                                hidden_dim_d=args.hidden_dim_d,
                                readout=args.readout,
                                dropout_prob=args.dropout_prob, )

    model.load_state_dict(saved_model['model_state_dict'])
    model.eval()
    model.to(device)
    model.eval()

    loss_valid = 0
    y_list = []
    pred_list = []

    with tqdm(total=len(test_loader)) as pbar:
        pbar.set_description('> TEST')
        for i, batch in enumerate(test_loader):
            pbar.update(1)

            graph_p, graph_l, y = batch[0], batch[1], batch[2]

            graph_p = graph_p.to(device)
            graph_l = graph_l.to(device)

            y = y.to(device)
            y = y.float()

            output = model(graph_p, graph_l)

            pred = output.squeeze()
            y_list.append(y)
            pred_list.append(output)

            loss = loss_fn(pred, y)

            time.sleep(0.1)

            del graph_p, graph_l, output

        test_metrics = evaluate_regression(y_list=y_list,
                                            pred_list=pred_list)

        rounded_test_metrics = list(map(lambda x: round(x, 3), test_metrics))
        # ci = round(ci, 3)

        dict_result = {'MSE': rounded_test_metrics[0],
                       'MAE': rounded_test_metrics[1],
                       'RMSE': rounded_test_metrics[2],
                       'R2': rounded_test_metrics[3]}

        mlflow.log_meterics(dict_result)

        df_result = pd.DataFrame.from_dict([dict_result]).transpose()
        df_result.columns = ['TEST']

        print(tabulate(df_result, headers='keys', tablefmt='psql', showindex=True))


def ex_test(args):
    # Set random seeds and device
    set_seed(seed=args.seed)
    device = set_device(use_gpu=args.use_gpu, gpu_idx=args.gpu_idx)

    loss_fn = nn.MSELoss()

    saved_model = torch.load('./checkpoint.pt')

    model = Model_vectorization(num_g_layers=args.num_g_layers,
                                num_d_layers=args.num_d_layers,
                                hidden_dim_g=args.hidden_dim_g,
                                hidden_dim_d=args.hidden_dim_d,
                                readout=args.readout,
                                dropout_prob=args.dropout_prob, )

    model.load_state_dict(saved_model['model_state_dict'])
    model.eval()
    model.to(device)
    model.eval()

    complex_list, complex_graph, y_list = preparation()

    print(f'number of test data: {len(complex_list)}')
    pred_list = []
    with torch.no_grad():
        # Test
        loss_test = 0
        num_batches = len(complex_list)


        graph_l = complex_graph[0]
        graph_p = complex_graph[1]

        graph_l = graph_l.to(device)
        graph_p = graph_p.to(device)

        output = model(graph_p, graph_l)

        pred = output.squeeze()
        pred_list.append(pred)

        test_metrics = evaluate_regression(y_list=y_list,
                                           pred_list=pred_list)
        ci = evaluate_regression_test(y_list=y_list, pred_list=pred_list)

        rounded_test_metrics = list(map(lambda x: round(x, 3), test_metrics))
        ci = round(ci, 3)

        dict_result = {'MSE': rounded_test_metrics[0],
                       'MAE': rounded_test_metrics[1],
                       'RMSE': rounded_test_metrics[2],
                       'R2': rounded_test_metrics[3],
                       'CI': ci}

        df_result = pd.DataFrame.from_dict([dict_result]).transpose()
        df_result.columns = ['TEST']

        print('\n',tabulate(df_result, headers='keys', tablefmt='psql', showindex=True))

    output_list = ['#code score\n']
    pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy()

    for i, complex in enumerate(complex_list):
        output_list.append(f'{complex} {pred_list[i]}\n')

    with open(f'./pred/CASF-2016_score_{args.mlflow_runname}.dat', 'w') as f:
        f.writelines(output_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## required conditions for conducting experiment
    parser.add_argument('--use_gpu', type=str, default='1', help='whether to use GPU device')
    parser.add_argument('--gpu_idx', type=str, default='0', help='index to gpu to use')
    parser.add_argument('--seed', type=int, default=999, help='Seed for all stochastic component and data sampling')

    ## hyper-parameters for model structure
    parser.add_argument('--interaction_type', type=str, default='vect', help='Type of interaction layer: dist, vect')
    parser.add_argument('--num_g_layers', type=int, default=4, help='Number of graph layers for ligand featurization')
    parser.add_argument('--num_d_layers', type=int, default=4, help='Number of dense layers for ligand featurization')
    parser.add_argument('--hidden_dim_g', type=int, default=64, help='Dimension of hidden features')
    parser.add_argument('--hidden_dim_d', type=int, default=64, help='Dimension of hidden features')
    parser.add_argument('--readout', type=str, default='mean', help='Readout method, Options: sum, mean, ...')
    parser.add_argument('--dropout_prob', type=float, default=0.0, help='Probability of dropout on node features')
    parser.add_argument('--mlflow_runname', type=str, default='')
    args = parser.parse_args()

    ex_test(args)