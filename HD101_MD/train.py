import time
import argparse
from tqdm import tqdm
from tabulate import tabulate
import mlflow
from itertools import chain
from collections import defaultdict

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from libs.io_utils import get_complex_list
from libs.io_utils import Complex_Dataset
from libs.io_utils import collate_func
# from libs.io_utils import collate_func_eTest

from libs.models import Model_vectorization

from libs.utils import str2bool
from libs.utils import set_seed
from libs.utils import set_device
from libs.utils import evaluate_regression
from libs.utils import evaluate_regression_test
from libs.utils import EarlyStopping

from test import test

# 오류 출력 명확하게 하기 위해 환경변수 설정
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
writer = SummaryWriter()

def mlflow_decorator(func):
    def decorated(*args, **kwargs):
        mlflow.log_param('max epoch', args[0].num_epochs)
        mlflow.log_param('batch_size', args[0].batch_size)
        mlflow.log_param('num workers', args[0].num_workers)
        mlflow.log_param('optimizer', args[0].optimizer)
        mlflow.log_params({'interaction type': args[0].interaction_type,
                           'number of graph learning layer': args[0].num_g_layers,
                           'number of interaction calculating layer': args[0].num_d_layers,
                           'number of hidden dimension of graph layers': args[0].hidden_dim_g,
                           'number of hidden dimension of dense layers': args[0].hidden_dim_d,
                           'readout method': args[0].readout,
                           'dropout probability': args[0].dropout_prob,
                           })
        func(*args, **kwargs)
        writer.flush()
        writer.close()
        mlflow.end_run()

    return decorated


def preparation(args, mlflow):
    # Prepare datasets and dataloaders
    if args.cross_validation == False:
        dataset_list = get_complex_list(
            data_seed=args.seed,
            frac=[0.7, 0.2, 0.1],
            cv=args.cross_validation,
            af_type = args.af_type
        )
    else:
        dataset_list = get_complex_list(
            data_seed=args.seed,
            frac=[0.2, 0.2, 0.2, 0.2, 0.2],
            cv=args.cross_validation,
            af_type=args.af_type
        )

    dataset = []
    for i, _ in enumerate(dataset_list):
        dataset.append(Complex_Dataset(splitted_set=dataset_list[i], is_coreset=False))

    dataset_loader = []
    for i, _ in enumerate(dataset):
        dataset_loader.append(DataLoader(dataset=dataset[i], batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, collate_fn=collate_func))

    if args.cross_validation == False:
        mlflow.log_param('data_length_train', dataset[0].__len__())
        mlflow.log_param('data_length_valid', dataset[1].__len__())
        mlflow.log_param('data_length_test', dataset[2].__len__())

    else:
        mlflow.log_param('data_length_train','cv')
        mlflow.log_param('data_length_valid','cv')
        mlflow.log_param('data_length_test', 'cv')
        # mlflow.log_param('data_length_eTest', dataset_eTest.__len__())

    return_value = dataset_loader
    del dataset_list, dataset

    return return_value


def batch_iter(data_loader, pbar, model, optimizer, loss_fn, device):
    loss_train = 0
    y_list = []
    pred_list = []

    for i, batch in enumerate(data_loader):
        pbar.update(1)

        optimizer.zero_grad()

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
        loss.backward()

        # loss function을 효율적으로 최소화할 수 있도록 파라미터 수
        optimizer.step()

        loss_train += loss.detach().cpu().numpy()

        # time.sleep(0.1)

        del graph_p, graph_l, output

    return y_list, pred_list, loss_train


@mlflow_decorator
def train(args, data_loader, mlflow):

    # Set random seeds and device
    set_seed(seed=args.seed)
    device = set_device(use_gpu=args.use_gpu, gpu_idx=args.gpu_idx)

    train_loader = data_loader[0]
    valid_loader = data_loader[1]

    # Construct model and load trained parameters if it is possible
    starting_point = 0
    model = Model_vectorization(num_g_layers=args.num_g_layers,
                                num_d_layers=args.num_d_layers,
                               hidden_dim_g=args.hidden_dim_g,
                               hidden_dim_d=args.hidden_dim_d,
                               readout=args.readout,
                               dropout_prob=args.dropout_prob,)
    model.to(device)

    # optimizer and learning rate scheduler setting
    if args.optimizer.lower() == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(),
                                        lr=args.lr,
                                        weight_decay=args.weight_decay,)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                step_size=40,
                                                gamma=0.1,)

    if args.load_saved_model == True:
        saved_model = torch.load('./checkpoint.pt')
        starting_point = int(saved_model['epoch'])
        model.load_state_dict(saved_model['model_state_dict'])
        optimizer.load_state_dict(saved_model['optimizer_state_dict'])
        model.eval()



    # loss function
    loss_fn = nn.MSELoss()

    # set EarlyStopping class
    early_stopping = EarlyStopping(verbose=True)

    train_result_list = []
    valid_result_list = []

    # training start
    for epoch in range(args.num_epochs):
        epoch += starting_point
        print(f'\n{epoch+1} EPOCH'.zfill(3))

        # set model to training state
        model.train()

        # variables for training
        if args.cross_validation == True:
            num_batches = 0
            for i, item in enumerate(train_loader):
                num_batches += len(item)
        else:
            num_batches = len(train_loader)

        loss_train = 0
        y_list = []
        pred_list = []
        # time.sleep(0.1)

        with tqdm(total=num_batches) as pbar:
            pbar.set_description("> TRAIN")
            if args.cross_validation == True:
                for _, fold in enumerate(train_loader):
                    for i, batch in enumerate(fold):
                        pbar.update(1)

                        optimizer.zero_grad()

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
                        loss.backward()

                        # loss function을 효율적으로 최소화할 수 있도록 파라미터 수
                        optimizer.step()

                        loss_train += loss.detach().cpu().numpy()

                        # time.sleep(0.1)

                        del graph_p, graph_l, output
            else:
                for i, batch in enumerate(train_loader):
                    pbar.update(1)

                    optimizer.zero_grad()

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
                    loss.backward()

                    # loss function을 효율적으로 최소화할 수 있도록 파라미터 수
                    optimizer.step()

                    loss_train += loss.detach().cpu().numpy()

                    # time.sleep(0.1)

                    del graph_p, graph_l, output

            scheduler.step()

            loss_train /= num_batches
            writer.add_scalar("Loss/Train", loss_train, epoch + 1)
            mlflow.log_metric(key="Loss/Train", value=loss_train, step=epoch)

            train_metrics = evaluate_regression(y_list=y_list,
                                                pred_list=pred_list)


        # set model to validation state and stop the gradient calculiation
        model.eval()
        with torch.no_grad():

            # Validation start for identifying the early stop point
            loss_valid = 0
            num_batches = len(valid_loader)
            y_list = []
            pred_list = []

            with tqdm(total=num_batches) as pbar:
                pbar.set_description('> VALID')
                for i, batch in enumerate(valid_loader):
                    pbar.update(1)

                    graph_p, graph_l, y = batch[0], batch[1], batch[2]
                    if args.interaction_type == 'dist':
                        coord = np.array(batch[3])

                    graph_p = graph_p.to(device)
                    graph_l = graph_l.to(device)

                    y = y.to(device)
                    y = y.float()

                    if args.interaction_type == 'dist':
                        output = model(graph_p, graph_l, coord)
                    else:
                        output = model(graph_p, graph_l)

                    pred = output.squeeze()
                    y_list.append(y)
                    pred_list.append(output)

                    loss = loss_fn(pred, y)
                    loss_valid += loss.detach().cpu().numpy()

                    # time.sleep(0.1)

                    del graph_p, graph_l, output

                loss_valid /= num_batches

                writer.add_scalar("Loss/Valid", loss_valid, epoch + 1)
                mlflow.log_metric(key="Loss/Valid", value=loss_valid, step=epoch)

                valid_metrics = evaluate_regression(y_list=y_list,
                                                    pred_list=pred_list)

                early_stopping(loss_valid, model, optimizer, epoch, mlflow)


        # REGRESSION     EVALUATION CRITERIA(3): mse, rmse, r2
        rounded_train_metrics = list(map(lambda x: round(x, 3), train_metrics))
        rounded_valid_metrics = list(map(lambda x: round(x, 3), valid_metrics))

        dict_result_train = {'MSE': rounded_train_metrics[0],
                               'MAE': rounded_train_metrics[1],
                               'RMSE': rounded_train_metrics[2],
                               'R2': rounded_train_metrics[3]}

        dict_result_valid = {'MSE': rounded_valid_metrics[0],
                               'MAE': rounded_valid_metrics[1],
                               'RMSE': rounded_valid_metrics[2],
                               'R2': rounded_valid_metrics[3]}

        dict_result = defaultdict(list)

        for k, v in chain(dict_result_train.items(), dict_result_valid.items()):
            dict_result[k].append(v)

        # mlflow.log_metrics(dict_result_train)
        mlflow.log_metrics(dict_result_valid)

        df_result = pd.DataFrame(dict_result).transpose()
        df_result.columns = ['TRAIN', 'VALID']

        print(tabulate(df_result, headers='keys', tablefmt='psql', showindex=True))

        if early_stopping.early_stop:
            mlflow.log_param('Early Stop epoch', epoch)
            print("Early stopping")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## required conditions for conducting experiment
    parser.add_argument('--job_title', type=str, default='train', help='Job titile of this execution')
    parser.add_argument('--use_gpu', type=str, default='1', help='whether to use GPU device')
    parser.add_argument('--gpu_idx', type=str, default='0', help='index to gpu to use')
    parser.add_argument('--seed', type=int, default=999, help='Seed for all stochastic component and data sampling')

    ## hyper-parameters for dataset
    parser.add_argument('--dataset_name', type=str, default='PDBbind 2020v', help='What dataset to use for model development')
    parser.add_argument('--split_method', type=str, default='random', help='How to split dataset')
    parser.add_argument('--af_type', type=str, default='kdki', help='Type of binding affinity value')

    ## hyper-parameters for model structure
    parser.add_argument('--interaction_type', type=str, default='vect', help='Type of interaction layer: dist, vect')
    parser.add_argument('--num_g_layers', type=int, default=4, help='Number of graph layers for ligand featurization')
    parser.add_argument('--num_d_layers', type=int, default=4, help='Number of dense layers for ligand featurization')
    parser.add_argument('--hidden_dim_g', type=int, default=64, help='Dimension of hidden features')
    parser.add_argument('--hidden_dim_d', type=int, default=64, help='Dimension of hidden features')
    parser.add_argument('--readout', type=str, default='mean', help='Readout method, Options: sum, mean, ...')
    parser.add_argument('--dropout_prob', type=float, default=0.0, help='Probability of dropout on node features')

    # vectorization layer : {num_workers - 4 / batch_size - 8}
    # distance layer : {num_workers - 1 / batch_size - 4 }
    ## hyper-parameters for model training
    parser.add_argument('--optimizer', type=str, default='adam', help='Options: adam, sgd, ...')
    parser.add_argument('--num_epochs', type=int, default=999, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers to run dataloaders')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of samples in a single batch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay coefficient')
    parser.add_argument('--cross_validation', type=bool, default=False, help='Cross validate a model')

    parser.add_argument('--save_model', type=str2bool, default=True, help='Whether to save model')
    parser.add_argument('--save_path', type=str, default='../BA_prediction_ignore/save/', help='Path for saving model')
    parser.add_argument('--save_loader', type=str, default='../BA_prediction_ignore/data/loader/', help='Path for saving data loader')
    parser.add_argument('--load_saved_model', type=bool, default=False)

    args = parser.parse_args()

    print("Arguments")

    print('------------------------------------------------------')
    for k, v in vars(args).items():
        print(k, ": ", v)

    # mlflow experiment setting
    version = 2  # 초기 버전 : vectorization 기반 모델
    experiment_name = f'BA_prediction({version})'
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        data_loader_all = preparation(args, mlflow)

        if args.cross_validation == False:
            data_loader = data_loader_all[0], data_loader_all[1]

            train(args, data_loader, mlflow)
        else:
            iter_loader = data_loader_all
            '''
             'batch_size',
             'optimizer',
             'number of graph learning layer',
             'number of interaction calculating layer',
             'number of hidden dimension of layers',
             'readout method',
             'dropout probability'
            '''

            hyperparams_list = [[64, 'adam', 4, 3, 64, 64, 'sum', 0.0],
                                [64, 'adam', 4, 3, 64, 64, 'mean', 0.0],
                                [64, 'adam', 4, 4, 64, 64, 'sum', 0.0],
                                [64, 'adam', 4, 3, 64, 64, 'sum', 0.2],
                                [64, 'adam', 4, 3, 64, 128, 'sum', 0.2]]


            # for i, _ in enumerate(iter_loader):
            #     train_loader = iter_loader[0:i] + iter_loader[i+1:]
            #     valid_loader = iter_loader[i]
            #
            #     data_loader = train_loader, valid_loader
            #
            #     # argument setting for k-fold valid
            #     args.batch_size = hyperparams_list[i][0]
            #     args.optimizer = hyperparams_list[i][1]
            #     args.num_g_layer = hyperparams_list[i][2]
            #     args.num_d_layer = hyperparams_list[i][3]
            #     args.hidden_dim_g = hyperparams_list[i][4]
            #     args.hidden_dim_d = hyperparams_list[i][5]
            #     args.readout = hyperparams_list[i][6]
            #     args.dropout_prob = hyperparams_list[i][7]
            #
            #     train(args, data_loader, mlflow)
            #     writer.flush()
            #     writer.close()
            #
            #     args.mlflow_runname = mlflow.runName
        test(args, data_loader_all[2], mlflow)
        test(args, data_loader_all[3], mlflow)

