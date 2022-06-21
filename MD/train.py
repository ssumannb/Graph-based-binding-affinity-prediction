import argparse
from tqdm import tqdm
from tabulate import tabulate
import mlflow

import os
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from libs.io_utils import get_complex_list
from libs.io_utils import Complex_Dataset
from libs.io_utils import collate_func

from libs.models import Model

from libs.utils import str2bool
from libs.utils import set_seed
from libs.utils import set_device
from libs.utils import evaluate_regression
from libs.utils import EarlyStopping
from libs.utils import draw_loss_curve
from libs.utils import get_mlflow_exp_id

# 오류 출력 명확하게 하기 위해 환경변수 설정
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
writer = SummaryWriter()


def prepare_data(args, mlflow):
    # Prepare datasets and dataloaders
    dataset_list = get_complex_list(
        data_seed=args.seed,
        frac=[0.7, 0.2, 0.1],
        dataset_name=args.dataset_name)

    dataset = []
    for i, _ in enumerate(dataset_list):
        dataset.append(Complex_Dataset(splitted_set=dataset_list[i]))

    dataset_loader = []
    for i, _ in enumerate(dataset):
        dataset_loader.append(DataLoader(dataset=dataset[i], batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, collate_fn=collate_func))

    mlflow.log_param('Data params - dataset name', args.dataset_name)
    mlflow.log_param('Data params - split method', args.split_method)
    mlflow.log_param('Data params - trainset length', dataset[0].__len__())
    mlflow.log_param('Data params - validation length', dataset[1].__len__())
    mlflow.log_param('Data params - test length', dataset[2].__len__())

    # return_value = dataset_loader
    del dataset_list, dataset

    return dataset_loader


def prepare_model(args, mlflow):
    # set device and random seed for model
    mlflow.log_params({'Model params - G.L. layer number': args.num_g_layers,
                       'Model params - A.P. layer number': args.num_d_layers,
                       'Model params - G.L. layer dimension': args.hidden_dim_g,
                       'Model params - A.P.layer dimension': args.hidden_dim_d,
                       'Model params - readout method': args.readout,
                       })

    set_seed(seed=args.seed)
    device = set_device(use_gpu=args.use_gpu, gpu_idx=args.gpu_idx)

    # Construct model and load trained parameters if it is possible
    starting_point = 0
    model = Model(num_g_layers=args.num_g_layers,
                  num_d_layers=args.num_d_layers,
                  hidden_dim_g=args.hidden_dim_g,
                  hidden_dim_d=args.hidden_dim_d,
                  readout=args.readout,
                  dropout_prob=args.dropout_prob, )
    model.to(device)

    # optimizer and learning rate scheduler setting
    opt = args.optimizer.lower()
    if opt == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.weight_decay, )
    elif opt == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay, )
    elif opt == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.weight_decay, )
    elif opt == 'radam':
        optimizer = torch.optim.RAdam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.weight_decay, )
    elif opt == 'adamax':
        optimizer = torch.optim.RAdam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.weight_decay, )

    # set a scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                step_size=40,
                                                gamma=0.1, )

    # load saved model if applicable
    if args.load_saved_model == True:
        saved_model = torch.load('./checkpoint.pt')
        starting_point = int(saved_model['epoch'])
        model.load_state_dict(saved_model['model_state_dict'])
        optimizer.load_state_dict(saved_model['optimizer_state_dict'])
        model.eval()

    # define loss function
    loss_fn = nn.MSELoss()

    # set EarlyStopping class
    early_stopping = EarlyStopping(verbose=True)

    return model, optimizer, loss_fn, scheduler, early_stopping, device


def learning(data_loader, pbar, model, optimizer, loss_fn, device):
    loss_sum = 0
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

        y_list.append(y)
        pred_list.append(output)

        pred = output.squeeze()
        loss = loss_fn(pred, y)

        if model.training == True:
            loss.backward()
            optimizer.step()

        loss_sum += loss.detach().cpu().numpy()

        del graph_p, graph_l, output

    return model, optimizer, loss_sum, y_list, pred_list


def evaluate(data_loader, pbar, model, device):
    y_list = []
    pred_list = []

    for i, batch in enumerate(data_loader):
        pbar.update(1)

        graph_p, graph_l, y = batch[0], batch[1], batch[2]

        graph_p = graph_p.to(device)
        graph_l = graph_l.to(device)

        y = y.to(device)
        y = y.float()

        output = model(graph_p, graph_l)

        y_list.append(y)
        pred_list.append(output)

        pred = output.squeeze()

        del graph_p, graph_l, output

    return y_list, pred_list


def print_metric(metrics, title, mlflow=None):
    rounded_train_metrics = list(map(lambda x: round(x, 3), metrics))
    dict_result = {'MSE': rounded_train_metrics[0],
                     'MAE': rounded_train_metrics[1],
                     'RMSE': rounded_train_metrics[2],
                     'R2': rounded_train_metrics[3]}
    if mlflow != None:
        mlflow.log_metrics(dict_result)

    df_result = pd.DataFrame(dict_result, index=[0]).transpose()
    df_result.columns = [title]

    print(f"\n{tabulate(df_result, headers='keys', tablefmt='psql', showindex=True)}")



def train(args, data_loader, mlflow, train_configs):
    experiment_id = get_mlflow_exp_id(args.experiment_name)

    model = train_configs[0]
    optimizer = train_configs[1]
    loss_fn = train_configs[2]
    scheduler = train_configs[3]
    early_stopping = train_configs[4]
    device = train_configs[5]

    # divide dataloader
    train_loader = data_loader[0]
    valid_loader = data_loader[1]

    train_losses = []
    valid_losses = []

    for epoch in range(args.num_epochs):
        print(f'\n{epoch+1} EPOCH'.zfill(3))

        # set model to training state
        model.train()

        # Training
        num_train_batch = len(train_loader)
        with tqdm(total=num_train_batch) as pbar:
            pbar.set_description("> TRAIN")
            model, optimizer, loss_train, y_list, pred_list \
                = learning(train_loader, pbar, model, optimizer, loss_fn, device)
            scheduler.step()

            loss_train /= num_train_batch
            train_losses.append(loss_train)
            writer.add_scalar("Loss/Train", loss_train, epoch + 1)
            train_metrics = evaluate_regression(y_list=y_list,pred_list=pred_list)

        # print_metric(train_metrics, mlflow, title='Train')

        # Validation
        # set model as validation state and stop the gradient calculiation
        num_valid_batch = len(valid_loader)
        with tqdm(total=num_valid_batch) as pbar:
            model.eval()
            with torch.no_grad():
                pbar.set_description('> VALID')
                model, optimizer, loss_valid, y_list, pred_list \
                    = learning(valid_loader, pbar, model, optimizer, loss_fn, device)

                loss_valid /= num_valid_batch
                valid_losses.append(loss_valid)
                writer.add_scalar("Loss/Valid", loss_valid, epoch + 1)
                valid_metrics = evaluate_regression(y_list=y_list, pred_list=pred_list)

        # print_metric(valid_metrics, mlflow, title='Valid')
        early_stopping(loss_valid, model, optimizer, epoch, mlflow)

        if early_stopping.early_stop:
            mlflow.log_param('Early Stop epoch', epoch)
            print("Early stopping")
            break

    draw_loss_curve(train_losses, valid_losses, id=experiment_id, label_list=['train', 'valid'])


def test(args, test_loader, mlflow, test_configs):

    model = test_configs[0]
    device = test_configs[5]

    model.to(device)

    # load saved model
    trained_model = torch.load('./checkpoint.pt')
    model.load_state_dict(trained_model['model_state_dict'])
    model.eval()

    with torch.no_grad():
        # test start for identifying the early stop point
        num_batches = len(test_loader)

        with tqdm(total=num_batches) as pbar:
            pbar.set_description('> TEST')
            y_list, pred_list = evaluate(test_loader, pbar, model, device)
            test_metrics = evaluate_regression(y_list=y_list, pred_list=pred_list)

    print_metric(test_metrics, title='Test', mlflow=mlflow)



def argument_define(parser):
    ## required conditions for conducting experiment
    parser.add_argument('--job_title', type=str, default='train', help='Job titile of this execution')
    parser.add_argument('--use_gpu', type=str, default='1', help='whether to use GPU device')
    parser.add_argument('--gpu_idx', type=str, default='0', help='index to gpu to use')
    parser.add_argument('--seed', type=int, default=999, help='Seed for all stochastic component and data sampling')
    parser.add_argument('--experiment_name', type=str, default='HD011', help='Experiment id used in mlflow')

    ## hyper-parameters for dataset
    parser.add_argument('--dataset_name', type=str, default='PDBbind 2020', help='What dataset to use for model development')
    parser.add_argument('--split_method', type=str, default='random', help='How to split dataset')

    ## hyper-parameters for model structure
    # parser.add_argument('--interaction_type', type=str, default='vect', help='Type of interaction layer: dist, vect')
    parser.add_argument('--num_g_layers', type=int, default=4, help='Number of graph layers for ligand featurization')
    parser.add_argument('--num_d_layers', type=int, default=5, help='Number of dense layers for ligand featurization')
    parser.add_argument('--hidden_dim_g', type=int, default=96, help='Dimension of hidden features')
    parser.add_argument('--hidden_dim_d', type=int, default=128, help='Dimension of hidden features')
    parser.add_argument('--readout', type=str, default='sum', help='Readout method, Options: sum, mean, ...')

    # hyper-parameters for experiment
    ## vectorization layer : {num_workers - 4 / batch_size - 8}
    ## distance layer : {num_workers - 1 / batch_size - 4 }
    parser.add_argument('--optimizer', type=str, default='nadam', help='Options: adam, sgd, ...')
    parser.add_argument('--num_epochs', type=int, default=999, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers to run dataloaders')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of samples in a single batch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay coefficient')
    parser.add_argument('--dropout_prob', type=float, default=0.0, help='Probability of dropout on node features')

    # hyper-parameters for model saving
    parser.add_argument('--save_model', type=str2bool, default=True, help='Whether to save model')
    parser.add_argument('--load_saved_model', type=bool, default=False)

    args = parser.parse_args()

    return args


def argument_print(args):
    print("Arguments")

    print('------------------------------------------------------')
    for k, v in vars(args).items():
        print(k, ": ", v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = argument_define(parser)
    argument_print(args)

    experiment_name = args.experiment_name
    mlflow.set_experiment(experiment_name)
    # data_loader_all = None

    with mlflow.start_run() as run:
        mlflow.log_param('Experiment params - optimizer', args.optimizer)
        mlflow.log_param('Experiment params - num workers', args.num_workers)
        mlflow.log_param('Experiment params - batch size', args.batch_size)
        mlflow.log_param('Experiment params - learning rate', args.lr)
        mlflow.log_param('Experiment params - weight decay', args.weight_decay)
        mlflow.log_param('Experiment params - dropout rate', args.dropout_prob)

        data_loader_all = prepare_data(args, mlflow)
        train_data_loader = data_loader_all[0], data_loader_all[1]
        test_data_loader = data_loader_all[2]

        model_configs = prepare_model(args, mlflow)
        train(args=args, data_loader=train_data_loader, mlflow=mlflow, train_configs=model_configs)
        test(args=args, test_loader=test_data_loader, mlflow=mlflow, test_configs=model_configs)

        writer.flush()
        writer.close()

    mlflow.end_run()


