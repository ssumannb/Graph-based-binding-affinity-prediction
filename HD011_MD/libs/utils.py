import argparse
import random       # random으로 숫자뽑는 라이브러리

import math

import mlflow.pytorch
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


def str2bool(v):
    if v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif v.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def set_seed(seed):
    # 모델의 reproducibility를 위해 Seed 고정
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                         # torch.rand(), torch.randint()와 같은 함수들에 대해 seed 적용
    torch.random.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)                # gpu로 만들어내는 결과들의 randomness를 통제
        torch.cuda.manual_seed_all(seed)            # 여러 gpu 사용 시 manual_seed 대신 사용
        torch.backends.cudnn.deterministic = True   # 딥러닝 프레임워크에서 사용되는 것들에 대해 randomness를 조절하기 위함
        torch.backends.cudnn.benchmark = False      # 딥러닝 프레임워크에서 사용되는 것들에 대해 randomness를 조절하기 위함


def set_device(
        use_gpu,
        gpu_idx
    ):
    if use_gpu:
        print('------------------------------------------------------')
        device = torch.device('cuda:'+str(gpu_idx))
        print("Pytorch version:", torch.__version__)
        print("Pytorch GPU count:", torch.cuda.device_count())
        print("Pytorch Current GPU:", device)
        print("Pytorch GPU name:", torch.cuda.get_device_name(device))
        print('------------------------------------------------------')
        return device
    else:
        device = torch.device('cpu')
        return device

def sigmoid(x):
    # torch.nn.functional에서 제공하는 sigmoid는 텐서연산을 위한 sigmoid
    # 해당 sigmoid function은 int를 input으로 받는 sigmoid
    return 1./1.+np.exp(-x)


def calibration(
        label,
        pred,
        bins=10
    ):
    # for calibration curve generation

    width = 1.0 /bins
    bin_center = np.linspace(0, 1.0-width, bins) + width/2

    conf_bin = []
    acc_bin = []
    counts = []

    for i, threshold in enumerate(bin_center):
        bin_idx = np.logical_and(           # np.logical_and(X1, X2): X1,X2논리곱(and)
            threshold - width/2 < pred,
            pred <= threshold + width
        )
        conf_mean = pred[bin_idx].mean()
        conf_sum = pred[bin_idx].sum()

        if (conf_mean != conf_mean) == False:
            conf_bin.append(conf_mean)
            counts.append(pred[bin_idx].shape[0])

        acc_mean = label[bin_idx].mean()
        acc_sum = label[bin_idx].sum()
        if (acc_mean != acc_mean) == False:
            acc_bin.append(acc_mean)

    conf_bin = np.asarray(conf_bin)
    acc_bin = np.asarray(acc_bin)
    counts = np.asarray(counts)

    ece = np.abs(conf_bin - acc_bin)
    ece = np.multiply(ece, counts)
    ece = ece.sum()
    ece /= np.sum(counts)

    return conf_bin, acc_bin, ece

def evaluate_classification(
        y_list,
        pred_list
    ):
    y_list = torch.cat(y_list, dim=0).detach().cpu().numpy()
    pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy()

    auroc = roc_auc_score(y_list, pred_list)
    _, _, ece = calibration(y_list, pred_list)

    '''
    To calculate metric in the below,
    scores should be presented in integer type
    because this evaluate the classification performance
    '''
    y_list = y_list.astype(int)
    pred_list = np.around(pred_list).astype(int)

    accuracy = accuracy_score(y_list, pred_list)
    precision = precision_score(y_list, pred_list)
    recall = recall_score(y_list, pred_list)
    f1 = 2.0 * precision * recall / (precision + recall)

    return accuracy, auroc, precision, recall, f1, ece


def evaluate_regression(
      y_list,
      pred_list
    ):
    try:
        y_list = torch.cat(y_list, dim=0).detach().cpu().numpy()
    except:
        y_list = np.array(y_list)

    try:
        pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy()
    except:
        pred_list = np.array(pred_list)

    mse = mean_squared_error(y_list, pred_list)
    mae = mean_absolute_error(y_list, pred_list)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_list, pred_list)


    # pearson_r = pearsonr(y_list, pred_list)
    # f = pred_list.shape[0]
    # pred_list_1D = np.reshape(pred_list, pred_list.shape[0])
    y_list_reshaped = np.reshape(y_list, (1, y_list.shape[0]))

    return mse, mae, rmse, r2

def evaluate_regression_test(
        y_list,
        pred_list
    ):
    try:
        y_list = torch.cat(y_list, dim=0).detach().cpu().numpy()
    except:
        y_list = np.array(y_list)

    try:
        pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy()
    except:
        pred_list = np.array(pred_list)

    sns.set_theme(color_codes=True)
    axis_x = np.array(pred_list)
    axis_y = np.array(y_list)

    sns.regplot(x=axis_x, y=axis_y)
    plt.savefig(f'../BA_prediction_ignore/prediction_plot.png', dpi=300)

    # CI index
    summ = 0
    pair = 0

    for i in range(1, len(y_list)):
        for j in range(0, i):
            if i is not j:
                if (y_list[i] > y_list[j]):
                    pair += 1
                    summ += 1 * (pred_list[i] > pred_list[j]) + 0.5 * (pred_list[i] == pred_list[j])

    if pair is not 0:
        ci =  summ/pair
    else:
        ci =  0

    return ci

class EarlyStopping:
    def __init__(
            self,
            patience=7,
            verbose=False,
            delta=0.01,
            path='checkpoint.pt',
            trace_func=print):
        '''
        Early stops the training if validation loss doesn't improve after a given patience
        :param patience: (int) How long to wait after last time validation loss improved.
        :param verbose: (bool): If True, prints a message for each validation loss improvement.
        :param delta: (float): Minimum change in the monitored quantity to qualify as an improvement.
        :param path: (str): Path for the checkpoint to be saved to.
        :param trace_func: (function): trace print function.
        '''
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(
            self,
            val_loss,
            model,
            optimizer,
            epoch,
            mlflow
        ):

        score = -val_loss

        if self.best_score is None:
            self.best_score= score
            self.save_checkpoint(val_loss, model, optimizer, epoch, mlflow)
            self.verbose = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, mlflow)
            self.counter = 0

    def save_checkpoint(
            self,
            val_loss,
            model,
            optimizer,
            epoch,
            mlflow
        ):
        '''
        Saves model when validation loss decrease
        '''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f}-->{val_loss:.6f}).  Saving model...')
        torch.save(model, self.path)

        torch.save({'epoch': epoch,
                    'optimizer_state_dict':optimizer.state_dict(),
                    'model_state_dict':model.state_dict(),},
                    self.path)
        mlflow.pytorch.log_model(model, 'Model')

        self.val_loss_min = val_loss