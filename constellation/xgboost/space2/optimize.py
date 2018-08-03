import os
import argparse
import numpy as np

import random
import pandas as pd

import xgboost as xgb

from mpi4py import MPI
from hyperspace import hyperdrive

from constellation.data.dataloaders import FashionMNIST


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class Log:
    
    def __init__(self, colnames, savepath, rank):
        self.colnames = colnames
        self.log = pd.DataFrame(columns=self.colnames)
        self.savepath = savepath
        self.rank = rank
        self.iter = 0

    def update(self, name, acc_train, acc_val, ll):
        name = name + str(self.iter)
        entry = pd.DataFrame([[name, acc_train*100, acc_val*100, ll]], columns=self.colnames)
        self.log = self.log.append(entry)
        self.iter += 1

    def save(self):
        filename = 'log' + str(self.rank)
        logfile = os.path.join(self.savepath, filename)
        self.log.to_csv(logfile)


def objective(params):
    """
    Objective function to be minimized.

    Parameters
    ----------
    * params [list, len(params)=n_hyperparameters]
        Settings of each hyperparameter for a given optimization iteration.
        - Controlled by hyperspaces's hyperdrive function.
        - Order preserved from list passed to hyperdrive's hyperparameters argument.
    """
    eta, max_depth = params

    param_list = [
      ("eta", eta), 
      ("max_depth", max_depth),
      ("objective", "multi:softmax"), 
      ("eval_metric", "merror"),
      ("num_class", 10)
    ]

    n_rounds = 2  
    early_stop = 2 

    train = xgb.DMatrix(X_train, label=y_train)
    val = xgb.DMatrix(X_val, label=y_val)

    eval_list = [(train, "train"), (val, "validation")]

    progress = {} 
    
    bst = xgb.train(
      param_list, train, n_rounds, 
      evals=eval_list, evals_result=progress,
      early_stopping_rounds=early_stop, verbose_eval=True
    )
   
    val_metric = progress['validation']
    merror_val = np.array(val_metric['merror'])

    merror_mean = merror_val.mean()
    print(f'Mean validation error: {merror_mean}')

    return merror_mean 


def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--data_path', type=str, default='/Users/youngtodd/constellation/constellation/data/fashion')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Path to save logs')
    args = parser.parse_args()

    # Logging for Visual Comparison
    global logger
    log_cols=["Classifier", "Train Accuracy", "Val Accuracy", "Log Loss"]
    logger = Log(colnames=log_cols, savepath=args.log_dir, rank=rank)

    fashion = FashionMNIST(path=args.data_path, stdscaling=True)
    fashion.create_validation()

    global X_train; global X_val; global X_test; global y_train; global y_val; global y_test
    X_train, y_train, X_val, y_val, X_test, y_test = fashion.get_data()

    hparams = [(0.001, 0.1),  # eta 
               (1, 10)]     # max_depth

    hyperdrive(objective=objective,
               hyperparameters=hparams,
               results_path=args.results_dir,
               model="GP",
               n_iterations=50,
               verbose=True,
               random_state=0,
               checkpoints=True)

    # Save the log data frame
    logger.save()


if __name__ == '__main__':
    main()
