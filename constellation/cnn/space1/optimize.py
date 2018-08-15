import argparse
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import math as mt

from mpi4py import MPI
from hyperspace import hyperdrive
from hyperspace.kepler import load_results
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense , Dropout , Flatten
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras import backend as K
K.set_image_dim_ordering('th')

# preprocessing/decomposition
from sklearn.preprocessing import StandardScaler
# model evaluation
from sklearn.model_selection import KFold


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# define path to save model
model_path_cnn = './output/fm_cnn_model.h5'

# training configuration
batch_size = 400
epochs = 1 
# prepare callbacks
callbacks = [
    EarlyStopping(
        monitor='val_acc', 
        patience=10,
        mode='max',
        verbose=1)
#    ModelCheckpoint(
#        model_path_cnn, 
#        monitor='val_acc', 
#        save_best_only=True, 
#        mode='max',
#        verbose=0)
]


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def mean(numbers):
    """Return the sample arithmetic mean of data."""
    return float(sum(numbers)) / max(len(numbers), 1)


def sum_of_square_deviation(numbers,mean):
    """Return sum of square deviations of sequence data."""
    return float(1/len(numbers) * sum((x - mean)** 2 for x in numbers)) 


def model_cnn(num_classes=10, kernel1=5, kernel2=5, kernel3=1, lr=0.01):
    # create model
    model = Sequential()
    model.add(Conv2D(32, (kernel1, kernel1), input_shape=(1, 28, 28), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    
    model.add(Conv2D(64, (kernel2, kernel2), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        
    model.add(Conv2D(128, (kernel3, kernel3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        
    model.add(Flatten())
    
    model.add(Dense(1024, activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    lrate = lr 
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


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
    lr = params[0]

    acc_scores = []
    for fold, (train_index, test_index) in enumerate(kf.split(X_train)):
        print('\n Fold %d'%(fold))

        X_tr, X_v = X_train[train_index], X_train[test_index]
        y_tr, y_v = y_train[train_index], y_train[test_index]
        # build the model
        model = model_cnn(lr=lr)
        # fit model
        model.fit(
            X_tr,
            y_tr,
            epochs=epochs,
            validation_data=(X_v, y_v),
            verbose=1,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=True
        )

        acc = model.evaluate(X_v, y_v, verbose=0)
        acc_scores.append(acc[1])

        print('Fold %d: Accuracy %.2f%%'%(fold, acc[1]*100))

    print('Accuracy scores: ', acc_scores)

    mean_acc = mean(acc_scores)
    standard_deviation_acc = mt.sqrt(sum_of_square_deviation(acc_scores,mean_acc))

    print('=====================')
    print( 'Mean Accuracy %f'%mean_acc)
    print('=====================')
    print('=====================')
    print( 'Stdev Accuracy %f'%standard_deviation_acc)
    print('=====================')

    # Final evaluation of the model
    scores = model.evaluate(X_val, y_val, verbose=0)
    loss_val = 100-scores[1]*100
    acc_val = scores[1]*100

    print("Error: %.2f%%" % loss_val)
    print("Accuracy: %.2f%%" % acc_val)

    name = 'CNN'
    logger.update(name, mean_acc, acc_val, loss_val)
    return loss_val

    
def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--results_dir', type=str, default='./results', help='Path to results directory.')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Path to save logs')
    parser.add_argument('--data', type=str, default='/lustre/atlas/proj-shared/csc249/yngtodd/data/fashion', help='Path to data')
    args = parser.parse_args()
    
    # k-fold configuration
    n_splits = 2

    # get data
    trainpath = os.path.join(args.data, 'fashion-mnist_train.csv')
    testpath = os.path.join(args.data, 'fashion-mnist_test.csv')

    test  = pd.read_csv(trainpath)
    train = pd.read_csv(testpath)

    global logger
    log_cols=["Classifier", "Train Accuracy (Mean)", "Val Accuracy", "Loss"]
    logger = Log(colnames=log_cols, savepath=args.log_dir, rank=rank)

    global y_train_CNN, X_train_CNN
    y_train_CNN = train.ix[:,0].values.astype('int32') # only labels i.e targets digits
    X_train_CNN = np.array(train.iloc[:,1:].values).reshape(train.shape[0], 1, 28, 28).astype(np.uint8)# reshape to be [samples][pixels][width][height]
   
    global y_test_CNN, X_test_CNN
    y_test_CNN = test.ix[:,0].values.astype('int32') # only labels i.e targets digits
    X_test_CNN = np.array(test.iloc[:,1:].values).reshape((test.shape[0], 1, 28, 28)).astype(np.uint8)
    
    # normalize inputs from 0-255 to 0-1
    X_train_CNN = X_train_CNN / 255
    X_test_CNN = X_test_CNN / 255
    
    global X_train, X_val, y_train, y_val
    X_train, X_val, y_train, y_val = \
            train_test_split(X_train_CNN, y_train_CNN, test_size=0.33, random_state=42)
    
    # one hot encode outputs
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test_CNN)
    num_classes = y_train.shape[1]
   
    global kf
    kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
    kf.get_n_splits(X_train)

    hparams = [
      (0.0001, 0.1)  # lr
    ]

    hyperdrive(objective=objective,
               hyperparameters=hparams,
               results_path=args.results_dir,
               model="GP",
               n_iterations=100,
               verbose=True,
               random_state=0,
               checkpoints=True)

    # Save the log data frame
    logger.save()


if __name__=="__main__":
    main()
