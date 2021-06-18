import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace, Dataset, Experiment, Run
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
import tensorflow.keras as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
 


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, dest='dataset', help='dataset')
parser.add_argument('--model_type', type=str, dest='model_type', help='model_type')
parser.add_argument('--n_epochs', type=int, dest='n_epochs', default=100, help='number of epochs')
parser.add_argument('--batch_size', type=int, dest='batch_size', default=64, help='batch_size')
parser.add_argument('--n_layers', type=int, dest='n_layers', default=1, help='number of layers')
parser.add_argument('--activation', type=int, dest='activation', default=100, help='number of neurons')
parser.add_argument('--window', type=int, dest='window', default=12, help='window')
parser.add_argument('--dropout', type=float, dest='dropout', default=0.0, help='dropout')
args = parser.parse_args()
dataset = args.dataset
model_type = args.model_type
n_epochs = args.n_epochs
batch_size = args.batch_size
n_layers = args.n_layers
activation = args.activation
window = args.window
dropout = args.dropout
run = Run.get_context()
workspace = run.experiment.workspace
dataset_prepared = Dataset.get_by_name(workspace=workspace, name=dataset, version='latest')
df = dataset_prepared.to_pandas_dataframe()
df = df.set_index('Date', inplace=False)


def data_prepared(df, window):
    X, Y =[], []
    for i in range(len(df)-window):
        d=i+window
        X.append(df[i:d,])
        Y.append(df[d,])
    return np.array(X), np.array(Y)


def model_loss(history):
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train_loss')
    plt.plot(history.history['val_loss'], label='Test_loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    sns.despine(top=True)
    plt.show()
    run.log_image('epoch_loss', plot=plt)

    return

def learning_curve(y_test, yhat,window):
    len_prediction=df.index[window:]
    plt.figure(figsize=(8,4))
    plt.plot(len_prediction, y_test[:], marker='.', label="actual")
    plt.plot(len_prediction, yhat[:], 'r', label="prediction")
    plt.tight_layout()
    sns.despine(top=True)
    plt.ylabel('Passenger Count', size=12)
    plt.xlabel('Time step', size=12)
    plt.legend(fontsize=12)    
    plt.show()
    run.log_image('Actual vs Predicted', plot=plt)

    return

class LogRunMetrics(Callback):
        def on_epoch_end(self, epoch, log):
            run.log('Loss', log['loss'])
            run.log('val_loss', log['val_loss'])


train_size = 160
def lstm(window, n_layers, activation, dropout):
    model=Sequential()    
    model.add(LSTM(activation, activation='relu', input_shape=(1,window), dropout=dropout))
    for i in range(n_layers):
        model.add(Dense(activation, activation='relu'))
    model.add(Dense(1))    
    model.compile(loss='mean_squared_error',  optimizer='adam',metrics = ['mse', 'mae'])    
    return model

if model_type == 'LSTM':
    df_arr= df.values
    df_arr = np.reshape(df_arr, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_arr = scaler.fit_transform(df_arr)
    test_size = len(df_arr)-train_size
    train, test = df_arr[0:train_size,:], df_arr[train_size:len(df_arr),:]
    x_train, y_train = data_prepared(train, window)
    x_test, y_test = data_prepared(test, window)
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    model=lstm(window, n_layers, activation, dropout)
    history = model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(x_test, y_test),
                        verbose=0, shuffle=False, callbacks=[LogRunMetrics()])
    train_predict = model.predict(x_train)
    yhat = model.predict(x_test)
    train_predict = scaler.inverse_transform(train_predict)
    y_train = scaler.inverse_transform(y_train)
    yhat = scaler.inverse_transform(yhat)
    y_test = scaler.inverse_transform(y_test)
    run.log('MAE_train', mean_absolute_error(y_train[:,0], train_predict[:,0]))
    run.log('MAE_val', mean_absolute_error(y_test[:,0], yhat[:,0]))
    run.log('RMSE_train', np.sqrt(mean_squared_error(y_train[:,0], train_predict[:,0])))
    run.log('RMSE_val', np.sqrt(mean_squared_error(y_test[:,0], yhat[:,0])))
    run.log('R2_train', r2_score(y_train[:,0], train_predict[:,0]))
    run.log('R2_val', r2_score(y_test[:,0], yhat[:,0]))
    run.tag("n_epochs", n_epochs)
    run.tag("batch_size", batch_size)
    run.tag("activation", activation)
    run.tag("n_layers", n_layers)
    run.tag("model_type", model_type)
    run.tag("window", window)


 

    run.tag("dropout", dropout)
    model_loss(history)
    X,y = data_prepared(df_arr,window)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    yhat = model.predict(X)
    y = scaler.inverse_transform(y)
    yhat = scaler.inverse_transform(yhat)    
    learning_curve(y, yhat,window)
    os.makedirs('outputs', exist_ok=True)
    model.save('outputs/'+str(run.id))

run.complete()