#!/usr/bin/env python
# coding: utf-8

# # Hyperparameter Tuning using HyperDrive
# 
# TODO: Import Dependencies. In the cell below, import all the dependencies that you will need to complete the project.

# In[6]:


import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
#!pip install pandas_datareader
from pandas_datareader import data
from azureml.core import Workspace, Dataset, Experiment, Run
from azureml.train.automl import AutoMLConfig
from azureml.widgets import RunDetails
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.train.automl.run import AutoMLRun
from azureml.core.model import Model, InferenceConfig
from azureml.core.environment import Environment
from azureml.core.webservice import AciWebservice

# Webservice
import urllib.request
import json
import os
import ssl

# Data Manipulation
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



# AzureML
from azureml.core import Workspace, Dataset, Experiment, Run
from azureml.train.hyperdrive import HyperDriveRun
from azureml.widgets import RunDetails
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.train.hyperdrive import RandomParameterSampling, BanditPolicy, HyperDriveConfig, PrimaryMetricGoal
from azureml.train.hyperdrive import choice
from azureml.train.dnn import TensorFlow

# Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, TimeDistributed, GRU
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Data manipulation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


# ## Dataset
# 
# TODO: Get data. In the cell below, write code to access the data you will be using in this project. Remember that the dataset needs to be external.

# In[7]:


ws = Workspace.from_config()

# choose a name for experiment
experiment_name = 'MSFT-stock'

experiment=Experiment(ws, experiment_name)


# In[3]:


dataset_prepared = Dataset.get_by_name(ws, name='MSFT-stock cleaned')

df = dataset_prepared.to_pandas_dataframe()
df = df.set_index('Date', inplace=False) 
df.head()


# ## Hyperdrive Configuration
# 
# TODO: Explain the model you are using and the reason for chosing the different hyperparameters, termination policy and config settings.

# In[ ]:


# TODO: Create an early termination policy. This is not required if you are using Bayesian sampling.
early_termination_policy = <your policy here>

#TODO: Create the different params that you will be using during training
param_sampling = <your params here>

#TODO: Create your estimator and hyperdrive config
estimator = <your estimator here>

get_ipython().set_next_input('hyperdrive_run_config = <your config here');get_ipython().run_line_magic('pinfo', 'here')


# In[8]:


#TODO: Submit your experiment
# CPU cluster name
amlcompute_cluster_name = "AMLForecasting"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=amlcompute_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    print('No cluster found')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS3_V2',                                                           
                                                           max_nodes=4,
                                                          min_nodes = 1, idle_seconds_before_scaledown = 600)
    compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True)


# ## Run Details
# 
# OPTIONAL: Write about the different models trained and their performance. Why do you think some models did better than others?
# 
# TODO: In the cell below, use the `RunDetails` widget to show the different experiments.

# In[37]:


# LSTM

def lstm_3d(df, look_back):
    X, Y =[], []
    for i in range(len(df)-look_back):
        d=i+look_back
        X.append(df[i:d,])
        Y.append(df[d,])
    return np.array(X), np.array(Y)


#create numpy.ndarray
df_arr= df.values
 
df_arr = np.reshape(df_arr, (-1, 1)) #LTSM requires more input features compared to RNN or DNN
scaler = MinMaxScaler(feature_range=(0, 1))#LTSM is senstive to the scale of features
df_arr = scaler.fit_transform(df_arr)

train_size =  10
test_size = len(df_arr)-train_size
train, test = df_arr[0:train_size,:], df_arr[train_size:len(df_arr),:]
look_back = 3
trainX, trainY = lstm_3d(train, look_back)
testX, testY = lstm_3d(test, look_back)
# reshape input to be [samples, time steps, features]
print(trainY.shape)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print(trainX.shape)

def model_lstm(look_back):
    model=Sequential()
#     model.add(LSTM(100, input_shape=(1, look_back), activation='relu'))
#     model.add(Dense(1))
    
    model.add(LSTM(64, activation='relu', input_shape=(1,look_back)))
    #lstm.add(LSTM(32, activation='relu', input_shape=(32,1)))
    model.add(Dense(10))
    model.add(Dropout(0.3))
    #lstm.add(Dense(10))
    #lstm.add(Dropout(0.3))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error',  optimizer='adam',metrics = ['mse', 'mae'])

    
    return model


model=model_lstm(look_back)
history = model.fit(trainX, trainY, epochs=100, batch_size=30, validation_data=(testX, testY),
#                     callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
                    verbose=0, shuffle=False)

train_predict = model.predict(trainX)
test_predict = model.predict(testX)

# invert predictions

train_predict = scaler.inverse_transform(train_predict)
trainY = scaler.inverse_transform(trainY)
test_predict = scaler.inverse_transform(test_predict)
testY = scaler.inverse_transform(testY)

print('Done')


### Hyperparameters
# layers
# nodes
# batch_size
# Model_type
# epochs
# look_back


# In[40]:


def model_loss(history):
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    


# In[41]:


print('Train Root Mean Squared Error(RMSE): %.2f; Train Mean Absolute Error(MAE) : %.2f '% (np.sqrt(mean_squared_error(trainY[:,0], train_predict[:,0])),(mean_absolute_error(trainY[:,0], train_predict[:,0]))))
print('Test Root Mean Squared Error(RMSE): %.2f; Test Mean Absolute Error(MAE) : %.2f '% (np.sqrt(mean_squared_error(testY[:,0], test_predict[:,0])),(mean_absolute_error(testY[:,0], test_predict[:,0]))))


model_loss(history)


# In[43]:


def prediction_plot(y_test, test_predict):
    len_prediction=df.index[look_back:]
    plt.figure(figsize=(8,4))
    plt.plot(len_prediction, y_test[:], marker='.', label="actual")    
    plt.plot(len_prediction, test_predict[:], 'r', label="prediction")
    plt.tight_layout()
    sns.despine(top=True)
    plt.ylabel('Passenger Count', size=12)
    plt.xlabel('Time step', size=12)
    plt.legend(fontsize=12)    
    plt.show()

Xa,ya = lstm_3d(df_arr,look_back)
Xa = np.reshape(Xa, (Xa.shape[0], 1, Xa.shape[1]))

test_predict = model.predict(Xa)
ya = scaler.inverse_transform(ya)

test_predict = scaler.inverse_transform(test_predict)


prediction_plot(ya, test_predict)


# ## Best Model
# 
# TODO: In the cell below, get the best model from the hyperdrive experiments and display all the properties of the model.

# In[26]:


train_size = 10
test_size = len(df_arr)-train_size
train, test = df_arr[0:train_size,:], df_arr[train_size:len(df_arr),:]
 
train


# In[ ]:


#TODO: Save the best model


# ## Model Deployment
# 
# Remember you have to deploy only one of the two models you trained.. Perform the steps in the rest of this notebook only if you wish to deploy this model.
# 
# TODO: In the cell below, register the model, create an inference config and deploy the model as a web service.

# In[ ]:





# TODO: In the cell below, send a request to the web service you deployed to test it.

# In[ ]:





# TODO: In the cell below, print the logs of the web service and delete the service

# In[ ]:




