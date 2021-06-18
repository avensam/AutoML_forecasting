 
# Predicting Microsoft stock 

In this project I consumed the latest stock data from yahoo strating 2019. Given the closing stock, the target here is to predict the next day's stock.

## Project Set Up and Installation
In this project I used AzureML forecasting tool (ensemble of forecasting models - [Set up AutoML training with Python](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train)

## Dataset
### Overview

The data is a daily Microsoft stock that I reads through Yahoo finance. pandas has a simple remote data access for the Yahoo Finance API data. data.DataReader returns a Panel object, which can be thought of as a 3D matrix. The first dimension consists of the various fields Yahoo Finance returns for a given instrument, namely, the Open, High, Low, Close and Adj Close prices for each date. The second dimension contain the dates.

### Task
Here in this project I used two main tools; AzureML and LSTM to predict daily MSstock.I have created a hyperparameter experiments to run LSTM as well as AzureML and compared their results.

### Access
    # import the dataset from Yahoo finance
     MSFT = data.DataReader('MSFT', 'yahoo',start='1/1/2019')
     
![Diagram1]( https://github.com/avensam/AutoML_forecasting/blob/master/starter_file/images/MSstock.PNG "MS stock") 


## Automated ML
    #
    automl_settings = {
     "experiment_timeout_minutes":30,
     "max_concurrent_iterations":4,
     "primary_metric": "normalized_root_mean_squared_error",
     'time_column_name': 'Date'
     }
     
     automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "forecasting",
                             training_data=dataset_train,
                             label_column_name="Adj Close",
                             enable_early_stopping= True,
                             n_cross_validations=5,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )

### Results
There were 36 models running, were the best model turned out to be voting ensemble with 0.94270894 MAE

![Diagram1]( https://github.com/avensam/AutoML_forecasting/blob/master/starter_file/images/autmlmodels.PNG "models running") 
![Diagram2]( https://github.com/avensam/AutoML_forecasting/blob/master/starter_file/images/autmlbestmodel.PNG "Rundetails") 
![Diagram3]( https://github.com/avensam/AutoML_forecasting/blob/master/starter_file/images/bestmodel.PNG "best model") 
![Diagram4]( https://github.com/avensam/AutoML_forecasting/blob/master/starter_file/images/1.automlcompleted1.PNG "best model") 


## Hyperparameter Tuning
34 different models were trained and compared The best model used the VotingEnsemble algorithm, reaching an accuracy of 87.30 %. Ten-fold cross validation was used. All the columns were used for training.

According to (1), serum creatinine and ejection fraction are the most relevant features and are enough to predict if a patient with heart failure will survive or not. Thus, using a dataset with only these two columns would provide better results, increased accuracy and higher performance.

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
