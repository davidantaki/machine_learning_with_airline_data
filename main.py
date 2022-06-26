from enum import unique
from logging import exception
import random
from scipy.stats import multivariate_normal  # MVN not univariate
import matplotlib.pyplot as plt  # For general plotting
import pandas
import numpy as np
import Airline_Funct as af
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

trainset_raw = pandas.read_csv('airline_satisfaction_train.csv')
testset_raw = pandas.read_csv('airline_satisfaction_test.csv')


# Load training set
trainset = trainset_raw
# Get rid of useless columns
del trainset["Unnamed: 0"]
del trainset["id"]
# TODO Convert these columns to integers
del trainset["Gender"]
del trainset["Customer Type"]
del trainset["Type of Travel"]
del trainset["Class"]
# print("trainset:\n{}".format(trainset))
# Separate labels
trainset_y = trainset_raw.loc[:, "satisfaction"]
print("trainset_y:\n{}".format(trainset_y))
trainset.drop(['satisfaction'], axis=1)
# Separate X samples
trainset_x = trainset
del trainset_x['satisfaction']
print("trainset_x:\n{}".format(trainset_x))
# Get list of feature names
features = trainset_x.columns
print(features)
# Convert to numpy arrays
trainset_y_names = np.array(trainset_y)
class_names = np.unique(trainset_y_names)
print("class_names:\n{}".format(class_names))
trainset_x = np.array(trainset_x)
# Convert string labels to integer labels corresponding to index of class_names
for i in range(0, len(trainset_y_names)):
    if trainset_y_names[i] == class_names[0]:
        trainset_y[i] = 0
    elif trainset_y_names[i] == class_names[1]:
        trainset_y[i] = 1
    else:
        raise Exception("Not valid class name: {}".format(trainset_y_names[i]))
print("trainset_y:\n{}".format(trainset_y.shape))
print("trainset_x:\n{}".format(trainset_x.shape))


# Load testing set
testset = testset_raw
del testset["Unnamed: 0"]
del testset["id"]
del testset["Gender"]
del testset["Customer Type"]
del testset["Type of Travel"]
del testset["Class"]
# print("testset:\n{}".format(testset))
testset_y = testset_raw.loc[:, "satisfaction"]
print("testset_y:\n{}".format(testset_y))
testset.drop(['satisfaction'], axis=1)
testset_x = testset
del testset_x['satisfaction']
print("testset_x:\n{}".format(testset_x))
features = testset_x.columns
print(features)
testset_y_names = np.array(testset_y)   # String labels
class_names = np.unique(testset_y_names)
print("class_names:\n{}".format(class_names))
# Convert string labels to integer labels corresponding to index of class_names
for i in range(0, len(testset_y_names)):
    if testset_y_names[i] == class_names[0]:
        testset_y[i] = 0
    elif testset_y_names[i] == class_names[1]:
        testset_y[i] = 1
    else:
        raise Exception("Not valid class name: {}".format(testset_y_names[i]))
testset_x = np.array(testset_x)
print("testset_y:\n{}".format(testset_y.shape))
print("testset_x:\n{}".format(testset_x.shape))


# Params
class_num = np.arange(0, len(class_names))
print("class_num:\n{}".format(class_num))
num_classes = len(class_names)
print("num_classes: {}".format(num_classes))


input_dim = trainset_x.shape[1]
# Cross validation to select hidden neurons
n_hidden_neurons = 2
output_dim = num_classes

model = af.MLP(input_dim, n_hidden_neurons, output_dim)
print(model)

# Stochastic GD with learning rate and momentum hyperparameters
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# The nn.CrossEntropyLoss() loss function automatically performs a log_softmax() to
# the output when validating, on top of calculating the negative log-likelihood using
# nn.NLLLoss(), while also being more stable numerically...
criterion = nn.CrossEntropyLoss()
num_epochs = 10

# Convert numpy structures to PyTorch tensors, as these are the data types required by the library
X_tensor = torch.FloatTensor(trainset_x)
y_tensor = torch.LongTensor(trainset_y)

# Trained model
model = af.train_model(model, X_tensor, y_tensor, criterion,
                       optimizer, num_epochs=num_epochs)

testset_y_pred = af.model_predict(model, testset_x)
print("testset_y_pred:\n{}".format(testset_y_pred))
print("testset_y_pred:\n{}".format(np.unique(testset_y_pred)))
print("testset_y:\n{}".format(np.unique(testset_y)))
print(testset_y_pred.shape)
print(confusion_matrix(testset_y, testset_y_pred))
