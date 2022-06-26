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
import seaborn as sns

cuda = torch.device('cuda')     # Default CUDA device
cuda0 = torch.device('cuda:0')
cuda2 = torch.device('cuda:2')  # GPU 2 (these are 0-indexed)
print(torch.cuda.is_available())

trainset_raw = pandas.read_csv('airline_satisfaction_train.csv')
testset_raw = pandas.read_csv('airline_satisfaction_test.csv')
trainset_raw.head()


def remove_null_samples(X):
    '''
    Returns dataset with all samples that have any feature with NaN value removed.
    '''
    # To numpy array
    X = np.array(X)
    # Remove all samples that have any feature with NaN (null) value
    nan_samples = np.argwhere(np.isnan(X))
    print("Number of null samples before Removing Them: {}".format(
        len(np.argwhere(np.isnan(X)))))
    print("Total number of samples before removing null samples: {}".format(
        len(X)))
    # X = X[~np.isnan(X).any(axis=1), :]
    X = np.delete(X, nan_samples.T[0], axis=0)  # Does same thing as line above
    print("Number of null samples After Removing Them: {}".format(
        len(np.argwhere(np.isnan(X)))))
    print("Total number of samples After removing null samples: {}".format(
        len(X)))
    return X


classes_str = ["neutral or dissatisfied", "satisfied"]
classes_int = [0, 1]
num_classes = len(classes_str)

# Load training set
trainset = trainset_raw
print("trainset:\n{}".format(trainset))
# Get rid of useless columns
del trainset["Unnamed: 0"]
del trainset["id"]
# TODO Convert these columns to integers
del trainset["Gender"]
del trainset["Customer Type"]
del trainset["Type of Travel"]
del trainset["Class"]
# Convert string labels to integer labels corresponding to index of class_names
trainset = trainset.replace(classes_str[0], value=classes_int[0], regex=True)
trainset = trainset.replace(classes_str[1], value=classes_int[1], regex=True)
# Get list of feature names
features = trainset.columns
print(features)
# Remove any sample with NULL value
trainset = remove_null_samples(trainset)

# Separate labels
# Get last column which is the labels
trainset_y = trainset[:, len(trainset[0])-1]
# Separate X samples
trainset_x = np.delete(trainset, len(trainset[0])-1, axis=1)

print("trainset_y:\n{}".format(trainset_y.shape))
print("trainset_x:\n{}".format(trainset_x.shape))


# Load testing set
testset = testset_raw
print("testset:\n{}".format(testset))
# Get rid of useless columns
del testset["Unnamed: 0"]
del testset["id"]
# TODO Convert these columns to integers
del testset["Gender"]
del testset["Customer Type"]
del testset["Type of Travel"]
del testset["Class"]
# Convert string labels to integer labels corresponding to index of class_names
testset = testset.replace(classes_str[0], value=classes_int[0], regex=True)
testset = testset.replace(classes_str[1], value=classes_int[1], regex=True)
# Get list of feature names
features = testset.columns
print(features)
# Remove any sample with NULL value
testset = remove_null_samples(testset)

# Separate labels
# Get last column which is the labels
testset_y = testset[:, len(testset[0])-1]
# print("testset_y:\n{}".format(testset_y))
# Separate X samples
testset_x = np.delete(testset, len(testset[0])-1, axis=1)
print("testset_x shape:\n{}".format(testset_x.shape))

print("testset_y:\n{}".format(testset_y.shape))
print("testset_x:\n{}".format(testset_x.shape))


# Params
input_dim = trainset_x.shape[1]
# TODO: Cross validation to select hidden neurons
n_hidden_neurons = 16
output_dim = num_classes

model = af.MLP(input_dim, n_hidden_neurons, output_dim)
print(model)

# Stochastic GD with learning rate and momentum hyperparameters
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# The nn.CrossEntropyLoss() loss function automatically performs a log_softmax() to
# the output when validating, on top of calculating the negative log-likelihood using
# nn.NLLLoss(), while also being more stable numerically...
criterion = nn.CrossEntropyLoss()
num_epochs = 100

# Convert numpy structures to PyTorch tensors, as these are the data types required by the library
trainset_x_tensor = torch.FloatTensor(trainset_x)
trainset_y_tensor = torch.LongTensor(trainset_y)

testset_x_tensor = torch.FloatTensor(testset_x)
testset_y_tensor = torch.LongTensor(testset_y)

# Trained model
model = af.train_model(model, trainset_x_tensor, trainset_y_tensor, criterion,
                       optimizer, num_epochs=num_epochs)
testset_y_pred = af.model_predict(model, testset_x_tensor)
# print("testset_y_pred:\n{}".format(np.unique(testset_y_pred)))
# print("testset_y:\np{}".format(np.unique(testset_y)))
print(testset_y_pred.shape)
print(testset_y.shape)

print(confusion_matrix(testset_y, testset_y_pred))
testset_y_pred_df = pandas.DataFrame(testset_y_pred, columns=['satisfaction'])
sns.countplot(x='satisfaction', data=testset_y_pred_df)
plt.title("Number of samples classified as Satisfied vs Unsatisfied")
plt.show()
