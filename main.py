from enum import unique
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



# Get sample labels aka "activity labels". These are the POSSIBLE sample labels but not the labels associated with data.
dataset = pandas.read_csv(
    '..\\ecece5644_machine_learning\\project\\flight_satisfaction_train.csv')
testset = pandas.read_csv(
    '..\\ecece5644_machine_learning\\project\\flight_satisfaction_test.csv')

dataset = dataset.drop(["Unnamed: 0", "id"], axis=1)
testdata = testset.drop(["Unnamed: 0", "id"], axis=1)

print(dataset)
print(dataset.shape)
dataset = np.array(dataset)
