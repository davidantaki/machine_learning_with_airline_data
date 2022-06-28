from enum import unique
from logging import exception
from turtle import color
from scipy.stats import multivariate_normal  # MVN not univariate
import matplotlib.pyplot as plt  # For general plotting

import numpy as np
import Airline_Funct as af
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as rf

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


import eli5
from eli5.sklearn import PermutationImportance
import statsmodels.api as sm
import time
from tabulate import tabulate
# np.set_printoptions(threshold=np.inf)
# plt.rc('font', size=22)          # controls default text sizes
# plt.rc('axes', titlesize=18)     # fontsize of the axes title
# plt.rc('axes', labelsize=18)     # fontsize of the x and y labels
# plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
# plt.rc('legend', fontsize=16)    # legend fontsize
# plt.rc('figure', titlesize=22)   # fontsize of the figure title

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)


def regularized_cov(X, lambda_reg):
    n = X.shape[0]
    sigma = np.cov(X)
    # Selecting the regularization parameter should be performed using CV and a separate data subset
    # As I only went by training set performance (overfitting) in this problem, I settled on lambda=1/n
    sigma += lambda_reg * np.eye(n)
    return sigma




trainset_raw = pd.read_csv('airline_satisfaction_train.csv')
testset_raw = pd.read_csv('airline_satisfaction_test.csv')
#print(trainset_raw[0:1])

del trainset_raw['Unnamed: 0']
del trainset_raw['id']
trainset= trainset_raw

del testset_raw['Unnamed: 0']
del testset_raw['id']
testset= testset_raw

trainset.columns = [c.replace(' ', '_') for c in trainset.columns]

testset.columns = [c.replace(' ', '_') for c in testset.columns]

#trainset.info()

fig = plt.figure(figsize = (10,10))
Freq=trainset['satisfaction'].value_counts(normalize=True)
Freq.plot(kind='bar', color= ['red','green'], rot=0)
plt.title('Frequency of Satisfaction in Training Set')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()
print(Freq)


trainset['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1},inplace = True)
testset['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1},inplace = True)

total = trainset.isnull().sum().sort_values(ascending=False)
percent = (trainset.isnull().sum()/trainset.isnull().count()).sort_values(ascending=False)
missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing.head()

# Imputing missing value with mean - Train and Test
trainset['Arrival_Delay_in_Minutes'] = trainset['Arrival_Delay_in_Minutes'].fillna(trainset['Arrival_Delay_in_Minutes'].mean())
testset['Arrival_Delay_in_Minutes'] = testset['Arrival_Delay_in_Minutes'].fillna(testset['Arrival_Delay_in_Minutes'].mean())

trainset.select_dtypes(include=['object']).columns

# Replace NaN with mode for categorical variables - Train and Test

trainset['Gender'] = trainset['Gender'].fillna(trainset['Gender'].mode()[0])
testset['Gender'] = testset['Gender'].fillna(testset['Gender'].mode()[0])

trainset['Customer_Type'] = trainset['Customer_Type'].fillna(trainset['Customer_Type'].mode()[0])
testset['Customer_Type'] = testset['Customer_Type'].fillna(testset['Customer_Type'].mode()[0])


trainset['Type_of_Travel'] = trainset['Type_of_Travel'].fillna(trainset['Type_of_Travel'].mode()[0])
testset['Type_of_Travel'] = testset['Type_of_Travel'].fillna(testset['Type_of_Travel'].mode()[0])

trainset['Class'] = trainset['Class'].fillna(trainset['Class'].mode()[0])
testset['Class'] = testset['Class'].fillna(testset['Class'].mode()[0])

# seaborn.catplot(*, x=None, y=None, hue=None, data=None, row=None, col=None, col_wrap=None, 
# estimator=<function mean at 0x7ff320f315e0>, ci=95, n_boot=1000, units=None, seed=None, 
# order=None, hue_order=None, row_order=None, col_order=None, kind='strip', height=5, aspect=1, 
# orient=None, color=None, palette=None, legend=True, legend_out=True, sharex=True, sharey=True, 
# margin_titles=False, facet_kws=None, **kwargs)

# Figure-level interface for drawing categorical plots onto a FacetGrid.

# This function provides access to several axes-level functions that show the relationship
# between a numerical and one or more categorical variables using one of several visual 
# representations. The kind parameter selects the underlying axes-level function to use:

#g = sns.catplot("satisfaction", col="Gender", col_wrap=2,data=trainset, kind="count")  
#plt.show()

# g = sns.catplot("satisfaction", col="Customer_Type", col_wrap=2, data=trainset, kind="count")
# plt.show()

# g = sns.catplot(x="Flight_Distance", y="Type_of_Travel", hue="satisfaction", col="Class", data=trainset, kind="bar", height=4.5, aspect=.8)
# plt.show()

# g = sns.catplot("Age", data=trainset, aspect=3.0, kind='count', hue='satisfaction', order=range(5, 80))
# plt.show()

# g.set_ylabels('Age vs Passenger Satisfaction')
# plt.show()

# g = sns.catplot(x="Class", y="Departure_Delay_in_Minutes", hue="satisfaction", col="Type_of_Travel", data=trainset, kind="bar", height=4.5, aspect=1.0)
# plt.show()

# g = sns.catplot(x="Class", y="Arrival_Delay_in_Minutes", hue="satisfaction", col="Type_of_Travel", data=trainset, kind="bar", height=4.5, aspect=1.0)
# plt.show()

#############################################################################################################

# Encode target labels with value between 0 and n_classes-1.
# This transformer should be used to encode target values, i.e. y, and not the input X.

#Encoded sets
labelenc_train = {}
labelenc_test = {}

# DataFrame.select_dtypes(include=None, exclude=None)
# Return a subset of the DataFrame’s columns based on the column dtypes.

# To select strings you must use the object dtype, 
# but note that this will return all object dtype columns

# Trainset Encoding
for c in trainset.select_dtypes(include=['object']).columns:
    labelenc_train[c] = LabelEncoder()
    #print(labelenc_train[c])
    trainset[c] = labelenc_train[c].fit_transform(trainset[c])


# Testset Encoding 
for c in testset.select_dtypes(include=['object']).columns:
    labelenc_test[c] = LabelEncoder()
    #print(labelenc_train[c])
    testset[c] = labelenc_test[c].fit_transform(testset[c])

#Outlier Detection

#numpy.quantile(a, q, axis=None, out=None, 
# overwrite_input=False, method='linear', keepdims=False, *, interpolation=None)
Q15 = trainset.quantile(0.15)
Q85 = trainset.quantile(0.85)
IQR = Q85 - Q15

# Removal of outliers from dataset
train_clean = trainset[~((trainset < (Q15 - 1.5 * IQR)) |(trainset > (Q85 + 1.5 * IQR))).any(axis=1)]

#Correlation Heatmap
corr = train_clean.corr()
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=np.bool)), cmap=sns.diverging_palette(150, 275, as_cmap=True), 
vmax=None, center=0,square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .9})
plt.show()


X_train = train_clean.drop('satisfaction', axis=1)
y_train = train_clean['satisfaction']
selector = SelectFromModel(rf(n_estimators=100, random_state=0))
selector.fit(X_train, y_train)
support = selector.get_support()
features = X_train.loc[:,support].columns.tolist()
print(features)


perm = PermutationImportance(rf(n_estimators=100, random_state=0).fit(X_train,y_train),random_state=1).fit(X_train,y_train)
eli5.show_weights(perm, feature_names = X_train.columns.tolist())

Feature = ['Type_of_Travel','Inflight_wifi_service','Online_boarding','Seat_comfort','Flight_Distance',
             'Inflight_entertainment','On-board_service','Leg_room_service','Cleanliness','Checkin_service', 
             'Inflight_service', 'Baggage_handling']
#features = ['Type_of_Travel']
Verdict = ['satisfaction']


#feature arrays
X_train = train_clean[features]
X_test = testset[features]

y_train = train_clean[Verdict].to_numpy()
y_test = testset[Verdict].to_numpy()

# Normalize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

params_lda = {}

model_lda = LinearDiscriminantAnalysis(**params_lda)
#model_lda, accuracy_lda, roc_auc_lda, tt_lda = run_model(model_lda, X_train, y_train, X_test, y_test)

init_t=time.time()

model_lda.fit(X_train,y_train.ravel())
Verd_predict = model_lda.predict(X_test)
acc = accuracy_score(y_test, Verd_predict)
roc_auc_score = roc_auc_score(y_test, Verd_predict) 
end_t = time.time()-init_t
print("Accuracy = {}".format(acc))
print("ROC Area under Curve = {}".format(roc_auc_score))
print("Runtime = {}".format(end_t))
plot_confusion_matrix(model_lda, X_test, y_test,cmap=plt.cm.pink, normalize = 'all')
plot_roc_curve(model_lda, X_test, y_test)                     
plt.show()

