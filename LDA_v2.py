from enum import unique
from logging import exception
from turtle import color
from scipy.stats import multivariate_normal  # MVN not univariate
import matplotlib.pyplot as plt  # For general plotting

import numpy as np
import sklearn.metrics as sm
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as rf



import eli5
from eli5.sklearn import PermutationImportance
from eli5 import show_weights
import time
from tabulate import tabulate

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

# Replacing empty values with means
trainset['Arrival_Delay_in_Minutes'] = trainset['Arrival_Delay_in_Minutes'].fillna(trainset['Arrival_Delay_in_Minutes'].mean())
testset['Arrival_Delay_in_Minutes'] = testset['Arrival_Delay_in_Minutes'].fillna(testset['Arrival_Delay_in_Minutes'].mean())

trainset.select_dtypes(include=['object']).columns

# Replace NaN with mode for features

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

graph = sns.catplot("satisfaction", col="Gender", col_wrap=2, data=trainset, kind="count") 
plt.show()

graph = sns.catplot("satisfaction", col="Customer_Type", col_wrap=2, data=trainset, kind="count")
plt.show()

graph = sns.catplot("satisfaction", col="Ease_of_Online_booking", col_wrap=6, data=trainset, kind="count")
plt.show()

graph = sns.catplot("satisfaction", col="Seat_comfort", col_wrap=6, data=trainset, kind="count")
plt.show()

graph = sns.catplot("satisfaction", col="Cleanliness", col_wrap=6, data=trainset, kind="count")
plt.show()

graph = sns.catplot("satisfaction", col="Food_and_drink", col_wrap=6, data=trainset, kind="count")
plt.show()

graph = sns.catplot("Age", data=trainset, kind='count', hue='satisfaction', order=range(0, 100))
graph.set_ylabels('Count')
plt.title('Satisfaction vs. Age')
plt.show()



#############################################################################################################

# Encode target labels with value between 0 and n_classes-1.
# This transformer should be used to encode target values, i.e. y, and not the input X.

# #Encoded sets
labelenc_train = {}
labelenc_test = {}

# # DataFrame.select_dtypes(include=None, exclude=None)
# # Return a subset of the DataFrame’s columns based on the column dtypes.

# # To select strings you must use the object dtype, 
# # but note that this will return all object dtype columns

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

# #Outlier Detection

#numpy.quantile(a, q, axis=None, out=None, 
# overwrite_input=False, method='linear', keepdims=False, *, interpolation=None)
Q15 = trainset.quantile(0.15)
Q85 = trainset.quantile(0.85)
diff = Q85 - Q15

# # Removal of outliers from dataset
train_clean = trainset[~((trainset < (Q15 - 1.5 * diff)) |(trainset > (Q85 + 1.5 * diff))).any(axis=1)]

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
#print(features)

perm = PermutationImportance(rf(n_estimators=100, random_state=0).fit(X_train,y_train),random_state=1).fit(X_train,y_train)
show_weights(perm, feature_names = X_train.columns.tolist())

Feature = ['Type_of_Travel','Inflight_wifi_service','Online_boarding','Seat_comfort','Flight_Distance',
             'Inflight_entertainment','On-board_service','Leg_room_service','Cleanliness','Checkin_service', 
              'Inflight_service', 'Baggage_handling']

Verdict = ['satisfaction']


#feature arrays
X_train = train_clean[features]
X_test = testset[features]

y_train = train_clean[Verdict].to_numpy()
y_test = testset[Verdict].to_numpy()

# Normalization of Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

parameters_lda = {}
model_lda = LinearDiscriminantAnalysis(**parameters_lda)

init_t=time.time()
model_lda.fit(X_train,y_train.ravel())
Verd_predict = model_lda.predict(X_test)
acc = sm.accuracy_score(y_test, Verd_predict)
roc_auc_score = sm.roc_auc_score(y_test, Verd_predict) 
end_t = time.time()-init_t

print("Runtime = {}".format(end_t))
print("Accuracy = {}".format(acc))
print("ROC Area under Curve = {}".format(roc_auc_score))

sm.plot_confusion_matrix(model_lda, X_test, y_test,cmap=plt.cm.pink, normalize = 'all')
sm.plot_roc_curve(model_lda, X_test, y_test)                     
plt.show()


