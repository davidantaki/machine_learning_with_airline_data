from random import shuffle
from skorch import NeuralNetClassifier
import matplotlib.pyplot as plt  # For general plotting
import pandas
import numpy as np
# import Airline_Funct as af
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
from sklearn import preprocessing


class MLP(nn.Module):
    # The nn.CrossEntropyLoss() loss function automatically performs a log_softmax() to
    # the output when validating, on top of calculating the negative-log-likelihood using
    # nn.NLLLoss(), while also being more stable numerically... So don't implement from scratch
    # https://pytorch.org/docs/stable/generated/torch.nn.Module.html

    def __init__(self, input_dim, C, hidden_dim=32):
        super(MLP, self).__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.input_activation = nn.ReLU()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation2 = nn.ReLU()
        self.output_fc = nn.Linear(hidden_dim, C)
        self.output_activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.output_fc(x)
        x = self.output_activation(x)
        return x


def model_predict(model, data):
    model.eval()
    X_test = torch.FloatTensor(data).to(torch.device('cuda'))

    # Evaluate nn on test data and compare to true labels
    predicted_labels = model(X_test)
    print("predicated_labels:\n {}".format(predicted_labels))
    # Back to numpy
    predicted_labels = predicted_labels.cpu().detach().numpy()

    return np.argmax(predicted_labels, 1)


def train_model(model, data, labels, criterion, optimizer, num_epochs=25, plot=False):

    # Apparently good practice to set this "flag" too before training
    # Does things like make sure Dropout layers are active, gradients are updated, etc.
    # Probably not a big deal for our toy network, but still worth developing good practice
    model.train()
    X_train = torch.FloatTensor(data).to(torch.device('cuda'))

    # Optimize the neural network
    y_train = torch.LongTensor(labels).to(torch.device('cuda'))

    # For storing loss vs epoch
    training_learning_data = []

    for epoch in range(num_epochs):
        # Set grads to zero explicitly before backprop
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        # Store loss vs epoch for graphing
        training_learning_data.append([epoch, loss.cpu().detach().numpy()])
        # Backward pass to compute the gradients through the network
        loss.backward()
        # GD step update
        optimizer.step()

    # Plot loss vs epoch
    training_learning_data = np.array(training_learning_data)
    if plot:
        plt.scatter(training_learning_data[:, 0], training_learning_data[:, 1])
        plt.xlabel("epoch")
        plt.ylabel("Loss function")
        plt.title("Value of the loss function")
        plt.ylim((0, 1))
        plt.show()

    return model, training_learning_data


def validate_model(model, test_data_x, test_data_y, plot=False):
    testset_y_pred = model_predict(model, test_data_x)
    # print(testset_y_pred.shape)
    # print(testset_y.shape)
    accuracy = accuracy_score(test_data_y, testset_y_pred)
    print("accuracy: {}".format(accuracy))
    print(confusion_matrix(test_data_y, testset_y_pred))

    if plot:
        testset_y_pred_df = pandas.DataFrame(
            testset_y_pred, columns=['satisfaction'])
        sns.countplot(x='satisfaction', data=testset_y_pred_df)
        plt.title("Number of samples classified as Satisfied vs Unsatisfied")
        plt.show()

    return accuracy


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


cuda = torch.device('cuda')     # Default CUDA device
print(torch.cuda.is_available())
print(torch.version.cuda)

trainset_raw = pandas.read_csv('airline_satisfaction_train.csv')
testset_raw = pandas.read_csv('airline_satisfaction_test.csv')
trainset_raw.head()


classes_str = ["neutral or dissatisfied", "satisfied"]
classes_int = [0, 1]
num_classes = len(classes_str)

# Load training set
trainset = trainset_raw
# print("trainset:\n{}".format(trainset))
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

# Temporary
# trainset_x = trainset_x[:, 0:1]

print("trainset_y:\n{}".format(trainset_y.shape))
print("trainset_x:\n{}".format(trainset_x.shape))
print("final trainset_x:\n{}".format(trainset_x))


# Load testing set
testset = testset_raw
# print("testset:\n{}".format(testset))
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

# testset_x = testset_x[:, 0:1]

print("testset_y:\n{}".format(testset_y.shape))
print("testset_x:\n{}".format(testset_x.shape))


# Visualize Dataset
# trainset_x_df = pandas.DataFrame(trainset_x, columns=features[0:18])
# print(trainset_x_df)

# plt.hist(trainset_x_df['Age'], color='blue', edgecolor='black',
#          bins=len(np.unique(trainset_x_df['Age'])))
# plt.title('Histogram of Arrival Delays')
# plt.xlabel('Delay (min)')
# plt.ylabel('Flights')
# plt.show()

# sns.pairplot(data=trainset_x_df)

# NORMALIZE DATA/FEATURE SCALING
# plt.scatter(np.arange(0, len(trainset_x)), trainset_x[:, 0])
# plt.show()
scaler = preprocessing.StandardScaler().fit(trainset_x)
trainset_x = scaler.transform(trainset_x)
scaler = preprocessing.StandardScaler().fit(testset_x)
testset_x = scaler.transform(testset_x)
print([np.dtype(testset_x[0, i]) for i in range(0, 18)])
print([np.dtype(trainset_x[0, i]) for i in range(0, 18)])
# plt.scatter(np.arange(0, len(trainset_x)), trainset_x[:, 0])
# plt.show()
# trainset_x_df = pandas.DataFrame(trainset_x, columns=features[0:18])
# print(trainset_x_df)
# plt.hist(trainset_x_df['Age'], color='blue', edgecolor='black',
#          bins=len(np.unique(trainset_x_df['Age'])))
# plt.title('Histogram of Arrival Delays')
# plt.xlabel('Delay (min)')
# plt.ylabel('Flights')
# plt.show()


def cross_validate(train_x, train_y, folds, input_dim, hidden_dim, output_dim, lr, momentum, n_epochs):
    '''
    Runs cross validation for a SINGLE set of hyperparmeters.
    Returns the accuracy score for the given set of hyperparameters.
    '''
    # Partition data
    kf = KFold(n_splits=folds, shuffle=True)
    # CROSS VALIDATION
    k = 0
    k_scores = []
    # NOTE that these subsets are of the TRAINING dataset
    for train_indices, valid_indices in kf.split(train_x):
        # Extract the training and validation sets from the K-fold split
        X_train_k = train_x[train_indices]
        X_valid_k = train_x[valid_indices]
        y_train_k = train_y[train_indices]
        y_valid_k = train_y[valid_indices]

        model = MLP(input_dim, hidden_dim,
                    output_dim).to(torch.device('cuda'))
        print(model)

        # Stochastic GD with learning rate and momentum hyperparameters
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # The nn.CrossEntropyLoss() loss function automatically performs a log_softmax() to
        # the output when validating, on top of calculating the negative log-likelihood using
        # nn.NLLLoss(), while also being more stable numerically...
        criterion = nn.NLLLoss()
        num_epochs = n_epochs

        # Train Model
        model, training_learning_data = train_model(model, X_train_k, y_train_k, criterion,
                                                                              optimizer, num_epochs=num_epochs)
        # Validate model and get accuracy score
        # Get log-likelihood on the validation set
        accuracy = validate_model(model, X_valid_k, y_valid_k)
        k_scores.append(accuracy)
        k += 1
        print(
            "k-fold: {}\taccuracy: {}".format(k, accuracy))

    k_scores = np.array(k_scores)
    print(k_scores)
    # Compute Average of scores
    k_scores = np.mean(k_scores, axis=0)
    print(k_scores)
    return k_scores

    # # Plot Cross Validation results
    # fig4, ax4 = plt.subplots(1, 1, figsize=(10, 10))
    # ax4.set_ylabel("Log Likelihood")
    # ax4.set_xlabel("# Components")
    # ax4.set_title("Cross Validation Resutls: Log Likelihood vs # Components")
    # ax4.plot(n_components_list, scores, marker='.')
    # ax4.plot(n_components_list[opt_s_i], scores[opt_s_i],
    #          marker='x', c='r', label="Optimal # Components")
    # ax4.legend()
    # plt.show()

    # opt_n_component = n_components_list[opt_s_i]
    # print("Optimal Num Components: {}".format(opt_n_component))
    # return opt_n_component


# Params
input_dim = trainset_x.shape[1]
# TODO: Cross validation to select hidden neurons
n_hidden_neurons = 64
output_dim = num_classes
lr = 0.01
num_epochs = 1000
momentum = 0.9


cross_validate(trainset_x, trainset_y, 2, input_dim,
               64, output_dim, lr, momentum, num_epochs)

'''
model = MLP(input_dim, n_hidden_neurons,
            output_dim).to(torch.device('cuda'))
print(model)

# Stochastic GD with learning rate and momentum hyperparameters
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# The nn.CrossEntropyLoss() loss function automatically performs a log_softmax() to
# the output when validating, on top of calculating the negative log-likelihood using
# nn.NLLLoss(), while also being more stable numerically...
criterion = nn.CrossEntropyLoss()


# Train Model
model, training_learning_data = train_model(
    model, trainset_x, trainset_y, criterion, optimizer, num_epochs=num_epochs)
'''