import itertools
import matplotlib.pyplot as plt
import pandas
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
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
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output_fc(x)
        # Don't need LogSoftMax if using CrossEntropyLoss
        # x = self.output_activation(x)
        return x


def model_predict(model, data):
    model.eval()
    X_test = torch.FloatTensor(data).to(torch.device('cuda'))

    # Evaluate nn on test data and compare to true labels
    predicted_labels = model(X_test)
    # print("predicted_labels:\n {}".format(predicted_labels))
    # Back to numpy
    predicted_labels = predicted_labels.cpu().detach().numpy()

    return np.argmax(predicted_labels, 1)


def train_model(model, X_train, y_train, X_valid, y_valid, criterion, optimizer, num_epochs=25, plot=False, early_stopping=False):
    # Early stopping Params
    last_loss = 100
    patience = 4
    threshold = 0.05
    trigger_times = 0
    min_epochs = 25

    # Apparently good practice to set this "flag" too before training
    # Does things like make sure Dropout layers are active, gradients are updated, etc.
    # Probably not a big deal for our toy network, but still worth developing good practice
    model.train()

    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(torch.device('cuda'))
    y_train = torch.LongTensor(y_train).to(torch.device('cuda'))
    X_valid = torch.FloatTensor(X_valid).to(torch.device('cuda'))
    y_valid = torch.LongTensor(y_valid).to(torch.device('cuda'))

    # For storing loss vs epoch
    training_loss_vs_epoch = []
    validation_loss_vs_epoch = []

    for epoch in range(num_epochs):

        # TRAIN STEP
        model.train()
        # Set grads to zero explicitly before backprop
        optimizer.zero_grad()
        outputs = model(X_train)
        train_loss = criterion(outputs, y_train)
        # Store training loss vs epoch for graphing
        training_loss_vs_epoch.append(
            [epoch, train_loss.cpu().detach().numpy()])
        # Backward pass to compute the gradients through the network
        train_loss.backward()
        # GD step update
        optimizer.step()

        # GET VALIDATION LOSS (EVAL)
        model.eval()
        y_valid_pred = model(X_valid)
        valid_loss = criterion(y_valid_pred, y_valid)
        validation_loss_vs_epoch.append(
            [epoch, valid_loss.cpu().detach().numpy()])

        # EARLY STOPPING
        if early_stopping:
            if not epoch < min_epochs and (valid_loss.item() > last_loss
                                           or np.abs(valid_loss.item() - train_loss.item()) > threshold
                                           or valid_loss.item() == np.nan):
                trigger_times += 1
                print('Trigger Times:', trigger_times)
                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    training_loss_vs_epoch = np.array(training_loss_vs_epoch)
                    validation_loss_vs_epoch = np.array(
                        validation_loss_vs_epoch)
                    return model, training_loss_vs_epoch, validation_loss_vs_epoch
            else:
                print('trigger times: 0')
                trigger_times = 0

            last_loss = valid_loss.item()

        print(f"Epoch: {epoch}/{num_epochs}, train_loss: {train_loss.cpu().detach().numpy()}, valid_loss: {valid_loss.cpu().detach().numpy()}")

    # Plot loss vs epoch
    training_loss_vs_epoch = np.array(training_loss_vs_epoch)
    validation_loss_vs_epoch = np.array(validation_loss_vs_epoch)
    if plot:
        plt.scatter(training_loss_vs_epoch[:, 0], training_loss_vs_epoch[:, 1])
        plt.xlabel("epoch")
        plt.ylabel("Loss function")
        plt.title("Value of the loss function")
        # plt.ylim((0, 1))
        plt.show()

    return model, training_loss_vs_epoch, validation_loss_vs_epoch


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
print(f"Is CUDA acceleration supported: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

trainset_raw = pandas.read_csv('airline_satisfaction_train.csv')
testset_raw = pandas.read_csv('airline_satisfaction_test.csv')
trainset_raw.head()


classes_str = ["neutral or dissatisfied", "satisfied"]
classes_int = [0, 1]
num_classes = len(classes_str)

print(trainset_raw.columns)
genders = np.unique(np.array(trainset_raw)[:,2])
customer_type = np.unique(np.array(trainset_raw)[:,3])
type_of_travel = np.unique(np.array(trainset_raw)[:,5])
seat_class = np.unique(np.array(trainset_raw)[:,6])
print(genders)
print(customer_type)
print(type_of_travel)
print(seat_class)

# Load training set
trainset = trainset_raw
# # Shuffle entire training set.
# trainset = trainset.sample(frac=1)
print("trainset:\n{}".format(trainset))
# Get rid of useless columns
del trainset["Unnamed: 0"]
del trainset["id"]
# Convert string labels to integer labels corresponding to index of class_names
trainset = trainset.replace(classes_str[0], value=classes_int[0], regex=True)
trainset = trainset.replace(classes_str[1], value=classes_int[1], regex=True)
# Replace other categorical data with numbers
for i in range(0,len(genders)):
    trainset = trainset.replace(genders[i], value=i, regex=True)
for i in range(0,len(customer_type)):
    trainset = trainset.replace(customer_type[i], value=i, regex=True)
for i in range(0,len(type_of_travel)):
    trainset = trainset.replace(type_of_travel[i], value=i, regex=True)
for i in range(0,len(seat_class)):
    trainset = trainset.replace(seat_class[i], value=i, regex=True)
# Get list of feature names
features = trainset.columns
print(features)
# Remove any sample with NULL value
trainset = remove_null_samples(trainset)
print(trainset)


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
# Convert string labels to integer labels corresponding to index of class_names
testset = testset.replace(classes_str[0], value=classes_int[0], regex=True)
testset = testset.replace(classes_str[1], value=classes_int[1], regex=True)
# Replace other categorical data with numbers
for i in range(0,len(genders)):
    testset = testset.replace(genders[i], value=i, regex=True)
for i in range(0,len(customer_type)):
    testset = testset.replace(customer_type[i], value=i, regex=True)
for i in range(0,len(type_of_travel)):
    testset = testset.replace(type_of_travel[i], value=i, regex=True)
for i in range(0,len(seat_class)):
    testset = testset.replace(seat_class[i], value=i, regex=True)

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


# VISUALIZE DATASET
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
# Visualize before scaling
# plt.scatter(np.arange(0, len(trainset_x)), trainset_x[:, 0])
# plt.show()
scaler = preprocessing.StandardScaler().fit(trainset_x)
trainset_x = scaler.transform(trainset_x)
scaler = preprocessing.StandardScaler().fit(testset_x)
testset_x = scaler.transform(testset_x)
# Visualize after scaling
# print([np.dtype(testset_x[0, i]) for i in range(0, 18)])
# print([np.dtype(trainset_x[0, i]) for i in range(0, 18)])
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


def cross_validate(_train_x, _train_y, _folds: int, _input_dim: int, _hidden_dim: int, _output_dim: int, _lr, _momentum, _n_epochs: int):
    '''
    Runs cross validation for a SINGLE set of hyperparmeters.
    Returns the accuracy score for the given set of hyperparameters.
    '''
    # Partition data
    kf = KFold(n_splits=_folds, shuffle=True)
    # CROSS VALIDATION
    k = 0
    k_scores = []
    # This is for storing the learning curve for the learning progress of the cross validation (NOT for generatlization error)
    training_loss_vs_epoch_k_list = []
    validation_loss_vs_epoch_k_list = []
    # NOTE that these subsets are of the TRAINING dataset
    for train_indices, valid_indices in kf.split(_train_x):
        # Extract the training and validation sets from the K-fold split
        X_train_k = _train_x[train_indices]
        X_valid_k = _train_x[valid_indices]
        y_train_k = _train_y[train_indices]
        y_valid_k = _train_y[valid_indices]

        model = MLP(_input_dim, _output_dim, _hidden_dim).to(
            torch.device('cuda'))
        # print(model)

        # Stochastic GD with learning rate and momentum hyperparameters
        optimizer = torch.optim.SGD(
            model.parameters(), lr=_lr, momentum=_momentum)

        # The nn.CrossEntropyLoss() loss function automatically performs a log_softmax() to
        # the output when validating, on top of calculating the negative log-likelihood using
        # nn.NLLLoss(), while also being more stable numerically...
        criterion = nn.CrossEntropyLoss()

        # Train Model
        model, training_loss_vs_epoch_k, validation_loss_vs_epoch_k = train_model(model, X_train_k, y_train_k, X_valid_k, y_valid_k, criterion,
                                                                                  optimizer, num_epochs=_n_epochs, plot=False)
        training_loss_vs_epoch_k_list.append(training_loss_vs_epoch_k)
        validation_loss_vs_epoch_k_list.append(validation_loss_vs_epoch_k)
        # Get accuracy score on validation set
        accuracy = validate_model(model, X_valid_k, y_valid_k)
        k_scores.append(accuracy)
        k += 1
        print(
            "k-fold: {}\taccuracy: {}".format(k, accuracy))

    # Average losses over all k-folds for training loss and validation loss
    training_loss_vs_epoch_k_list = np.array(training_loss_vs_epoch_k_list)
    training_loss_vs_epoch_k_list = np.mean(
        training_loss_vs_epoch_k_list, axis=0)
    validation_loss_vs_epoch_k_list = np.array(validation_loss_vs_epoch_k_list)
    validation_loss_vs_epoch_k_list = np.mean(
        validation_loss_vs_epoch_k_list, axis=0)

    k_scores = np.array(k_scores)
    print(k_scores)
    # Compute Average of scores
    k_score = np.mean(k_scores, axis=0)
    print(k_score)
    return k_score, training_loss_vs_epoch_k_list, validation_loss_vs_epoch_k_list

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


def generate_hyper_params_grid(params):
    params_grid = list(itertools.product(*params))
    params_grid = np.array(params_grid)
    # print(params_grid)
    print(f"param_grid.shape: {params_grid.shape}")
    return params_grid.tolist()


def grid_search_cv():
    # Permanent Hyperparams
    input_dim = trainset_x.shape[1]
    output_dim = num_classes
    k_folds = 5
    # Params to try for Cross validation
    n_hidden_neurons_list = [512]
    lr_list = [.1, .3]
    # lr_list = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]
    momentum_list = [0.99]
    num_epochs_list = [200]

    # n_hidden_neurons = [2, 256]
    # lr = [.01]
    # momentum = [0.9]
    # num_epochs = [1000]

    # Get grid of all param combinations
    params_grid = generate_hyper_params_grid(
        [n_hidden_neurons_list, lr_list, momentum_list, num_epochs_list])

    # For Plotting Results
    print(len(params_grid))
    fig, ax = plt.subplots(len(params_grid), 2, figsize=(10, 10))

    grid_search_scores = []
    param_counter = 1
    for param in params_grid:
        neurons = int(param[0])
        lr = param[1]
        momentum = param[2]
        epochs = int(param[3])

        percent_done = (param_counter/len(params_grid))*100
        print(
            f"GridSearch:\thidden_neurons: {neurons}\tlr: {lr}\tmomentum: {momentum}\tepochs: {epochs}\tpercent_done: {percent_done}%")

        cv_score, training_loss_vs_epoch_k, validation_loss_vs_epoch_k = cross_validate(_train_x=trainset_x, _train_y=trainset_y, _folds=k_folds, _input_dim=input_dim, _hidden_dim=neurons,
                                                                                        _output_dim=output_dim, _lr=lr, _momentum=momentum, _n_epochs=epochs)
        grid_search_scores.append(cv_score)

        # Plot training and validation loss vs epoch (using the training loss from CV averaged over k-folds and the validation loss from CV averaged over k-folds)
        # This is an indicator of how well the CV is learning. (NOT for generalizability as we are not using an unseen dataset here)
        # ax[param_counter-1, 0].scatter(
        #     training_loss_vs_epoch_k[:, 0], training_loss_vs_epoch_k[:, 1], label="Training loss")
        # ax[param_counter-1, 0].scatter(
        #     validation_loss_vs_epoch_k[:, 0], validation_loss_vs_epoch_k[:, 1], label="Validation loss")
        # ax[param_counter-1, 0].legend()
        # ax[param_counter-1, 0].set_xlabel("Epoch")
        # ax[param_counter-1, 0].set_ylabel("Loss function")
        # ax[param_counter-1, 0].set_title(
        #     f"Cross Validation Training Loss vs. Validation Loss\nParams: hidden_neurons: {neurons} lr: {lr} momentum: {momentum} epochs: {epochs}")
        # ax[param_counter-1, 0].set_ylim((0, 1))

        # # Plot training and test loss vs epoch (using the full training and test datasets)
        # # This is an indicator of the generalizability of the model.
        # model = MLP(input_dim, output_dim, neurons).to(
        #     torch.device('cuda'))
        # optimizer = torch.optim.SGD(
        #     model.parameters(), lr=lr, momentum=momentum, )
        # criterion = nn.CrossEntropyLoss()
        # model, training_loss_vs_epoch_k, validation_loss_vs_epoch_k = train_model(model, trainset_x, trainset_y, testset_x, testset_y, criterion,
        #                                                                           optimizer, num_epochs=epochs, plot=False)
        # ax[param_counter-1, 1].scatter(
        #     training_loss_vs_epoch_k[:, 0], training_loss_vs_epoch_k[:, 1], label="Training loss")
        # ax[param_counter-1, 1].scatter(
        #     validation_loss_vs_epoch_k[:, 0], validation_loss_vs_epoch_k[:, 1], label="Test loss")
        # ax[param_counter-1, 1].legend()
        # ax[param_counter-1, 1].set_xlabel("Epoch")
        # ax[param_counter-1, 1].set_ylabel("Loss function")
        # ax[param_counter-1, 1].set_title(
        #     f"Training Loss vs. Test Loss\nParams: hidden_neurons: {neurons} lr: {lr} momentum: {momentum} epochs: {epochs}")
        # ax[param_counter-1, 1].set_ylim((0, 1))
        # testset_y_pred = model_predict(model, testset_x)
        # print(classification_report(testset_y, testset_y_pred))

        param_counter = param_counter + 1

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(
        np.array(lr_list), np.array(grid_search_scores))
    ax.legend()
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("CV Score/Accuracy (higher=better)")
    ax.set_title(
        f"Accuracy vs. Learning Rate")
    # ax.set_ylim((0, 1))
    plt.show()

    grid_search_scores = np.array(grid_search_scores)
    print(f"grid_search_scores:\n {grid_search_scores}")
    optimal_params = params_grid[np.argmax(grid_search_scores)]
    print(optimal_params)
    plt.show()


def demonstrate_underfitting_overfitting():
    global trainset_x
    global trainset_y
    trainset_x = trainset_x[0:5000, :]
    trainset_y = trainset_y[0:5000]
    # Hyperparams
    input_dim = trainset_x.shape[1]
    output_dim = num_classes
    neurons = 128
    lr = 0.1
    momentum = 0.99
    epochs = 500

    model = MLP(input_dim, output_dim, neurons).to(
        torch.device('cuda'))
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, )
    criterion = nn.CrossEntropyLoss()
    model, training_loss_vs_epoch_k, validation_loss_vs_epoch_k = train_model(model, trainset_x, trainset_y, testset_x, testset_y, criterion,
                                                                              optimizer, num_epochs=epochs, plot=False, early_stopping=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(
        training_loss_vs_epoch_k[:, 0], training_loss_vs_epoch_k[:, 1], label="Training loss")
    ax.plot(
        validation_loss_vs_epoch_k[:, 0], validation_loss_vs_epoch_k[:, 1], label="Test loss")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss function")
    ax.set_title(
        f"Training Loss vs. Test Loss\nParams: hidden_neurons: {neurons} lr: {lr} momentum: {momentum} epochs: {epochs}")
    # ax.set_ylim((0, 1))
    testset_y_pred = model_predict(model, testset_x)
    print(f"Test Accuracy: {accuracy_score(testset_y, testset_y_pred)}")
    print(classification_report(testset_y, testset_y_pred))
    plt.show()


def train_final_model():
    # Cross Validation selected params:
    input_dim = trainset_x.shape[1]
    n_hidden_neurons = 512
    output_dim = num_classes
    lr = 0.3
    num_epochs = 200
    momentum = 0.99

    model = MLP(input_dim, output_dim, n_hidden_neurons).to(
        torch.device('cuda'))
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    model, training_loss_vs_epoch_k, validation_loss_vs_epoch_k = train_model(model, trainset_x, trainset_y, testset_x, testset_y, criterion,
                                                                              optimizer, num_epochs=num_epochs, plot=False, early_stopping=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(
        training_loss_vs_epoch_k[:, 0], training_loss_vs_epoch_k[:, 1], label="Training loss")
    ax.scatter(
        validation_loss_vs_epoch_k[:, 0], validation_loss_vs_epoch_k[:, 1], label="Test loss")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss function")
    ax.set_title(
        f"Training Loss vs. Test Loss\nParams: hidden_neurons: {n_hidden_neurons} lr: {lr} momentum: {momentum} epochs: {num_epochs}")
    # ax.set_ylim((0, 1))
    testset_y_pred = model_predict(model, testset_x)
    print(f"Test Accuracy: {accuracy_score(testset_y, testset_y_pred)}")
    print(classification_report(testset_y, testset_y_pred))
    plt.show()


if __name__ == "__main__":
    # demonstrate_underfitting_overfitting()
    # grid_search_cv()
    train_final_model()
