import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt  # For general plotting


class MLP(nn.Module):
    # The nn.CrossEntropyLoss() loss function automatically performs a log_softmax() to
    # the output when validating, on top of calculating the negative-log-likelihood using
    # nn.NLLLoss(), while also being more stable numerically... So don't implement from scratch
    # https://pytorch.org/docs/stable/generated/torch.nn.Module.html

    def __init__(self, input_dim, hidden_dim, C):
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


def train_model(model, data, labels, criterion, optimizer, num_epochs=25):

    # Apparently good practice to set this "flag" too before training
    # Does things like make sure Dropout layers are active, gradients are updated, etc.
    # Probably not a big deal for our toy network, but still worth developing good practice
    X_train = torch.FloatTensor(data).to(torch.device('cuda'))

    # Optimize the neural network
    y_train = torch.LongTensor(labels).to(torch.device('cuda'))

    # For storing loss vs iteration
    loss_vs_iteration = []

    for epoch in range(num_epochs):
        # Set grads to zero explicitly before backprop
        optimizer.zero_grad()
        outputs = model(X_train)
        # Criterion computes the cross entropy loss between input and target
        loss = criterion(outputs, y_train)
        # Store loss vs epoch for graphing
        loss_vs_iteration.append([epoch, loss.cpu().detach().numpy()])
        # Backward pass to compute the gradients through the network
        loss.backward()
        # GD step update
        optimizer.step()

    # Plot loss vs epoch
    loss_vs_iteration = np.array(loss_vs_iteration)
    plt.scatter(loss_vs_iteration[:,0], loss_vs_iteration[:,1])
    plt.xlabel("epoch")
    plt.ylabel("Loss function")
    plt.title("Value of the loss function")
    plt.ylim((0,1))
    plt.show()

    return model


def model_predict(model, data):
    # Similar idea to model.train(), set a flag to let network know your in "inference" mode
    X_test = torch.FloatTensor(data).to(torch.device('cuda'))

    # Evaluate nn on test data and compare to true labels
    predicted_labels = model(X_test)
    print("predicated_labels:\n {}".format(predicted_labels))
    # Back to numpy
    predicted_labels = predicted_labels.cpu().detach().numpy()

    return np.argmax(predicted_labels, 1)
