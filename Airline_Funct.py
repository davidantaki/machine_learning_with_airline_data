import torch
import torch.nn as nn
import numpy as np


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
        # self.output_activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.input(x)
        x = self.input_activation(x)
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.output_fc(x)
        # x = self.output_activation(x)
        return x


def train_model(model, data, labels, criterion, optimizer, num_epochs=25):

    # Apparently good practice to set this "flag" too before training
    # Does things like make sure Dropout layers are active, gradients are updated, etc.
    # Probably not a big deal for our toy network, but still worth developing good practice
    X_train = torch.FloatTensor(data)

    # Optimize the neural network
    y_train = torch.LongTensor(labels)
    for epoch in range(num_epochs):
        # Set grads to zero explicitly before backprop
        optimizer.zero_grad()
        outputs = model(X_train)
        # Criterion computes the cross entropy loss between input and target
        loss = criterion(outputs, y_train)
        # Backward pass to compute the gradients through the network
        loss.backward()
        # GD step update
        optimizer.step()

    return model


def model_predict(model, data):
    # Similar idea to model.train(), set a flag to let network know your in "inference" mode
    X_test = torch.FloatTensor(data)

    # Evaluate nn on test data and compare to true labels
    predicted_labels = model(X_test)
    print("predicated_labels:\n {}".format(predicted_labels))
    # Back to numpy
    predicted_labels = predicted_labels.detach().numpy()

    return np.argmax(predicted_labels, 1)
