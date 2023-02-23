# Airline Passenger Satisfaction Classification
The goal of this was to use machine learning to predict how satisfied an airline passenger would be after their flight based on a 25 feature dataset/survey.
Both a MLP neural network and an LDA classifier were used to fit the dataset. Both models were compared.

## Dataset Info
Dataset: kragle
Data from airline passenger survey
25 features (Gender, Customer Type, Age, Type of Travel, etc.)
Testing Sample Size: 26,986 samples
Training Sample Size: 103,904 samples 
Problem Type: Supervised, classification.
The Plan:
* Train a ML model to predict whether a passenger will be satisfied based on features in dataset
  * e.g. age, ease of online booking, gate location, arrival delay, and departure delay.
* Multi-layer neural network
* GridSearch + Cross validation for selecting hyperparameters.
* # layers and # neurons, momentum, learning rate, etc.
* Binary classification: neutral/dissatisfied, satisfied
* Ensure not overfitting
* Will use early stopping

## See the report for the process and further findings
[David Antaki and Maulik Patel Final ML Report 2.pdf](https://github.com/davidantaki/machine_learning_with_airline_data/files/10815508/David.Antaki.and.Maulik.Patel.Final.ML.Report.2.pdf)


## MLP vs LDA Findings
An LDA classifier was trained on the dataset and produced an accuracy of 83.4%. A 4 layer MLP was trained that
produced a final accuracy of 93.5%, nearly 10% better than the LDA classifier. To train the MLP model, feature scaling
was first performed on the data which had a significant impact (i.e. a neural network must have data scaled).
Hyperparameters were then tuned using grid search 10-fold cross validation, and regularization was introduced to the
training process via early stopping and dropout though the model was not significantly overfit to begin with.
