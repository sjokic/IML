# IML
Project assignments for the IML course @ ETHZ

All assignments passed the hardest grading baseline.

---
### Task 1:
+ These were simple, introductory tasks. The other tasks were more involved.
+ In task 1a the goal was to perform 10-fold cross validaiton (CV) for different choices of the regularisation parameter lambda when fitting a ridge regression model and then to compute the RMSE for each fold. Finally, the mean RMSE over all 10 folds is computed.
+ In task 1b the goal was to perform a feature transformation on data and then fit a lasso regression model.
---
### Task 2:
+ This task dealt with medical data from patients (e.g. their Lactate, SaO2, EtCO2, etc. levels) and the goal was to make predictions on whether a medical test of a certain kind would have to be performed based on these measurements.
+ Before solving the subtasks, data imputation had to be performed because of missing values. The data was then also normalized to bring each of the different features to the same scale.
+ In subtask 1, the goal was to predict whether further  medical tests are required, with 0 meaning that no further tests are required and 1 meaning that at least one of a test of a particular kind is required. The labels which have to be predicted for this task are: LABEL_BaseExcess, LABEL_Fibrinogen, LABEL_AST, LABEL_Alkalinephos, LABEL_Bilirubin_total, LABEL_Lactate, LABEL_TroponinI, LABEL_SaO2, LABEL_Bilirubin_direct, LABEL_EtCO2. This subtask was solved using a One-vs-rest (OVR) classifier with sklearn's HistGradientBoostingClassifier. 
+ In subtask 2, the goal was to predict whether sepsis will occur, with 0 meaning that sepsis will not occur and 1 that it will occur. The label that has to be predicted is LABEL_Sepsis. This subtask was solved using sklearn's HistGradientBoostingClassifier.
+ In subtask 3, the goal was to predict future mean values of key vital signs. The labels which have to be predicted are: LABEL_RRate, LABEL_ABPm, LABEL_SpO2, LABEL_Heartrate. This subtask was solved using sklearn's HistGradientBoostingRegressor. 

---
### Task 3:

+ The goal in this task was to classify human proteins as active (1) or inactive (0) based on amino acid differences. The sequences of amino acids (length 4, we only considered 4 fixed mutation sites) were one-hot encoded before being passed as input to a multilayer perceptron neural network (using sklearn's MLPClassifier) which consisted of a single hidden layer of 300 perceptrons. The MLP then predicts 0 if the mutation of the 4 amino acid sites causes the protein to become inactive and 1 otherwise.

---
### Task 4:

+ Given a query triplet of images of food (A, B, C), the goal in this task was to predict 1 if A is more similar in taste to B and 0 if A is more similar in taste to C.
The task was solved using a triplet network (in particular it employed triplet loss), i.e. an identical neural network with shared weights was used to compute the embedding for A, B and  C and then evaluate the triplet loss function. In particular, the neural network architecture that was used to compute the embeddings consisted of 3 convolutional neural networks (CNNs) in parallel: two small CNNs to extract low res features and one deep CNN to capture the main features (which used the ResNet50 pre-trained model). The outputs of each of the CNNs were concatenated and then normalized to produce the final embedding. This task was solved using TensorFlow.
+ The implementation is based on the paper ['Learning Fine-grained Image Similarity with Deep Ranking' by Jian Wang et al.](https://arxiv.org/abs/1404.4661)
