import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import RidgeCV
import time

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

train_features = pd.read_csv('train_features.csv')
train_labels = pd.read_csv('train_labels.csv')
test_features = pd.read_csv('test_features.csv')

#compute mean for each feature in the training and test set
means_train = pd.DataFrame(data = np.nanmean(train_features.to_numpy()[:,1:], axis=0)).T
means_train.columns = train_features.drop(columns=['pid']).columns
means_train = means_train.to_dict('records')

means_test = pd.DataFrame(data = np.nanmean(test_features.to_numpy()[:,1:], axis=0)).T
means_test.columns = test_features.drop(columns=['pid']).columns
means_test = means_test.to_dict('records')

features = train_features.columns.to_numpy()

### training data imputation ###

#per patient imputation for columns containing at least one finite value (using per patient median of values for each feature)
for f in features:
    train_features[f] = train_features.groupby('pid')[f].apply(lambda x:x.fillna(x.median()))

#imputation for the left over columns containing only NaNs for a given patient (using global mean for each feature)
imp_train_features = train_features.fillna(value = means_train[0])
imp_train_features.to_csv('imp_train_features.csv', index=False, header=True)


### test data imputation ###

#per patient imputation for columns containing at least one finite value (using per patient median of values for each feature)
for f in features:
    test_features[f] = test_features.groupby('pid')[f].apply(lambda x:x.fillna(x.median()))

#imputation for the left over columns containing only NaNs for a given patient (using global mean for each feature)
imp_test_features = test_features.fillna(value = means_test[0])
imp_test_features.to_csv('imp_test_features.csv', index=False, header=True)


imp_train_features = pd.read_csv('imp_train_features.csv').drop(columns=['pid'])
imp_test_features = pd.read_csv('imp_test_features.csv').drop(columns=['pid'])

#standardization and normalization of training and test set
scaler = preprocessing.StandardScaler().fit(imp_train_features)
norm = preprocessing.Normalizer()

imp_train_features = pd.DataFrame(data = norm.transform(scaler.transform(imp_train_features)))
imp_test_features = pd.DataFrame(data = norm.transform(scaler.transform(imp_test_features)))

imp_train_features = pd.concat([train_features.iloc[:, 0], imp_train_features], axis=1, ignore_index=True)
imp_train_features.columns=train_features.columns

imp_test_features = pd.concat([test_features.iloc[:, 0], imp_test_features], axis=1, ignore_index=True)
imp_test_features.columns=test_features.columns

#put the 12 measurements per patient into a single vector each
train_pids = train_features.pid.unique()
grps_train = imp_train_features.groupby('pid')

patient_train_data = []

for pid in train_pids:
    patient_vector = np.ndarray.flatten(grps_train.get_group(pid).drop(columns=['pid']).to_numpy())
    patient_train_data.append(patient_vector)

test_pids = test_features.pid.unique()
grps_test = imp_test_features.groupby('pid')

patient_test_data = []

for pid in test_pids:
    patient_vector = np.ndarray.flatten(grps_test.get_group(pid).drop(columns=['pid']).to_numpy())
    patient_test_data.append(patient_vector)


#subtask 1
t1 = time.time()
clf = HistGradientBoostingClassifier(random_state=1510)
                
multi_clf = OneVsRestClassifier(clf, n_jobs=-1)

multi_clf.fit(patient_train_data, train_labels.to_numpy()[:, 1:11])

res = 1 / (1 + np.exp(-multi_clf.decision_function(patient_test_data)))

task1_df = pd.DataFrame(data=res, columns=train_labels.columns[1:11])

task1_df.to_csv('subtask1.csv', index=False, header=True)

t2 = time.time()
print('subtask1, time taken: ', t2-t1)
print(task1_df)


#subtask 2
t1 = time.time()
clf.fit(patient_train_data, train_labels.to_numpy()[:, 11])

res = 1 / (1 + np.exp(-clf.decision_function(patient_test_data)))

task2_df = pd.DataFrame(data=res, columns=[train_labels.columns[11]])

task2_df.to_csv('subtask2.csv', index=False, header=True)

t2 = time.time()
print('subtask2, time taken: ', t2-t1)
print(task2_df)


#subtask 3
t1 = time.time()
reg = HistGradientBoostingRegressor(random_state=1510)

res = np.empty( shape=(len(patient_test_data), 4))
for feat in range(12, 16):
    reg.fit(patient_train_data, train_labels.to_numpy()[:, feat])
    res[:,feat-12] = reg.predict(patient_test_data)

task3_df = pd.DataFrame(data=res, columns=train_labels.columns[12:])

task3_df.to_csv('subtask3.csv', index=False, header=True)

t2 = time.time()
print('subtask3, time taken: ', t2-t1)
print(task3_df)


#task1_df = pd.read_csv('subtask1.csv')
#task2_df = pd.read_csv('subtask2.csv')
#task3_df = pd.read_csv('subtask3.csv')


#combine results
pids = pd.DataFrame(data=test_features.pid.unique(), columns=['pid'])
sol = pd.concat([pids, task1_df, task2_df, task3_df], axis=1, ignore_index=True)
sol.columns = train_labels.columns
sol.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')