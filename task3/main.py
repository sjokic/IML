import pandas as pd
from sklearn.neural_network import MLPClassifier
import time



#read training and test data
print("Loading Files")
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

print("Preprocessing data")
train_length = train_data.shape[0]
test_length = test_data.shape[0]

t1 = time.time()
columns = []
alphabet ="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
for i in range(1, 5):
    for a in alphabet:
        columns.append(a+str(i))
columns.append('Active')


train_data_array = []
for i in range(0, train_length):
    pos = 1
    arr = [0]*105
    for mut in train_data.loc[ i , 'Sequence']:
        pos_in_array = alphabet.find(mut) + (pos-1)*26
        arr[pos_in_array] = 1
        pos+=1
    if train_data.loc[ i , 'Active'] == 1:
        arr[104] = 1
    train_data_array.append(arr)    
train_processed = pd.DataFrame(data=train_data_array, columns = columns)
#train_processed.to_csv('train_processed.csv', index = False)

test_data_array = []
for i in range(0, test_length):
    pos = 1
    arr = [0]*104
    for mut in test_data.loc[ i , 'Sequence']:
        pos_in_array = alphabet.find(mut) + (pos-1)*26
        arr[pos_in_array] = 1
        pos+=1
    test_data_array.append(arr)    
test_processed = pd.DataFrame(data=test_data_array, columns = columns[:-1])
#test_processed.to_csv('test_processed.csv', index = False)
    
t2 = time.time()
print("Time for preprocessing: ", (t2-t1), "s")

print("Training Network")
t1 = time.time()
X_train = train_processed.loc[:, 'A1':'Z4']
y_train = train_processed.loc[::, 'Active']
clf = MLPClassifier(hidden_layer_sizes=(300), random_state=1510)
clf.fit(X_train, y_train)
t2 = time.time()
print("Time for training: ", (t2-t1), "s")

print("Making prediction")
t1 = time.time()
res = clf.predict(test_processed)
t2 = time.time()
print("Time for prediction: ", (t2-t1), "s")

print(res)

#We dont use dataframe.to_csv here because the first line would be the column name '1'
#and there would be 48001 lines in total
f = open('prediction.csv', 'w')
for i in res:
    f.write(str(i)+"\n")
f.close()



