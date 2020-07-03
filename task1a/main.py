import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
  
# read data

data = pd.read_csv("train.csv")
x = pd.DataFrame(data.loc[:,'x1':])
y = pd.DataFrame(data.loc[:,'y'])

# initialize array of lambdas specified in the task description 

lambdas = [0.01, 0.1, 1, 10, 100]

result = []

for lam in lambdas:

    rmse = [] # rmse for each test fold
    
    for i in range(0, 10):
    
        # generate folds
        
        fold_size = int(len(y)/10)
        index = data.index.isin(range(i*fold_size, (i+1)*fold_size))
    
        x_test = x[index]
        x_train = x[~index]
        
        y_test = y[index]
        y_train = y[~index]
        
        # closed form solution (np.eye creates an identity matrix)
        
        weights = np.dot(np.linalg.pinv(np.dot(x_train.T, x_train) + lam * np.eye(13)), np.dot(x_train.T, y_train))
        
        y_predicted = np.matmul(x_test.to_numpy(), weights)
        
        rmse.append(mean_squared_error(y_test, y_predicted, squared = False))
        
    result.append(np.mean(rmse))

# print results and store in submission.csv

print(result)
df = pd.DataFrame(result)
df.to_csv('submission.csv', index=False, header=False)