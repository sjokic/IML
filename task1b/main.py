import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

#Read data
data = pd.read_csv("train.csv")

#Feature transformation
arr = np.transpose(np.array([data['x1'] ,data['x2'], data['x3'], data['x4'], data['x5'], data['x1']**2, data['x2']**2, data['x3']**2, data['x4']**2, data['x5']**2,  np.exp(data['x1']), np.exp(data['x2']),np.exp(data['x3']), np.exp(data['x4']), np.exp(data['x5']), np.cos(data['x1']), np.cos(data['x2']), np.cos(data['x3']), np.cos(data['x4']), np.cos(data['x5']), np.ones(700)]))

phi = pd.DataFrame(arr, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21'])

#Apply Lasso linear model and select best model via leave-one-out cross-validation
reg = LassoCV(alphas = np.logspace(-2, 2, 100), fit_intercept = False, max_iter = 10000, tol = 0.00001, cv = 700, n_jobs = -1).fit(phi,data['y'])

np.set_printoptions(precision=12)

#Print results and store in submission.csv
print('Weights: \n', reg.coef_)

print('RMSE: ', mean_squared_error(data['y'],reg.predict(phi.loc[:,'x1':'x21']), squared=False))

df = pd.DataFrame(reg.coef_)
df.to_csv('submission.csv', index=False, header=False)