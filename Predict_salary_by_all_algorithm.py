import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\paruc\Desktop\DS\2. Dec\18th\emp_sal.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#SVR
from sklearn.svm import SVR
regressor_SVR = SVR(degree = 3,kernel = 'poly')
regressor_SVR.fit(X, y)

y_pred_SVR = regressor_SVR.predict([[6.5]])
y_pred_SVR

#Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor_RF = RandomForestRegressor(criterion='squared_error',random_state=0,n_estimators=10)
regressor_RF.fit(X, y)

y_pred_RF = regressor_RF.predict([[6.5]])
y_pred_RF

#Decision Tree
from sklearn.tree import DecisionTreeRegressor
regressor_DT = DecisionTreeRegressor(criterion="absolute_error", splitter="random", random_state = 0)
regressor_DT.fit(X, y)

y_pred_DT = regressor_DT.predict([[6.5]])
y_pred_DT

#KNN
from sklearn.neighbors import KNeighborsRegressor
regressor_KNN = KNeighborsRegressor(n_neighbors = 4,algorithm = "ball_tree",weights = 'uniform')
regressor_KNN.fit(X, y)

y_pred_KNN = regressor_KNN.predict([[6.5]])
y_pred_KNN

#Polynomial
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5,order = 'C')
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

poly_model_pred = lin_reg.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred

