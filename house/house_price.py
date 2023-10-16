import os #provides functions for interacting with the operating system
import numpy as np
from matplotlib import *
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
#####################################
data=pd.read_csv('house price.csv')
print(data.head())
print(data.columns)
print(data.info())
print(data.shape)
print(data.corr())
###########X1 & house age have less affect on house price so we can not select  them.
# Heatmap showing the correlations
#sns.heatmap(data.corr(), annot=True,cmap='Reds')
#plt.show()
sns.displot(data= data, x='Y house price of unit area' , bins=25 , kde=True, height=6)
plt.title("Distribution of House Price per Unit Area")
sns.displot(data= data, x='X2 house age' , bins=25 , kde=True, height=6)
plt.title("Distribution of House Age")
sns.displot(data= data, x='X3 distance to the nearest MRT station' , bins=25 , kde=True, height=6)
plt.title("Distribution of distance to the nearest MRT station")
sns.displot(data= data, x='X4 number of convenience stores' , bins=25 , kde=True, height=6)
plt.title("Distribution of number of Nearby convenience stores")
sns.displot(data= data, x='X5 latitude' , bins=25 , kde=True, height=6)
plt.title("Distribution of latitude")
sns.displot(data= data, x='X6 longitude' , bins=25 , kde=True, height=6)
plt.title("Distribution of longitude")
#sns.scatterplot(data=data, y=data['Y house price of unit area'], x=data['X1 transaction date'] , hue= 'X2 house age', palette="rocket")
plt.show()
sns.scatterplot(data=data, y=data['Y house price of unit area'], x=data['X1 transaction date'] , hue= 'X2 house age', palette="rocket")
plt.figure(figsize=(5, 5), dpi=100)
sns.scatterplot(data=data, y=data['Y house price of unit area'], x=data['X3 distance to the nearest MRT station'] , hue= 'X4 number of convenience stores', palette="rocket")
plt.show()
##############3most of price is about 40$$$$$2- most of house age is about 15 years3- most of houses is nearest 0to 800 m to MRT 4-as the date is nearest now and age is small the price is high
#5-the price increase as house is near MRT and has more convienence stores
################split data into x ,y
from sklearn.preprocessing import PolynomialFeatures
X = data.drop('Y house price of unit area',axis=1).values
y = data['Y house price of unit area'].values
PF=PolynomialFeatures(degree=2)
x_poly=PF.fit_transform(X)
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(x_poly, y, test_size=0.3, random_state=101)
polymodel=linear_model.LinearRegression()
polymodel.fit(X_train_poly, y_train_poly)
y_pred=polymodel.predict(X_test_poly)
##########to show error resiudual
print(pd.DataFrame({'Y_Test': y_test_poly,'Y_Pred':y_pred, 'Residuals':(y_test_poly-y_pred) }).head(5))
#######################evaluation
MAE_Poly = mean_absolute_error(y_test_poly,y_pred)
MSE_Poly = mean_squared_error(y_test_poly,y_pred)
print('MSE = ',MSE_Poly)
print('MAE = ',MAE_Poly)
###############TO use linear regression
XS_train, XS_test, ys_train, ys_test = train_test_split(X, y, test_size=0.3, random_state=101)
simplemodel=linear_model.LinearRegression()
simplemodel.fit(XS_train, ys_train)
ys_pred=simplemodel.predict(XS_test)
MAE_simple = mean_absolute_error(ys_test,ys_pred)
MSE_simple = mean_squared_error(ys_test,ys_pred)
print('MSE(linear) = ',MSE_simple)
print('MAE(linear) = ',MAE_simple)
############################using polynomial decreases error
##########################################################################ridege regression
from sklearn.linear_model import RidgeCV
ridge_cv_model=RidgeCV(alphas=(0.1, 1.0, 10.0), scoring='neg_mean_absolute_error')
ridge_cv_model.fit(X_train_poly, y_train_poly)
y_pred_ridge_cv = ridge_cv_model.predict(X_test_poly)
MAE_ridge_cv= mean_absolute_error(y_test_poly, y_pred_ridge_cv)
MSE_ridge_cv= mean_squared_error(y_test_poly, y_pred_ridge_cv)
print('MSE_RIDGE = ',MSE_ridge_cv)
print('MAE_RIDGE = ',MAE_ridge_cv)############IT has less error than linear
##################################Lasso Regression
from sklearn.linear_model import LassoCV
lasso_cv_model= LassoCV(eps=0.01, n_alphas=100, cv=5,max_iter=10000, tol=0.0001)
lasso_cv_model.fit(X_train_poly, y_train_poly)
y_pred_lasso= lasso_cv_model.predict(X_test_poly)
MAE_Lasso_cv= mean_absolute_error(y_test_poly, y_pred_lasso)
MSE_Lasso_cv= mean_squared_error(y_test_poly, y_pred_lasso)
print('MAE_Lasso_cv = ',MAE_Lasso_cv)
print('MSE_Lasso_cv = ',MSE_Lasso_cv)






