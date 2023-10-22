import pandas as pd
import numpy as np
import os #provides functions for interacting with the operating system
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlxtend.evaluate import bias_variance_decomp
from sklearn.tree import DecisionTreeRegressor
##########################
df=pd.read_csv('Predict Price of Airline Tickets.csv')
print(df.head())
print(df.describe(include='all'))
print(df.columns)
df.dropna(inplace=True)
print(df.isnull().sum())
###############converting date of journey(Date_of_Journey) to date type
df['day_of_Journey']=pd.to_datetime(df.Date_of_Journey, format="%d/%m/%Y").dt.day
df['month_of_Journey']=pd.to_datetime(df.Date_of_Journey, format="%d/%m/%Y").dt.month
df.drop(["Date_of_Journey"], axis = 1, inplace = True)
# Departure time is when a plane leaves the gate
#df["Dep_hour"] = pd.to_datetime(df["Dep_Time"],format='mixed').dt.hour
#df["Dep_min"] = pd.to_datetime(df["Dep_Time"],format='mixed').dt.minute
#df.drop(["Dep_Time"], axis = 1, inplace = True)
##################arrival of air plane
df["Dep_hour"] = pd.to_datetime(df["Dep_Time"],format='mixed').dt.hour
df["Dep_min"] = pd.to_datetime(df["Dep_Time"],format='mixed').dt.minute
df.drop(["Dep_Time"], axis = 1, inplace = True)
###########################converting duration
duration = list(df["Duration"])
for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i] #adds 0h
duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))
df["Duration_hours"] = duration_hours
df["Duration_mins"] = duration_mins
df.drop(["Duration"], axis = 1, inplace = True)
######################stop station
df.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

#########################
sns.barplot(x='Airline',y='Price' ,data=df)
plt.show()
sns.countplot(x='Airline',data=df)
plt.show()
print(df.Airline.value_counts())
plt.figure(figsize = (10, 10))
plt.title('Price VS Additional Information')
plt.scatter(df['Additional_Info'], df['Price'])
plt.xticks(rotation = 90)
plt.xlabel('Information')
plt.ylabel('Price of ticket')
plt.show()
plt.title('Count of flights month wise')
sns.countplot(x = 'month_of_Journey', data = df)
plt.xlabel('Month')
plt.ylabel('Count of flights')
plt.show()
print(df.month_of_Journey.unique())
plt.figure(figsize = (10, 10))
plt.title('Count of flights with different Airlines')
sns.countplot(x = 'Airline', data = df)
plt.xlabel('Airline')
plt.ylabel('Count of flights')
plt.xticks(rotation = 90)
plt.show()
##############1-Jet airways business is the most price 2-jet airway is the most company make trips 3-business class has gighest price 
#4- the most 2 monthes has flights are 5 & 6. 5-jet airways has the most number of trips 6-
##########how to invert obj columns into int
from sklearn.preprocessing import LabelEncoder
Le_encoder = LabelEncoder()
df[['Airline','Source','Destination','Additional_Info']]=df[['Airline','Source','Destination','Additional_Info']].apply(Le_encoder.fit_transform)
print(df.dtypes)
df.drop(['Arrival_Time'], axis = 1, inplace = True)
df.drop(['Route'], axis = 1, inplace = True)
print(df.head())
##################################all of columns now are ready to use regression with them
#############trying polynomial
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
x=df.drop(['Price'],axis=1).values
y=df['Price'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 12,test_size = 0.3)
#for i in range(1,15):
 # poly_reg = PolynomialFeatures(degree =i)
  #x_poly_train = poly_reg.fit_transform(x_train)
  #x_poly_test = poly_reg.fit_transform(x_test)
  #PL_model = linear_model.LinearRegression()
  #PL_model.fit(x_poly_train,y_train)
  #y_pred_poly = PL_model.predict(x_poly_test)
  #avg_Error, avg_bais, avg_var = bias_variance_decomp(PL_model, x_poly_train,y_train, x_poly_test, y_test, loss='mse', random_seed = 1)
  #print(' * '*10, 'Degree = ',i,' * '*10)
  #print('Error MSE = ',avg_Error )
  #print('Error bais = ',avg_bais )
  #print('Error variance = ',avg_var )
####from results polynomial not good regression 
#########################using decision tree regression ###################
from sklearn.model_selection import GridSearchCV
D_T_reg_model = DecisionTreeRegressor(random_state=101) # ,max_depth=3
D_T_reg_model.fit(x_train,y_train)
depth  =list(range(3,30))
param_grid =dict(max_depth =depth)
tree =GridSearchCV(DecisionTreeRegressor(random_state=101),param_grid,cv =10)
tree.fit(x_train,y_train)
print(tree.best_estimator_)
##############max depth=10
y_pred=tree.predict(x_test)
print("Train Results for Decision Tree Regressor Model:")
print(50 * '-')
print("Root mean squared error: ", np.sqrt(mean_squared_error(y_test, y_pred)))
####################the best result is rms=2192
###########################using forest model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_percentage_error
R_F_model = RandomForestRegressor()
R_F_model.fit(x_train,y_train)
y_pred_fo = R_F_model.predict(x_test)
MAE = mean_absolute_error(y_test,y_pred_fo)
MAPE = mean_absolute_percentage_error(y_test,y_pred_fo)
MSE = mean_squared_error(y_test,y_pred_fo)
RMSE = np.sqrt(MSE)
R2_score = r2_score(y_test,y_pred_fo)
print(" MAE on R_f model = ",MAE)
print(" MAPE on R_f model = ",MAPE)
print(" MSE on R_f model = ",MSE)
print(" RMSE on R_f model = ",RMSE)
print(" R2_score on R_f model = ",R2_score)
y_mean = y.mean()
print('RMS/ymean =',(RMSE/y_mean))
############RMS=1827
import matplotlib.pyplot as plt
columns_names = df.drop('Price',axis=1).columns.tolist()
importance = R_F_model.feature_importances_
indices = np.argsort(importance)
plt.barh(range(len(indices)),importance[indices], color='b')
plt.yticks(range(len(indices)), [columns_names[i] for i in indices])
plt.show()############3duration is the most importance
##############################using k fold
from sklearn.model_selection import KFold, cross_val_score
R_F_model = RandomForestRegressor()
cv_conf = KFold(n_splits=8, random_state=101 , shuffle=True)
scores = cross_val_score(R_F_model, x, y, cv =cv_conf , scoring='neg_root_mean_squared_error')
print(np.absolute(scores))
print(' Final score = ',(np.mean(np.absolute(scores))))
##################rmse=1592
#########################################
from sklearn.model_selection import GridSearchCV
Param_ranges = {
    'n_estimators': list(range(50,60)),
    'max_depth': [5,10,15,20,22,None],
    'min_samples_split':[2,3,4,5]
}
R_F_model = RandomForestRegressor()
g_search = GridSearchCV(estimator=R_F_model, param_grid =Param_ranges, scoring='neg_root_mean_squared_error', return_train_score=True, verbose=3  )
g_search.fit(x,y)
print('Best parameters',g_search.best_params_)
print('Best score',np.absolute(g_search.best_score_))

##################best score=1583
#######################################################








