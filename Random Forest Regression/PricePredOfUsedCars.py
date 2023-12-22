import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


df = pd.read_csv('https://raw.githubusercontent.com/sahilrahman12/Price_prediction_of_used_Cars_-Predictive_Analysis-/master/cardekho_data.csv')

print(df.head())

df['current_year'] = 2021
df['no. of year'] = df['current_year'] - df['Year']

df.drop(['Car_Name', 'Year', 'current_year'],axis=1, inplace=True)

df = pd.get_dummies(df)
print(df)
df.drop(['Fuel_Type_CNG', 'Seller_Type_Dealer', 'Transmission_Automatic'],axis=1, inplace=True)


X = df.iloc[:,1:]
y = df.iloc[:,0]

# Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,
                                                    random_state=0)


# Model Selection
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()






# Hyper Parameter Tuning
n_estimators = [int(i) for i in np.linspace(start=100, stop=1200, num=12)]#linespace create an array -[100 200 300...1200]

max_features = ['auto','sqrt']

max_depth = [int(i) for i in np.linspace(start=5, stop=30, num=6)]

min_samples_split = [2, 5, 10, 15, 100]

min_samples_leaf = [1, 2, 5,10]

random_grid = {'n_estimators':n_estimators,
               'max_features':max_features,
               'max_depth':max_depth,
               'min_samples_split':min_samples_split,
               'min_samples_leaf':min_samples_leaf}
print(random_grid)

from sklearn.model_selection import RandomizedSearchCV

rf_regressor = RandomizedSearchCV(estimator=regressor,#v have to give the model
                                  param_distributions=random_grid,#dict of parameters
                                  scoring='neg_mean_squared_error',
                                  cv=5,
                                  verbose = 2,
                                  random_state=42,
                                  n_jobs=1)

# Training the model
rf_regressor.fit(X_train, y_train)


#testing
y_pred = rf_regressor.predict(X_test)


finaldf = pd.DataFrame({"Actual":y_test, "Predicted":y_pred})
print(finaldf)

# Performance/Accuracy
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)