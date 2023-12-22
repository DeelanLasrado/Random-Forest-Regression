import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

df = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan/Regression/master/IceCreamData.csv')
print(df)

# Assigning features and target
X = np.array(df.Temperature)
y = np.array(df.Revenue)


# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.05)

# Model Selection
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)

# Training the model
regressor.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

# Testing the model
y_pred = regressor.predict(X_test.reshape(-1,1))


pred = pd.DataFrame({"Actual":y_test.reshape(-1), 
                     "Prediction":y_pred.reshape(-1)})

print(pred.head())

sns.heatmap(pred.corr(), annot=True, cmap='Greens')
plt.show()

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))