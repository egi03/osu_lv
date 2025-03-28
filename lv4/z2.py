from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sklearn.linear_model as lm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score


data = pd.read_csv('data_C02_emission.csv')

features = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 
            'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)', 'Fuel Type']

X = data[features]
y = data['CO2 Emissions (g/km)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

ohe = OneHotEncoder()
X_encoded_train = ohe.fit_transform(X_train[['Fuel Type']]).toarray()
X_encoded_test = ohe.transform(X_test[['Fuel Type']]).toarray()


X_train_final = pd.concat([X_train.drop(columns=['Fuel Type']), pd.DataFrame(X_encoded_train)], axis=1)
X_test_final = pd.concat([X_test.drop(columns=['Fuel Type']), pd.DataFrame(X_encoded_test)], axis=1)


linear_model = lm.LinearRegression()
linear_model.fit(X_train_final, y_train)


y_test_p = linear_model.predict(X_test_final)


MSE = mean_squared_error(y_test, y_test_p)
RMSE = root_mean_squared_error(y_test, y_test_p)
MAE = mean_absolute_error(y_test, y_test_p)
MAPE = mean_absolute_percentage_error(y_test, y_test_p)
R2 = r2_score(y_test, y_test_p)
print("\n\nMSE: ", MSE)
print("RMSE: ", RMSE)
print("MAE: ", MAE)
print("MAPE: ", MAPE)
print("R2: ", R2)


errors = np.abs(y_test - y_test_p)
max_err = np.max(errors)
max_err_ind = np.argmax(errors)
print(type(max_err_ind))
print("Max error: ", max_err)
print("Model vozila: ", data.iloc[max_err_ind]["Model"])