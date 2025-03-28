from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import sklearn.linear_model as lm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score


# Odaberite željene numericke velicine specificiranjem liste s nazivima stupaca. 
# Podijelite podatke na skup za ucenje i skup za testiranje u omjeru 80%-20%.
features = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)']
data = pd.read_csv('data_C02_emission.csv')

X = data[features]
y = data['CO2 Emissions (g/km)']
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state=1)

#Pomocu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova ´
#o jednoj numerickoj veli ˇ cini. Pri tome podatke koji pripadaju skupu za u ˇ cenje ozna ˇ cite ˇ
#plavom bojom, a podatke koji pripadaju skupu za testiranje oznacite crvenom bojom.
""" plt.figure()
plt.scatter(X_train["Engine Size (L)"], y_train, c="b")
plt.scatter(X_test["Engine Size (L)"], y_test, c="r")
plt.xlabel("Engine Size (L)")
plt.ylabel("CO2 Emissions (g/km)")
plt.show() """

#Izvršite standardizaciju ulaznih velicina skupa za ucenje. Prikažite histogram vrijednosti ˇ
#jedne ulazne velicine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja ˇ
#transformirajte ulazne velicine skupa podataka za testiranje.

""" fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].hist(X_train["Engine Size (L)"])
axes[0].set_title("Prije skaliranja") """

sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)
""" axes[1].hist(X_train_n[:, X_train.columns.get_loc("Engine Size (L)")])
axes[1].set_title("Poslje skaliranja") 
plt.show()
"""

# Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i povežite ih s izrazom 4.6.

linear_model = lm.LinearRegression()
linear_model.fit(X_train_n, y_train)

print("Koeficijenti:", linear_model.coef_)
print("Slobodni clan:", linear_model.intercept_)

# Izvršite procjenu izlazne velicine na temelju ulaznih veli ˇ cina skupa za testiranje. Prikažite ˇ
# pomocu dijagrama raspršenja odnos izme ´ du stvarnih vrijednosti izlazne veli ¯ cine i procjene ˇ
# dobivene modelom.

y_test_p = linear_model.predict(X_test_n)
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_test, c='b', label="Stvarne vrijednosti")
plt.scatter(y_test, y_test_p, c='r', label="Predviđene vrijednosti")

plt.xlabel("Stvarne emisije CO2 (g/km)")
plt.ylabel("Predviđene emisije CO2 (g/km)")
plt.legend()
plt.show()


# Izvršite vrednovanje modela na nacin da izracunate vrijednosti regresijskih metrika na ˇ
# skupu podataka za testiranje

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



# Što se dogada s vrijednostima evaluacijskih metrika na testnom skupu kada mijenjate broj ¯
# ulaznih velicina?

for i in range(len(features)):
    X = data[features[i:]]
    y = data['CO2 Emissions (g/km)']
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state=1)

    sc = MinMaxScaler()
    X_train_n = sc.fit_transform(X_train)
    X_test_n = sc.transform(X_test)

    linear_model = lm.LinearRegression()
    linear_model.fit(X_train_n, y_train)

    print("\n\n\nModel sa: ", ", ".join(features[i:]))

    print("Koeficijenti:", linear_model.coef_)
    print("Slobodni clan:", linear_model.intercept_)


    y_test_p = linear_model.predict(X_test_n)

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
    
# Model je losiji sa manje ulaznih velicina