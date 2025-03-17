import pandas as pd
import numpy as np

data = pd.read_csv("data_C02_emission.csv", sep=",")

print("Mjerenja: ", data.shape[0])
print("\nTipovi velicina: \n", data.dtypes)

data_wo_null = data.dropna()
print("Mjerenja nakon obrisanih izostalih vrijednosti: ", data_wo_null.shape[0])
# Ne postoje izostale vrijednosti

# Object u category
object_cols = ["Make", "Model", "Vehicle Class", "Transmission", "Fuel Type"]
for col in object_cols:
    data[col] = data[col].astype('category')


# Min max 
sorted_by_fule = data.sort_values(by="Fuel Consumption City (L/100km)")[["Make", "Model", "Fuel Consumption City (L/100km)"]]
print("\nMin gradska potrosnja: \n",sorted_by_fule.head(3))
print("\Max gradska potrosnja: \n",sorted_by_fule.tail(3))

# Velicina izmedju 2.5 i 3.5
engline_capacity_interval = data.where(data["Engine Size (L)"] <= 3.5).where(data["Engine Size (L)"] >= 2.5).dropna()
print("\n\nUkupno izmedju 2.5 i 3.5L motora: ", engline_capacity_interval.count(axis=0).iloc()[0])
print("Prosjecan CO2 za motore izmedju 2.5 i 3.5L: ", engline_capacity_interval["CO2 Emissions (g/km)"].mean())

# Audis
audis = data[data["Make"] == "Audi"]
print("\n\nTotal audis: ", audis.count().iloc()[0])
print("Prosjecan CO2 za audi 4 cilindra: ", audis[audis["Cylinders"] == 4]["CO2 Emissions (g/km)"].mean())

# Cars with cylinders
cylinders = data["Cylinders"].unique()
for cy in cylinders:
    print(f"Broj auta sa {cy} cilindara: ", data[data["Cylinders"] == cy].count().iloc()[0])

# Dizel benzin
diesel = data[data["Fuel Type"] == "Z"]
benzin = data[data["Fuel Type"] == "X"]

print("\n\nProsjecna gradska potrosnja dizel: ", diesel["Fuel Consumption City (L/100km)"].mean())
print("Medijal gradska potrosnja dizel: ", diesel["Fuel Consumption City (L/100km)"].median())
print("\nProsjecna gradska potrosnja benzin: ", benzin["Fuel Consumption City (L/100km)"].mean())
print("Medijal gradska potrosnja benzin: ", benzin["Fuel Consumption City (L/100km)"].median())


# Najveci 4 cilindra dizel potrosac
worst_fuel = diesel[diesel["Cylinders"] == 4].sort_values(by="Fuel Consumption City (L/100km)")[["Make", "Model", "Fuel Consumption City (L/100km)"]].tail(1)
print("\n\n4 cilindra dizel sa najvecom potrosnjom:\n ", worst_fuel)

# Koliko vozila runci mjenjac
num_of_manuals = len([1 for d in data["Transmission"] if d[0] == 'M' or d[0:2] == 'AM'])
print("\n\nUkupan broj manualnih mjenjaca: ", num_of_manuals)

# Korelacija
print(data.corr(numeric_only=True))
# CO2 Emisija i velicina motora zajedno linearno rastu i padaju korelacija je 0.83
# Potrosnja Comb i velicina motora zajedno linearno padaju i padaju korelacija je -0.72