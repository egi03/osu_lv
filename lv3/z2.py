import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data_C02_emission.csv", sep=",")

# Hist
plt.figure()
data["CO2 Emissions (g/km)"].plot(kind="hist")
plt.title("CO2")

# Scatter
data['Fuel Type'] = pd.Categorical(data['Fuel Type'])

plt.figure()
data.plot.scatter(
    x="Fuel Consumption City (L/100km)",
    y="CO2 Emissions (g/km)",
    c="Fuel Type", cmap="Set1", s=50,
)

plt.figure()
# Box
data.boxplot(column=["Fuel Consumption Hwy (L/100km)"], by="Fuel Type")
# Veliki outlier u Fuel Type Z koje je preko 20.0 dok je prosjek 9


plt.figure()
# Vozila/gorivu
data_by_fuel = data.groupby(by="Fuel Type").size()
data_by_fuel.plot(kind="bar")






plt.show()