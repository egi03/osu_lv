import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data.csv", delimiter=',', dtype=str)
data = data[1:].astype(float)

# Ukupno osoba
print(np.shape(data)[0])


#Scatter plot visina-masa
x = data[:,1]
y = data[:,2]
plt.scatter(x, y)
plt.xlabel("Visina")
plt.xlabel("Tezina")
plt.title("Zadatak2")
plt.show()


# Svaka 50. osoba
x = data[:,1][::50]
y = data[:,2][::50]

plt.scatter(x, y)
plt.xlabel("Visina")
plt.xlabel("Tezina")
plt.title("Svaka 50. osoba")
plt.show()


# Max min i mean visine
height = data[:,1]
print(height.max())
print(height.min())
print(height.mean())


# Max min i mean visine posebno za muskace i zene
man_ind = (data[:,0] == 1)
woman_ind = (data[:,0] == 0)
men = data[man_ind, 1]
women = data[woman_ind, 1]

print("Men:")
print(men.max())
print(men.min())
print(men.mean())
print("Women:")
print(women.max())
print(women.min())
print(women.mean())

