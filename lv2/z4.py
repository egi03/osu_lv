import numpy as np
import matplotlib.pyplot as plt

first_black = np.zeros((50, 50))
second_black = np.zeros((50, 50))

first_white = np.ones((50, 50))
second_white = np.ones((50, 50))

first_row = np.hstack((first_black, first_white))
second_row = np.hstack((second_white, second_black))

img = np.vstack((first_row, second_row))
plt.figure()
plt.imshow(img, cmap="gray")
plt.show()