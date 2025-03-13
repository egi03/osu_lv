import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")
img = img [:,:,0].copy()



figure, axis = plt.subplots(2, 2)

# Bright
bright_img = img + 50
bright_img[bright_img < 50] = 255

axis[0, 0].imshow(bright_img, cmap="gray")
axis[0, 0].set_title("Posvjetljena")

# Druga cetvrtina
_, width = img.shape
first_bound = width // 4
second_bound = width // 2

second_q = img[:,first_bound:second_bound]
axis[0, 1].imshow(second_q, cmap="gray")
axis[0, 1].set_title("Druga cetvrtina")


# Rotate 90
rot_90 = np.rot90(img, axes=(1, 0))
axis[1, 0].imshow(rot_90, cmap="gray")
axis[1, 0].set_title("Rotirana")



# Mirror
mirrored = np.fliplr(img)
axis[1, 1].imshow(mirrored, cmap="gray")
axis[1, 1].set_title("Zrcaljena")


plt.show()