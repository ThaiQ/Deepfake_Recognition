import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("C:/Github/Deepfake_Recognition_SSD/frame_00000.png")
plt.imshow(img, cmap="gray")

plt.imshow(img, cmap="gray") # plot image
plt.scatter(508, 68, 50, c="r", marker="+") # plot markers
plt.scatter(700, 275, 50, c="r", marker="+") # plot markers
plt.show()