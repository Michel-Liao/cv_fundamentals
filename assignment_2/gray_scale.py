import numpy as np
import matplotlib.pyplot as plt
# import cv2 # doesn't work
from PIL import Image

im = Image.open("englishcreams.png")
im = im.convert('RGB')
pic = np.array(im)
# height, width, channels (rgba)
height, width, channels = pic.shape

# print image before grayscale

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.imshow(pic)
# plt.show()

# changing to grayscale
for i in range(height):
    for j in range(width):
        pic[i,j] = np.sum(pic[i,j]) / 3
breakpoint()
pic = np.delete(pic, [1,2], axis=1)
# delets columns 1 and 2
print(pic)
# not sure why each pixel still has 3 values though

# save grayscale image
img = Image.fromarray(pic)
img.save('englishcreams_grayscale.png')