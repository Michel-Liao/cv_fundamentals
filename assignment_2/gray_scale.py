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

print(pic)

pic = np.delete(pic, pic[:, :, :3], axis=1)
# not exactly sure why axis=1 here? otherwise the image gets flattened

# save grayscale image
img = Image.fromarray(pic)
img.save('englishcreams_grayscale.png')