import numpy as np
import matplotlib.pyplot as plt
# import cv2 # doesn't work
from PIL import Image

img = Image.open("englishcreams.png")
pic = np.array(img)
# height, width, channels (rgba)

height, width, channels = pic.shape

print('height, width, channels')
print(pic[:3])
print(pic.shape)
print()

# reshaping
pic1 = pic.reshape(channels, height, width)
print('channels, height, width')
print(pic1[:3])
print(pic1.shape)
print()

print('width, height, channels')
pic2 = pic.reshape(width, height, channels)
print(pic2[:3])
print(pic2.shape)

print('width, channels, height')
pic3 = pic.reshape(width, channels, height)
print(pic3[:3])
print(pic3.shape)