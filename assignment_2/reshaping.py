import numpy as np
import matplotlib.pyplot as plt
# import cv2 # doesn't work
from PIL import Image

img = Image.open("englishcreams.png")
pic = np.array(img)
pic = np.delete(pic, 3, axis=2)
# height, width, channels (rgba)

height, width, channels = pic.shape
print(f'before transpose {pic[:2]} \n shape is {pic.shape} \n \n')

# desired shape: channels, height, width

pic_T1 = np.transpose(pic, (2, 0, 1))
print(f'after transpose {pic_T1[:2]} \n shape is {pic_T1.shape}')
plt.imshow(pic_T1[1])
plt.show()
