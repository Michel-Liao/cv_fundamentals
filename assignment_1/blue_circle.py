import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

im = Image.open('unsplash.jpg')
pic = np.array(im)

height, width, channels = pic.shape
center_height = int(height/2)
center_width = int(width/2)

# blue circle

# r = 200
# for i in range(height):
#     for j in range(width):
#         if (i - center_height)**2 + (j - center_width)**2 <= r**2:
#             pic[i,j,:] = [0,0,255]

# overlapping circles

r = 300
for i in range(height):
    for j in range(width):
        if ((i - center_height)**2 + (j - center_width - math.floor(width/18))**2 <= r**2) and ((i - center_height)**2 + (j - center_width + math.floor(width/18))**2 <= r**2):
            pic[i,j,:] = [128,0,128]
        elif (i - center_height)**2 + (j - center_width - math.floor(width/18))**2 <= r**2:
            pic[i,j,:] = [255,0,0]
        elif (i - center_height)**2 + (j - center_width + math.floor(width/18))**2 <= r**2:
            pic[i,j,:] = [0,0,255]

# plotting
fig, ax = plt.subplots(1, 1, figsize=(15,10))
ax.imshow(pic)
ax.axis('off')
plt.show()