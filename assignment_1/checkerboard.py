import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="choose filename for image")
parser.add_argument("size", help="choose pixel size of squares in checkerboard", type=int)
args = parser.parse_args()

im = Image.open(args.filename)

# flower = Image.open("flower.jpeg")
# flower = Image.open("unsplash.jpg")
checker_pic = np.array(im)
height, width, channel = checker_pic.shape
# print(height, width, channel)

n = args.size # square size

for i in range(width):
    for j in range(height):
        if ((i % (2 * n)) < n) and ((j % (2 * n)) < n):
            checker_pic[j,i,:] = [0,0,0]
        elif ((i % (2 * n)) >= n) and ((j % (2 * n)) >= n):
            checker_pic[j,i,:] = [0,0,0]

# plotting
fig, ax = plt.subplots(1, 1, figsize=(15,10))
ax.imshow(checker_pic)
ax.axis('off')
plt.show()