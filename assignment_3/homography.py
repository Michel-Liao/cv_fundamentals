import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def evenDims(image):
    """
    Removes row/column of image if dimension is odd.

    Args:
        image (ndarray): The input image.

    Returns:
        image (ndarray): The image with even dimensions.
    """
    height, width, channels = image.shape
    if height % 2 != 0:
        image = image[: height - 1, :, :]
    if width % 2 != 0:
        image = image[:, : width - 1, :]

    return image


# Read and show image
img = cv.imread("central_park.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = evenDims(img)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.imshow(img)
plt.show()
