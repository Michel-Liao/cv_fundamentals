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
    del channels
    if height % 2 != 0:
        image = image[: height - 1, :, :]
    if width % 2 != 0:
        image = image[:, : width - 1, :]

    return image


def drawAxes(image, size=16):
    """
    Draw axes on the image.

    Parameters:
    image (numpy.ndarray): The input image.
    size (int): Width of the axes, must be even. Default is 16 pixels.

    Returns:
    image (numpy.ndarray): The image with axes drawn.
    """

    if size % 2 != 0:
        raise ValueError("Size must be even.")

    height, width, channels = image.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("Image must have even dimensions.")

    # Draw x-axis
    xAx = np.zeros((size, width, channels), dtype=image.dtype)
    image[int((height / 2 - size / 2)) : int((height / 2 + size / 2)), :] = xAx

    # Draw y-axis
    yAx = np.zeros((height, size, channels), dtype=image.dtype)
    image[:, int((width / 2 - size / 2)) : int((width / 2 + size / 2))] = yAx

    return image


# Read and show image
img = cv.imread("central_park.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = evenDims(img)
img = drawAxes(img)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.imshow(img)
plt.show()
