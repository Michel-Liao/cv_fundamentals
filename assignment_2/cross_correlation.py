import torch
import numpy as np
import matplotlib as plt
import cv2 as cv

# def add_border(orig, border_px=100):
#     a, b, c = orig.shape
#     result = np.zeros((a + 2 * border_px, b + 2 * border_px, c))
#     print(result)
#     result[border_px:a + border_px, border_px:b + border_px, :] = orig
#     return result


def add_border(orig, border_px=100):
    """
    Applies border around an image
    Args:
      orig (ndarray (a,b,c)) : original image
      border_px (int)        : border size, nonnegative, default value = 100
    Returns
      bordered (ndarray (a + 2*border_px, b + 2*border_px, c)) : bordered image
    """
    a, b, c = orig.shape
    new_shape = (a + 2 * border_px, b + 2 * border_px, c)

    result = np.zeros(new_shape, dtype=orig.dtype)
    result[border_px : a + border_px, border_px : b + border_px, :] = orig

    return result


# orig = np.ones((1, 2, 3))
# bordered = add_border(orig, 1)
# print(orig)
# print(orig.shape)
# print("\n\n\n")
# print(bordered.shape)
# print(bordered)

filter = np.array(
    [
        [1, 0.5, 0, -0.5, -1],
        [2, 1, 0, -1, 2],
        [3, 1.5, 0, -1.5, -3],
        [2, 1, 0, -1, -2],
        [1, 0.5, 0, -0.5, -1],
    ]
)

# Load image to grayscale
img = cv.imread("golden-retriever.jpg", cv.IMREAD_GRAYSCALE)
height, width = img.shape
img = img.reshape(height, width, 1)

# Change image size to (1, 1, height, width) and convert to torch tensor
img = np.transpose(img, (2, 0, 1)).reshape(1, 1, height, width)
img = torch.from_numpy(img).float()

# Unfold image
unfold_op = torch.nn.Unfold(kernel_size=(5, 5), dilation=1, padding=2, stride=1)
unfolded_img = unfold_op(img)
unfolded_img = unfolded_img.numpy()
print(type(unfolded_img))
print(unfolded_img)


# Cross-correlation function
def cross_correlation(filter, image):
    a, b = image.shape
    result = np.zeros((a, b))
    for i in range(a):
        for j in range(b):
            result[i, j] = filter[i, j] * image[i, j]
    return result


# cv.imshow("Image", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# print(f"this is img {img}")
