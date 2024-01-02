import torch
import numpy as np
import matplotlib as plt
import cv2 as cv


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


# Cross-correlation functions


def cross_correlation(filter, image):
    """
    Calculates the cross-correlation between a filter and an image.

    Parameters:
    filter (ndarray): The 2D filter to be applied.
    image (ndarray): A segment of the 1D image vector.

    Returns:
    float: The result of the cross-correlation.
    """

    filter = filter.flatten().reshape(25, -1)

    output = np.sum(filter * image)

    return output


def threshold_cross_correlation(filter, image):
    """
    Performs cross-correlation between a filter and an image and thresholds the result.

    Args:
      filter (ndarray): The filter to be applied.
      image (ndarray): The image on which the filter is applied.

    Returns:
      int: 0 if the cross-correlation is negative, 1 otherwise.
    """

    filter = filter.flatten()
    output = 0

    result = np.sum(filter * image)

    if result > 0:
        output = 1

    return output


def normalized_cross_correlation(filter, image_segment):
    """
    Calculates the normalized cross-correlation between a filter and an image segment.

    Args:
      filter (ndarray): The filter to be applied.
      image_segment (ndarray): The image segment on which the filter is applied.

    Returns:
      float: The result of the normalized cross-correlation.
    """

    filter = filter.flatten().reshape(25, -1)
    image_segment = image_segment.flatten().reshape(25, -1)

    output = np.sum(filter * image_segment) / np.sqrt(
        np.sum(filter**2) * np.sum(image_segment**2)
    )
    print(output)

    return output


def ncc_filter(image_segment):
    f = filter
    f = f.flatten().reshape(25, -1)

    output = np.sum(f * image_segment) / np.sqrt(
        np.sum(f**2) * np.sum(image_segment**2)
    )

    return output


filter = np.array(
    [
        [1, 0.5, 0, -0.5, -1],
        [2, 1, 0, -1, 2],
        [3, 1.5, 0, -1.5, -3],
        [2, 1, 0, -1, -2],
        [1, 0.5, 0, -0.5, -1],
    ]
)

sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

three_gaussian_blur_filter = np.array(
    [[1 / 16, 2 / 16, 1 / 16], [2 / 16, 4 / 16, 2 / 16], [1 / 16, 2 / 16, 1 / 16]]
)

five_gaussian_blur_filter = np.array(
    [
        [1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256],
    ]
)

# Load image to grayscale
img = cv.imread("princeton_low_def.jpg", cv.IMREAD_GRAYSCALE)
height, width = img.shape
img = img.reshape(height, width, 1)

# Display image
# cv.imshow("Image", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Change image size to (1, 1, height, width) and convert to torch tensor
img = np.transpose(img, (2, 0, 1)).reshape(1, 1, height, width)
img = torch.from_numpy(img).float()

# Unfold image
unfold_op = torch.nn.Unfold(
    kernel_size=(filter.shape[0], filter.shape[1]),
    dilation=1,
    padding=int((filter.shape[0] - 1) / 2),
    stride=1,
)
unfolded_img = unfold_op(img)
unfolded_img = unfolded_img.numpy()
print(type(unfolded_img))
print(unfolded_img)


# Apply cross-correlation to img using broadcasting
# result = np.array([])
# result = np.sum(
#     five_gaussian_blur_filter.flatten().reshape(25, -1) * unfolded_img[0, :, :], axis=0
# )


# Normalized cross-correlation using for loop
# result = np.array([])
# for i in range(unfolded_img.shape[2]):
#     result = np.append(
#         result,
#         normalized_cross_correlation(filter, unfolded_img[0, :, i]),
#     )

# ? Does this actually work as intended? Is this mapping?
# Normalized cross-correlation using mapping
result = np.array([])
result = np.append(result, np.apply_along_axis(ncc_filter, 0, unfolded_img[0, :, :]))

# Reshape result to dimensions of img
result = result.reshape(height, width, 1)
# Normalize result. Solves issue of imwrite and imshow showing different images.
result = cv.normalize(result, None, 1, 0, cv.NORM_MINMAX, dtype=cv.CV_32F)

# Save image
cv.imwrite("ncc_filter.png", result)
cv.imshow("Cross-correlation ", result)
cv.waitKey(0)
cv.destroyAllWindows()
