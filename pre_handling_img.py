import numpy as np
import scipy.ndimage
import cv2

def center_digit(img):
    """Centers a digit in a 28x28 image (numpy array)."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    
    # Bounding box coordinates
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Compute the current centroid
    cy, cx = (rmin + rmax) / 2, (cmin + cmax) / 2
    
    # Compute the required shift
    shift_y, shift_x = 14 - cy, 14 - cx
    
    # Apply translation
    centered_img = scipy.ndimage.shift(img, shift=(shift_y, shift_x), mode='constant')
    
    return centered_img

def scale_digit_only(image, scale_factor=0.8):
    """
    Scales only the digit inside a 28x28 image without changing the image size.
    
    Args:
    - image (numpy array): 28x28 grayscale image.
    - scale_factor (float): Scaling factor for the digit (e.g., 0.8 for 80% size).
    
    Returns:
    - New 28x28 image with the scaled digit centered.
    """
    # Find bounding box of the digit
    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return image  # Return unchanged if empty image

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Extract the digit
    digit = image[rmin:rmax+1, cmin:cmax+1]

    # Compute new size after scaling
    new_h = max(1, int(digit.shape[0] * scale_factor))
    new_w = max(1, int(digit.shape[1] * scale_factor))

    # Resize the digit using OpenCV
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a blank 28x28 image
    new_image = np.zeros((28, 28), dtype=np.uint8)

    # Compute new position to center the resized digit
    start_r = (28 - new_h) // 2
    start_c = (28 - new_w) // 2

    # Place the resized digit in the center
    new_image[start_r:start_r+new_h, start_c:start_c+new_w] = digit_resized

    return new_image