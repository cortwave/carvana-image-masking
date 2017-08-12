from cv2 import flip
from scipy.ndimage import rotate
import numpy as np

rotate_angles = [0, 90, 180, 270]

def tta(image):
    images = []
    for rotate_angle in rotate_angles:
        img = rotate(image, rotate_angle) if rotate_angle != 0 else image
        images.append(img)
    return np.array(images)


def back_tta(images):
    backed = []
    i = 0
    for rotate_angle in rotate_angles:
        image = images[i]
        i += 1
        img = rotate(image, 360 - rotate_angle) if rotate_angle != 0 else image
        backed.append(img)
    return backed
