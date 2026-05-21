import numpy as np
from scipy.ndimage import label


def apply_mmu(mask, min_pixels):

    structure = np.ones((3,3))

    labeled, num = label(mask, structure)

    output = np.zeros_like(mask)

    for i in range(1, num+1):

        component = labeled == i

        if component.sum() >= min_pixels:

            output[component] = 1

    return output