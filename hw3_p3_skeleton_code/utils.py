"""
EECS 445 - Introduction to Machine Learning
Winter 2019 - Homework 3
Utility functions
"""

import numpy as np

def get(attr):
    """
    Retrieves the queried attribute value from the config file. Loads the
    config file on first call.
    """
    if not hasattr(get, 'config'):
        with open('config.json') as f:
            get.config = eval(f.read())
    node = get.config
    for part in attr.split('.'):
        node = node[part]
    return node

def denormalize_image(image):
    """ Rescale the image's color space from (min, max) to (0, 1) """
    ptp = np.max(image, axis=(0,1)) - np.min(image, axis=(0,1))
    return (image - np.min(image, axis=(0,1))) / ptp
