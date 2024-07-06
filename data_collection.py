# In that file, we will implement the data collection algorithm.
"""
    Using camera and take pictures of the hand region.
    1. Choose hand gestures to collect data.
    2. Take pictures of the hand region.
    3. Save the images in the specified directory.


"""

# Import the necessary libraries
import cv2
import os
import time

# Setup file path for saving images
data_path = 'imagedata' # path existed

# Create a new directory to save the images
if not os.path.exists(data_path):
    os.makedirs(data_path)

# Create directory for each hand gesture
gestures = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb', '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']

# Check for large folder name in imagedata by getting the maximum folder name
max_folder = max([int(folder) for folder in os.listdir(data_path)]) if os.listdir(data_path) else 0

# Create a new folder by max_folder + 1 if max folder is found and not empty
if max_folder:
    data_path = os.path.join(data_path, str(max_folder + 1))
    os.makedirs(data_path)
else:
    data_path = os.path.join(data_path, max_folder)
    os.makedirs(data_path)