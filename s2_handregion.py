"""
    From the given grayscale image, the hand region module will detect the hand region and return list with all region maybe is hand region, by the list with 4 values per element: x, y, w, h.
    After that, by each element in the list, we will get it and use the hand_region_model to predict it is hand region or not.
    If it is not hand region, we will remove it from the list.
    Finally, we will return the list with all hand region.

    The hand region module will be implemented in the s2_model.ipynb file.

    We use gaussian blur to decrease the noise in the image.
    After that, we use canny edge detection to detect the edges in the image.
    We use mathematical dilation to make the edges more visible.
    We use findContours to find the contours in the image.



    We use HOG to detect the region maybe is hand region.
    After that, we will conpare the region parameters with criteria to get the hand region and add it to the list.
    Criteria:
        Hand region square percentage 10% < square percentage < 50%
        Hand region width > 1/5 * width of the image
        Hand region height > 1/4 * height of the image



    Criteria of input:
        - Type: grayscale image (png, jpg, jpeg)
        - Size: max 1280x720
    Output format:
        - Type: list
        - Value: list with 4 values per element: x, y, w, h
"""

# Importing the required libraries
import numpy as np
import pandas as pd
import cv2
import os
from skimage import feature
from skimage.transform import resize

class Hand_Region:
    def __init__(self):
        self.image = None

    def preprocess_image(self, image):
        # Try to convert the image to grayscale
        try:
            # Convert the image to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        # Use thresholding to decrease the light in the image
        image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1]
        
        # Use Gaussian blur to decrease the noise in the image
        image = cv2.GaussianBlur(image, (3, 3), 0)

        # Use mathematical dilation to make the edges more visible
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        image = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)

        return image
        
    def detect_hand_region(self, img):
        # Apply Gaussian blur to the image
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Decrease the light
        img_blur = cv2.convertScaleAbs(img_blur, alpha=0.5, beta=10)

        # Detect the edges in the image
        edges = cv2.Canny(img_blur, 150, 200)

        # Apply mathematical dilation to the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)

        # Find the contours in the image
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If contours are found, find the largest one (hand region)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            # Show the image
            return x, y, w, h
        else:
            return None  
    def get_hand_region(self, img):
        # Get the image size
        height, width = img.shape

        # Detect the hand region in the image
        x, y, w, h = self.detect_hand_region(img)

        # Check if the detected region meets the criteria
        if x is not None and y is not None and w is not None and h is not None:
            # Calculate the square percentage of the detected region
            square_percentage = (w * h) / (width * height) * 100

            # Check if the detected region meets the criteria
            if square_percentage > 10 and square_percentage < 50 and w > width / 5 and h > height / 4:
                return x, y, w, h
            else:
                return None
        else:
            return None  
    def get_all_hand_regions(self, img):
        # Get the image size
        height, width = img.shape

        # Detect the hand region in the image
        x, y, w, h = self.detect_hand_region(img)

        # Check if the detected region meets the criteria
        if x is not None and y is not None and w is not None and h is not None:
            # Calculate the square percentage of the detected region
            square_percentage = (w * h) / (width * height) * 100

            # Check if the detected region meets the criteria
            if square_percentage > 10 and square_percentage < 50 and w > width / 5 and h > height / 4:
                return [(x, y, w, h)]
            else:
                return []
        else:
            return []
        


# Test the Hand_Region class
if __name__ == '__main__':
    # Create an instance of the Hand_Region class
    hand_region = Hand_Region()

    # Load the image
    img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

    # Use hog and sliding window to show the different regions
    hogFeature, hogImage = hand_region.hog_feature(img)
    # max_score, maxr, maxc, response_map = hand_region.sliding_window(img, hogFeature, 8, (64, 128))

    # SShow hog image
    cv2.imshow('HOG Image', hogImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


        


