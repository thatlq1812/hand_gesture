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
        # Use Canny edge detection to detect the edges in the image
        image = cv2.Canny(image, 50, 100)

        # Use mathematical dilation to make the edges more visible
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        image = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)

        return image
        
    def hog_feature(self, img, pixel_per_cell=8):
        """
        Compute HOG feature for a given image.

        Args:
            image: an image with object that we want to detect.
            pixel_per_cell: number of pixels in each cell, an argument for hog descriptor.

        Returns:
            hogFeature: a vector of hog representation.
            hogImage: an image representation of hog provided by skimage.
        """
        hogFeature, hogImage = feature.hog(img,
                                           pixels_per_cell=(pixel_per_cell, pixel_per_cell),
                                           cells_per_block=(3, 3),
                                           block_norm='L1',
                                           visualize=True,
                                           feature_vector=True)
        return hogFeature, hogImage
    def sliding_window(self,image, base_score, stepSize, windowSize, pixel_per_cell=8):
        """ A sliding window that checks each different location in the image,
            and finds which location has the highest hog score. The hog score is computed
            as the dot product between the hog feature of the sliding window and the hog feature
            of the template. It generates a response map where each location of the
            response map is a corresponding score. And you will need to resize the response map
            so that it has the same shape as the image.

        Hint: use the resize function provided by skimage.

        Args:
            image: an np array of size (h,w).
            base_score: hog representation of the object you want to find, an array of size (m,).
            stepSize: an int of the step size to move the window.
            windowSize: a pair of ints that is the height and width of the window.
        Returns:
            max_score: float of the highest hog score.
            maxr: int of row where the max_score is found (top-left of window).
            maxc: int of column where the max_score is found (top-left of window).
            response_map: an np array of size (h,w).
        """

        (max_score, maxr, maxc) = (0, 0, 0)
        winH, winW = windowSize
        H, W = image.shape
        pad_image = np.lib.pad(
            image,
            ((winH // 2,
                winH - winH // 2),
                (winW // 2,
                winW - winW // 2)),
            mode='constant')
        response_map = np.zeros((H // stepSize + 1, W // stepSize + 1))

        for r in range(0, H, stepSize):
                for c in range(0, W, stepSize):
                    window = pad_image[r:r + winH, c:c + winW]
                    hogFeature = feature.hog(window,
                                            pixels_per_cell=(pixel_per_cell, pixel_per_cell),
                                            cells_per_block=(3, 3),
                                            block_norm='L1',
                                            visualize=False,
                                            feature_vector=True)
                    score = np.dot(hogFeature, base_score)
                    response_map[r // stepSize, c // stepSize] = score
                    if score > max_score:
                        max_score = score
                        maxr, maxc = r, c
        maxr = maxr - winH // 2
        maxc = maxc - winW // 2
        response_map = resize(response_map, (H, W))

        return (max_score, maxr, maxc, response_map)
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

    # Get the hand region in the image
    get_image = hand_region.preprocess_image(img)

    # Show the image
    cv2.imshow('Image', get_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


        


