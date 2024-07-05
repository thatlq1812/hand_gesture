# In that file, we will implement the detection algorithm.
"""
    In that file, we will implement the detection algorithm.
    You can change the parameters of the detection algorithm to improve the detection accuracy.
    Parameters:
        - GaussianBlur: The size of the Gaussian kernel used to blur the image.
        - Canny: The threshold values used to detect the edges in the image.
        - MORPH_DILATE: The size of the kernel used to apply mathematical dilation to the image.
        - findContours: The retrieval mode and approximation method used to find the contours in the image.
        - draw_rectangle: The color and thickness of the rectangle drawn around the hand region.
"""
# Importing the necessary libraries
import cv2
import numpy as np
import os
import time

class Region_Detection: # Detection class
    def __init__(self):
        self.image = None
    
    def detect_hand_region(self, img):

        # Apply Gaussian blur to the image
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Decrease the light
        img_blur = cv2.convertScaleAbs(img_blur, alpha=0.3, beta=10)

        # Detect the edges in the image
        edges = cv2.Canny(img_blur, 50, 100)

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
            
    def draw_rectangle(self,img):
        # Call the detect_hand_region method to get the bounding box
        bbox = self.detect_hand_region(img)
        # Create a copy of the image
        img_output = img.copy()
        
        if bbox:
            x, y, w, h = bbox
            # Draw a rectangle around the hand region
            cv2.rectangle(img_output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Show
            cv2.imshow('Detected Hand', img_output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return img_output
        else:
            return None

# Example
if __name__ == "__main__":
    # Example image path
    image_path = r"c:\Users\ASUS\Pictures\Camera Roll\WIN_20240704_13_08_13_Pro.jpg"
    
    # Create Detection object
    test_detection = Region_Detection()
    
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Ensure the image was loaded successfully
    if img is not None:
        # Call draw_rectangle method to detect and draw rectangle around hand region
        result_image = test_detection.draw_rectangle(img)
        
        # Display the image with the rectangle
    else:
        print(f"Failed to detect hand region in image: {image_path}")
