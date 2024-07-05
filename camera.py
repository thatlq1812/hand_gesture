# In that file, we will implement the camera setup algorithm.
"""



"""

# Import the necessary libraries

import cv2
import os
import time

from region_detection import Region_Detection

class Camera_Setup:
    def __init__(self) -> None:
        pass

    def setup_camera(self,cameranumb):
        cap = cv2.VideoCapture(cameranumb)
        
        if not cap.isOpened():
            print("Cannot open the camera.")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot receive data from the camera.")
                break
            
            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    

# Gọi hàm setup_camera để chạy
if __name__ == "__main__":
    cam_test = Camera_Setup()
    cam_test.setup_camera(0)

