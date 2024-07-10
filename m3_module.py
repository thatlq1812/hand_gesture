# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import csv

import getpass
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump, load

warnings.filterwarnings("ignore", category=DeprecationWarning)

#Config: Setup the global variables here

# For MediaPipe
gl_static_image_mode = False
gl_max_num_hands = 2

# For Camera
gl_cam_index = 0 # IF YOU CANT SEE THE CAMERA, CHANGE THIS TO 1 OR HIGHER
gl_cam_width = 1280
gl_cam_height = 720
gl_cam_fps = 30

# For Data Processing
gl_round_decimal = 6



# Class for MediaPipe
class MediaPipe:
    def __init__(self, static_image_mode, max_num_hands):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.hands = mp.solutions.hands.Hands(static_image_mode, max_num_hands)
        self.draw = mp.solutions.drawing_utils
    
    def process_image(self, image):
        # Convert image to RGB format expected by MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image and get results
        self.results = self.hands.process(image_rgb)
        
        return self.results 

    def draw_landmarks(self, image):
        """
        Draw landmarks on the input image using the results from the last processed image.
        
        Parameters:
        - image: numpy array, the input image in BGR format
        
        Returns:
        - image: numpy array, the input image with landmarks drawn
        """
        # Check if there are any detected hand landmarks
        if self.results.multi_hand_landmarks:
            # Iterate through each detected hand
            for hand_landmarks in self.results.multi_hand_landmarks:
                # Draw landmarks on the image 
                self.draw.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS, 
                                         self.draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                                         self.draw.DrawingSpec(color=(0, 255, 0), thickness=2))
        return image

    def get_hand_landmarks(self):
        """
        Retrieve the landmarks of the detected hands from the last processed image.
        
        Returns:
        - landmarks_list: list of numpy arrays, each array contains landmarks for a hand
                          Each landmark is represented by its normalized coordinates (x, y, z)
        """
        landmarks_list = []
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    # Store normalized coordinates of each landmark
                    landmarks.append((landmark.x, landmark.y, landmark.z))
                landmarks_list.append(np.array(landmarks))
        
        return landmarks_list

    def num_detected_hands(self):
        """
        Get the number of hands detected in the last processed image.
        
        Returns:
        - num_hands: int, the number of detected hands
        """
        return len(self.results.multi_hand_landmarks) if self.results.multi_hand_landmarks else 0
    
    def get_minmax_xy(self,landmarks):
        """
        Get the minimum and maximum x, y coordinates of the detected hand landmarks.
        
        Returns:
        - min_x, max_x, min_y, max_y: float, the minimum and maximum x, y coordinates
        """
        width = gl_cam_width
        height = gl_cam_height
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        min_x, max_x = np.min(x_coords)*width, np.max(x_coords)*width
        min_y, max_y = np.min(y_coords)*height, np.max(y_coords)*height
        return [int(min_x), int(min_y), int(max_x), int(max_y)]
    
    def draw_bounding_box(self, image, minmax_xy):
        """
        Draw a bounding box on the image using the min and max x, y values.
        
        Parameters:
        - image: numpy array, the input image in BGR format
        - minmax_xy: list, containing min_x, min_y, max_x, max_y values
        
        Returns:
        - image: numpy array, the input image with the bounding box drawn
        """
        min_x, min_y, max_x, max_y = minmax_xy
        cv2.rectangle(image, (min_x-20, min_y-20), (max_x+20, max_y+20), (255, 255, 0), 1)
        return image

# Class for Camera
class Camera:
    def __init__(self, cam_index, cam_width, cam_height, cam_fps):
        self.cam_index = cam_index
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.cam_fps = cam_fps
        self.cap = None
    
    def initialize_camera(self):
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.cam_index}.")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cam_fps)
        
        return True
    
    def close_camera(self):
        if self.cap is not None:
            self.cap.release()
    
    def test_camera(self):
        if self.cap is None or not self.cap.isOpened():
            print("Error: Camera is not initialized or could not open.")
            return
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Cannot receive frame.")
                break

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
    
    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            print("Error: Camera is not initialized or could not open.")
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            return None
        
        return frame

# Class for Data Processing
class DataProcessing:
    def __init__(self):
        pass

    def calculate(self, landmarks_list):
        """
        Calculate distances and angles between landmarks for each hand.

        Parameters:
        - landmarks_list: list of numpy arrays, each array contains landmarks for a hand

        Returns:
        - features_list: list of numpy arrays, each array contains distances and angles between landmarks for a hand
        """
        thumb_flag = 0
        if landmarks_list[0][4][1] < landmarks_list[0][0][1]:
            thumb_flag = 1

        features_list = np.concatenate((self.calculate_distances(landmarks_list), self.calculate_angles(landmarks_list)), axis=1)
        features_list = features_list.flatten()
        features_list = np.append(features_list, thumb_flag)
        features_list = np.round(features_list, gl_round_decimal)
        return features_list

    def calculate_distances(self, landmarks_list):
        """
        Calculate distances between landmarks in each hand and normalize within a specified range.

        Parameters:
        - landmarks_list: list of numpy arrays, each array contains landmarks for a hand

        Returns:
        - normalized_distances: list of numpy arrays, each array contains normalized distances between landmarks for a hand
        """
        distances_list = []
        for landmarks in landmarks_list:
            distances = []
            # Calculate distances between consecutive landmarks
            for i in range(len(landmarks)):
                for j in range(i + 1, len(landmarks)):
                    x1, y1 = landmarks[i][0:2]  # Take only x, y coordinates
                    x2, y2 = landmarks[j][0:2]  # Take only x, y coordinates
                    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    distances.append(distance)
            distances_list.append(np.array(distances))
    
        # Normalize distances within the range [0, 1]
        normalized_distances = []
        max_distance = max([np.max(distances) for distances in distances_list])
        min_distance = min([np.min(distances) for distances in distances_list])

        for distances in distances_list:
            normalized_distances.append((distances - min_distance) / (max_distance - min_distance))

        return normalized_distances

    def calculate_angles(self, landmarks_list):
        """
        Calculate angles between consecutive landmarks for each hand.

        Parameters:
        - landmarks_list: list of numpy arrays, each array contains landmarks for a hand

        Returns:
        - angles_list: list of numpy arrays, each array contains angles between landmarks for a hand
        """
        angles_list = []

        # Calculate angles between landmarks for each hand
        for landmarks in landmarks_list:
            angles = []

            # Angles between fingertips
            angles.append(self.calculate_angle(landmarks[4], landmarks[0], landmarks[8]))
            angles.append(self.calculate_angle(landmarks[8], landmarks[0], landmarks[12]))
            angles.append(self.calculate_angle(landmarks[12], landmarks[0], landmarks[16]))
            angles.append(self.calculate_angle(landmarks[16], landmarks[0], landmarks[20]))

            # Angles between knuckles
            angles.append(self.calculate_angle(landmarks[1], landmarks[2], landmarks[3]))
            angles.append(self.calculate_angle(landmarks[5], landmarks[6], landmarks[7]))
            angles.append(self.calculate_angle(landmarks[9], landmarks[10], landmarks[11]))
            angles.append(self.calculate_angle(landmarks[13], landmarks[14], landmarks[15]))
            angles.append(self.calculate_angle(landmarks[17], landmarks[18], landmarks[19]))

            # Angles between fingers and wrist
            angles.append(self.calculate_angle(landmarks[0], landmarks[1], landmarks[2]))
            angles.append(self.calculate_angle(landmarks[0], landmarks[5], landmarks[6]))
            angles.append(self.calculate_angle(landmarks[0], landmarks[9], landmarks[10]))
            angles.append(self.calculate_angle(landmarks[0], landmarks[13], landmarks[14]))
            angles.append(self.calculate_angle(landmarks[0], landmarks[17], landmarks[18]))

            # Add angles to the numpy array
            angles_list.append(np.array(angles))

            # Normalize angles within the range [0, 1]
            normalized_angles = []
            max_angle = max([np.max(angles) for angles in angles_list])
            min_angle = min([np.min(angles) for angles in angles_list])

            for angles in angles_list:
                normalized_angles.append((angles - min_angle) / (max_angle - min_angle))

        return normalized_angles

    def calculate_angle(self, point1, point2, point3):
        """
        Calculate angle between three points.

        Parameters:
        - point1: tuple, (x, y) coordinates of the first point
        - point2: tuple, (x, y) coordinates of the second point
        - point3: tuple, (x, y) coordinates of the third point

        Returns:
        - angle: float, the angle between the three points in degrees
        """
        v1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        v2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        dot_product = np.dot(v1, v2)
        v1_magnitude = np.linalg.norm(v1)
        v2_magnitude = np.linalg.norm(v2)
        angle_rad = np.arccos(dot_product / (v1_magnitude * v2_magnitude))
        angle_deg = np.degrees(angle_rad)

        return angle_deg

# Class for Data Saving
class DataSaver:
    def __init__(self, csv_file):
        self.csv_file = csv_file

        # Create CSV file if it does not exist on current directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        if not os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    header = ['person','gesture']
                    for i in range(1, 211):
                        header.append(f'f{i}')
                    for i in range(1, 16):
                        header.append(f'g{i}')
                    writer.writerow(header)
                    
            except IOError as e:
                print(f"Error creating CSV file: {str(e)}")
                raise

        # Notify if creation was successful
        print(f"CSV file '{self.csv_file}' is ready.")

    def save_data(self, gesture, capture_person, features_list):
        label = gesture
        features = features_list
        data = [capture_person, label]
        data.extend(features)

        with open(self.csv_file, mode='a', newline='') as file: 
            writer = csv.writer(file)
            writer.writerow(data)

# Class for Model Training
class RandomForestTrainer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = pd.read_csv(csv_file,header=0)
        self.model = None
        self.all_columns = list(self.data.columns)

    def preprocess_data(self):
        print(self.data.info())
        if self.data.isnull().sum().sum() > 0:
            self.data = self.data.dropna()
        print(self.data['gesture'].nunique())
        print(self.data['person'].nunique())
        self.data = self.data.drop('person', axis=1) 

    def split_data(self, test_size=0.2, random_state=42):
        X = self.data.drop('gesture', axis=1)
        y = self.data['gesture']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Print the shapes of the training and testing datasets
        print('Shape of X_train:', np.array(self.X_train).shape)
        print('Shape of X_test:', np.array(self.X_test).shape)
        print('Shape of y_train:', self.y_train.shape)
        print('Shape of y_test:', self.y_test.shape)

    def create_model(self,ip_n_estimators,ip_max_depth,ip_min_samples_split,ip_min_samples_leaf,ip_max_features,ip_bootstrap,ip_random_state):
        self.model = RandomForestClassifier(
            n_estimators=ip_n_estimators,
            max_depth=ip_max_depth,
            min_samples_split=ip_min_samples_split,
            min_samples_leaf=ip_min_samples_leaf,
            max_features=ip_max_features,
            bootstrap=ip_bootstrap,
            random_state=ip_random_state
        )

    def train_model(self,save_location='m3_rf_model.joblib'):
        self.model.fit(self.X_train, self.y_train)
        model_file = save_location
        dump(self.model, model_file)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        print(self.X_test.info())
        print(classification_report(self.y_test, y_pred))
        print('Accuracy:', self.model.score(self.X_test, self.y_test))