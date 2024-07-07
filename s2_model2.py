import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import csv

class HandDetector:
    def __init__(self):
        self.svm = None
        self.scaler = StandardScaler()

    def train(self, csv_file):
        # Load training data from CSV
        images = []
        labels = []

        with open(csv_file, newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                image_path = row[0]
                num_hands = int(row[1])
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if num_hands == 0:
                    images.append(image)
                    labels.append(0)
                else:
                    boxes = []
                    for i in range(num_hands):
                        x_min = int(row[2 + i*4])
                        y_min = int(row[3 + i*4])
                        x_max = int(row[4 + i*4])
                        y_max = int(row[5 + i*4])
                        boxes.append((x_min, y_min, x_max, y_max))
                    
                    for box in boxes:
                        x_min, y_min, x_max, y_max = box
                        hand_region = image[y_min:y_max, x_min:x_max]
                        images.append(hand_region)
                        labels.append(1)

        # Extract HOG features
        features = [hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2)) for img in images]
        features = self.scaler.fit_transform(features)

        # Train SVM
        self.svm = SVC(kernel='linear', probability=True)
        self.svm.fit(features, labels)

        # Save the model
        dump(self.svm, 'svm_hand_detector.joblib')
        dump(self.scaler, 'scaler.joblib')

    def load_model(self):
        self.svm = load('svm_hand_detector.joblib')
        self.scaler = load('scaler.joblib')

    def detect(self, image, win_size=(64, 128), step_size=16, scale_factor=1.25):
        detected_boxes = []
        orig_image = image.copy()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for scale in range(0, int(np.log(min(image.shape[0], image.shape[1]) / win_size[0]) / np.log(scale_factor)) + 1):
            resized_image = cv2.resize(gray_image, (int(image.shape[1] / (scale_factor ** scale)), int(image.shape[0] / (scale_factor ** scale))))
            for y in range(0, resized_image.shape[0] - win_size[1], step_size):
                for x in range(0, resized_image.shape[1] - win_size[0], step_size):
                    window = resized_image[y:y + win_size[1], x:x + win_size[0]]
                    feature = hog(window, pixels_per_cell=(8, 8), cells_per_block=(2, 2)).reshape(1, -1)
                    feature = self.scaler.transform(feature)
                    prediction = self.svm.predict_proba(feature)
                    if prediction[0][1] > 0.5:  # threshold
                        detected_boxes.append((int(x * (scale_factor ** scale)), int(y * (scale_factor ** scale)), int((x + win_size[0]) * (scale_factor ** scale)), int((y + win_size[1]) * (scale_factor ** scale))))

        # Apply Non-Maximum Suppression
        final_boxes = self.non_max_suppression(detected_boxes, 0.3)
        for box in final_boxes:
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        return orig_image, final_boxes

    def non_max_suppression(self, boxes, overlap_thresh):
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        pick = []
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        return boxes[pick].astype("int")

# Example usage
if __name__ == "__main__":
    detector = HandDetector()
    
    # Train the model (only once, you can comment out after training)
    detector.train('data/data_info.csv')

    # Load the trained model
    detector.load_model()

    # Read an image for testing
    test_image = cv2.imread('path_to_test_image.jpg')

    # Detect hands in the image
    result_image, hand_boxes = detector.detect(test_image)
    
    # Save the result image
    cv2.imwrite('result.jpg', result_image)
    
    # Show the result image
    cv2.imshow('Hand Detection', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
