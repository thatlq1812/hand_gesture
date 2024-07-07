import cv2
import os
import mediapipe as mp
import csv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_hands(self, image):
        if image is None:
            raise ValueError("Cannot read image")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        boxes = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min, y_min = float('inf'), float('inf')
                x_max, y_max = float('-inf'), float('-inf')

                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                    
                x_min = int(max(0, x_min - 20))
                y_min = int(max(0, y_min - 20))
                x_max = int(min(image.shape[1], x_max + 20))
                y_max = int(min(image.shape[0], y_max + 20))

                boxes.append((x_min, y_min, x_max, y_max))
        
        return boxes

class DataSaver:
    def __init__(self, image_folder, csv_file):
        self.image_folder = image_folder
        self.csv_file = csv_file

        # Create image folder if it does not exist
        if not os.path.exists(self.image_folder):
            try:
                os.makedirs(self.image_folder)
            except OSError as e:
                print(f"Error creating image folder: {str(e)}")
                raise

        # Create CSV file if it does not exist
        if not os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['image_path', 'num_hands', '1_x_1', '1_y_1', '1_x_2', '1_y_2', '2_x_1', '2_y_1', '2_x_2', '2_y_2'])
            except IOError as e:
                print(f"Error creating CSV file: {str(e)}")
                raise

        # Notify if creation was successful
        print(f"Image folder '{self.image_folder}' and CSV file '{self.csv_file}' are ready.")


        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['image_path', 'num_hands', '1_x_1', '1_y_1', '1_x_2', '1_y_2', '2_x_1', '2_y_1', '2_x_2', '2_y_2'])

    def save_data(self, image, boxes, capture_person):
        image_path = os.path.join(self.image_folder, f'{capture_person}_img_{len(os.listdir(self.image_folder)) + 1}.jpg')
        cv2.imwrite(image_path, image)
        
        num_hands = len(boxes)
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            data = [image_path, num_hands]
            for box in boxes:
                data.extend(box)

            writer.writerow(data)

class HandProcessor:
    def __init__(self, image_folder, csv_file):
        self.detector = HandDetector()
        self.saver = DataSaver(image_folder, csv_file)

    def process_video(self, video_path, capture_person):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video file: {video_path}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes = self.detector.detect_hands(frame)
            try:
                self.saver.save_data(frame, boxes, capture_person)
                print(f'Saved {capture_person}_img_{len(os.listdir(self.saver.image_folder))}.jpg')
            except Exception as e:
                print(f"Error saving data: {str(e)}")

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_camera(self, capture_person, cam_index, frame_width, frame_height, frame_rate):
        cap = cv2.VideoCapture(cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        cap.set(cv2.CAP_PROP_FPS, frame_rate)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow('Frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('s'):
                frame = cv2.resize(frame, (256, 144))
                boxes = self.detector.detect_hands(frame)
                try:
                    self.saver.save_data(frame, boxes, capture_person)
                    print(f'Saved {capture_person}_img_{len(os.listdir(self.saver.image_folder)) + 1}.jpg')
                except Exception as e:
                    print(f"Error saving data: {str(e)}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Data information
    capture_person = 'thatlq'
    image_folder = 'data/images/'
    csv_file = 'data/info.csv'

    # Camera settings
    cam_index = 0
    frame_width = 1280
    frame_height = 720
    frame_rate = 10

    # Video setting
    video_path = 'data/video.mp4'

    # Process data
    hand_processor = HandProcessor(image_folder, csv_file)

    # Camera processing
    hand_processor.process_camera(capture_person, cam_index, frame_width, frame_height, frame_rate)

    # Or process video
    # hand_processor.process_video(video_path, capture_person)
