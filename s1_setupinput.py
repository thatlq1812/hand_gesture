import cv2
import os
import mediapipe as mp
import csv

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_hands(self, image):
        # Đọc ảnh
        if image is None:
            raise ValueError("Cannot read image")
        
        # Chuyển ảnh sang RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Phát hiện bàn tay
        results = self.hands.process(image_rgb)
        boxes = []
        # Vẽ rectangle bao quanh bàn tay
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min, y_min = float('inf'), float('inf')
                x_max, y_max = float('-inf'), float('-inf')

                # Tìm tọa độ của bounding box
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                    
                # Mở rộng bounding box thêm 20 pixels ở tất cả các cạnh
                x_min = int(max(0, x_min - 20))
                y_min = int(max(0, y_min - 20))
                x_max = int(min(image.shape[1], x_max + 20))
                y_max = int(min(image.shape[0], y_max + 20))

                boxes.append((x_min, y_min, x_max, y_max))
        
        return boxes

class Data_Saver:
    def __init__(self):
        self.data = []
    
    def save_data(self, image, boxes, image_path, csv_file):
        # Save image
        cv2.imwrite(image_path, image)
        
        # Count number of hands
        num_hands = len(boxes)
        
        # Save data to CSV file
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            data = [image_path, num_hands]
            for box in boxes:
                data.extend(box)

            writer.writerow(data)

# Use the code below to test the HandDetector class
if __name__ == "__main__":
    detector = HandDetector()
    saver = Data_Saver()
    cap = cv2.VideoCapture(0)
    # Setup camera size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 10)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # Information of saving data
    capture_person = 'thatlq'
    image_folder = 'data/images/'
    csv_file = 'data/info.csv'
    saving = False

    # Create folder and csv file if not exist
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['image_path', 'num_hands', '1_x_1', '1_y_1', '1_x_2', '1_y_2', '2_x_1', '2_y_1', '2_x_2', '2_y_2'])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (256, 144))
        cv2.imshow('Frame', frame)
        box = detector.detect_hands(frame)
        if saving:
            try:
                saver.save_data(frame, box, image_folder + f'{capture_person}_img_{len(os.listdir(image_folder)) + 1}.jpg', csv_file)
                print(f'Saved {capture_person}_img_{len(os.listdir(image_folder)) + 1}.jpg')
            except:
                pass

        if cv2.waitKey(1) & 0xFF == ord('s'):
            saving = not saving
            print('Started saving images' if saving else 'Ended saving images')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
