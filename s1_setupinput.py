import cv2
import os
import mediapipe as mp
import keyboard
import csv

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_hands(self, image):
        # Đọc ảnh
        if image is None:
            raise ValueError("Không thể đọc ảnh")
        
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
                x_min = max(0, x_min - 20)
                y_min = max(0, y_min - 20)
                x_max = min(image.shape[1], x_max + 20)
                y_max = min(image.shape[0], y_max + 20)

                boxes.append((x_min, y_min, x_max, y_max))
        
        return boxes

class Data_Saver:
    def __init__(self):
        self.data = []
    
    def save_data(self, image, boxes, image_path, csv_file):

        # Lưu ảnh
        cv2.imwrite(image_path, image)
        
        # Đếm số lượng tay
        num_hands = len(boxes) 

        # Lưu thông tin vào file CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            for box in boxes:
                # Ghi thông tin của mỗi tay vào file CSV
                writer.writerow([image_path, num_hands] + list(box))


# Sử dụng lớp HandDetector để phát hiện bàn tay trong camera
if __name__ == "__main__":
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    # Setup camere size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 10)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # Information of saving data
    image_folder = 'data/images/'
    csv_file = 'data/data_info.csv'

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        box = detector.detect_hands(frame)

        # Hiển thị ảnh đã được chú thích box 1 and box 2
        try:
            detector.save_data(frame, box, image_folder + f'image_{len(os.listdir(image_folder)) + 1}.jpg', csv_file)
            print(f'Lưu ảnh và thông tin vào {image_folder} và {csv_file}')
        except:
            pass

        if keyboard.is_pressed('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()