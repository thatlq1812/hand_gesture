import cv2
import mediapipe as mp
import numpy as np
import threading
import math
import time

# Global constants
gCamIndex = 0
gFPS = 30

#class VectorOperations:

class HandDetector:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_hands(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        return results

    def draw_hand_landmarks(self, image, hand_landmarks):
        self.mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        )

    def get_landmark_coordinates(self, hand_landmarks, image_shape):
        image_height, image_width, _ = image_shape
        landmarks = {}
        for landmark in self.mp_hands.HandLandmark:
            lm = hand_landmarks.landmark[landmark]
            landmarks[landmark] = (int(lm.x * image_width), int(lm.y * image_height))
        return landmarks

class HandDetection:
    def __init__(self, hand_detector):
        self.cap = cv2.VideoCapture(gCamIndex)
        self.hand_detector = hand_detector
        self.gesture_recognition = gesture_recognition

        # Thiết lập kích thước cửa sổ
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, gFPS)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        # FPS counter
        self.prev_frame_time = 0
        self.new_frame_time = 0

    def run_detection(self):
        while True:
            # Đọc một khung hình từ camera
            ret, frame = self.cap.read()

            # Ghi thời gian hiện tại
            self.new_frame_time = time.time()

            # Phát hiện các tay trên khung hình
            results = self.hand_detector.detect_hands(frame)

            # Vẽ các điểm landmark của tay trên khung hình
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.hand_detector.draw_hand_landmarks(frame, hand_landmarks)

            # Vẽ tọa độ các điểm landmark trên tay
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = self.hand_detector.get_landmark_coordinates(hand_landmarks, frame.shape)          
                    gesture = self.gesture_recognition.recognize_finger_gesture(landmarks)
                    cv2.putText(frame, gesture, (800, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    space = 0
                    for landmark, (x, y) in landmarks.items():
                        cv2.putText(frame, f"{landmark.name}: ({x}, {y})", (50, 25 + space), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        space += 15

            # Tính FPS
            fps = 1 / (self.new_frame_time - self.prev_frame_time)
            self.prev_frame_time = self.new_frame_time

            # Hiển thị FPS trên khung hình
            cv2.putText(frame, f"FPS: {int(fps)}", (50, 680), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # Hiển thị khung hình đã xử lý
            cv2.imshow('Hand Detection', frame)

            # Thoát vòng lặp nếu phím 'q' được nhấn
            if cv2.waitKey(1) != -1:
                break

        # Giải phóng tài nguyên và đóng cửa sổ
        self.cap.release()
        cv2.destroyAllWindows()

class GestureRecognition:
    def __init__(self):
        pass

    def distance(self, point1, point2):
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    def angle(self, point1, point2, point3):
        # Vector AB
        AB = (point1[0] - point2[0], point1[1] - point2[1])
        # Vector BC
        BC = (point3[0] - point2[0], point3[1] - point2[1])
        # Dot product
        dot_product = AB[0] * BC[0] + AB[1] * BC[1]
        # Magnitudes
        magnitude_AB = self.distance(point1, point2)
        magnitude_BC = self.distance(point2, point3)
        # Cosine of the angle
        if magnitude_AB == 0 or magnitude_BC == 0:
            return 0
        cos_angle = dot_product / (magnitude_AB * magnitude_BC)
        # Angle in radians
        if abs(cos_angle) > 1:
            return 0
        angle = math.acos(cos_angle)
        # Convert to degrees
        angle_degrees = math.degrees(angle)
        return angle_degrees

    def recognize_finger_gesture(self, landmarks):
        """
        
        """
        # Lấy tọa độ của các điểm landmark
        wrist = landmarks[mp.solutions.hands.HandLandmark.WRIST]
        thumb_mcp = landmarks[mp.solutions.hands.HandLandmark.THUMB_MCP]
        thumb_cmc = landmarks[mp.solutions.hands.HandLandmark.THUMB_CMC]
        thumb_ip = landmarks[mp.solutions.hands.HandLandmark.THUMB_IP]
        thumb_tip = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]
        index_mcp = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
        index_pip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]
        index_dip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP]
        index_tip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        middle_mcp = landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
        middle_pip = landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
        middle_dip = landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP]
        middle_tip = landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_mcp = landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_MCP]
        ring_pip = landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_PIP]
        ring_dip = landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_DIP]
        ring_tip = landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
        pinky_mcp = landmarks[mp.solutions.hands.HandLandmark.PINKY_MCP]
        pinky_pip = landmarks[mp.solutions.hands.HandLandmark.PINKY_PIP]
        pinky_dip = landmarks[mp.solutions.hands.HandLandmark.PINKY_DIP]
        pinky_tip = landmarks[mp.solutions.hands.HandLandmark.PINKY_TIP]

        # Tính góc từ mcp - pip và pip - dip 4 ngón chính
        angle_index_mcp_pip_dip = self.angle(index_mcp,index_pip,index_dip)
        angle_middle_mcp_pip_dip = self.angle(middle_mcp,middle_pip,middle_dip)
        angle_ring_mcp_pip_dip = self.angle(ring_mcp,ring_pip,ring_dip)
        angle_pinky_mcp_pip_dip = self.angle(pinky_mcp,pinky_pip,pinky_dip)

        # Các nhận diện

        if (max(angle_index_mcp_pip_dip, angle_ring_mcp_pip_dip, angle_pinky_mcp_pip_dip) < 135) and (angle_middle_mcp_pip_dip > 135):
            # Nhận diện cử chỉ "I love u"
            return "I love u!"
        elif (max(angle_index_mcp_pip_dip, angle_middle_mcp_pip_dip, angle_ring_mcp_pip_dip, angle_pinky_mcp_pip_dip) < 135) and (thumb_tip[1] < thumb_ip[1]) and (thumb_tip[1] < index_mcp[1]):
            # Nhận diện cử chỉ thumps up
            return "Thumps up!"
        elif (max(angle_index_mcp_pip_dip, angle_middle_mcp_pip_dip, angle_ring_mcp_pip_dip, angle_pinky_mcp_pip_dip) < 135) and (thumb_tip[1] > thumb_ip[1]) and (thumb_tip[1] > index_mcp[1]):
            # Nhận diện cử chỉ thumps down
            return "Thumps down!"
        elif (max(angle_middle_mcp_pip_dip,angle_ring_mcp_pip_dip)<135) and (angle_index_mcp_pip_dip>135) and (angle_pinky_mcp_pip_dip>135):
            # Nhận diện cử chỉ spiderman
            return "Spiderman!"
        elif (max(angle_ring_mcp_pip_dip, angle_pinky_mcp_pip_dip) < 135) and (angle_middle_mcp_pip_dip > 135) and (angle_index_mcp_pip_dip > 135):
            # Nhận diện cử chỉ Hi
            return "Hi!"
        elif (angle_middle_mcp_pip_dip > 135) and (angle_ring_mcp_pip_dip > 135) and (angle_pinky_mcp_pip_dip > 135) and (angle_index_mcp_pip_dip < 135) and (self.distance(index_tip,thumb_tip) < self.distance(index_tip,index_mcp)):
            # Nhận diện cử chỉ "OK"
            return "OK!"
        else:
            return "Gesture not recognized"

if __name__ == "__main__":
    gesture_recognition = GestureRecognition()
    hand_detector = HandDetector()
    hand_detection = HandDetection(hand_detector)
    hand_detection_thread = threading.Thread(target=hand_detection.run_detection)
    hand_detection_thread.start()
    hand_detection_thread.join()