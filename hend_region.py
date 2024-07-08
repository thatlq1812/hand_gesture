import cv2
import joblib
import numpy as np

rf_model = joblib.load('D:\CODE\hand_gesture\models\hand_region_detection_model.pkl')

def process_video(video_path, output_path, model):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Predict hand regions
        predicted_positions = predict_and_return_coordinates(frame, model)

        # Draw the predicted bounding boxes on the frame
        for pos in predicted_positions:
            x_min, y_min, x_max, y_max = pos
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

    # Release the video objects
    cap.release()
    out.release()
    print("Video processing complete. Output saved to:", output_path)

# Use the function
video_path = 'input_video.mp4'  # Path to your input video
output_path = 'output_video.avi'  # Path to save the output video
process_video(video_path, output_path, rf_model)

