from m3_module import *

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print("Program started...")

    lmmodel = load("m3_model.joblib")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1 # From 0 to 1
    text_color_1 = (255, 0, 255)
    
    # Create objects
    hand_detector = MediaPipe(gl_static_image_mode, 10) # Set 1 hand to create dataset
    data_processor = DataProcessing()

    # Read video file
    video = cv2.VideoCapture('m3_video_test.mp4')

    # Set the video frame width and height to 1280x720
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Read the gesture names from the csv file
    gesturesdf = pd.read_csv('m3_gesture.csv')
    gestures = gesturesdf['name'].values.tolist()   
    for i in range(len(gestures)):
        print(f"{i}: {gestures[i]}")

    if not video.isOpened():
        print("Error: Video could not be opened.")
        exit()
    else:
        print("Video opened.")

    # Process video frames
    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Cannot receive frame or video has ended.")
            break
        
        # Process frame with MediaPipe
        hand_detector.process_image(frame)  # Process the frame to detect hands
        
        # Process the frame and print features
        num_hands = hand_detector.num_detected_hands()  # Get the number of detected hands
        if num_hands > 0:
            # frame = hand_detector.draw_landmarks(frame)  # Draw landmarks on the frame
            
            # Get landmarks of detected hands
            landmarks_list = hand_detector.get_hand_landmarks()

            for i, landmarks in enumerate(landmarks_list):
                features = data_processor.calculate([landmarks])  # Calculate features for each hand separately
                features = pd.DataFrame(features).T
                
                # Draw the bounding box around the hand
                mmxy = hand_detector.get_minmax_xy(landmarks)
                frame = hand_detector.draw_bounding_box(frame, mmxy)

                # Predict the hand gesture
                gesture = gestures[int(lmmodel.predict(features))]
                # Put the number of hands detected on the top left corner of the hand bounding box
                cv2.putText(frame, gesture, (mmxy[0]-10, mmxy[1]+5), font, font_scale, text_color_1, 2, cv2.LINE_AA)

        # Put text on the frame for the total number of hands detected
        cv2.putText(frame, f"Number of hands: {num_hands}", (10, 100), font, font_scale, text_color_1, 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Processing frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    # Release the video
    video.release()

# End of file, print the result
print("The program has ended...")
