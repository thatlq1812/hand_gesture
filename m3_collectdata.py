from m3_module import * # Import all class in module

warnings.filterwarnings("ignore") # Ignore warnings

if __name__ == "__main__":
    print("Program started...")
    # Data information
    # Get PC username
    username = getpass.getuser()
    capture_person = username
    csv_file = 'm3_info.csv'
    gesture_file = 'm3_gesture.csv'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1 # From 0 to 1
    text_color_1 = (255, 0, 255)
    count_down = 0.25 # Set the count down time - seconds
    

    # Create objects
    hand_detector = MediaPipe(gl_static_image_mode, 1) # Set 1 hand to create dataset
    cam = Camera(gl_cam_index, gl_cam_width, gl_cam_height, gl_cam_fps)
    data_processor = DataProcessing()
    saving = DataSaver(csv_file)

    # Initialize the camera
    cap = cam.initialize_camera()
    if not cap:
        print("Error: Camera could not be initialized.")
        exit()
    else:
        print("Camera initialized.")

    # Show gesture list
    print("Gesture list:")
    with open(gesture_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            print(f'Label: {row[0]}, Name: {row[1]}')

    # Input gesture name
    gesture = input("Enter the gesture label: ")
    sec = int(count_down * gl_cam_fps)
    save = False
    count = 0
    # Test camera functionality
    while True:
        frame = cam.get_frame()
        if frame is None:
            print("Error: Cannot receive frame.")
            break
        
        # Process frame with MediaPipe
        hand_detector.process_image(frame) # Process the frame to detect hands
        
        # Process the frame and print features
        num_hands = hand_detector.num_detected_hands() # Get the number of detected hands
        if num_hands > 0:
            frame = hand_detector.draw_landmarks(frame) # Draw landmarks on the frame
            landmarks_list = hand_detector.get_hand_landmarks() # Get landmarks of detected hands
            features_list = data_processor.calculate(landmarks_list) # Calculate features

        if sec == 0:
            cv2.putText(frame, "Recording...", (10, 30), font, font_scale, text_color_1, 2, cv2.LINE_AA)
            # Save the features to a CSV file
            if save and num_hands > 0:
                saving.save_data(gesture, capture_person, features_list)
                count += 1
            sec = int(count_down * gl_cam_fps)
      
        else:
            cv2.putText(frame, f"Count down: {sec}", (10, 30), font, font_scale, text_color_1, 2, cv2.LINE_AA)
            sec -= 1

        # Put text on the frame
        cv2.putText(frame, f"Number of hands: {num_hands}", (10, 60), font, font_scale, text_color_1, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Hold 'q' to quit.", (10, 90), font, font_scale, text_color_1, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Gesture: {gesture}", (10, 120), font, font_scale, text_color_1, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Capture person: {capture_person}", (10, 150), font, font_scale, text_color_1, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {gl_cam_fps}", (10, 180), font, font_scale, text_color_1, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Count: {count}", (10, 240), font, font_scale, text_color_1, 2, cv2.LINE_AA)
        if save:
            cv2.putText(frame, "Press 's' to stop saving.", (10, 210), font, font_scale, text_color_1, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Press 's' to start saving.", (10, 210), font, font_scale, text_color_1, 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Processing frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
        # Save toggle button
        if cv2.waitKey(10) & 0xFF == ord('s'):
            save = not save
    
    cv2.destroyAllWindows()
    # Release the camera
    cam.close_camera()

# End of file, print the result
print("The program has ended...")