import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Serial communication setup
arduino_port = 'COM8'  # Change to your Arduino port
baud_rate = 9600
ser = serial.Serial(arduino_port, baud_rate)
time.sleep(2)  # Wait for Arduino to reset

# Function to detect black color in the upper body region
def is_wearing_black(frame, landmarks):
    h, w, _ = frame.shape
    # Define region of interest (ROI) based on upper body landmarks
    upper_body_landmarks = [11, 12, 23, 24]  # Shoulders and hips
    x_min = int(min([landmarks.landmark[i].x for i in upper_body_landmarks]) * w)
    x_max = int(max([landmarks.landmark[i].x for i in upper_body_landmarks]) * w)
    y_min = int(min([landmarks.landmark[i].y for i in upper_body_landmarks]) * h)
    y_max = int(max([landmarks.landmark[i].y for i in upper_body_landmarks]) * h)

    roi = frame[y_min:y_max, x_min:x_max]

    if roi.size == 0:
        return False

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])  # Lower bound of black
    upper_black = np.array([180, 255, 40])  # Upper bound of black

    mask = cv2.inRange(hsv_roi, lower_black, upper_black)
    black_pixels = cv2.countNonZero(mask)
    total_pixels = roi.size // 3
    black_ratio = black_pixels / total_pixels if total_pixels > 0 else 0

    return black_ratio > 0.6  # Returns True if more than 60% black detected

# Capture video from webcam
cap = cv2.VideoCapture(2)  # Use appropriate camera index

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            if is_wearing_black(frame, results.pose_landmarks):
                # Draw the landmarks of the person wearing black
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

                # Logic to control the robot based on position of the landmarks
                center_x = int(
                    (results.pose_landmarks.landmark[11].x + results.pose_landmarks.landmark[12].x) * frame.shape[1] / 2
                )

                # Decide movement command based on the center position
                frame_center_x = frame.shape[1] // 2
                if center_x < frame_center_x - 50:  # If the detected person is too far left
                    command = b'L'
                    print("Sending command: L (Turn Left)")
                elif center_x > frame_center_x + 50:  # If too far right
                    command = b'R'
                    print("Sending command: R (Turn Right)")
                else:
                    command = b'F'
                    print("Sending command: F (Move Forward)")

                ser.write(command)  # Send the command to Arduino

            else:
                command = b'S'
                print("Sending command: S (Stop)")
                ser.write(command)  # Stop if no black clothing detected

        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
ser.close()  # Close the serial connection
cv2.destroyAllWindows()
