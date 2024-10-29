import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to detect black color
def is_wearing_black(frame, landmarks):
    # Get bounding box around the person using pose landmarks
    h, w, _ = frame.shape
    x_min = int(min([landmarks.landmark[i].x for i in range(11, 15)]) * w)
    x_max = int(max([landmarks.landmark[i].x for i in range(11, 15)]) * w)
    y_min = int(min([landmarks.landmark[i].y for i in range(11, 15)]) * h)
    y_max = int(max([landmarks.landmark[i].y for i in range(11, 15)]) * h)

    # Crop the region of interest (ROI)
    roi = frame[y_min:y_max, x_min:x_max]

    # Check if the ROI is valid
    if roi.size == 0:
        return False
    
    # Convert the ROI to HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define the range for black color in HSV
    lower_black = np.array([0, 0, 0])  # Lower bound of black
    upper_black = np.array([180, 255, 50])  # Upper bound of black

    # Create a mask for black color
    mask = cv2.inRange(hsv_roi, lower_black, upper_black)

    # Calculate the percentage of black pixels
    black_pixels = cv2.countNonZero(mask)
    total_pixels = roi.size // 3  # Divide by 3 for color channels
    black_ratio = black_pixels / total_pixels if total_pixels > 0 else 0

    # Check if the ratio of black pixels is above a threshold
    return black_ratio > 0.5  # Adjust threshold as needed

# Capture video from webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR image to RGB for MediaPipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detect the pose
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check if landmarks are detected
        if results.pose_landmarks:
            # Check if the person is wearing black
            if is_wearing_black(frame, results.pose_landmarks):
                # Draw the landmarks of the person wearing black
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
