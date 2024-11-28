import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize Mediapipe Pose and Drawing
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseMeasurement:
    def __init__(self, user_height_cm):
        self.user_height_cm = user_height_cm
        self.body_measurements = []

    # Function to calculate distance between two points
    def calculate_distance(self, point1, point2):
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

    # Function to calculate angle
    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle
        return angle

    # Function to calculate hip angle using arccos formula
    def calculate_hip_angle(self, hip_left, hip_right, initial_hip_length):
        # Calculate the new distance between left_hip and right_hip (L)
        new_hip_length = abs(hip_left.x - hip_right.x)

        # Avoid division by zero
        if new_hip_length == 0 or initial_hip_length == 0:
            return 0

        # Calculate theta (hip angle) using arccos formula: theta = arccos(L / x)
        cos_theta = new_hip_length / initial_hip_length
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Clamp value to avoid numerical errors
        theta_radians = np.arccos(cos_theta)
        theta_degrees = np.degrees(theta_radians)

        print("init : ",initial_hip_length)
        
        return theta_degrees
    
    def calculate_cheast_angle(self, shoulder_left, shoulder_right, initial_shoulder_length):
        # Calculate the new distance between left_hip and right_hip (L)
        new_shoulder_length = abs(shoulder_left.x - shoulder_right.x)

        # Avoid division by zero
        if new_shoulder_length == 0 or initial_shoulder_length == 0:
            return 0

        # Calculate theta (hip angle) using arccos formula: theta = arccos(L / x)
        cos_theta = new_shoulder_length / initial_shoulder_length
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Clamp value to avoid numerical errors
        theta_radians = np.arccos(cos_theta)
        theta_degrees = np.degrees(theta_radians)

        print("init1 : ",initial_shoulder_length)
        
        return theta_degrees
    
    def calculate_shoulder_angle(self, shoulder, elbow ,tmp):
        shoulder_pos = [shoulder.x, shoulder.y]
        elbow_pos = [elbow.x, elbow.y]
        tmp_pos = [tmp.x, tmp.y]
        angle = self.calculate_angle(shoulder_pos, elbow_pos,tmp_pos)
        return angle

    # Function to estimate user's height
    def estimate_user_height(self, frame, nose, left_ankle):
        height_normalized = self.calculate_distance(nose, left_ankle)
        height_in_pixels = height_normalized * frame.shape[0]
        scale_factor = self.user_height_cm / height_in_pixels if height_in_pixels > 0 else 1
        return height_in_pixels * scale_factor

    # Function to process pose landmarks and calculate body measurements
    def process_pose(self, frame, results):
        try:
            landmarks = results.pose_landmarks.landmark

            # Extract key landmarks
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            nose = landmarks[mp_pose.PoseLandmark.NOSE]

            # Estimate user height
            user_height_estimated = self.estimate_user_height(frame, nose, left_ankle)

            left_shoulder_angle = self.calculate_shoulder_angle(left_elbow,left_shoulder,right_shoulder)
            right_shoulder_angle = self.calculate_shoulder_angle(right_elbow,right_shoulder,left_shoulder)



            # Calculate arm lengths
            # Upper_Left_arm = self.calculate_distance(left_shoulder, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW])* frame.shape[0]
            # Lower_left_arm = self.calculate_distance(left_elbow, landmarks[mp_pose.PoseLandmark.LEFT_WRIST])* frame.shape[0]
            # Upper_Right_arm = self.calculate_distance(right_shoulder, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW])* frame.shape[0]
            # Lower_right_arm = self.calculate_distance(right_elbow, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])* frame.shape[0]

            # Leg lengths
            # Upper_left_leg = self.calculate_distance(left_hip, landmarks[mp_pose.PoseLandmark.LEFT_KNEE])* frame.shape[0]
            # Lower_left_leg = self.calculate_distance(left_knee, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])* frame.shape[0]
            # Upper_rigth_leg = self.calculate_distance(right_hip, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE])* frame.shape[0]
            # Lower_right_leg = self.calculate_distance(right_knee, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])* frame.shape[0]

            # Body lengths
            # left_body_length = self.calculate_distance(left_shoulder, left_hip) * frame.shape[0]
            # right_body_length = self.calculate_distance(right_shoulder, right_hip) * frame.shape[0]
            # chest_length = self.calculate_distance(left_shoulder, right_shoulder) * frame.shape[0]

            # Elbow angles
            left_elbow_angle = self.calculate_angle(
                [left_shoulder.x, left_shoulder.y],
                [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y])
            right_elbow_angle = self.calculate_angle(
                [right_shoulder.x, right_shoulder.y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y])

            # Hip length must be static morikawa
            hip_length_initial = 0.07923012971878052
            # hip_length_initial = abs(left_hip.x-right_hip.x)

            hip_lenght = {"Morikawa" : 0.07446172833442688, "KO" : 0.13735729455947876,"Wood" :0.06401515007019043
                           ,"Korda" :0.09038853645324707 ,
                          "Mcilroy" : 0.07923012971878052}


            shoulder_length_initial =  0.1309712529182434
            # shoulder_length_initial = abs(left_shoulder.x-right_shoulder.x)
            
            shoulder_lenght = {"Morikawa" : 0.12147849798202515, "KO":0.2453744113445282,"Wood" : 0.10393267869949341
                           ,"Korda" :0.15254566073417664 ,
                           "Mcilroy" : 0.1309712529182434}

            # Calculate hip angle
            hip_angle = self.calculate_hip_angle(left_hip, right_hip, hip_length_initial)

            # Calculate cheast angle
            cheast_angle = self.calculate_cheast_angle(left_shoulder, right_shoulder,shoulder_length_initial)

            # Add new measurement to array
            self.body_measurements.append([ 
                user_height_estimated, 
                # Upper_Left_arm, 
                # Lower_left_arm,
                # Upper_Right_arm,
                # Lower_right_arm,
                # Upper_left_leg,
                # Lower_left_leg,
                # Upper_rigth_leg, 
                # Lower_right_leg, 
                # left_body_length, 
                # right_body_length,
                # chest_length,
                left_elbow_angle, 
                right_elbow_angle,
                left_shoulder_angle, 
                right_shoulder_angle,
                hip_length_initial,
                hip_angle,shoulder_length_initial,cheast_angle
            ])

            # Debug prints
            print(f"User Estimated Height: {user_height_estimated:.2f} cm")
            print(f"Hip Angle: {hip_angle:.2f}°")
            print(f"Measurements: [Left Elbow Angle: {left_elbow_angle}, Right Elbow Angle: {right_elbow_angle}, Left Shoulder Angle: {left_shoulder_angle}, Right Shoulder Angle: {right_shoulder_angle}, Hip Angle: {hip_angle}],Cheast Angle: {cheast_angle}]")

        except Exception as e:
            print(f"Error processing landmarks: {e}")

    def save_measurements(self, filename):
        with open(filename, "w") as array_file:
            array_file.write("body_measurements = " + repr(self.body_measurements))

def Capture(n):
    vid = cv2.VideoCapture(n)

    while vid.isOpened():
        ret, frame = vid.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                img_name = "Capture"
                cv2.imwrite(img_name, image)
                image = cv2.imread(img_name)
                return frame
            else:
                cv2.imshow(image)

# Example usage:
if __name__ == "__main__":
    # Set user height in cm
    user_height_cm = 175
    pose_measurement = PoseMeasurement(user_height_cm)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame = Capture(1)
        if frame is None:
            print("Error: Could not load image.")
            exit()

        # Check if the image loads correctly by showing it
        cv2.imshow("Input Image", frame)
        cv2.waitKey(0)

        # Convert frame from BGR to RGB (original frame remains in BGR)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Set the flag to allow modifications
        image_rgb.flags.writeable = True

        # Detect pose
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            print("No landmarks detected. Please check the input image.")
            exit()

        # Draw landmarks on the image
        mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Convert back to BGR for OpenCV to display or save the image
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Process pose and calculate measurements
        pose_measurement.process_pose(frame, results)

        # Save the body measurements to a file
        pose_measurement.save_measurements("body_measurements.py")

        # Show the image with pose landmarks
        cv2.imshow('Pose Landmarks', image_bgr)
        cv2.imwrite('output_pose_with_measurements.png', image_bgr)

        # Wait for a key press to close the window
        cv2.waitKey(0)

    # Close resources
    cv2.destroyAllWindows()
