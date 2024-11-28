import cv2
import mediapipe as mp
import math
import numpy as np
import csv

class BodyMeasurement:
    def __init__(self, user_height_cm=174):
        # Initialize Mediapipe Pose and Segmentation
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        
        # Initialize camera and user height
        self.user_height_cm = user_height_cm
        self.cap = cv2.VideoCapture(1)

        # Open CSV file to log data
        #self.csv_file = open('body_measurements.csv', mode='w', newline='')
        #self.csv_writer = csv.writer(self.csv_file)
        #self.csv_writer.writerow([
        #    "Estimated Height (cm)", 
        #    "Left Arm Length (cm)", "Right Arm Length (cm)", 
        #    "Left Leg Length (cm)", "Right Leg Length (cm)", 
        #    "Left Body Length (cm)", "Right Body Length (cm)", 
        #    "Chest Length (cm)",
        #   "Left Elbow Angle (°)", "Right Elbow Angle (°)", 
        #    "Left Knee Angle (°)", "Right Knee Angle (°)"
        #])
        
    # Function to calculate distance between two points
    @staticmethod
    def calculate_distance(point1, point2):
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

    # Function to calculate angle
    @staticmethod
    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle
        return angle
    
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
    
    def process_frame(self):    
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
             self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
            
            while self.cap.isOpened():
                leave = 0
                ret, frame = self.cap.read()
    
                if not ret:
                    print("Failed to capture frame. Exiting...")
                    break

                # Convert frame from BGR to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Detect pose
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Perform segmentation
                segmentation_results = selfie_segmentation.process(image)
                mask = segmentation_results.segmentation_mask > 0.5
                body_mask = (mask.astype(np.uint8) * 255)

                # Calculate frame dimensions
                frame_height, frame_width = frame.shape[:2]

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Height from nose to ankle
                    try:
                        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
                        height_normalized = self.calculate_distance(nose, left_ankle)
                        height_in_pixels = height_normalized * frame_height
                        scale_factor = self.user_height_cm / height_in_pixels if height_in_pixels > 0 else 1
                        user_height_estimated = height_in_pixels * scale_factor

                        # Extract key landmarks
                        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
                        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
                        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
                        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
                        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
                        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
                        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
                        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]

                        # Calculate arm lengths
                        Upper_Left_arm = self.calculate_distance(left_shoulder, left_elbow) * frame.shape[0] * scale_factor
                        Lower_left_arm = self.calculate_distance(left_elbow, left_wrist) * frame.shape[0] * scale_factor
                        Upper_Right_arm = self.calculate_distance(right_shoulder, right_elbow) * frame.shape[0] * scale_factor
                        Lower_right_arm = self.calculate_distance(right_elbow, right_wrist) * frame.shape[0] * scale_factor

                        # Calculate leg lengths
                        Upper_left_leg = self.calculate_distance(left_hip, left_knee) * frame.shape[0] * scale_factor
                        Lower_left_leg = self.calculate_distance(left_knee, left_ankle) * frame.shape[0] * scale_factor
                        Upper_rigth_leg = self.calculate_distance(right_hip, right_knee) * frame.shape[0] * scale_factor
                        Lower_right_leg = self.calculate_distance(right_knee, right_ankle) * frame.shape[0] * scale_factor

                        # Calculate body lengths
                        left_body_length = self.calculate_distance(left_shoulder, left_hip) * frame.shape[0] * scale_factor
                        right_body_length = self.calculate_distance(right_shoulder, right_hip) * frame.shape[0] * scale_factor
                        chest_length = self.calculate_distance(left_shoulder, right_shoulder) * frame.shape[0] * scale_factor

                        # Calculate elbow angles
                        left_elbow_pos = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                                          landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
                        angle_left_elbow = self.calculate_angle(
                            [left_shoulder.x, left_shoulder.y], left_elbow_pos,
                            [left_wrist.x, left_wrist.y])

                        right_elbow_pos = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                                           landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                        angle_right_elbow = self.calculate_angle(
                            [right_shoulder.x, right_shoulder.y], right_elbow_pos,
                            [right_wrist.x, right_wrist.y])
                        
                        #calculate shoulder angle
                        left_shoulder_angle = self.calculate_shoulder_angle(left_elbow,left_shoulder,right_shoulder)
                        right_shoulder_angle = self.calculate_shoulder_angle(right_elbow,right_shoulder,left_shoulder)


                        # Calculate knee angles
                        left_foot_index = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                        right_foot_index = landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

                        left_knee_pos = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                                         landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
                        angle_left_knee = self.calculate_angle(
                            [left_hip.x, left_hip.y], left_knee_pos,
                            [left_foot_index.x, left_foot_index.y])

                        right_knee_pos = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].x,
                                          landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].y]
                        angle_right_knee = self.calculate_angle(
                            [right_hip.x, right_hip.y], right_knee_pos,
                            [right_foot_index.x, right_foot_index.y])
                        
                        #calcualte hip angle
                        # hip_length_initial = abs(left_hip.x-right_hip.x)
                        #hip_angle = self.calculate_hip_angle(left_hip, right_hip, hip_length_initial)

                        #calculate cheast angle
                        #shoulder_length_initial = abs(left_shoulder.x-right_shoulder.x)
                        #cheast_angle = self.calculate_cheast_angle(left_shoulder, right_shoulder,shoulder_length_initial)

                        # Log data to CSV
                        #self.csv_writer.writerow([
                        #   user_height_estimated, 
                            # Upper_Left_arm + Lower_left_arm, Upper_Right_arm + Lower_right_arm, 
                            # Upper_left_leg + Lower_left_leg, Upper_rigth_leg + Lower_right_leg, 
                        #    angle_left_elbow, angle_right_elbow, 
                        #    angle_left_knee, angle_right_knee,
                        #    left_shoulder_angle, right_shoulder_angle
                            # left_body_length, right_body_length, chest_length
                        #])

                        # Debug Prints
                        # print(f"Height: {user_height_estimated:.2f} cm")
                        # print(f"Upper Left Arm Length: {Upper_Left_arm:.2f} cm") 
                        # print(f"Lower Left Arm Length: {Lower_left_arm:.2f} cm") 
                        # print(f"Upper Right Arm Length: {Upper_Right_arm:.2f} cm")
                        # print(f"Lower Right Arm Length: {Lower_right_arm:.2f} cm")
                        # print(f"Upper Left Leg Length: {Upper_left_leg :.2f} cm") 
                        # print(f"Lower Left Leg Length: {Lower_left_leg:.2f} cm") 
                        # print(f"Upper Right Leg Length: {Upper_rigth_leg :.2f} cm")
                        # print(f"Lower Right Leg Length: {Lower_right_leg:.2f} cm")
                        # print(f"Left Body Length: {left_body_length:.2f} cm")
                        # print(f"Right Body Length: {right_body_length:.2f} cm")
                        # print(f"Chest Length: {chest_length:.2f} cm")
                        # print(f"Left Shoulder Angle: {left_shoulder_angle:.2f}°")
                        # print(f"Right Shoulder Angle: {right_shoulder_angle:.2f}°")
                        # print(f"Left Elbow Angle: {angle_left_elbow:.2f}°") 
                        # print(f"Right Elbow Angle: {angle_right_elbow:.2f}°")
                        # print(f"Left Knee Angle: {angle_left_knee:.2f}°")
                        # print(f"Right Knee Angle: {angle_right_knee:.2f}°")
                        # print(f"Hip Angle: {hip_angle:.2f}°")
                        # print(f"cheast_angle: {cheast_angle:.2f}°")
                        # print(f"hip_length_initial: {hip_length_initial:.2f}°")
                        # print(f"shoulder_length_initial: {shoulder_length_initial:.2f}°")
                        cv2.putText(frame, f"Left Elbow: {angle_left_elbow:.2f}°", 
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"Right Elbow: {angle_right_elbow:.2f}°", 
                                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"Left Knee: {angle_left_knee:.2f}°", 
                                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"Right Knee: {angle_right_knee:.2f}°",
                                    (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"Left Shoulder: {left_shoulder_angle:.2f}°", 
                                    (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"Right Shoulder: {right_shoulder_angle:.2f}°", 
                                    (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    except Exception as e:
                        print(f"Error processing landmarks: {e}")

                # Display the frame
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                body_mask_colored = cv2.applyColorMap(body_mask, cv2.COLORMAP_JET)
                combined_frame = cv2.addWeighted(frame, 0.6, body_mask_colored, 0.4, 0)
                cv2.imshow('Body Measurements', combined_frame)

                # Break loop on 'q' key press
                break
            
            while(1):
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                body_mask_colored = cv2.applyColorMap(body_mask, cv2.COLORMAP_JET)
                combined_frame = cv2.addWeighted(frame, 0.6, body_mask_colored, 0.4, 0)
                cv2.imshow('Body Measurements', combined_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            #self.cap.release()
            #self.csv_file.close()
            #cv2.destroyAllWindows()

# Usage
body_measurement = BodyMeasurement(user_height_cm=174)
body_measurement.process_frame()
