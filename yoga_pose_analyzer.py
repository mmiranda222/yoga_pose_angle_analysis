def determine_variant(self, landmarks, pose_type: str, features=None) -> str:
    """
    Determine if the pose is active or passive.
    
    Args:
        landmarks: The pose landmarks
        pose_type: The identified pose type
        features: Pre-extracted features (optional)
        
    Returns:
        String indicating "active" or "passive"
    """
    # For most poses, the active variant has a smaller angle than the passive variant
    # This is a general rule that we'll use as a baseline
    
    if pose_type == "unknown":
        return "unknown"
    
    # Extract the angle from the pose detection
    angle = None
    for pose_name, pose_checker in self.poses.items():
        if pose_name == pose_type:
            _, _, angle_value = pose_checker(landmarks, (0, 0, 0))  # Image shape doesn't matter here
            angle = angle_value
            break
    
    if angle is None:
        return "unknown"
            
    # As you mentioned, generally passive angles are greater than active angles
    if pose_type.startswith("lunge"):
        # For lunges, smaller angle typically means more active
        return "active" if angle < 90 else "passive"
            
    elif pose_type == "cobra":
        # For cobra, more extension (larger angle) typically means more active
        return "active" if angle > 140 else "passive"
            
    elif pose_type.startswith("hip_opening"):
        # For hip opening, smaller angle typically means more active
        return "active" if angle < 60 else "passive"
            
    elif pose_type == "side_splits":
        # For side splits, wider angle typically means more active
        return "active" if angle > 155 else "passive"
            
    elif pose_type.startswith("front_splits"):
        # For front splits, wider angle typically means more active
        return "active" if angle > 155 else "passive"
    
    # If we don't have specific criteria for this pose type
    return "unknown"

import cv2
import mediapipe as mp
import numpy as np
import os
import csv
from typing import List, Tuple, Dict, Optional

class YogaPoseAnalyzer:
    def __init__(self, image_folder: str, output_csv: str):
        """
        Initialize the Yoga Pose Analyzer.
        
        Args:
            image_folder: Path to the folder containing yoga pose images
            output_csv: Path where the output CSV will be saved
        """
        self.image_folder = image_folder
        self.output_csv = output_csv
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        
        # Define the pose types and their corresponding detection functions
        self.poses = {
            "lunge_right": self.is_lunge_right,
            "lunge_left": self.is_lunge_left,
            "cobra": self.is_cobra,
            "hip_opening_right": self.is_hip_opening_right,
            "hip_opening_left": self.is_hip_opening_left,
            "side_splits": self.is_side_splits,
            "side_splits_lying": self.is_side_splits_lying,
            "front_splits_right": self.is_front_splits_right,
            "front_splits_left": self.is_front_splits_left
        }
        
        # Define pose-specific landmarks and angle calculations
        self.pose_landmarks = {
            "lunge_right": [(28, 26, "derived_right")],
            "lunge_left": [(27, 25, "derived_left")],
            "cobra": [(11, "derived_hip", "derived_point")],
            "hip_opening_right": [(28, 24, 27)],
            "hip_opening_left": [(27, 24, 28)],
            "side_splits": [(28, "derived_hip", 27)],
            "side_splits_lying": [(28, "mid_hip", 27)],
            "front_splits_right": [(28, "mid_hip", 27)],
            "front_splits_left": [(27, "mid_hip", 28)]
        }
        
        # Results storage
        self.results = []
        
        # Visualization settings
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def process_images(self):
        """Process all images in the folder and analyze the yoga poses."""
        image_files = [f for f in os.listdir(self.image_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(self.image_folder, img_file)
            self.analyze_image(img_path, img_file)
        
        # Save results to CSV
        self.save_results()
    
    def analyze_image(self, img_path: str, img_filename: str):
        """
        Analyze a single image to identify pose and calculate angles.
        
        Args:
            img_path: Path to the image file
            img_filename: Filename of the image (for recording purposes)
        """
        # Read the image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error reading image: {img_path}")
            return
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        # Process the image with MediaPipe
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            print(f"No pose landmarks found in: {img_path}")
            return
        
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Identify the pose using the available detection functions
        pose_type = "unknown"
        pose_variant = "unknown"  # active or passive
        pose_side = "unknown"     # left, right, or none
        angle = None
        confidence = 0
        
        # Try each pose detection function
        for pose_name, pose_checker in self.poses.items():
            is_pose, side, angle_value = pose_checker(landmarks, image.shape)
            if is_pose:
                pose_type = pose_name
                pose_side = side
                angle = angle_value
                break
        
        # If we've identified a pose, determine if it's active or passive
        if pose_type != "unknown":
            pose_variant = self.determine_variant(landmarks, pose_type)
        
        # Create a debug image with landmarks and angle visualization
        debug_image = self.visualize_pose(image, landmarks, pose_type, angle)
        
        # Save the debug image for reference
        debug_dir = os.path.join(os.path.dirname(self.output_csv), "debug_images")
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"debug_{img_filename}")
        cv2.imwrite(debug_path, debug_image)
        
        # Store the results
        self.results.append({
            "filename": img_filename,
            "pose_type": pose_type,
            "pose_variant": pose_variant,
            "pose_side": pose_side,
            "angle": angle
        })
        
        print(f"Processed {img_filename}: {pose_type} ({pose_variant}, {pose_side}), angle: {angle}")

    
    def extract_pose_features(self, landmarks):
        """
        Extract features from pose landmarks for classification.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Dictionary of extracted features
        """
        # Calculate midpoints
        mid_shoulder = self.get_midpoint(landmarks[11], landmarks[12])
        mid_hip = self.get_midpoint(landmarks[23], landmarks[24])
        
        # Calculate general alignment features
        shoulder_hip_alignment = abs(mid_shoulder.x - mid_hip.x)
        hip_height_ratio = mid_hip.y
        
        # Calculate knee-ankle distances
        left_knee_ankle_dist = self.calculate_distance(landmarks[25], landmarks[27])
        right_knee_ankle_dist = self.calculate_distance(landmarks[26], landmarks[28])
        knee_ankle_distance = (left_knee_ankle_dist + right_knee_ankle_dist) / 2
        
        # Calculate angles for left and right sides
        left_hip_knee_ankle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        right_hip_knee_ankle = self.calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        left_shoulder_hip_knee = self.calculate_angle(landmarks[11], landmarks[23], landmarks[25])
        right_shoulder_hip_knee = self.calculate_angle(landmarks[12], landmarks[24], landmarks[26])
        
        # Calculate hip openness (angle between legs, measured at hips)
        hip_openness = self.calculate_angle(landmarks[27], mid_hip, landmarks[28])
        
        # Calculate leg alignment for splits
        leg_alignment = abs(landmarks[25].y - landmarks[26].y)
        
        # Calculate back extension angle for cobra
        back_extension = self.calculate_angle(landmarks[0], mid_shoulder, mid_hip)
        
        # Store Y positions for classification
        left_knee_y = landmarks[25].y
        right_knee_y = landmarks[26].y
        left_ankle_y = landmarks[27].y
        right_ankle_y = landmarks[28].y
        
        # Return all features as a dictionary
        return {
            "shoulder_hip_alignment": shoulder_hip_alignment,
            "hip_height_ratio": hip_height_ratio,
            "knee_ankle_distance": knee_ankle_distance,
            "left_hip_knee_ankle": left_hip_knee_ankle,
            "right_hip_knee_ankle": right_hip_knee_ankle,
            "left_shoulder_hip_knee": left_shoulder_hip_knee,
            "right_shoulder_hip_knee": right_shoulder_hip_knee,
            "hip_openness": hip_openness,
            "leg_alignment": leg_alignment,
            "back_extension": back_extension,
            "left_knee_y": left_knee_y,
            "right_knee_y": right_knee_y,
            "left_ankle_y": left_ankle_y,
            "right_ankle_y": right_ankle_y
        }
    
    def classify_pose_by_features(self, features, landmarks):
        """
        Classify the pose based on extracted features if the primary method fails.
        
        Args:
            features: Dictionary of extracted pose features
            landmarks: MediaPipe pose landmarks
            
        Returns:
            (pose_type, pose_side, angle) tuple
        """
        # Get midpoints that might be needed
        mid_hip = self.get_midpoint(landmarks[23], landmarks[24])
        
        # Check for side splits first (usually has the largest hip openness)
        if features["hip_openness"] > 120 and features["leg_alignment"] < 0.1:
            return "side_splits", "none", features["hip_openness"]
        
        # Check for front splits
        if features["hip_openness"] > 120 and features["leg_alignment"] > 0.2:
            if landmarks[27].y > landmarks[28].y:
                return "front_splits_left", "left", features["hip_openness"]
            else:
                return "front_splits_right", "right", features["hip_openness"]
        
        # Check for cobra pose
        if 120 < features["back_extension"] < 170 and mid_hip.y > 0.6:
            return "cobra", "none", features["back_extension"]
        
        # Check for lunge poses
        if 70 < features["left_hip_knee_ankle"] < 110 and features["left_hip_knee_ankle"] < features["right_hip_knee_ankle"]:
            return "lunge_left", "left", features["left_hip_knee_ankle"]
        
        if 70 < features["right_hip_knee_ankle"] < 110 and features["right_hip_knee_ankle"] < features["left_hip_knee_ankle"]:
            return "lunge_right", "right", features["right_hip_knee_ankle"]
        
        # Check for hip opening poses
        if 40 < features["hip_openness"] < 100:
            if landmarks[27].y > landmarks[28].y:
                return "hip_opening_left", "left", features["hip_openness"]
            else:
                return "hip_opening_right", "right", features["hip_openness"]
        
        # Default if no pose can be confidently classified
        return "unknown", "unknown", None
    
    def determine_variant(self, landmarks, pose_type: str, features=None) -> str:
        """
        Determine if the pose is active or passive.
        
        Args:
            landmarks: The pose landmarks
            pose_type: The identified pose type
            features: Pre-extracted features (optional)
            
        Returns:
            String indicating "active" or "passive"
        """
        if features is None and pose_type != "unknown":
            # If features weren't provided, extract them now
            features = self.extract_pose_features(landmarks)
        
        # Define pose-specific criteria for active vs passive
        if pose_type == "lunge_right" or pose_type == "lunge_left":
            # In active lunges, the back is more upright
            if pose_type == "lunge_right":
                back_angle = self.calculate_angle(landmarks[12], landmarks[24], landmarks[26])
            else:
                back_angle = self.calculate_angle(landmarks[11], landmarks[23], landmarks[25])
            
            # More upright back indicates active pose
            if back_angle > 150:
                return "active"
            else:
                return "passive"
                
        elif pose_type == "cobra":
            # In active cobra, arms are more extended and back is more arched
            left_elbow_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
            right_elbow_angle = self.calculate_angle(landmarks[12], landmarks[14], landmarks[16])
            
            # Straighter arms in active cobra
            if (left_elbow_angle + right_elbow_angle) / 2 > 150:
                return "active"
            else:
                return "passive"
                
        elif "hip_opening" in pose_type:
            # In active hip opening, the back is more upright
            mid_shoulder = self.get_midpoint(landmarks[11], landmarks[12])
            mid_hip = self.get_midpoint(landmarks[23], landmarks[24])
            
            # Calculate torso angle relative to vertical
            vertical_angle = abs(90 - self.calculate_vertical_angle(mid_shoulder, mid_hip))
            
            # More upright torso in active variant
            if vertical_angle < 30:
                return "active"
            else:
                return "passive"
                
        elif "splits" in pose_type:
            # In active splits, the back is more upright
            mid_shoulder = self.get_midpoint(landmarks[11], landmarks[12])
            mid_hip = self.get_midpoint(landmarks[23], landmarks[24])
            
            # Calculate torso angle relative to vertical
            vertical_angle = abs(90 - self.calculate_vertical_angle(mid_shoulder, mid_hip))
            
            # More upright torso in active variant
            if vertical_angle < 30:
                return "active"
            else:
                return "passive"
        
        # Default if we can't determine
        return "unknown"
    
    def calculate_vertical_angle(self, p1, p2):
        """Calculate the angle between a line and the vertical axis."""
        dy = p2.y - p1.y
        dx = p2.x - p1.x
        angle = np.degrees(np.arctan2(dy, dx)) + 90
        return angle
    
    def calculate_angle(self, p1, p2, p3) -> float:
        """
        Calculate angle between three points.
        
        Args:
            p1, p2, p3: Three points where p2 is the vertex
            
        Returns:
            Angle in degrees
        """
        # Convert to numpy arrays
        a = np.array([p1.x, p1.y])
        b = np.array([p2.x, p2.y])
        c = np.array([p3.x, p3.y])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate cosine of the angle
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine = np.clip(cosine, -1.0, 1.0)  # Ensure it's within valid range
        
        # Calculate the angle in degrees
        angle = np.arccos(cosine) * 180.0 / np.pi
        
        return angle
    
    def calculate_distance(self, p1, p2) -> float:
        """Calculate the Euclidean distance between two landmarks."""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def get_midpoint(self, p1, p2):
        """Calculate the midpoint between two landmarks."""
        return type('obj', (object,), {
            'x': (p1.x + p2.x) / 2,
            'y': (p1.y + p2.y) / 2,
            'z': (p1.z + p2.z) / 2
        })
    
    # Individual pose detection methods
    def is_lunge_right(self, landmarks, img_shape) -> Tuple[bool, str, Optional[float]]:
        """Check if the pose is a right-side lunge."""
        # Extract relevant landmarks
        right_ankle = landmarks[28]
        right_knee = landmarks[26]
        right_hip = landmarks[24]
        left_ankle = landmarks[27]
        
        # Create derived point for outer knee angle
        derived_point = type('obj', (object,), {
            'x': right_knee.x + 0.1,  # Add 10% of image width
            'y': right_knee.y,       # Same y value
            'z': right_knee.z
        })
        
        # Calculate angle
        angle = self.calculate_angle(right_ankle, right_knee, derived_point)
        
        # Define criteria for right lunge
        # 1. Angle should be in range for lunges
        angle_in_range = 50 < angle < 130
        
        # 2. Right ankle should be roughly under the right knee
        ankle_under_knee = abs(right_ankle.x - right_knee.x) < 0.20
        
        # 3. Left ankle should be behind right ankle
        left_ankle_behind = left_ankle.y > right_ankle.y
        
        # 4. Right hip should be above right knee
        hip_above_knee = right_hip.y < right_knee.y
        
        # Combined criteria
        is_lunge = angle_in_range and ankle_under_knee and left_ankle_behind and hip_above_knee
        
        return is_lunge, "right", angle if is_lunge else None

    def is_lunge_left(self, landmarks, img_shape) -> Tuple[bool, str, Optional[float]]:
        """Check if the pose is a left-side lunge."""
        # Extract relevant landmarks
        left_ankle = landmarks[27]
        left_knee = landmarks[25]
        left_hip = landmarks[23]
        right_ankle = landmarks[28]
        
        # Create derived point for outer knee angle
        derived_point = type('obj', (object,), {
            'x': left_knee.x - 0.1,  # Subtract 10% of image width
            'y': left_knee.y,       # Same y value
            'z': left_knee.z
        })
        
        # Calculate angle
        angle = self.calculate_angle(left_ankle, left_knee, derived_point)
        
        # Define criteria for left lunge
        # 1. Angle should be in range for lunges
        angle_in_range = 50 < angle < 130
        
        # 2. Left ankle should be roughly under the left knee
        ankle_under_knee = abs(left_ankle.x - left_knee.x) < 0.20
        
        # 3. Right ankle should be behind left ankle
        right_ankle_behind = right_ankle.y > left_ankle.y
        
        # 4. Left hip should be above left knee
        hip_above_knee = left_hip.y < left_knee.y
        
        # Combined criteria
        is_lunge = angle_in_range and ankle_under_knee and right_ankle_behind and hip_above_knee
        
        return is_lunge, "left", angle if is_lunge else None

    def is_hip_opening_right(self, landmarks, img_shape) -> Tuple[bool, str, Optional[float]]:
        """Check if the pose is right-side hip opening with height-based criteria."""
        # Extract landmarks
        right_ankle = landmarks[28]
        hip_right = landmarks[24]
        left_ankle = landmarks[27]
        right_wrist = landmarks[16]
        nose = landmarks[0]  # For head position
        
        # Calculate angle
        angle = self.calculate_angle(right_ankle, hip_right, left_ankle)
        
        # Key distinction: In hip opening, person is standing, head in upper third
        head_in_upper_third = nose.y < 0.33
        knee_in_upper_two_thirds = landmarks[26].y < 0.66
        
        # Hand near foot for passive version
        hand_near_foot = abs(right_wrist.x - right_ankle.x) < 0.15
        
        # Angle criteria
        angle_in_range = 30 < angle < 130
        
        # Combined criteria with emphasis on vertical positioning
        is_hip_opening = angle_in_range and head_in_upper_third and knee_in_upper_two_thirds
        
        return is_hip_opening, "right", angle if is_hip_opening else None

    def is_hip_opening_left(self, landmarks, img_shape) -> Tuple[bool, str, Optional[float]]:
        """Check if the pose is left-side hip opening with height-based criteria."""
        # Extract landmarks
        left_ankle = landmarks[27]
        hip_left = landmarks[23]
        right_ankle = landmarks[28]
        left_wrist = landmarks[15]
        nose = landmarks[0]  # For head position
        
        # Calculate angle
        angle = self.calculate_angle(left_ankle, hip_left, right_ankle)
        
        # Key distinction: In hip opening, person is standing, head in upper third
        head_in_upper_third = nose.y < 0.33
        knee_in_upper_two_thirds = landmarks[25].y < 0.66
        
        # Hand near foot for passive version
        hand_near_foot = abs(left_wrist.x - left_ankle.x) < 0.15
        
        # Angle criteria
        angle_in_range = 30 < angle < 130
        
        # Combined criteria with emphasis on vertical positioning
        is_hip_opening = angle_in_range and head_in_upper_third and knee_in_upper_two_thirds
        
        return is_hip_opening, "left", angle if is_hip_opening else None

    def is_side_splits(self, landmarks, img_shape) -> Tuple[bool, str, Optional[float]]:
        """Check if the pose is side splits."""
        # Extract relevant landmarks
        right_ankle = landmarks[28]
        left_ankle = landmarks[27]
        mid_hip = self.get_midpoint(landmarks[23], landmarks[24])
        
        # NO vertical displacement for derived point
        
        # Calculate angle
        angle = self.calculate_angle(right_ankle, mid_hip, left_ankle)
        
        # Define criteria for side splits
        # 1. Angle between legs should be wide
        angle_wide_enough = angle > 140
        
        # 2. Legs should be to the sides (symmetric positions)
        leg_symmetry = abs(abs(left_ankle.x - mid_hip.x) - abs(right_ankle.x - mid_hip.x)) < 0.15
        
        # 3. Legs should be at similar height (y-value)
        legs_at_similar_height = abs(left_ankle.y - right_ankle.y) < 0.15
        
        # 4. Ankles should be wider apart than knees (side splits characteristic)
        ankles_wide_apart = abs(right_ankle.x - left_ankle.x) > abs(landmarks[26].x - landmarks[25].x)
        
        # Combined criteria
        is_side_splits = angle_wide_enough and leg_symmetry and legs_at_similar_height and ankles_wide_apart
        
        return is_side_splits, "none", angle if is_side_splits else None

    def is_front_splits_right(self, landmarks, img_shape) -> Tuple[bool, str, Optional[float]]:
        """Check if the pose is right-side front splits."""
        # Extract relevant landmarks
        right_ankle = landmarks[28]
        mid_hip = self.get_midpoint(landmarks[23], landmarks[24])
        left_ankle = landmarks[27]
        
        # Calculate angle
        angle = self.calculate_angle(right_ankle, mid_hip, left_ankle)
        
        # Define criteria for right-side front splits
        # 1. Angle between legs should be wide
        angle_wide_enough = angle > 120
        
        # 2. Right leg should be forward, left leg back
        right_leg_forward = right_ankle.y < left_ankle.y
        
        # 3. Clear front-back distinction (significant difference in y-values)
        clear_front_back = abs(right_ankle.y - left_ankle.y) > 0.2
        
        # 4. Angle must be wide enough to differentiate from lunge
        not_a_lunge = angle > 130
        
        # Combined criteria - more emphasis on front-back positioning
        is_front_splits = angle_wide_enough and right_leg_forward and clear_front_back and not_a_lunge
        
        return is_front_splits, "right", angle if is_front_splits else None

    def is_front_splits_left(self, landmarks, img_shape) -> Tuple[bool, str, Optional[float]]:
        """Check if the pose is left-side front splits."""
        # Extract relevant landmarks
        left_ankle = landmarks[27]
        mid_hip = self.get_midpoint(landmarks[23], landmarks[24])
        right_ankle = landmarks[28]
        
        # Calculate angle
        angle = self.calculate_angle(left_ankle, mid_hip, right_ankle)
        
        # Define criteria for left-side front splits
        # 1. Angle between legs should be wide
        angle_wide_enough = angle > 120
        
        # 2. Left leg should be forward, right leg back
        left_leg_forward = left_ankle.y < right_ankle.y
        
        # 3. Clear front-back distinction (significant difference in y-values)
        clear_front_back = abs(left_ankle.y - right_ankle.y) > 0.2
        
        # 4. Angle must be wide enough to differentiate from lunge
        not_a_lunge = angle > 130
        
        # Combined criteria - more emphasis on front-back positioning
        is_front_splits = angle_wide_enough and left_leg_forward and clear_front_back and not_a_lunge
        
        return is_front_splits, "left", angle if is_front_splits else None

    def is_cobra(self, landmarks, img_shape) -> Tuple[bool, str, Optional[float]]:
        """Check if the pose is cobra with emphasis on feet positioning."""
        # Extract landmarks
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        left_shoulder = landmarks[11]  # This is the correct landmark we want to use
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        nose = landmarks[0]
        
        # Create derived points with MINIMAL vertical displacement
        derived_hip = type('obj', (object,), {
            'x': left_hip.x,
            'y': left_hip.y + 0.002,  # Even less displacement (half of 0.01)
            'z': left_hip.z
        })
        
        # Keep derived_point exactly level with derived_hip
        derived_point = type('obj', (object,), {
            'x': derived_hip.x - 0.15,
            'y': derived_hip.y,  # EXACT same y as derived_hip
            'z': derived_hip.z
        })
        
        # Use left_shoulder directly to avoid any confusion with variable names
        angle = self.calculate_angle(left_shoulder, derived_hip, derived_point)
        
        # Keep the rest of the identification logic that's working
        feet_close_together = abs(left_ankle.x - right_ankle.x) < 0.05
        profile_view = 0.2 < nose.x < 0.8 and abs(nose.x - 0.5) > 0.1
        upper_third_empty = nose.y > 0.33
        lying_flat = abs(nose.y - (left_ankle.y + right_ankle.y)/2) < 0.2
        shoulders_elevated = (left_shoulder.y + right_shoulder.y)/2 < (left_hip.y + right_hip.y)/2
        
        is_cobra = feet_close_together and profile_view and upper_third_empty and lying_flat and shoulders_elevated
        
        return is_cobra, "none", angle if is_cobra else None

    def is_side_splits_lying(self, landmarks, img_shape) -> Tuple[bool, str, Optional[float]]:
        """
        Special case for side splits lying down with focus on visible landmarks.
        """
        # Compute the average y for the hips
        hip_mean = (landmarks[23].y + landmarks[24].y) / 2

        # Check leg landmarks: if they are not sufficiently lower than the hips,
        # we assume they're misdetected and clustered on the torso.
        legs_on_torso = True
        for idx in [25, 26, 27, 28]:
            # For a proper detection, legs should be at least 0.1 lower than the hip_mean.
            if landmarks[idx].y > hip_mean + 0.1:
                legs_on_torso = False
                break

        # If legs are misdetected (i.e., remain too close to the hips)
        if legs_on_torso:
            # Additionally, verify that the hips are roughly aligned:
            hips_aligned = abs(landmarks[23].y - landmarks[24].y) < 0.1
            # Use an approximate angle for side splits lying.
            angle = 160.0
            return hips_aligned, "none", angle

        # If legs are detected correctly, we don't consider it side splits lying.
        return False, "unknown", None
        
    def save_results(self):
        """Save the analysis results to a CSV file."""
        with open(self.output_csv, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'pose_type', 'pose_variant', 'pose_side', 'angle']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
            writer.writeheader()
            for result in self.results:
                writer.writerow(result)
            
        print(f"Results saved to {self.output_csv}")
        
    def visualize_pose(self, image, landmarks, pose_type, angle):
        """
        Visualize the pose landmarks and calculated angle.
        This is useful for debugging and verification.
            
        Args:
            image: The input image
            landmarks: The detected pose landmarks
            pose_type: The identified pose type
            angle: The calculated angle
                
        Returns:
            Image with visualization overlays
        """
        # Create a copy of the image to draw on
        vis_image = image.copy()
        h, w, _ = vis_image.shape
            
        # Draw all landmarks detected by MediaPipe
        if landmarks:
            # Convert landmarks to pixel coordinates for visualization
            landmark_points = []
            for i, landmark in enumerate(landmarks):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                # Draw landmark point
                cv2.circle(vis_image, (x, y), 3, (0, 255, 0), -1)
                # Add landmark number
                cv2.putText(vis_image, str(i), (x + 5, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                landmark_points.append((x, y))
                
            # Draw connections between landmarks (simplified)
            connections = [
                # Torso
                (11, 12), (11, 23), (12, 24), (23, 24),
                # Arms
                (11, 13), (13, 15), (12, 14), (14, 16),
                # Legs
                (23, 25), (25, 27), (24, 26), (26, 28)
            ]
                
            for connection in connections:
                if (0 <= connection[0] < len(landmark_points) and 
                    0 <= connection[1] < len(landmark_points)):
                    cv2.line(vis_image, landmark_points[connection[0]], 
                            landmark_points[connection[1]], (0, 255, 255), 2)
            
        # If a pose was detected, draw the angle and pose name
        if pose_type != "unknown" and angle is not None:
            # Get landmarks used for angle calculation based on pose type
            if pose_type in self.pose_landmarks:
                landmark_indices = self.pose_landmarks[pose_type][0]
                    
                # Create derived points for visualization
                derived_points = {}
                    
                # For lunge right - derived point to the right of the knee
                if pose_type == "lunge_right":
                    right_knee = landmarks[26]
                    derived_points["derived_right"] = (
                        int((right_knee.x + 0.1) * w), 
                        int(right_knee.y * h)
                    )
                    
                # For lunge left - derived point to the left of the knee
                elif pose_type == "lunge_left":
                    left_knee = landmarks[25]
                    derived_points["derived_left"] = (
                        int((left_knee.x - 0.1) * w), 
                        int(left_knee.y * h)
                    )
                    
                # For cobra - derived points
                elif pose_type == "cobra":
                    left_hip = landmarks[23]
                    derived_points["derived_hip"] = (
                        int(left_hip.x * w), 
                        int((left_hip.y + 0.025) * h)  # Use the value that works (0.025)
                    )
                    derived_points["derived_point"] = (
                        int((left_hip.x - 0.15) * w), 
                        int((left_hip.y + 0.025) * h)  # Same height as derived_hip
                    )
                
                # For poses that need mid_hip
                if "mid_hip" not in derived_points:
                    mid_hip = (
                        int(((landmarks[23].x + landmarks[24].x) / 2) * w),
                        int(((landmarks[23].y + landmarks[24].y) / 2) * h)
                    )
                    derived_points["mid_hip"] = mid_hip
                
                # For poses that need derived_hip but don't have it set yet
                if "derived_hip" not in derived_points and pose_type != "cobra":
                    derived_points["derived_hip"] = (
                        derived_points["mid_hip"][0],
                        int(derived_points["mid_hip"][1] + 0.025 * h)  # Reduced from 0.1 to 0.025
                    )
                    
                # Extract points for angle visualization
                points = []
                for idx in landmark_indices:
                    if isinstance(idx, int) and idx < len(landmarks):
                        point = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
                        points.append(point)
                    elif idx in derived_points:
                        points.append(derived_points[idx])
                    
                # Draw angle lines and vertex
                if len(points) == 3:
                    # Draw lines forming the angle
                    cv2.line(vis_image, points[0], points[1], (0, 255, 0), 2)
                    cv2.line(vis_image, points[1], points[2], (0, 255, 0), 2)
                        
                    # Mark the vertex
                    cv2.circle(vis_image, points[1], 5, (0, 0, 255), -1)
                        
                    # Add angle text
                    text_pos = (points[1][0] + 10, points[1][1] - 10)
                    cv2.putText(vis_image, f"{angle:.1f}°", text_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
            # Add pose type at the top of the image
            cv2.putText(vis_image, f"Pose: {pose_type}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
        return vis_image
    def analyze_image_debug(self, img_path: str, img_filename: str):
        """
        Analyze an image with detailed debugging output.
        This helps understand why poses are being classified or rejected.
        
        Args:
            img_path: Path to the image file
            img_filename: Filename of the image (for recording purposes)
        """
        # Read the image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error reading image: {img_path}")
            return
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        # Process the image with MediaPipe
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            print(f"No pose landmarks found in: {img_path}")
            return
        
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        
        print(f"\n----- DEBUG: {img_filename} -----")
        
        # Print some key landmark positions and visibilities
        print("Key landmarks (x, y, visibility):")
        print(f"Nose (0): ({landmarks[0].x:.2f}, {landmarks[0].y:.2f}, {landmarks[0].visibility:.2f})")
        print(f"L Shoulder (11): ({landmarks[11].x:.2f}, {landmarks[11].y:.2f}, {landmarks[11].visibility:.2f})")
        print(f"R Shoulder (12): ({landmarks[12].x:.2f}, {landmarks[12].y:.2f}, {landmarks[12].visibility:.2f})")
        print(f"L Hip (23): ({landmarks[23].x:.2f}, {landmarks[23].y:.2f}, {landmarks[23].visibility:.2f})")
        print(f"R Hip (24): ({landmarks[24].x:.2f}, {landmarks[24].y:.2f}, {landmarks[24].visibility:.2f})")
        print(f"L Knee (25): ({landmarks[25].x:.2f}, {landmarks[25].y:.2f}, {landmarks[25].visibility:.2f})")
        print(f"R Knee (26): ({landmarks[26].x:.2f}, {landmarks[26].y:.2f}, {landmarks[26].visibility:.2f})")
        print(f"L Ankle (27): ({landmarks[27].x:.2f}, {landmarks[27].y:.2f}, {landmarks[27].visibility:.2f})")
        print(f"R Ankle (28): ({landmarks[28].x:.2f}, {landmarks[28].y:.2f}, {landmarks[28].visibility:.2f})")
        
        # Print some relationship metrics
        ankles_apart = abs(landmarks[27].x - landmarks[28].x)
        head_height = landmarks[0].y
        feet_level_diff = abs(landmarks[27].y - landmarks[28].y)
        
        print("\nRelationship metrics:")
        print(f"Ankles apart (x-distance): {ankles_apart:.2f}")
        print(f"Head height (y-position, 0=top): {head_height:.2f}")
        print(f"Feet level difference: {feet_level_diff:.2f}")
        
        # Try each pose detection and print detailed results
        print("\nPose detection results:")
        detected_pose = None
        
        for pose_name, pose_checker in self.poses.items():
            is_pose, side, angle = pose_checker(landmarks, image.shape)
            
            if is_pose:
                print(f"✓ {pose_name} detected! Side: {side}, Angle: {angle:.2f}°")
                detected_pose = (pose_name, side, angle)
            else:
                # More detailed rejection reasons for specific poses
                if pose_name == "lunge_right":
                    angle = self.calculate_angle(landmarks[28], landmarks[26], type('obj', (object,), {'x': landmarks[26].x + 0.1, 'y': landmarks[26].y, 'z': landmarks[26].z}))
                    back_foot_higher = landmarks[27].y < landmarks[28].y
                    not_too_high = landmarks[27].y > 0.5
                    head_upper = landmarks[0].y < 0.5
                    print(f"✗ {pose_name}: angle={angle:.2f}° (needs 70-120), " +
                        f"back foot higher={back_foot_higher}, not too high={not_too_high}, " +
                        f"head upper={head_upper}")
                
                elif pose_name == "lunge_left":
                    angle = self.calculate_angle(landmarks[27], landmarks[25], type('obj', (object,), {'x': landmarks[25].x - 0.1, 'y': landmarks[25].y, 'z': landmarks[25].z}))
                    back_foot_higher = landmarks[28].y < landmarks[27].y
                    not_too_high = landmarks[28].y > 0.5
                    head_upper = landmarks[0].y < 0.5
                    print(f"✗ {pose_name}: angle={angle:.2f}° (needs 70-120), " +
                        f"back foot higher={back_foot_higher}, not too high={not_too_high}, " +
                        f"head upper={head_upper}")
                
                elif pose_name == "cobra":
                    angle = self.calculate_angle(landmarks[11], 
                                        type('obj', (object,), {'x': landmarks[23].x, 'y': landmarks[23].y + 0.002, 'z': landmarks[23].z}), 
                                        type('obj', (object,), {'x': landmarks[23].x - 0.2, 'y': landmarks[23].y + 0.002, 'z': landmarks[23].z}))
                    feet_close = abs(landmarks[27].x - landmarks[28].x) < 0.08
                    upper_body_elevated = (landmarks[11].y + landmarks[12].y)/2 < (landmarks[23].y + landmarks[24].y)/2
                    print(f"✗ {pose_name}: angle={angle:.2f}° (needs 100-180), " +
                        f"feet close together={feet_close}, upper body elevated={upper_body_elevated}")
                
                elif pose_name == "side_splits":
                    mid_hip = self.get_midpoint(landmarks[23], landmarks[24])
                    derived_point = type('obj', (object,), {'x': mid_hip.x, 'y': mid_hip.y + 0.05, 'z': mid_hip.z})
                    angle = self.calculate_angle(landmarks[28], derived_point, landmarks[27])
                    head_lower = landmarks[0].y > 0.5
                    leg_symmetry = abs(abs(landmarks[27].x - mid_hip.x) - abs(landmarks[28].x - mid_hip.x)) < 0.15
                    print(f"✗ {pose_name}: angle={angle:.2f}° (needs >130), " +
                        f"head lower={head_lower}, leg symmetry={leg_symmetry}")
                
                elif pose_name == "hip_opening_right" or pose_name == "hip_opening_left":
                    if pose_name == "hip_opening_right":
                        angle = self.calculate_angle(landmarks[28], landmarks[24], landmarks[27])
                    else:
                        angle = self.calculate_angle(landmarks[27], landmarks[23], landmarks[28])
                    head_upper = landmarks[0].y < 0.5
                    ankles_below = landmarks[27].y > landmarks[23].y and landmarks[28].y > landmarks[24].y
                    print(f"✗ {pose_name}: angle={angle:.2f}° (needs 30-130), " +
                        f"head upper={head_upper}, ankles below hips={ankles_below}")
                
                else:
                    print(f"✗ {pose_name} not detected")
        
        # Determine active/passive if pose detected
        if detected_pose:
            pose_type, pose_side, angle = detected_pose
            pose_variant = self.determine_variant(landmarks, pose_type)
            print(f"\nFinal classification: {pose_type}, {pose_variant}, {pose_side}, angle={angle:.2f}°")
        else:
            print("\nNo pose detected!")
        
        # Create and save debug image
        debug_image = self.visualize_pose(image, landmarks, 
                                        detected_pose[0] if detected_pose else "unknown", 
                                        detected_pose[2] if detected_pose else None)
        
        debug_dir = os.path.join(os.path.dirname(self.output_csv), "debug_images")
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"debug_{img_filename}")
        cv2.imwrite(debug_path, debug_image)
        
        # Store the results
        self.results.append({
            "filename": img_filename,
            "pose_type": detected_pose[0] if detected_pose else "unknown",
            "pose_variant": pose_variant if detected_pose else "unknown",
            "pose_side": detected_pose[1] if detected_pose else "unknown",
            "angle": detected_pose[2] if detected_pose else None
        })
        
        print(f"Debug image saved to: {debug_path}")
        print("-" * 40)

def main():
    # Example usage
    analyzer = YogaPoseAnalyzer(
        image_folder="path/to/yoga/images",
        output_csv="yoga_pose_analysis_results.csv"
    )
    analyzer.process_images()


if __name__ == "__main__":
    main()