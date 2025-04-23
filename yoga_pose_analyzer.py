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
import argparse

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
            "side_splits": self.is_side_splits,
            "cobra": self.is_cobra,
            "hip_opening_right": self.is_hip_opening_right,
            "hip_opening_left": self.is_hip_opening_left,
            "front_splits_right": self.is_front_splits_right,
            "front_splits_left": self.is_front_splits_left,
            "front_splits_passive_right": self.is_front_splits_passive_right,
            "front_splits_passive_left": self.is_front_splits_passive_left,
            "lunge": self.is_lunge,
        }

        self.pose_landmarks = {
            "lunge_right": [(28, 26, "derived_right")],
            "lunge_left": [(27, 25, "derived_left")],
            "cobra": [(11, "derived_hip", "derived_point")],
            "hip_opening_right": [(28, "mid_hip", 27)],
            "hip_opening_left": [(27, "mid_hip", 28)],
            "side_splits": [(28, "mid_hip", 27)],
            "front_splits_right": [(28, "mid_hip", 27)],
            "front_splits_left": [(27, "mid_hip", 28)],
            "front_splits_passive_right": [(28, "mid_hip", 27)],
            "front_splits_passive_left": [(27, "mid_hip", 28)]
        }
        
        # Results storage
        self.results = []
        
        # Visualization settings
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def process_images(self):
        """Process all images in the folder and analyze the yoga poses."""
        # Recursively traverse all subdirectories for images
        for root, _, files in os.walk(self.image_folder):
            for img_file in files:
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, img_file)
                    self.analyze_image(img_path, img_file)
        # Save results to CSV
        self.save_results()

    def process_images_debug(self):
        """Process all images in debug mode (detailed per-image debug)."""
        # Recursively traverse all subdirectories for images in debug mode
        for root, _, files in os.walk(self.image_folder):
            for img_file in files:
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, img_file)
                    self.analyze_image_debug(img_path, img_file)
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
            # Ensure entry exists for summary
            self.results.append({
                "filename": img_filename,
                "pose_type": "unknown",
                "pose_variant": "unknown",
                "pose_side": "unknown",
                "angle": None
            })
            return
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        # Process the image with MediaPipe
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            print(f"No pose landmarks found in: {img_path}")
            # Ensure entry exists for summary
            self.results.append({
                "filename": img_filename,
                "pose_type": "unknown",
                "pose_variant": "unknown",
                "pose_side": "unknown",
                "angle": None
            })
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
        
        # Store the results with detailed columns
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
        # Handle generic lunge label
        if pose_type == "lunge":
            # Compute the back-leg angle for generic lunge
            _, _, angle_val = self.poses["lunge"](landmarks, (0, 0, 0))
            if angle_val is None:
                return "unknown"
            # Active if smaller angle
            return "active" if angle_val < 90 else "passive"
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
    def is_lunge(self, landmarks, img_shape) -> Tuple[bool, str, Optional[float]]:
        """
        Determine if the stance is a lunge, based on back-leg knee angle and stance width.
        """
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        # Decide which leg is back by its ankle being higher in the image (smaller y)
        if left_ankle.y < right_ankle.y:
            back_knee = landmarks[25]
            back_ankle = landmarks[27]
            derived_point = type('obj', (object,), {
                'x': back_knee.x - 0.1,
                'y': back_knee.y,
                'z': back_knee.z
            })
            side = "right"
        else:
            back_knee = landmarks[26]
            back_ankle = landmarks[28]
            derived_point = type('obj', (object,), {
                'x': back_knee.x + 0.1,
                'y': back_knee.y,
                'z': back_knee.z
            })
            side = "left"

        # Compute the knee angle at the back leg
        angle = self.calculate_angle(back_ankle, back_knee, derived_point)

        # Criteria: angle in lunge range, knee aligned, and ankles not too wide (to exclude splits)
        angle_in_range = 50 < angle < 120
        knee_under_ankle = abs(back_knee.x - back_ankle.x) < 0.5
        ankles_not_wide = abs(left_ankle.x - right_ankle.x) < 0.4

        is_lunge = angle_in_range and knee_under_ankle and ankles_not_wide
        return is_lunge, side, angle if is_lunge else (False, side, None)


    def is_hip_opening_right(self, landmarks, img_shape) -> Tuple[bool, str, Optional[float]]:
        """Check if the pose is right-side hip opening."""
        # Extract relevant landmarks
        right_ankle = landmarks[28]
        mid_hip = self.get_midpoint(landmarks[23], landmarks[24])
        left_ankle = landmarks[27]
        right_wrist = landmarks[16]
        left_wrist = landmarks[15]
        nose = landmarks[0]
        
        # Calculate angle
        angle = self.calculate_angle(right_ankle, mid_hip, left_ankle)
        
        # Basic criteria
        angle_in_range = 30 < angle < 170
        # Require head in upper third of the frame for hip opening
        head_in_upper_third = nose.y < (1.0 / 3.0)
        
        # Look at the position of the entire body
        # If the whole body is shifted to the right, it's likely right hip opening
        body_shift_right = nose.x > 0.5
        
        # Check which foot is more raised/extended
        right_foot_extension = abs(right_ankle.x - mid_hip.x)
        left_foot_extension = abs(left_ankle.x - mid_hip.x)
        # Prevent extremely wide stances (e.g. side splits) from being classified as hip opening
        ankles_apart = abs(left_ankle.x - right_ankle.x)
        ankles_not_too_wide = ankles_apart < 0.3
        right_foot_more_extended = right_foot_extension > left_foot_extension
        
        # Check hand-to-foot proximity
        right_hand_near_foot = abs(right_wrist.x - right_ankle.x) < 0.2
        left_hand_near_foot = abs(left_wrist.x - left_ankle.x) < 0.2
        only_right_hand_near = right_hand_near_foot and not left_hand_near_foot
        
        # Combined indicators of right side
        is_right_side = body_shift_right or right_foot_more_extended or only_right_hand_near
        
        # Final criterion
        is_hip_opening = angle_in_range and head_in_upper_third and is_right_side and ankles_not_too_wide
        
        if is_hip_opening and right_ankle.y < left_ankle.y:
            return True, "right", angle
        return False, "unknown", None

    def is_hip_opening_left(self, landmarks, img_shape) -> Tuple[bool, str, Optional[float]]:
        """Check if the pose is left-side hip opening."""
        # Extract relevant landmarks
        left_ankle = landmarks[27]
        mid_hip = self.get_midpoint(landmarks[23], landmarks[24])
        right_ankle = landmarks[28]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        nose = landmarks[0]
        
        # Calculate angle
        angle = self.calculate_angle(left_ankle, mid_hip, right_ankle)
        
        # Basic criteria
        angle_in_range = 30 < angle < 170
        # Require head in upper third of the frame for hip opening
        head_in_upper_third = nose.y < (1.0 / 3.0)
        
        # Look at the position of the entire body
        # If the whole body is shifted to the left, it's likely left hip opening
        body_shift_left = nose.x < 0.5
        
        # Check which foot is more raised/extended
        left_foot_extension = abs(left_ankle.x - mid_hip.x)
        right_foot_extension = abs(right_ankle.x - mid_hip.x)
        # Prevent extremely wide stances (e.g. side splits) from being classified as hip opening
        ankles_apart = abs(left_ankle.x - right_ankle.x)
        ankles_not_too_wide = ankles_apart < 0.3
        left_foot_more_extended = left_foot_extension > right_foot_extension
        
        # Check hand-to-foot proximity
        left_hand_near_foot = abs(left_wrist.x - left_ankle.x) < 0.2
        right_hand_near_foot = abs(right_wrist.x - right_ankle.x) < 0.2
        only_left_hand_near = left_hand_near_foot and not right_hand_near_foot
        
        # Combined indicators of left side
        is_left_side = body_shift_left or left_foot_more_extended or only_left_hand_near
        
        # Final criterion
        is_hip_opening = angle_in_range and head_in_upper_third and is_left_side and ankles_not_too_wide
        
        if is_hip_opening and left_ankle.y < right_ankle.y:
            return True, "left", angle
        return False, "unknown", None
    
    def is_side_splits(self, landmarks, img_shape) -> Tuple[bool, str, Optional[float]]:
        """Check if the pose is side splits based on actual data patterns."""
        # Extract relevant landmarks
        right_ankle = landmarks[28]
        left_ankle = landmarks[27]
        right_knee = landmarks[26]
        left_knee = landmarks[25]
        hip_left = landmarks[23]
        hip_right = landmarks[24]

        # Calculate midpoint between hips
        mid_hip = self.get_midpoint(hip_left, hip_right)

        # Calculate angle
        angle = self.calculate_angle(right_ankle, mid_hip, left_ankle)

        # Conditionally exclude passive front splits: both ankles below hips, similar height, and wide angle
        ankles_below_hips = (left_ankle.y > mid_hip.y) and (right_ankle.y > mid_hip.y)
        feet_level = abs(left_ankle.y - right_ankle.y) < 0.15
        if ankles_below_hips and feet_level and angle > 110:
            return False, "unknown", None

        # Based on your debug output, side splits have a wide range of angles (lowered threshold)
        angle_acceptable = angle > 30

        # Key characteristics from your data:
        # 1. Ankles are at similar height (y-value)
        ankles_at_similar_height = abs(left_ankle.y - right_ankle.y) < 0.15

        # 2. Ankles are wide apart horizontally
        ankles_wide_apart = abs(left_ankle.x - right_ankle.x) > 0.30  # Increased threshold to avoid lunge misclassification

        # 3. Legs should be spread outward from hips (knees outside of hips)
        legs_spread_outward = (
            left_knee.x > hip_left.x or
            right_knee.x < hip_right.x
        )

        # Combined criteria with less emphasis on angle
        is_side_splits = (
            angle_acceptable and
            ankles_at_similar_height and
            ankles_wide_apart
        )

        return is_side_splits, "none", angle if is_side_splits else None

    def is_front_splits_right(self, landmarks, img_shape) -> Tuple[bool, str, Optional[float]]:
        """Check if the pose is right-side front splits (active variant)."""
        # Extract relevant landmarks
        right_ankle = landmarks[28]
        mid_hip = self.get_midpoint(landmarks[23], landmarks[24])
        left_ankle = landmarks[27]
        nose = landmarks[0]
        
        # Calculate angle
        angle = self.calculate_angle(right_ankle, mid_hip, left_ankle)
        
        # Active front splits criteria
        angle_wide_enough = angle > 90
        right_leg_forward = right_ankle.y < left_ankle.y
        legs_at_different_heights = abs(right_ankle.y - left_ankle.y) > 0.10
        
        # Key distinction: Head in lower third for active front splits
        head_in_lower_third = nose.y > 0.6
        
        # Combined criteria focused on active variant
        is_front_splits = angle_wide_enough and right_leg_forward and legs_at_different_heights and head_in_lower_third
        
        return is_front_splits, "right", angle if is_front_splits else None

    def is_front_splits_left(self, landmarks, img_shape) -> Tuple[bool, str, Optional[float]]:
        """Check if the pose is left-side front splits (active variant)."""
        # Extract relevant landmarks
        left_ankle = landmarks[27]
        mid_hip = self.get_midpoint(landmarks[23], landmarks[24])
        right_ankle = landmarks[28]
        nose = landmarks[0]
        
        # Calculate angle
        angle = self.calculate_angle(left_ankle, mid_hip, right_ankle)
        
        # Active front splits criteria
        angle_wide_enough = angle > 90
        left_leg_forward = left_ankle.y < right_ankle.y
        legs_at_different_heights = abs(left_ankle.y - right_ankle.y) > 0.10
        
        # Key distinction: Head in lower third for active front splits
        head_in_lower_third = nose.y > 0.6
        
        # Combined criteria focused on active variant
        is_front_splits = angle_wide_enough and left_leg_forward and legs_at_different_heights and head_in_lower_third
        
        return is_front_splits, "left", angle if is_front_splits else None
    
    def is_front_splits_passive_right(self, landmarks, img_shape) -> Tuple[bool, str, Optional[float]]:
        """Check if the pose is right-side front splits (passive variant)."""
        # Extract relevant landmarks based on 'passive (right) 2.JPG'
        right_ankle = landmarks[28]
        mid_hip = self.get_midpoint(landmarks[23], landmarks[24])
        left_ankle = landmarks[27]
        
        # Calculate angle
        angle = self.calculate_angle(right_ankle, mid_hip, left_ankle)
        
        # Relaxed angle range: angle > 110
        angle_in_range = angle > 110
        
        # For this specific image:
        # Legs at similar height (legs_level=True)
        legs_at_similar_height = abs(left_ankle.y - right_ankle.y) < 0.25
        
        # Position relative to hips
        ankles_below_hips = (left_ankle.y > mid_hip.y) and (right_ankle.y > mid_hip.y)

        # Exclude narrow stances (e.g., lunges)
        ankles_apart = abs(left_ankle.x - right_ankle.x)
        sufficient_width = ankles_apart > 0.30
        
        # Combined criteria specifically for this image
        is_front_splits_passive = angle_in_range and legs_at_similar_height and ankles_below_hips and sufficient_width
        
        if is_front_splits_passive and right_ankle.y < left_ankle.y:
            return True, "right", angle
        return False, "unknown", None

    def is_front_splits_passive_left(self, landmarks, img_shape) -> Tuple[bool, str, Optional[float]]:
        """Check if the pose is left-side front splits (passive variant)."""
        # Extract relevant landmarks
        left_ankle = landmarks[27]
        mid_hip = self.get_midpoint(landmarks[23], landmarks[24])
        right_ankle = landmarks[28]
        
        # Calculate angle
        angle = self.calculate_angle(left_ankle, mid_hip, right_ankle)
        
        # Relaxed angle range: angle > 110
        angle_in_range = angle > 110
        # Legs should be at similar height
        legs_at_similar_height = abs(left_ankle.y - right_ankle.y) < 0.25
        # Both ankles below hips to ensure lying position
        ankles_below_hips = (left_ankle.y > mid_hip.y) and (right_ankle.y > mid_hip.y)

        # Exclude narrow stances (e.g., lunges)
        ankles_apart = abs(left_ankle.x - right_ankle.x)
        sufficient_width = ankles_apart > 0.30

        # Combined criteria specifically for passive front splits
        is_front_splits_passive = angle_in_range and legs_at_similar_height and ankles_below_hips and sufficient_width

        if is_front_splits_passive and left_ankle.y < right_ankle.y:
            return True, "left", angle
        return False, "unknown", None

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
        
        # Relaxed thresholds for more robust cobra detection
        feet_close_together = abs(left_ankle.x - right_ankle.x) < 0.10
        # Ensure both feet lie on the same horizontal side of the face (nose)
        feet_same_side = (left_ankle.x < nose.x and right_ankle.x < nose.x) or \
                         (left_ankle.x > nose.x and right_ankle.x > nose.x)
        # Ensure ankles are at the same vertical level (feet flat)
        ankles_level = abs(left_ankle.y - right_ankle.y) < 0.07
        profile_view = 0.1 < nose.x < 0.9 and abs(nose.x - 0.5) > 0.05
        upper_third_empty = nose.y > 0.40
        lying_flat = abs(nose.y - (left_ankle.y + right_ankle.y) / 2) < 0.30
        shoulders_elevated = (left_shoulder.y + right_shoulder.y) / 2 < (left_hip.y + right_hip.y) / 2 - 0.05
        
        is_cobra = feet_close_together and feet_same_side and ankles_level and profile_view and upper_third_empty and lying_flat and shoulders_elevated
        
        return is_cobra, "none", angle if is_cobra else None

    # (is_side_splits_lying method removed)
        
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
            # Unified lunge visualization
            if pose_type.startswith("lunge"):
                left_ankle = landmarks[27]
                right_ankle = landmarks[28]
                # Decide back leg by higher ankle (smaller y)
                if left_ankle.y < right_ankle.y:
                    bk_idx, ba_idx = 25, 27
                    derived_x = landmarks[25].x - 0.1
                else:
                    bk_idx, ba_idx = 26, 28
                    derived_x = landmarks[26].x + 0.1
                # Build the three visualization points
                p1 = (int(landmarks[ba_idx].x * w), int(landmarks[ba_idx].y * h))
                p2 = (int(landmarks[bk_idx].x * w), int(landmarks[bk_idx].y * h))
                p3 = (int(derived_x * w), int(landmarks[bk_idx].y * h))
                points = [p1, p2, p3]
                # Draw lunge angle lines and label
                cv2.line(vis_image, p1, p2, (0, 255, 0), 2)
                cv2.line(vis_image, p2, p3, (0, 255, 0), 2)
                cv2.circle(vis_image, p2, 5, (0, 0, 255), -1)
                text_pos = (p2[0] + 10, p2[1] - 10)
                cv2.putText(vis_image, f"{angle:.1f}°", text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            elif pose_type in self.pose_landmarks:
                # Existing pose_landmarks logic
                landmark_indices = self.pose_landmarks[pose_type][0]
                derived_points = {}
                # For cobra - derived points
                if pose_type == "cobra":
                    left_hip = landmarks[23]
                    derived_points["derived_hip"] = (
                        int(left_hip.x * w), 
                        int((left_hip.y + 0.025) * h)
                    )
                    derived_points["derived_point"] = (
                        int((left_hip.x - 0.15) * w), 
                        int((left_hip.y + 0.025) * h)
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
                        int(derived_points["mid_hip"][1] + 0.025 * h)
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
                # Unified lunge debug handling
                if pose_name == "lunge":
                    # Debug: show back-leg points for generic lunge
                    left_ankle = landmarks[27]
                    right_ankle = landmarks[28]
                    if left_ankle.y > right_ankle.y:
                        back_knee, back_ankle = landmarks[25], landmarks[27]
                        derived = type('obj',(object,),{'x':back_knee.x - 0.1,'y':back_knee.y,'z':back_knee.z})
                    else:
                        back_knee, back_ankle = landmarks[26], landmarks[28]
                        derived = type('obj',(object,),{'x':back_knee.x + 0.1,'y':back_knee.y,'z':back_knee.z})
                    print(f"    Points for angle: back_ankle ({back_ankle.x:.2f}, {back_ankle.y:.2f}), back_knee ({back_knee.x:.2f}, {back_knee.y:.2f}), derived ({derived.x:.2f}, {derived.y:.2f})")
                detected_pose = (pose_name, side, angle)
                break
            else:
                # More detailed rejection reasons for specific poses
                if pose_name == "cobra":
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
                    print(f"✗ {pose_name}: angle={angle:.2f}° (needs >70), " +
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
                
                elif pose_name == "lunge":
                    # Debug rejection for generic lunge
                    left_ankle = landmarks[27]
                    right_ankle = landmarks[28]
                    if left_ankle.y > right_ankle.y:
                        # left leg back
                        p1, p2 = landmarks[27], landmarks[25]
                        dir_text = "left-back"
                        derived_pt = type('obj',(object,),{'x':p2.x - 0.1,'y':p2.y,'z':p2.z})
                    else:
                        # right leg back
                        p1, p2 = landmarks[28], landmarks[26]
                        dir_text = "right-back"
                        derived_pt = type('obj',(object,),{'x':p2.x + 0.1,'y':p2.y,'z':p2.z})
                    angle_dbg = self.calculate_angle(p1, p2, derived_pt)
                    back_higher = (p1.y < (landmarks[27].y + landmarks[28].y)/2)
                    print(f"✗ {pose_name}: angle={angle_dbg:.2f}° (needs 60-120), back_dir={dir_text}, back_higher={back_higher}")
                    print(f"    Derived point (x,y): ({derived_pt.x:.2f}, {derived_pt.y:.2f})")
                
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['analyze','debug'], default='analyze',
                        help='Run mode: "analyze" for standard analysis, "debug" for per-image debug output')
    parser.add_argument('--image_folder', default='path/to/yoga/images',
                        help='Folder containing yoga images')
    parser.add_argument('--output_csv', default='yoga_pose_analysis_results.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    analyzer = YogaPoseAnalyzer(
        image_folder=args.image_folder,
        output_csv=args.output_csv
    )
    if args.mode == 'debug':
        analyzer.process_images_debug()
    else:
        analyzer.process_images()


if __name__ == "__main__":
    main()