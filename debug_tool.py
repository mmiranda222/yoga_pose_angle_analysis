import os
import cv2
import mediapipe as mp
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider

class YogaPoseDebugger:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not self.image_files:
            raise ValueError(f"No image files found in {image_folder}")
            
        self.current_idx = 0
        self.landmarks = None
        self.image = None
        self.image_rgb = None
        self.landmark_visibility = [True] * 33  # Show all landmarks initially
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        
        # Setup the figure and axes
        self.setup_plot()
        
        # Process the first image
        self.process_current_image()
        
    def setup_plot(self):
        """Set up the matplotlib figure and controls."""
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.suptitle("Yoga Pose Debugger", fontsize=16)
        
        # Image display area
        self.ax_image = self.fig.add_subplot(121)
        self.ax_image.set_title("Pose Detection")
        
        # Controls and information area
        self.ax_controls = self.fig.add_subplot(122)
        self.ax_controls.set_title("Controls and Information")
        self.ax_controls.axis('off')
        
        # Buttons for navigation
        ax_prev = plt.axes([0.65, 0.05, 0.1, 0.05])
        ax_next = plt.axes([0.8, 0.05, 0.1, 0.05])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)
        
        # Radio buttons for landmark visualization options
        ax_radio = plt.axes([0.65, 0.15, 0.3, 0.3])
        self.radio = RadioButtons(ax_radio, ('All Landmarks', 'Body Only', 'Custom'))
        self.radio.on_clicked(self.update_landmark_display)
        
        # Angle measurement controls
        ax_point1 = plt.axes([0.65, 0.5, 0.3, 0.03])
        ax_point2 = plt.axes([0.65, 0.55, 0.3, 0.03])
        ax_point3 = plt.axes([0.65, 0.6, 0.3, 0.03])
        
        self.slider_p1 = Slider(ax_point1, 'Point 1', 0, 32, valinit=0, valstep=1)
        self.slider_p2 = Slider(ax_point2, 'Point 2', 0, 32, valinit=1, valstep=1)
        self.slider_p3 = Slider(ax_point3, 'Point 3', 0, 32, valinit=2, valstep=1)
        
        self.slider_p1.on_changed(self.update_angle_calculation)
        self.slider_p2.on_changed(self.update_angle_calculation)
        self.slider_p3.on_changed(self.update_angle_calculation)
        
        # Text annotation for angle value
        self.angle_text = self.ax_controls.text(0.1, 0.8, "Angle: N/A", fontsize=12)
        
        # Text annotation for landmark indices reference
        landmark_names = [
            "0: Nose", "1-10: Face", 
            "11: Left Shoulder", "12: Right Shoulder", 
            "13: Left Elbow", "14: Right Elbow",
            "15: Left Wrist", "16: Right Wrist",
            "23: Left Hip", "24: Right Hip",
            "25: Left Knee", "26: Right Knee",
            "27: Left Ankle", "28: Right Ankle"
        ]
        self.landmark_ref = self.ax_controls.text(0.1, 0.3, "\n".join(landmark_names), fontsize=10)
    
    def process_current_image(self):
        """Process the current image and update the display."""
        img_path = os.path.join(self.image_folder, self.image_files[self.current_idx])
        self.image = cv2.imread(img_path)
        
        if self.image is None:
            print(f"Error reading image: {img_path}")
            return
            
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w, _ = self.image.shape
        
        # Process with MediaPipe
        results = self.pose.process(self.image_rgb)
        
        if not results.pose_landmarks:
            print(f"No pose landmarks found in: {img_path}")
            self.landmarks = None
        else:
            self.landmarks = results.pose_landmarks.landmark
        
        # Update the display
        self.update_display()
        
    def update_display(self):
        """Update the image display with landmarks and annotations."""
        self.ax_image.clear()
        
        if self.image_rgb is not None:
            self.ax_image.imshow(self.image_rgb)
            
            if self.landmarks:
                # Get image dimensions
                h, w, _ = self.image.shape
                
                # Draw landmarks that are set to visible
                for i, landmark in enumerate(self.landmarks):
                    if self.landmark_visibility[i]:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        self.ax_image.plot(x, y, 'ro', markersize=5)
                        self.ax_image.text(x, y, str(i), fontsize=8, color='white',
                                          bbox=dict(facecolor='black', alpha=0.5))
                
                # Draw the angle markers if all points are valid
                p1_idx = int(self.slider_p1.val)
                p2_idx = int(self.slider_p2.val)
                p3_idx = int(self.slider_p3.val)
                
                if all(self.landmark_visibility[i] for i in [p1_idx, p2_idx, p3_idx]):
                    p1 = (int(self.landmarks[p1_idx].x * w), int(self.landmarks[p1_idx].y * h))
                    p2 = (int(self.landmarks[p2_idx].x * w), int(self.landmarks[p2_idx].y * h))
                    p3 = (int(self.landmarks[p3_idx].x * w), int(self.landmarks[p3_idx].y * h))
                    
                    # Draw lines
                    self.ax_image.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=2)
                    self.ax_image.plot([p2[0], p3[0]], [p2[1], p3[1]], 'g-', linewidth=2)
            
            self.ax_image.set_title(f"Image: {self.image_files[self.current_idx]}")
        
        self.fig.canvas.draw_idle()
        self.update_angle_calculation(None)  # Update angle display
    
    def update_landmark_display(self, label):
        """Update which landmarks are displayed based on radio button selection."""
        if label == 'All Landmarks':
            self.landmark_visibility = [True] * 33
        elif label == 'Body Only':
            # Hide face landmarks (0-10), show only body (11-32)
            self.landmark_visibility = [False] * 11 + [True] * 22
        elif label == 'Custom':
            # Set a custom configuration focusing on key yoga pose landmarks
            key_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            self.landmark_visibility = [i in key_landmarks for i in range(33)]
        
        self.update_display()
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate the angle between three landmarks."""
        if not all([p1, p2, p3]):
            return None
            
        a = np.array([p1.x, p1.y])
        b = np.array([p2.x, p2.y])
        c = np.array([p3.x, p3.y])
        
        ba = a - b
        bc = c - b
        
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine = np.clip(cosine, -1.0, 1.0)
        
        angle = np.arccos(cosine) * 180.0 / np.pi
        return angle
    
    def update_angle_calculation(self, val):
        """Update the angle calculation based on the selected landmarks."""
        if self.landmarks is None:
            self.angle_text.set_text("Angle: No landmarks detected")
            return
            
        p1_idx = int(self.slider_p1.val)
        p2_idx = int(self.slider_p2.val)
        p3_idx = int(self.slider_p3.val)
        
        angle = self.calculate_angle(
            self.landmarks[p1_idx],
            self.landmarks[p2_idx],
            self.landmarks[p3_idx]
        )
        
        if angle is not None:
            self.angle_text.set_text(f"Angle: {angle:.2f}° between landmarks {p1_idx}, {p2_idx}, {p3_idx}")
        else:
            self.angle_text.set_text("Angle: Unable to calculate")
            
        self.fig.canvas.draw_idle()
    
    def next_image(self, event):
        """Load the next image."""
        if self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self.process_current_image()
    
    def prev_image(self, event):
        """Load the previous image."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.process_current_image()
    
    def show(self):
        """Show the debugger interface."""
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Yoga Pose Debugging Tool")
    parser.add_argument("--input_folder", type=str, required=True, 
                        help="Path to folder containing yoga pose images")
    
    args = parser.parse_args()
    
    try:
        debugger = YogaPoseDebugger(args.input_folder)
        debugger.show()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()