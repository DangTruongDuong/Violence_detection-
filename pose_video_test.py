import torch
import cv2
import numpy as np
import time
import os
from collections import deque
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import mediapipe as mp
from config import Config
from model import create_model

class PoseVideoViolenceDetector:
    def __init__(self, model_path, model_type="resnet_lstm"):
        self.device = torch.device(Config.DEVICE)
        self.model_type = model_type
        
        # Load violence detection model
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Initialize MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Frame buffer for sequence processing
        self.frame_buffer = deque(maxlen=Config.SEQUENCE_LENGTH)
        
        # Detection results with smoothing
        self.prediction_history = deque(maxlen=8)  # Last 8 predictions
        self.confidence_history = deque(maxlen=8)  # Last 8 confidences
        
        # Current results
        self.current_prediction = "Non-Violence"
        self.current_confidence = 0.0
        self.smoothed_confidence = 0.0
        
        # Parameters
        self.confidence_threshold = 0.65
        self.violence_count_threshold = 3  # Need 3 consecutive violence detections
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Detection statistics
        self.total_detections = 0
        self.violence_detections = 0
        
        # Pose analysis
        self.pose_analysis_enabled = True
        self.movement_threshold = 0.1
        self.previous_landmarks = None
        
    def load_model(self, model_path):
        """Load trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        model = create_model(self.model_type)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        print(f"Model loaded from {model_path}")
        print(f"Model type: {self.model_type}")
        print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'Unknown'):.2f}%")
        
        return model
    
    def preprocess_frame(self, frame):
        """Enhanced preprocessing for better detection"""
        # Resize frame
        frame = cv2.resize(frame, Config.FRAME_SIZE)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply histogram equalization for better contrast
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        frame[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(frame[:, :, 0])
        frame = cv2.cvtColor(frame, cv2.COLOR_LAB2RGB)
        
        return frame
    
    def predict_violence(self, frames):
        """Predict violence in frame sequence"""
        with torch.no_grad():
            # Convert frames to tensor
            if self.model_type == "convlstm3d":
                frames_tensor = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)
                frames_tensor = frames_tensor.unsqueeze(0)
            else:
                frames_tensor = torch.tensor(frames, dtype=torch.float32)
                frames_tensor = frames_tensor.permute(0, 3, 1, 2)
                frames_tensor = frames_tensor.unsqueeze(0)
            
            # Normalize to [0, 1] range
            frames_tensor = frames_tensor / 255.0
            
            frames_tensor = frames_tensor.to(self.device)
            
            # Get prediction
            outputs = self.model(frames_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]
    
    def analyze_pose_movement(self, landmarks):
        """Analyze pose movement for violence detection"""
        if self.previous_landmarks is None:
            self.previous_landmarks = landmarks
            return 0.0
        
        # Calculate movement between frames
        movement = 0.0
        if landmarks and self.previous_landmarks:
            for i in range(min(len(landmarks.landmark), len(self.previous_landmarks.landmark))):
                prev_point = self.previous_landmarks.landmark[i]
                curr_point = landmarks.landmark[i]
                
                # Calculate Euclidean distance
                dx = curr_point.x - prev_point.x
                dy = curr_point.y - prev_point.y
                movement += np.sqrt(dx*dx + dy*dy)
        
        self.previous_landmarks = landmarks
        return movement / len(landmarks.landmark) if landmarks else 0.0
    
    def apply_temporal_smoothing(self, prediction, confidence, movement_score=0.0):
        """Apply temporal smoothing with pose analysis"""
        # Add to history
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)
        
        # Calculate smoothed confidence
        if len(self.confidence_history) > 1:
            self.smoothed_confidence = (self.smoothed_confidence * 0.7 + 
                                      confidence * 0.3)
        else:
            self.smoothed_confidence = confidence
        
        # Apply majority voting for prediction
        if len(self.prediction_history) >= 3:
            violence_count = sum(1 for p in self.prediction_history if p == 1)
            
            # Adjust threshold based on movement
            adjusted_threshold = self.violence_count_threshold
            if movement_score > self.movement_threshold:
                adjusted_threshold = max(1, adjusted_threshold - 1)
            
            # Need majority of recent predictions to be violence
            if violence_count >= adjusted_threshold:
                final_prediction = 1
            else:
                final_prediction = 0
        else:
            final_prediction = prediction
        
        return final_prediction, self.smoothed_confidence
    
    def process_frame(self, frame):
        """Process single frame with pose estimation"""
        # Preprocess frame for violence detection
        processed_frame = self.preprocess_frame(frame)
        
        # Add to buffer
        self.frame_buffer.append(processed_frame)
        
        # Process if buffer is full
        if len(self.frame_buffer) == Config.SEQUENCE_LENGTH:
            # Convert buffer to numpy array
            frames_array = np.array(list(self.frame_buffer))
            
            # Predict violence
            prediction, confidence, probabilities = self.predict_violence(frames_array)
            
            # Analyze pose movement
            movement_score = 0.0
            if self.pose_analysis_enabled:
                # Convert frame to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    movement_score = self.analyze_pose_movement(results.pose_landmarks)
            
            # Apply smoothing with pose analysis
            final_prediction, smoothed_confidence = self.apply_temporal_smoothing(
                prediction, confidence, movement_score
            )
            
            # Update current prediction
            if smoothed_confidence >= self.confidence_threshold:
                if final_prediction == 1:
                    self.current_prediction = "Violence"
                    self.violence_detections += 1
                else:
                    self.current_prediction = "Non-Violence"
            else:
                self.current_prediction = "Uncertain"
            
            self.current_confidence = smoothed_confidence
            self.total_detections += 1
            
            # Update FPS counter
            self.fps_counter += 1
            if self.fps_counter % 30 == 0:
                current_time = time.time()
                self.current_fps = 30 / (current_time - self.fps_start_time)
                self.fps_start_time = current_time
    
    def draw_pose_and_detection(self, frame):
        """Draw pose landmarks and detection information"""
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Define colors based on confidence
        if self.current_prediction == "Violence":
            if self.current_confidence > 0.8:
                color = (0, 0, 255)  # Red - High confidence violence
            else:
                color = (0, 165, 255)  # Orange - Medium confidence violence
        elif self.current_prediction == "Non-Violence":
            color = (0, 255, 0)  # Green
        else:
            color = (0, 255, 255)  # Yellow - Uncertain
        
        # Draw prediction text with confidence
        text = f"{self.current_prediction} ({self.current_confidence:.3f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw FPS
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw detection statistics
        stats_text = f"Violence: {self.violence_detections}/{self.total_detections}"
        cv2.putText(frame, stats_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw confidence bar with color coding
        bar_width = 300
        bar_height = 25
        bar_x = 10
        bar_y = 150
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Confidence bar
        confidence_width = int(bar_width * self.current_confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), color, -1)
        
        # Confidence text
        conf_text = f"Confidence: {self.current_confidence:.3f}"
        cv2.putText(frame, conf_text, (bar_x, bar_y + bar_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw threshold line
        threshold_x = bar_x + int(bar_width * self.confidence_threshold)
        cv2.line(frame, (threshold_x, bar_y), (threshold_x, bar_y + bar_height), (255, 255, 255), 2)
        
        # Draw prediction history
        history_y = 220
        for i, (pred, conf) in enumerate(zip(self.prediction_history, self.confidence_history)):
            x = 10 + i * 30
            if pred == 1:
                cv2.circle(frame, (x, history_y), 10, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (x, history_y), 10, (0, 255, 0), -1)
            # Draw confidence as circle size
            radius = int(5 + conf * 5)
            cv2.circle(frame, (x, history_y), radius, (255, 255, 255), 1)
        
        # Draw pose analysis info
        if self.pose_analysis_enabled:
            pose_text = "Pose Analysis: ON"
            cv2.putText(frame, pose_text, (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def get_detection_summary(self):
        """Get enhanced detection summary"""
        if self.total_detections == 0:
            return {
                'total_frames': 0,
                'violence_frames': 0,
                'non_violence_frames': 0,
                'violence_percentage': 0.0,
                'avg_confidence': 0.0,
                'pose_analysis': self.pose_analysis_enabled
            }
        
        violence_percentage = (self.violence_detections / self.total_detections) * 100
        avg_confidence = np.mean(list(self.confidence_history)) if self.confidence_history else 0.0
        
        return {
            'total_frames': self.total_detections,
            'violence_frames': self.violence_detections,
            'non_violence_frames': self.total_detections - self.violence_detections,
            'violence_percentage': violence_percentage,
            'avg_confidence': avg_confidence,
            'pose_analysis': self.pose_analysis_enabled
        }

class PoseVideoTestGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Pose-based Violence Detection - Video Test")
        self.root.geometry("600x450")
        self.root.resizable(False, False)
        
        # Variables
        self.video_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.model_path = os.path.join(Config.SAVE_MODEL_PATH, "best_resnet_lstm_model.pth")
        
        # Advanced parameters
        self.confidence_threshold = tk.DoubleVar(value=0.65)
        self.violence_count_threshold = tk.IntVar(value=3)
        self.pose_analysis_enabled = tk.BooleanVar(value=True)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Pose-based Violence Detection", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Model status
        model_status = "✅ Available" if os.path.exists(self.model_path) else "❌ Not found"
        model_label = ttk.Label(main_frame, text=f"Model: {model_status}")
        model_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        
        # Video selection
        ttk.Label(main_frame, text="Select Video File:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.video_path, width=50).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_video).grid(row=2, column=2, pady=5)
        
        # Output selection
        ttk.Label(main_frame, text="Output Video (Optional):").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_path, width=50).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(row=3, column=2, pady=5)
        
        # Advanced parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Advanced Parameters", padding="10")
        params_frame.grid(row=4, column=0, columnspan=3, pady=20, sticky=(tk.W, tk.E))
        
        # Confidence threshold
        ttk.Label(params_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Scale(params_frame, from_=0.1, to=0.9, variable=self.confidence_threshold, 
                 orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(params_frame, textvariable=self.confidence_threshold).grid(row=0, column=2, pady=5)
        
        # Violence count threshold
        ttk.Label(params_frame, text="Violence Count Threshold:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Scale(params_frame, from_=1, to=10, variable=self.violence_count_threshold, 
                 orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(params_frame, textvariable=self.violence_count_threshold).grid(row=1, column=2, pady=5)
        
        # Pose analysis checkbox
        ttk.Checkbutton(params_frame, text="Enable Pose Analysis", 
                       variable=self.pose_analysis_enabled).grid(row=2, column=0, columnspan=3, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="Start Detection", command=self.start_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_fields).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5)
        
        # Info text
        info_text = """
Pose-based Features:
• Real-time pose estimation with MediaPipe (33 landmarks)
• Movement analysis for violence detection
• Temporal smoothing to reduce false positives
• Enhanced confidence visualization
• Pose landmarks and connections overlay

Controls: 'q'=quit, 's'=screenshot, 'p'=pause/resume, 'c'=clear history
Supported formats: MP4, AVI, MOV, MKV
        """
        info_label = ttk.Label(main_frame, text=info_text, justify=tk.LEFT)
        info_label.grid(row=6, column=0, columnspan=3, pady=10)
        
    def browse_video(self):
        """Browse for video file"""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )
        if filename:
            self.video_path.set(filename)
    
    def browse_output(self):
        """Browse for output file"""
        filetypes = [
            ("MP4 files", "*.mp4"),
            ("AVI files", "*.avi"),
            ("All files", "*.*")
        ]
        filename = filedialog.asksaveasfilename(
            title="Save Output Video As",
            filetypes=filetypes,
            defaultextension=".mp4"
        )
        if filename:
            self.output_path.set(filename)
    
    def clear_fields(self):
        """Clear all fields"""
        self.video_path.set("")
        self.output_path.set("")
        self.confidence_threshold.set(0.65)
        self.violence_count_threshold.set(3)
        self.pose_analysis_enabled.set(True)
    
    def start_detection(self):
        """Start pose-based video detection"""
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file!")
            return
        
        if not os.path.exists(self.video_path.get()):
            messagebox.showerror("Error", "Video file not found!")
            return
        
        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", "Model not found! Please train a model first.")
            return
        
        # Get output path
        output_path = self.output_path.get() if self.output_path.get() else None
        
        # Hide GUI
        self.root.withdraw()
        
        try:
            # Run pose-based detection
            test_pose_video_detection(
                self.video_path.get(), 
                self.model_path, 
                output_path,
                self.confidence_threshold.get(),
                self.violence_count_threshold.get(),
                self.pose_analysis_enabled.get()
            )
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
        finally:
            # Show GUI again
            self.root.deiconify()
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

def test_pose_video_detection(video_path, model_path, output_path=None, 
                            confidence_threshold=0.65, violence_count_threshold=3,
                            pose_analysis_enabled=True):
    """Test pose-based violence detection on video file"""
    # Initialize detector with advanced parameters
    detector = PoseVideoViolenceDetector(model_path, "resnet_lstm")
    detector.confidence_threshold = confidence_threshold
    detector.violence_count_threshold = violence_count_threshold
    detector.pose_analysis_enabled = pose_analysis_enabled
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Pose-based Video Detection Started")
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.2f} seconds")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Violence count threshold: {violence_count_threshold}")
    print(f"Pose analysis: {'Enabled' if pose_analysis_enabled else 'Disabled'}")
    print("\nPress 'q' to quit, 's' to save screenshot, 'p' to pause/resume, 'c' to clear history")
    
    # Setup output video if specified
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output video will be saved to: {output_path}")
    
    frame_count = 0
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                detector.process_frame(frame)
                
                # Draw pose and detection overlay
                frame = detector.draw_pose_and_detection(frame)
                
                # Draw progress
                progress = (frame_count / total_frames) * 100
                progress_text = f"Progress: {progress:.1f}% ({frame_count}/{total_frames})"
                cv2.putText(frame, progress_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Write to output video if specified
                if out:
                    out.write(frame)
                
                # Display frame
                cv2.imshow('Pose-based Violence Detection - Video', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = int(time.time())
                filename = f"pose_video_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved as {filename}")
            elif key == ord('p'):
                paused = not paused
                print(f"Video {'paused' if paused else 'resumed'}")
            elif key == ord('c'):
                # Clear detection history
                detector.prediction_history.clear()
                detector.confidence_history.clear()
                detector.previous_landmarks = None
                print("Detection history cleared")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Print enhanced summary
        summary = detector.get_detection_summary()
        print("\n" + "="*60)
        print("POSE-BASED VIDEO DETECTION SUMMARY")
        print("="*60)
        print(f"Total frames processed: {summary['total_frames']}")
        print(f"Violence frames: {summary['violence_frames']}")
        print(f"Non-violence frames: {summary['non_violence_frames']}")
        print(f"Violence percentage: {summary['violence_percentage']:.2f}%")
        print(f"Average confidence: {summary['avg_confidence']:.3f}")
        print(f"Pose analysis: {'Enabled' if summary['pose_analysis'] else 'Disabled'}")
        print(f"Confidence threshold used: {confidence_threshold}")
        print(f"Violence count threshold used: {violence_count_threshold}")
        
        if output_path:
            print(f"Output video saved to: {output_path}")

def main():
    """Main function"""
    # Check if model exists
    model_path = os.path.join(Config.SAVE_MODEL_PATH, "best_resnet_lstm_model.pth")
    
    if not os.path.exists(model_path):
        print("Model not found! Please train a model first using train.py")
        return
    
    # Run pose-based GUI
    app = PoseVideoTestGUI()
    app.run()

if __name__ == "__main__":
    main()
