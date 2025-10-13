import torch
import cv2
import numpy as np
import time
import os
from collections import deque
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from config import Config
from model import create_model

class VideoViolenceDetector:
    def __init__(self, model_path, model_type="resnet_lstm"):
        self.device = torch.device(Config.DEVICE)
        self.model_type = model_type
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD
        
        # Clear GPU cache at start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Frame buffer for sequence processing
        self.frame_buffer = deque(maxlen=Config.SEQUENCE_LENGTH)
        
        # Detection results
        self.current_prediction = "Non-Violence"
        self.current_confidence = 0.0
        self.detection_history = []
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
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
        """Preprocess frame for model input"""
        # Resize frame
        frame = cv2.resize(frame, Config.FRAME_SIZE)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def predict_violence(self, frames):
        """Predict violence in frame sequence"""
        with torch.no_grad():
            # Convert frames to tensor
            if self.model_type == "convlstm3d":
                # For 3D CNN: (batch_size, channels, sequence_length, height, width)
                frames_tensor = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)
                frames_tensor = frames_tensor.unsqueeze(0)  # Add batch dimension
            else:
                # For LSTM models: (batch_size, sequence_length, channels, height, width)
                # frames shape: (sequence_length, height, width, channels)
                frames_tensor = torch.tensor(frames, dtype=torch.float32)
                # Permute to (sequence_length, channels, height, width)
                frames_tensor = frames_tensor.permute(0, 3, 1, 2)
                frames_tensor = frames_tensor.unsqueeze(0)  # Add batch dimension
            
            # Normalize to [0, 1] range
            frames_tensor = frames_tensor / 255.0
            
            frames_tensor = frames_tensor.to(self.device)
            
            # Get prediction
            outputs = self.model(frames_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get results and move to CPU immediately
            predicted_value = predicted.item()
            confidence_value = confidence.item()
            probabilities_value = probabilities.cpu().numpy()[0]
            
            # Clear GPU memory
            del frames_tensor, outputs, probabilities, confidence, predicted
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return predicted_value, confidence_value, probabilities_value
    
    def process_frame(self, frame):
        """Process single frame"""
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Add to buffer
        self.frame_buffer.append(processed_frame)
        
        # Process if buffer is full
        if len(self.frame_buffer) == Config.SEQUENCE_LENGTH:
            # Convert buffer to numpy array
            frames_array = np.array(list(self.frame_buffer))
            
            # Predict violence
            prediction, confidence, probabilities = self.predict_violence(frames_array)
            
            # Update current prediction
            if confidence >= self.confidence_threshold:
                self.current_prediction = "Violence" if prediction == 1 else "Non-Violence"
            else:
                self.current_prediction = "Uncertain"
            
            self.current_confidence = confidence
            
            # Store in history
            self.detection_history.append({
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': time.time(),
                'probabilities': probabilities
            })
            
            # Update FPS counter
            self.fps_counter += 1
            if self.fps_counter % 30 == 0:  # Update FPS every 30 frames
                current_time = time.time()
                self.current_fps = 30 / (current_time - self.fps_start_time)
                self.fps_start_time = current_time
                
                # Periodic GPU cleanup every 90 frames to prevent memory leak
                if torch.cuda.is_available() and self.fps_counter % 90 == 0:
                    torch.cuda.empty_cache()
    
    def draw_detection_overlay(self, frame):
        """Draw detection information on frame"""
        # Define colors
        if self.current_prediction == "Violence":
            color = (0, 0, 255)  # Red
        elif self.current_prediction == "Non-Violence":
            color = (0, 255, 0)  # Green
        else:
            color = (0, 255, 255)  # Yellow
        
        # Draw prediction text
        text = f"{self.current_prediction} ({self.current_confidence:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw FPS
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw confidence bar
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = 100
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Confidence bar
        confidence_width = int(bar_width * self.current_confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), color, -1)
        
        # Confidence text
        conf_text = f"Confidence: {self.current_confidence:.2f}"
        cv2.putText(frame, conf_text, (bar_x, bar_y + bar_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def get_detection_summary(self):
        """Get summary of detections"""
        if not self.detection_history:
            return {
                'total_frames': 0,
                'violence_frames': 0,
                'non_violence_frames': 0,
                'violence_percentage': 0.0,
                'avg_confidence': 0.0
            }
        
        total_frames = len(self.detection_history)
        violence_frames = sum(1 for d in self.detection_history if d['prediction'] == 1)
        non_violence_frames = total_frames - violence_frames
        violence_percentage = (violence_frames / total_frames) * 100
        avg_confidence = np.mean([d['confidence'] for d in self.detection_history])
        
        return {
            'total_frames': total_frames,
            'violence_frames': violence_frames,
            'non_violence_frames': non_violence_frames,
            'violence_percentage': violence_percentage,
            'avg_confidence': avg_confidence
        }

def test_video_detection(video_path, model_path, output_path=None):
    """Test violence detection on video file"""
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train a model first using train.py")
        return
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Video not found at {video_path}")
        return
    
    # Initialize detector
    detector = VideoViolenceDetector(model_path, "resnet_lstm")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.2f} seconds")
    print("\nPress 'q' to quit, 's' to save screenshot, 'p' to pause/resume")
    
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
                
                # Draw detection overlay
                frame = detector.draw_detection_overlay(frame)
                
                # Draw progress
                progress = (frame_count / total_frames) * 100
                progress_text = f"Progress: {progress:.1f}% ({frame_count}/{total_frames})"
                cv2.putText(frame, progress_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Write to output video if specified
                if out:
                    out.write(frame)
                
                # Display frame
                cv2.imshow('Violence Detection - Video Test', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = int(time.time())
                filename = f"video_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved as {filename}")
            elif key == ord('p'):
                paused = not paused
                print(f"Video {'paused' if paused else 'resumed'}")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Print summary
        summary = detector.get_detection_summary()
        print("\n" + "="*50)
        print("DETECTION SUMMARY")
        print("="*50)
        print(f"Total frames processed: {summary['total_frames']}")
        print(f"Violence frames: {summary['violence_frames']}")
        print(f"Non-violence frames: {summary['non_violence_frames']}")
        print(f"Violence percentage: {summary['violence_percentage']:.2f}%")
        print(f"Average confidence: {summary['avg_confidence']:.2f}")
        
        if output_path:
            print(f"Output video saved to: {output_path}")

class VideoTestGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Violence Detection - Video Test")
        self.root.geometry("600x400")
        self.root.resizable(False, False)
        
        # Variables
        self.video_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.model_path = os.path.join(Config.SAVE_MODEL_PATH, "best_resnet_lstm_model.pth")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Violence Detection - Video Test", 
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
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="Start Detection", command=self.start_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_fields).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5)
        
        # Info text
        info_text = """
Instructions:
1. Select a video file to test
2. Optionally select output path to save result video
3. Click 'Start Detection' to begin
4. In the video window:
   - Press 'q' to quit
   - Press 's' to save screenshot
   - Press 'p' to pause/resume

Supported formats: MP4, AVI, MOV, MKV
        """
        info_label = ttk.Label(main_frame, text=info_text, justify=tk.LEFT)
        info_label.grid(row=5, column=0, columnspan=3, pady=10)
        
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
    
    def start_detection(self):
        """Start video detection"""
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
            # Run detection
            test_video_detection(self.video_path.get(), self.model_path, output_path)
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
        finally:
            # Show GUI again
            self.root.deiconify()
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

def main():
    """Main function"""
    # Check if model exists
    model_path = os.path.join(Config.SAVE_MODEL_PATH, "best_resnet_lstm_model.pth")
    
    if not os.path.exists(model_path):
        print("Model not found! Please train a model first using train.py")
        return
    
    # Run GUI
    app = VideoTestGUI()
    app.run()

if __name__ == "__main__":
    main()
