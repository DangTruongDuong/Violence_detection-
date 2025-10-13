import torch
import cv2
import numpy as np
import time
import threading
from collections import deque
from queue import Queue
import os
from config import Config
from model import create_model
from data_loader import ViolenceDataset

class RealtimeViolenceDetector:
    def __init__(self, model_path, model_type="resnet_lstm"):
        self.device = torch.device(Config.DEVICE)
        self.model_type = model_type
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Frame buffer for sequence processing
        self.frame_buffer = deque(maxlen=Config.SEQUENCE_LENGTH)
        
        # Detection results
        self.current_prediction = "Non-Violence"
        self.current_confidence = 0.0
        self.detection_history = deque(maxlen=30)  # Keep last 30 detections
        
        # Threading
        self.detection_queue = Queue(maxsize=10)
        self.running = False
        self.detection_thread = None
        
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
            
            return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]
    
    def detection_worker(self):
        """Worker thread for processing detections"""
        while self.running:
            try:
                # Get frames from queue
                frames = self.detection_queue.get(timeout=1.0)
                
                if frames is None:  # Shutdown signal
                    break
                
                # Predict violence
                prediction, confidence, probabilities = self.predict_violence(frames)
                
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
                
                self.detection_queue.task_done()
                
            except Exception as e:
                print(f"Detection error: {e}")
                continue
    
    def add_frame(self, frame):
        """Add frame to detection pipeline"""
        if not self.running:
            return
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Add to buffer
        self.frame_buffer.append(processed_frame)
        
        # Process if buffer is full
        if len(self.frame_buffer) == Config.SEQUENCE_LENGTH:
            # Convert buffer to numpy array
            frames_array = np.array(list(self.frame_buffer))
            
            # Add to detection queue (non-blocking)
            try:
                self.detection_queue.put_nowait(frames_array)
            except:
                pass  # Queue full, skip this detection
    
    def start_detection(self):
        """Start realtime detection"""
        if self.running:
            return
        
        self.running = True
        self.detection_thread = threading.Thread(target=self.detection_worker)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        print("Realtime detection started")
    
    def stop_detection(self):
        """Stop realtime detection"""
        if not self.running:
            return
        
        self.running = False
        
        # Send shutdown signal
        try:
            self.detection_queue.put_nowait(None)
        except:
            pass
        
        # Wait for thread to finish
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        
        print("Realtime detection stopped")
    
    def get_detection_stats(self):
        """Get detection statistics"""
        if not self.detection_history:
            return {
                'current_prediction': self.current_prediction,
                'current_confidence': self.current_confidence,
                'fps': self.current_fps,
                'violence_percentage': 0.0,
                'avg_confidence': 0.0,
                'total_detections': 0
            }
        
        # Calculate violence percentage in last 30 detections
        violence_count = sum(1 for d in self.detection_history if d['prediction'] == 1)
        violence_percentage = (violence_count / len(self.detection_history)) * 100
        
        # Calculate average confidence
        avg_confidence = np.mean([d['confidence'] for d in self.detection_history])
        
        return {
            'current_prediction': self.current_prediction,
            'current_confidence': self.current_confidence,
            'fps': self.current_fps,
            'violence_percentage': violence_percentage,
            'avg_confidence': avg_confidence,
            'total_detections': len(self.detection_history)
        }
    
    def draw_detection_overlay(self, frame):
        """Draw detection information on frame"""
        stats = self.get_detection_stats()
        
        # Define colors
        if stats['current_prediction'] == "Violence":
            color = (0, 0, 255)  # Red
        elif stats['current_prediction'] == "Non-Violence":
            color = (0, 255, 0)  # Green
        else:
            color = (0, 255, 255)  # Yellow
        
        # Draw prediction text
        text = f"{stats['current_prediction']} ({stats['current_confidence']:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw FPS
        fps_text = f"FPS: {stats['fps']:.1f}"
        cv2.putText(frame, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw violence percentage
        violence_text = f"Violence: {stats['violence_percentage']:.1f}%"
        cv2.putText(frame, violence_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw detection count
        count_text = f"Detections: {stats['total_detections']}"
        cv2.putText(frame, count_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw confidence bar
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = 180
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Confidence bar
        confidence_width = int(bar_width * stats['current_confidence'])
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), color, -1)
        
        # Confidence text
        conf_text = f"Confidence: {stats['current_confidence']:.2f}"
        cv2.putText(frame, conf_text, (bar_x, bar_y + bar_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

def test_realtime_detection():
    """Test realtime detection with webcam"""
    # You need to train a model first or provide a pre-trained model path
    model_path = os.path.join(Config.SAVE_MODEL_PATH, "best_resnet_lstm_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train a model first using train.py")
        return
    
    # Initialize detector
    detector = RealtimeViolenceDetector(model_path, "resnet_lstm")
    detector.start_detection()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Press 'q' to quit, 's' to save screenshot")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add frame to detection pipeline
            detector.add_frame(frame)
            
            # Draw detection overlay
            frame = detector.draw_detection_overlay(frame)
            
            # Display frame
            cv2.imshow('Violence Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = int(time.time())
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved as {filename}")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Cleanup
        detector.stop_detection()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_realtime_detection()


