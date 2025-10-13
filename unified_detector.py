"""
Unified Violence Detection System
Hệ thống phát hiện bạo lực tích hợp cho cả video và camera
- Phát hiện người: Bounding Box (HOG) hoặc Pose Skeleton (MediaPipe)
- Vẽ khung đỏ hoặc pose skeleton khi phát hiện bạo lực
- Hỗ trợ cả camera và video file
- Giao diện GUI thân thiện với thanh điều chỉnh confidence
"""

import torch
import cv2
import numpy as np
import time
import os
from collections import deque
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import mediapipe as mp
from config import Config
from model import create_model


class UnifiedViolenceDetector:
    def __init__(self, model_path, model_type="resnet_lstm", use_person_detection=True, use_pose_mode=False):
        self.device = torch.device(Config.DEVICE)
        self.model_type = model_type
        self.use_person_detection = use_person_detection
        self.use_pose_mode = use_pose_mode  # True = pose, False = bounding box
        self.draw_mode = "bbox"  # Default: "none", "bbox", or "pose" - can be set later
        
        # Load violence detection model
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Initialize HOG Person Detector (used by both modes)
        if self.use_person_detection:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Initialize MediaPipe Pose for pose mode
        if self.use_person_detection and self.use_pose_mode:
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
        
        # Store pose landmarks for all detected persons
        self.pose_landmarks_list = []
        
        # Frame buffer for sequence processing
        self.frame_buffer = deque(maxlen=Config.SEQUENCE_LENGTH)
        
        # Detection results with smoothing
        self.prediction_history = deque(maxlen=10)
        self.confidence_history = deque(maxlen=10)
        
        # Current results
        self.current_prediction = "Non-Violence"
        self.current_confidence = 0.0
        self.is_violence_detected = False
        
        # Person bounding boxes (cached)
        self.person_boxes = []
        self.person_boxes_cache_frame = 0
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Detection statistics
        self.total_detections = 0
        self.violence_detections = 0
        
        # Settings
        self.confidence_threshold = 0.70
        self.violence_frames_threshold = 3  # Consecutive frames for violence detection
        self.consecutive_violence_count = 0
        
        # Performance optimization
        self.frame_skip = 2  # Process every N frames (1=no skip, 2=skip 1 frame, 3=skip 2 frames)
        self.frame_count_total = 0
        self.person_detect_interval = 5  # Detect persons every N frames
        
    def load_model(self, model_path):
        """Load trained violence detection model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        model = create_model(self.model_type)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        print(f"✅ Model loaded from {model_path}")
        print(f"   Model type: {self.model_type}")
        print(f"   Best validation accuracy: {checkpoint.get('best_val_acc', 'Unknown'):.2f}%")
        
        return model
    
    def preprocess_frame(self, frame):
        """Preprocess frame for violence detection model"""
        frame_resized = cv2.resize(frame, Config.FRAME_SIZE)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        return frame_rgb
    
    def detect_persons(self, frame, force_detect=False):
        """Detect persons using HOG (bounding box) or MediaPipe (pose)"""
        # Use cached results if not forced and within interval
        if not force_detect and self.frame_count_total % self.person_detect_interval != 0:
            return self.person_boxes
        
        person_boxes = []
        self.pose_landmarks_list = []
        
        if not self.use_person_detection:
            return person_boxes
        
        h, w, _ = frame.shape
        
        # MODE 1: POSE SKELETON - Improved multi-person detection  
        if self.use_pose_mode and self.draw_mode == "pose":
            # Step 1: Use HOG to detect person bounding boxes
            scale = 0.5
            frame_resized = cv2.resize(frame, None, fx=scale, fy=scale)
            
            try:
                (rects, weights) = self.hog.detectMultiScale(
                    frame_resized,
                    winStride=(8, 8),
                    padding=(4, 4),
                    scale=1.1,
                    useMeanshiftGrouping=False
                )
            except:
                rects = []
            
            # Step 2: For each detected person, run pose estimation
            for i, (x, y, w_box, h_box) in enumerate(rects):
                # Scale back to original
                x = int(x / scale)
                y = int(y / scale)
                w_box = int(w_box / scale)
                h_box = int(h_box / scale)
                
                # Add padding
                padding = 20
                x_min = max(0, x - padding)
                y_min = max(0, y - padding)
                x_max = min(w, x + w_box + padding)
                y_max = min(h, y + h_box + padding)
                
                # Crop person region
                person_crop = frame[y_min:y_max, x_min:x_max]
                
                if person_crop.size == 0:
                    continue
                
                # Run MediaPipe Pose on cropped region
                rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_crop)
                
                if results.pose_landmarks:
                    # Adjust landmarks to full frame coordinates
                    adjusted_landmarks = self.mp_pose.PoseLandmark
                    pose_landmarks = results.pose_landmarks
                    
                    # Store for drawing
                    self.pose_landmarks_list.append({
                        'landmarks': pose_landmarks,
                        'offset': (x_min, y_min)
                    })
                    
                    person_boxes.append({
                        'bbox': (x_min, y_min, x_max, y_max),
                        'confidence': float(weights[i]) if i < len(weights) else 1.0,
                        'pose_landmarks': pose_landmarks,
                        'pose_offset': (x_min, y_min)
                    })
        
        # MODE 2: BOUNDING BOX
        else:
            # Resize frame for faster detection
            scale = 0.5
            frame_resized = cv2.resize(frame, None, fx=scale, fy=scale)
            
            try:
                (rects, weights) = self.hog.detectMultiScale(
                    frame_resized,
                    winStride=(8, 8),
                    padding=(4, 4),
                    scale=1.1,
                    useMeanshiftGrouping=False
                )
            except:
                return self.person_boxes
            
            for i, (x, y, w_box, h_box) in enumerate(rects):
                x = int(x / scale)
                y = int(y / scale)
                w_box = int(w_box / scale)
                h_box = int(h_box / scale)
                
                padding = 10
                x_min = max(0, x - padding)
                y_min = max(0, y - padding)
                x_max = min(w, x + w_box + padding)
                y_max = min(h, y + h_box + padding)
                
                person_boxes.append({
                    'bbox': (x_min, y_min, x_max, y_max),
                    'confidence': float(weights[i]) if i < len(weights) else 1.0
                })
        
        # Cache results
        self.person_boxes = person_boxes
        self.person_boxes_cache_frame = self.frame_count_total
        
        return person_boxes
    
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
            
            # Normalize
            frames_tensor = frames_tensor / 255.0
            frames_tensor = frames_tensor.to(self.device)
            
            # Get prediction
            outputs = self.model(frames_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Move to CPU
            predicted_value = predicted.item()
            confidence_value = confidence.item()
            probabilities_value = probabilities.cpu().numpy()[0]
            
            # Clear GPU memory
            del frames_tensor, outputs, probabilities, confidence, predicted
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return predicted_value, confidence_value, probabilities_value
    
    def apply_smoothing(self, prediction, confidence):
        """Apply temporal smoothing to reduce false positives"""
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)
        
        if len(self.prediction_history) < 5:
            return prediction, confidence
        
        # Calculate smoothed values
        violence_count = sum(1 for p in self.prediction_history if p == 1)
        avg_confidence = np.mean(list(self.confidence_history))
        
        # If majority of recent predictions are violence with high confidence
        if violence_count >= 3 and avg_confidence >= self.confidence_threshold:
            return 1, avg_confidence
        else:
            return 0, avg_confidence
    
    def process_frame(self, frame):
        """Process single frame for violence detection (with frame skipping)"""
        self.frame_count_total += 1
        
        # Skip frames for better performance
        if self.frame_count_total % self.frame_skip != 0:
            return  # Skip this frame
        
        # Detect persons in frame (with caching)
        if self.use_person_detection:
            self.person_boxes = self.detect_persons(frame)
        
        # Preprocess frame for violence detection
        processed_frame = self.preprocess_frame(frame)
        
        # Add to buffer
        self.frame_buffer.append(processed_frame)
        
        # Process if buffer is full
        if len(self.frame_buffer) == Config.SEQUENCE_LENGTH:
            frames_array = np.array(list(self.frame_buffer))
            
            # Predict violence
            prediction, confidence, probabilities = self.predict_violence(frames_array)
            
            # Apply smoothing
            smoothed_prediction, smoothed_confidence = self.apply_smoothing(prediction, confidence)
            
            # Update violence detection status
            if smoothed_prediction == 1 and smoothed_confidence >= self.confidence_threshold:
                self.consecutive_violence_count += 1
                if self.consecutive_violence_count >= self.violence_frames_threshold:
                    self.is_violence_detected = True
                    self.current_prediction = "Violence"
                    self.violence_detections += 1
                else:
                    self.is_violence_detected = False
                    self.current_prediction = "Checking..."
            else:
                self.consecutive_violence_count = 0
                self.is_violence_detected = False
                self.current_prediction = "Non-Violence"
            
            self.current_confidence = smoothed_confidence
            self.total_detections += 1
            
            # Update FPS
            self.fps_counter += 1
            if self.fps_counter % 20 == 0:  # Update FPS mỗi 20 frames
                current_time = time.time()
                elapsed = current_time - self.fps_start_time
                if elapsed > 0:
                    self.current_fps = 20 / elapsed
                self.fps_start_time = current_time
    
    def draw_person_boxes(self, frame):
        """Draw bounding box OR pose skeleton when violence is detected"""
        # CHỈ vẽ khi phát hiện bạo lực
        if not self.is_violence_detected:
            return frame
        
        if not self.person_boxes:
            return frame
        
        # KHÔNG VẼ nếu draw_mode = "none"
        if self.draw_mode == "none":
            return frame
        
        num_persons = len(self.person_boxes)
        h, w, _ = frame.shape
        
        # MODE 1: VẼ POSE SKELETON - Multi-person support
        if self.draw_mode == "pose":
            # Vẽ pose landmarks cho tất cả người được detect
            for i, pose_data in enumerate(self.pose_landmarks_list):
                landmarks = pose_data['landmarks']
                offset_x, offset_y = pose_data['offset']
                
                # Get corresponding bbox for size calculation
                if i < len(self.person_boxes):
                    bbox = self.person_boxes[i]['bbox']
                    bbox_width = bbox[2] - bbox[0]
                    bbox_height = bbox[3] - bbox[1]
                else:
                    bbox_width = w
                    bbox_height = h
                
                # Draw landmarks with offset
                for idx, landmark in enumerate(landmarks.landmark):
                    # Calculate actual position with offset
                    x = int(landmark.x * bbox_width + offset_x)
                    y = int(landmark.y * bbox_height + offset_y)
                    
                    # Draw landmark point
                    if landmark.visibility > 0.5:
                        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                        cv2.circle(frame, (x, y), 6, (0, 150, 255), 2)
                
                # Draw connections
                for connection in self.mp_pose.POSE_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    start_landmark = landmarks.landmark[start_idx]
                    end_landmark = landmarks.landmark[end_idx]
                    
                    if start_landmark.visibility > 0.5 and end_landmark.visibility > 0.5:
                        start_x = int(start_landmark.x * bbox_width + offset_x)
                        start_y = int(start_landmark.y * bbox_height + offset_y)
                        end_x = int(end_landmark.x * bbox_width + offset_x)
                        end_y = int(end_landmark.y * bbox_height + offset_y)
                        
                        cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 150, 255), 4)
            
            # Vẽ warning text
            if num_persons >= 2:
                warning_text = f"VIOLENCE - {num_persons} PERSONS"
            else:
                warning_text = "VIOLENCE DETECTED"
            
            # Compact warning on top bar area (avoid overlap with overlay)
            cv2.putText(frame, warning_text, (w // 2 - 150, 70),
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, warning_text, (w // 2 - 150, 70),
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # MODE 2: VẼ BOUNDING BOX
        else:
            # Tính toán 1 VÙNG DUY NHẤT bao quanh TẤT CẢ người
            all_x_min = float('inf')
            all_y_min = float('inf')
            all_x_max = 0
            all_y_max = 0
            
            for person in self.person_boxes:
                x_min, y_min, x_max, y_max = person['bbox']
                all_x_min = min(all_x_min, x_min)
                all_y_min = min(all_y_min, y_min)
                all_x_max = max(all_x_max, x_max)
                all_y_max = max(all_y_max, y_max)
            
            # Thêm padding
            padding = 25
            all_x_min = max(0, int(all_x_min) - padding)
            all_y_min = max(0, int(all_y_min) - padding)
            all_x_max = min(w, int(all_x_max) + padding)
            all_y_max = min(h, int(all_y_max) + padding)
            
            # Màu đỏ
            color = (0, 0, 255)
            color_bright = (0, 100, 255)
            thickness = 8
            
            # VẼ KHUNG NHÁY
            blink = int(time.time() * 3) % 2
            if blink:
                cv2.rectangle(frame, (all_x_min, all_y_min), (all_x_max, all_y_max), color, thickness)
                cv2.rectangle(frame, (all_x_min-3, all_y_min-3), (all_x_max+3, all_y_max+3), color_bright, 2)
            else:
                cv2.rectangle(frame, (all_x_min, all_y_min), (all_x_max, all_y_max), color, thickness-2)
            
            # Corner markers
            corner_len = 40
            corner_thickness = 6
            # Top-left
            cv2.line(frame, (all_x_min, all_y_min), (all_x_min + corner_len, all_y_min), (255, 255, 255), corner_thickness)
            cv2.line(frame, (all_x_min, all_y_min), (all_x_min, all_y_min + corner_len), (255, 255, 255), corner_thickness)
            # Top-right
            cv2.line(frame, (all_x_max, all_y_min), (all_x_max - corner_len, all_y_min), (255, 255, 255), corner_thickness)
            cv2.line(frame, (all_x_max, all_y_min), (all_x_max, all_y_min + corner_len), (255, 255, 255), corner_thickness)
            # Bottom-left
            cv2.line(frame, (all_x_min, all_y_max), (all_x_min + corner_len, all_y_max), (255, 255, 255), corner_thickness)
            cv2.line(frame, (all_x_min, all_y_max), (all_x_min, all_y_max - corner_len), (255, 255, 255), corner_thickness)
            # Bottom-right
            cv2.line(frame, (all_x_max, all_y_max), (all_x_max - corner_len, all_y_max), (255, 255, 255), corner_thickness)
            cv2.line(frame, (all_x_max, all_y_max), (all_x_max, all_y_max - corner_len), (255, 255, 255), corner_thickness)
            
            # Label
            if num_persons >= 2:
                label = f"VIOLENCE - {num_persons} PERSONS"
            else:
                label = "VIOLENCE DETECTED"
            
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
            label_x = all_x_min
            label_y = max(all_y_min - 20, label_size[1] + 20)
            
            # Background shadow
            cv2.rectangle(frame, 
                         (label_x - 12, label_y - label_size[1] - 12),
                         (label_x + label_size[0] + 12, label_y + 8),
                         (0, 0, 0), -1)
            
            # Background main
            cv2.rectangle(frame, 
                         (label_x - 10, label_y - label_size[1] - 10),
                         (label_x + label_size[0] + 10, label_y + 6),
                         color, -1)
            
            # Label text
            cv2.putText(frame, label, (label_x, label_y - 2),
                       cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame
    
    def draw_overlay(self, frame):
        """Draw detection information overlay - Beautiful & No Overlap"""
        h, w, _ = frame.shape
        
        # Determine status
        if self.is_violence_detected:
            status_color = (0, 0, 255)  # Red
            status_text = "VIOLENCE DETECTED"
            status_bg = (0, 0, 180)
        elif self.current_prediction == "Checking...":
            status_color = (0, 200, 255)  # Orange
            status_text = "ANALYZING..."
            status_bg = (0, 140, 200)
        else:
            status_color = (0, 255, 0)  # Green
            status_text = "SAFE"
            status_bg = (0, 180, 0)
        
        # ===== TOP BANNER (Status) - Compact =====
        banner_height = 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Status indicator (left)
        indicator_size = 40
        cv2.rectangle(frame, (15, 10), (15 + indicator_size, 10 + indicator_size), status_bg, -1)
        cv2.rectangle(frame, (15, 10), (15 + indicator_size, 10 + indicator_size), status_color, 3)
        
        # Status text
        cv2.putText(frame, status_text, (65, 38),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # FPS Badge (right)
        fps_text = f"FPS: {self.current_fps:.0f}"
        fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        fps_x = w - fps_size[0] - 25
        cv2.rectangle(frame, (fps_x - 8, 15), (fps_x + fps_size[0] + 8, 45), (50, 50, 50), -1)
        cv2.putText(frame, fps_text, (fps_x, 36),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2, cv2.LINE_AA)
        
        # ===== LEFT PANEL (Info Cards) - Smaller & Lower =====
        panel_x = 15
        panel_y = 75  # Bắt đầu sau top banner
        card_width = 200  # Nhỏ hơn
        card_height = 45  # Nhỏ hơn
        card_spacing = 8
        
        # Card 1: Confidence
        self._draw_info_card(frame, panel_x, panel_y, card_width, card_height,
                            "CONF", f"{self.current_confidence:.0%}", status_color)
        
        # Card 2: Total Detections
        self._draw_info_card(frame, panel_x, panel_y + card_height + card_spacing,
                            card_width, card_height,
                            "EVENTS", str(self.violence_detections), (200, 200, 200))
        
        # ===== RIGHT PANEL: Confidence Bar - Smaller =====
        bar_width = 200  # Nhỏ hơn
        bar_height = 25
        bar_x = w - bar_width - 20
        bar_y = 75
        
        # Bar background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (40, 40, 40), -1)
        
        # Confidence fill
        conf_width = int(bar_width * self.current_confidence)
        cv2.rectangle(frame, (bar_x + 2, bar_y + 2), 
                     (bar_x + conf_width - 2, bar_y + bar_height - 2),
                     status_color, -1)
        
        # Bar border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (200, 200, 200), 2)
        
        # Bar label (smaller)
        cv2.putText(frame, "CONFIDENCE", (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        # ===== BOTTOM BAR (Statistics) =====
        bottom_height = 35
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - bottom_height), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay2, 0.85, frame, 0.15, 0, frame)
        
        # Stats text
        violence_rate = (self.violence_detections / max(self.total_detections, 1)) * 100
        stats_text = f"Violence Rate: {violence_rate:.1f}%  |  Total Frames: {self.total_detections}"
        
        cv2.putText(frame, stats_text, (15, h - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        return frame
    
    def _draw_info_card(self, frame, x, y, width, height, label, value, color):
        """Draw info card with label and value - Compact version"""
        # Card background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Border
        cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), 2)
        
        # Label (left side, small)
        cv2.putText(frame, label, (x + 10, y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Value (left side, large)
        cv2.putText(frame, value, (x + 10, y + 36),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2, cv2.LINE_AA)
    
    
    def render_frame(self, frame):
        """Render frame with all visualizations"""
        # Draw person bounding boxes
        if self.use_person_detection:
            frame = self.draw_person_boxes(frame)
        else:
            # If no person detection, draw full frame box when violence detected
            if self.is_violence_detected:
                h, w, _ = frame.shape
                cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 0, 255), 5)
                cv2.putText(frame, "VIOLENCE DETECTED!", (w//2 - 200, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        
        # Draw overlay information
        frame = self.draw_overlay(frame)
        
        return frame


def run_camera_detection(model_path, camera_id=0, confidence_threshold=0.70, performance_mode="balanced", use_pose_mode=False, enable_person_detection=True, draw_mode="none"):
    """Run violence detection on camera feed"""
    print("="*60)
    print("CAMERA VIOLENCE DETECTION")
    print("="*60)
    
    # Initialize detector
    detector = UnifiedViolenceDetector(model_path, use_person_detection=enable_person_detection, use_pose_mode=use_pose_mode)
    detector.confidence_threshold = confidence_threshold
    detector.draw_mode = draw_mode  # Set draw mode
    
    # Display mode
    if draw_mode == "none":
        print("⭕ Draw Mode: NONE (Detect but don't draw)")
    elif draw_mode == "pose":
        print("🦴 Draw Mode: POSE SKELETON")
    else:
        print("📦 Draw Mode: BOUNDING BOX")
    
    # Apply performance settings
    if performance_mode == "fast":
        detector.frame_skip = 3  # Skip 2 frames
        detector.person_detect_interval = 10  # Detect persons every 10 frames
        print("⚡ Performance Mode: FAST (Ít lag nhất)")
    elif performance_mode == "accurate":
        detector.frame_skip = 1  # No skip
        detector.person_detect_interval = 3  # Detect persons every 3 frames
        print("🎯 Performance Mode: ACCURATE (Chính xác nhất)")
    else:  # balanced
        detector.frame_skip = 2  # Skip 1 frame
        detector.person_detect_interval = 5  # Default
        print("⚖️ Performance Mode: BALANCED (Cân bằng)")
    
    print(f"📊 Confidence Threshold: {confidence_threshold:.0%}")
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_id}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print(f"✅ Camera {camera_id} opened successfully")
    print("\nControls:")
    print("  Q - Quit")
    print("  S - Save screenshot")
    print("  P - Toggle person detection")
    print("  R - Reset statistics")
    print("="*60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Process frame
            detector.process_frame(frame)
            
            # Render frame
            frame = detector.render_frame(frame)
            
            # Display
            cv2.imshow('Violence Detection - Camera', frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                filename = f"camera_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('p'):
                detector.use_person_detection = not detector.use_person_detection
                status = "ON" if detector.use_person_detection else "OFF"
                print(f"👤 Person detection: {status}")
            elif key == ord('r'):
                detector.total_detections = 0
                detector.violence_detections = 0
                print("🔄 Statistics reset")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Camera detection stopped")


def run_video_detection(model_path, video_path, output_path=None, confidence_threshold=0.70, performance_mode="balanced", use_pose_mode=False, enable_person_detection=True, draw_mode="none"):
    """Run violence detection on video file"""
    print("="*60)
    print("VIDEO VIOLENCE DETECTION")
    print("="*60)
    
    # Check video exists
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    # Initialize detector
    detector = UnifiedViolenceDetector(model_path, use_person_detection=enable_person_detection, use_pose_mode=use_pose_mode)
    detector.confidence_threshold = confidence_threshold
    detector.draw_mode = draw_mode  # Set draw mode
    
    # Display mode
    if draw_mode == "none":
        print("⭕ Draw Mode: NONE (Detect but don't draw)")
    elif draw_mode == "pose":
        print("🦴 Draw Mode: POSE SKELETON")
    else:
        print("📦 Draw Mode: BOUNDING BOX")
    
    # Apply performance settings
    if performance_mode == "fast":
        detector.frame_skip = 3  # Skip 2 frames
        detector.person_detect_interval = 10  # Detect persons every 10 frames
        print("⚡ Performance Mode: FAST (Ít lag nhất)")
    elif performance_mode == "accurate":
        detector.frame_skip = 1  # No skip
        detector.person_detect_interval = 3  # Detect persons every 3 frames
        print("🎯 Performance Mode: ACCURATE (Chính xác nhất)")
    else:  # balanced
        detector.frame_skip = 2  # Skip 1 frame
        detector.person_detect_interval = 5  # Default
        print("⚖️ Performance Mode: BALANCED (Cân bằng)")
    
    print(f"📊 Confidence Threshold: {confidence_threshold:.0%}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"📹 Video: {os.path.basename(video_path)}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {duration:.2f} seconds")
    print("\nControls:")
    print("  Q - Quit")
    print("  S - Save screenshot")
    print("  SPACE - Pause/Resume")
    print("="*60)
    
    # Setup output video
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Output: {output_path}")
    
    frame_count = 0
    paused = False
    start_time = time.time()
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                detector.process_frame(frame)
                
                # Render frame
                rendered_frame = detector.render_frame(frame.copy())
                
                # Add progress bar
                progress = frame_count / total_frames
                progress_bar_width = width - 40
                progress_bar_height = 20
                progress_x = 20
                progress_y = height - 30
                
                cv2.rectangle(rendered_frame,
                            (progress_x, progress_y),
                            (progress_x + progress_bar_width, progress_y + progress_bar_height),
                            (50, 50, 50), -1)
                
                cv2.rectangle(rendered_frame,
                            (progress_x, progress_y),
                            (progress_x + int(progress_bar_width * progress), progress_y + progress_bar_height),
                            (0, 255, 0), -1)
                
                cv2.rectangle(rendered_frame,
                            (progress_x, progress_y),
                            (progress_x + progress_bar_width, progress_y + progress_bar_height),
                            (255, 255, 255), 2)
                
                progress_text = f"{progress*100:.1f}% ({frame_count}/{total_frames})"
                cv2.putText(rendered_frame, progress_text,
                           (progress_x + 5, progress_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Write to output
                if out:
                    out.write(rendered_frame)
                
                # Display
                cv2.imshow('Violence Detection - Video', rendered_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                filename = f"video_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, rendered_frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord(' '):
                paused = not paused
                print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Print summary
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("DETECTION SUMMARY")
        print("="*60)
        print(f"Frames processed: {frame_count}/{total_frames}")
        print(f"Processing time: {elapsed_time:.2f} seconds")
        print(f"Average FPS: {frame_count/elapsed_time:.2f}")
        print(f"Total detections: {detector.total_detections}")
        print(f"Violence detections: {detector.violence_detections}")
        violence_rate = (detector.violence_detections / max(detector.total_detections, 1)) * 100
        print(f"Violence rate: {violence_rate:.1f}%")
        
        if output_path:
            print(f"\n💾 Output saved: {output_path}")
        print("="*60)


class ViolenceDetectionGUI:
    """Modern GUI for Violence Detection System"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Violence Detection AI - Professional System")
        self.root.geometry("1200x700")
        self.root.resizable(True, True)
        
        # Modern color scheme
        self.bg_color = "#f0f2f5"
        self.card_bg = "#ffffff"
        self.primary_color = "#4a90e2"
        self.success_color = "#5cb85c"
        self.danger_color = "#d9534f"
        self.dark_bg = "#2c3e50"
        
        self.root.configure(bg=self.bg_color)
        
        # Variables
        self.mode = tk.StringVar(value="camera")
        self.video_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.camera_id = tk.StringVar(value="0")
        self.save_output = tk.BooleanVar(value=False)
        self.detection_mode = tk.StringVar(value="none")  # "none", "bbox" or "pose"
        self.confidence_threshold = tk.DoubleVar(value=0.70)
        self.performance_mode = tk.StringVar(value="balanced")
        
        # Model path
        self.model_path = os.path.join(Config.SAVE_MODEL_PATH, "best_resnet_lstm_model.pth")
        
        # Detection state
        self.is_running = False
        self.detection_thread = None
        
        # Setup UI
        self.setup_ui()
        
        # Center window
        self.center_window()
    
    def center_window(self):
        """Center window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def setup_ui(self):
        """Setup modern user interface - 2 column layout with scrollable left panel"""
        # Main container
        container = tk.Frame(self.root, bg=self.bg_color)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # LEFT COLUMN - Settings with Scrollbar (70%)
        left_container = tk.Frame(container, bg=self.bg_color)
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        # Create Canvas and Scrollbar for left column
        left_canvas = tk.Canvas(left_container, bg=self.bg_color, highlightthickness=0)
        left_scrollbar = tk.Scrollbar(left_container, orient=tk.VERTICAL, command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        
        # Scrollable frame for settings
        left_column = tk.Frame(left_canvas, bg=self.bg_color)
        
        # Pack scrollbar and canvas
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create window in canvas
        canvas_window = left_canvas.create_window((0, 0), window=left_column, anchor=tk.NW)
        
        # Update scroll region
        def update_scroll_region(event=None):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
            left_canvas.itemconfig(canvas_window, width=left_canvas.winfo_width())
        
        left_column.bind("<Configure>", update_scroll_region)
        left_canvas.bind("<Configure>", lambda e: left_canvas.itemconfig(canvas_window, width=e.width))
        
        # Mouse wheel scrolling - only when mouse is over canvas
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        def _bind_mousewheel(event):
            left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_mousewheel(event):
            left_canvas.unbind_all("<MouseWheel>")
        
        left_canvas.bind("<Enter>", _bind_mousewheel)
        left_canvas.bind("<Leave>", _unbind_mousewheel)
        
        # RIGHT COLUMN - Action Buttons (30%)
        right_column = tk.Frame(container, bg=self.bg_color, width=300)
        right_column.pack(side=tk.RIGHT, fill=tk.Y)
        right_column.pack_propagate(False)
        
        # Use left_column as main for settings
        main = left_column
        
        # ===== HEADER =====
        header = tk.Frame(main, bg=self.bg_color)
        header.pack(fill=tk.X, pady=(0, 25))
        
        tk.Label(header, text="Violence Detection System", 
                font=("Segoe UI", 28, "bold"), 
                fg=self.dark_bg, bg=self.bg_color).pack(anchor=tk.W)
        
        model_exists = os.path.exists(self.model_path)
        status_text = "● Model Ready" if model_exists else "● Model Not Found"
        status_color = self.success_color if model_exists else self.danger_color
        
        tk.Label(header, text=status_text, 
                font=("Segoe UI", 11), 
                fg=status_color, bg=self.bg_color).pack(anchor=tk.W, pady=(5, 0))
        
        # ===== MODE SELECTION =====
        mode_card = tk.Frame(main, bg=self.card_bg, relief=tk.FLAT)
        mode_card.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(mode_card, text="Source", 
                font=("Segoe UI", 14, "bold"), 
                fg=self.dark_bg, bg=self.card_bg).pack(anchor=tk.W, padx=20, pady=(15, 10))
        
        mode_btns = tk.Frame(mode_card, bg=self.card_bg)
        mode_btns.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        tk.Radiobutton(mode_btns, text="📷 Camera", variable=self.mode, value="camera",
                      command=self.on_mode_change, font=("Segoe UI", 11),
                      bg=self.card_bg, activebackground=self.card_bg).pack(side=tk.LEFT, padx=(0, 30))
        
        tk.Radiobutton(mode_btns, text="🎬 Video File", variable=self.mode, value="video",
                      command=self.on_mode_change, font=("Segoe UI", 11),
                      bg=self.card_bg, activebackground=self.card_bg).pack(side=tk.LEFT)
        
        # ===== SETTINGS =====
        settings_card = tk.Frame(main, bg=self.card_bg, relief=tk.FLAT)
        settings_card.pack(fill=tk.X, pady=(0, 15))
        
        # Camera Settings
        self.camera_settings = tk.Frame(settings_card, bg=self.card_bg)
        self.camera_settings.pack(fill=tk.X, padx=20, pady=15)
        
        tk.Label(self.camera_settings, text="Camera ID:", 
                font=("Segoe UI", 11), fg=self.dark_bg, bg=self.card_bg).pack(anchor=tk.W)
        tk.Entry(self.camera_settings, textvariable=self.camera_id,
                font=("Segoe UI", 11), width=15).pack(anchor=tk.W, pady=(5, 0))
        
        # Video Settings
        self.video_settings = tk.Frame(settings_card, bg=self.card_bg)
        
        tk.Label(self.video_settings, text="Video File:", 
                font=("Segoe UI", 11), fg=self.dark_bg, bg=self.card_bg).pack(anchor=tk.W)
        
        vid_row = tk.Frame(self.video_settings, bg=self.card_bg)
        vid_row.pack(fill=tk.X, pady=(5, 10))
        
        tk.Entry(vid_row, textvariable=self.video_path,
                font=("Segoe UI", 10), width=50).pack(side=tk.LEFT, padx=(0, 10))
        tk.Button(vid_row, text="Browse", command=self.browse_video,
                 font=("Segoe UI", 10), bg=self.primary_color, fg="white",
                 relief=tk.FLAT, padx=15, cursor="hand2").pack(side=tk.LEFT)
        
        tk.Checkbutton(self.video_settings, text="💾 Save output video",
                      variable=self.save_output, command=self.toggle_output,
                      font=("Segoe UI", 10), bg=self.card_bg,
                      activebackground=self.card_bg).pack(anchor=tk.W, pady=(0, 10))
        
        self.output_widgets = tk.Frame(self.video_settings, bg=self.card_bg)
        
        tk.Label(self.output_widgets, text="Output File:", 
                font=("Segoe UI", 10), fg=self.dark_bg, bg=self.card_bg).pack(anchor=tk.W)
        
        out_row = tk.Frame(self.output_widgets, bg=self.card_bg)
        out_row.pack(fill=tk.X, pady=(5, 0))
        
        tk.Entry(out_row, textvariable=self.output_path,
                font=("Segoe UI", 10), width=50).pack(side=tk.LEFT, padx=(0, 10))
        tk.Button(out_row, text="Browse", command=self.browse_output,
                 font=("Segoe UI", 10), bg=self.primary_color, fg="white",
                 relief=tk.FLAT, padx=15, cursor="hand2").pack(side=tk.LEFT)
        
        # ===== PARAMETERS =====
        params_card = tk.Frame(main, bg=self.card_bg, relief=tk.FLAT)
        params_card.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(params_card, text="Detection Parameters", 
                font=("Segoe UI", 14, "bold"), 
                fg=self.dark_bg, bg=self.card_bg).pack(anchor=tk.W, padx=20, pady=(15, 15))
        
        # Visualization mode
        tk.Label(params_card, text="Visualization Mode:",
                font=("Segoe UI", 11), fg=self.dark_bg, bg=self.card_bg).pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        detect_mode_frame = tk.Frame(params_card, bg=self.card_bg)
        detect_mode_frame.pack(fill=tk.X, padx=40, pady=(0, 15))
        
        tk.Radiobutton(detect_mode_frame, text="⭕ None", 
                      variable=self.detection_mode, value="none",
                      font=("Segoe UI", 10), bg=self.card_bg, 
                      activebackground=self.card_bg).pack(side=tk.LEFT, padx=(0, 15))
        
        tk.Radiobutton(detect_mode_frame, text="📦 Bounding Box", 
                      variable=self.detection_mode, value="bbox",
                      font=("Segoe UI", 10), bg=self.card_bg, 
                      activebackground=self.card_bg).pack(side=tk.LEFT, padx=(0, 15))
        
        tk.Radiobutton(detect_mode_frame, text="🦴 Pose Skeleton", 
                      variable=self.detection_mode, value="pose",
                      font=("Segoe UI", 10), bg=self.card_bg, 
                      activebackground=self.card_bg).pack(side=tk.LEFT)
        
        # Confidence
        conf_frame = tk.Frame(params_card, bg=self.card_bg)
        conf_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        conf_header = tk.Frame(conf_frame, bg=self.card_bg)
        conf_header.pack(fill=tk.X)
        
        tk.Label(conf_header, text="Confidence Threshold", 
                font=("Segoe UI", 11, "bold"), fg=self.dark_bg, bg=self.card_bg).pack(side=tk.LEFT)
        
        self.conf_label = tk.Label(conf_header, text=f"{self.confidence_threshold.get():.0%}",
                                   font=("Segoe UI", 16, "bold"), fg=self.primary_color, bg=self.card_bg)
        self.conf_label.pack(side=tk.RIGHT)
        
        self.conf_slider = tk.Scale(conf_frame, from_=0.0, to=1.0, resolution=0.05,
                                    orient=tk.HORIZONTAL, variable=self.confidence_threshold,
                                    command=self.on_confidence_change, length=600,
                                    bg=self.card_bg, highlightthickness=0, showvalue=0)
        self.conf_slider.pack(fill=tk.X, pady=(10, 0))
        
        # Performance
        perf_frame = tk.Frame(params_card, bg=self.card_bg)
        perf_frame.pack(fill=tk.X, padx=20, pady=(15, 15))
        
        tk.Label(perf_frame, text="Performance Mode", 
                font=("Segoe UI", 11, "bold"), fg=self.dark_bg, bg=self.card_bg).pack(anchor=tk.W, pady=(0, 10))
        
        perf_btns = tk.Frame(perf_frame, bg=self.card_bg)
        perf_btns.pack(fill=tk.X)
        
        tk.Radiobutton(perf_btns, text="🚀 Fast", variable=self.performance_mode, value="fast",
                      font=("Segoe UI", 10), bg=self.card_bg, activebackground=self.card_bg).pack(side=tk.LEFT, padx=(0, 20))
        tk.Radiobutton(perf_btns, text="⚖️ Balanced", variable=self.performance_mode, value="balanced",
                      font=("Segoe UI", 10), bg=self.card_bg, activebackground=self.card_bg).pack(side=tk.LEFT, padx=(0, 20))
        tk.Radiobutton(perf_btns, text="🎯 Accurate", variable=self.performance_mode, value="accurate",
                      font=("Segoe UI", 10), bg=self.card_bg, activebackground=self.card_bg).pack(side=tk.LEFT)
        
        # ===== RIGHT COLUMN - ACTION BUTTONS =====
        # Title for right panel
        tk.Label(right_column, text="Actions", 
                font=("Segoe UI", 16, "bold"), 
                fg=self.dark_bg, bg=self.bg_color).pack(pady=(0, 20))
        
        # START button
        self.start_btn = tk.Button(right_column, text="▶ START\nDETECTION",
                                   command=self.start_detection,
                                   font=("Segoe UI", 14, "bold"),
                                   bg=self.success_color, fg="white",
                                   relief=tk.FLAT, padx=20, pady=25,
                                   cursor="hand2", width=15, height=3)
        self.start_btn.pack(pady=(0, 15), padx=10)
        
        # STOP button
        self.stop_btn = tk.Button(right_column, text="■ STOP",
                                  command=self.stop_detection,
                                  font=("Segoe UI", 14, "bold"),
                                  bg="#6c757d", fg="white",
                                  relief=tk.FLAT, padx=20, pady=25,
                                  cursor="hand2", state=tk.DISABLED, width=15, height=2)
        self.stop_btn.pack(pady=(0, 30), padx=10)
        
        # Shortcuts card
        shortcuts_card = tk.Frame(right_column, bg=self.card_bg, relief=tk.FLAT)
        shortcuts_card.pack(fill=tk.BOTH, expand=True, padx=10)
        
        tk.Label(shortcuts_card, text="Keyboard Shortcuts", 
                font=("Segoe UI", 11, "bold"), 
                fg=self.dark_bg, bg=self.card_bg).pack(pady=(15, 10))
        
        shortcuts = [
            ("Q", "Quit"),
            ("S", "Screenshot"),
            ("P", "Toggle Detection"),
            ("R", "Reset Stats"),
            ("SPACE", "Pause/Resume")
        ]
        
        for key, desc in shortcuts:
            s_frame = tk.Frame(shortcuts_card, bg=self.card_bg)
            s_frame.pack(fill=tk.X, padx=15, pady=3)
            
            tk.Label(s_frame, text=key, 
                    font=("Courier", 9, "bold"), 
                    fg="white", bg=self.primary_color,
                    width=8).pack(side=tk.LEFT, padx=(0, 10))
            tk.Label(s_frame, text=desc, 
                    font=("Segoe UI", 9), 
                    fg=self.dark_bg, bg=self.card_bg,
                    anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(shortcuts_card, text="", bg=self.card_bg).pack(pady=10)
        
        # Initial setup
        self.on_mode_change()
    
    def on_mode_change(self):
        """Handle mode change"""
        if self.mode.get() == "camera":
            self.camera_settings.pack(fill=tk.X, padx=20, pady=15)
            self.video_settings.pack_forget()
        else:
            self.camera_settings.pack_forget()
            self.video_settings.pack(fill=tk.X, padx=20, pady=15)
    
    def on_confidence_change(self, value):
        """Handle confidence change"""
        self.conf_label.config(text=f"{float(value):.0%}")
    
    def toggle_output(self):
        """Toggle output widgets"""
        if self.save_output.get():
            self.output_widgets.pack(fill=tk.X, pady=(10, 0))
        else:
            self.output_widgets.pack_forget()
    
    def browse_video(self):
        """Browse video file"""
        filename = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.video_path.set(filename)
            if not self.output_path.get():
                base = os.path.splitext(filename)[0]
                self.output_path.set(f"{base}_detected.mp4")
    
    def browse_output(self):
        """Browse output file"""
        filename = filedialog.asksaveasfilename(
            title="Save Output",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi")],
            defaultextension=".mp4"
        )
        if filename:
            self.output_path.set(filename)
    
    def start_detection(self):
        """Start detection"""
        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", "Model not found! Train model first.")
            return
        
        if self.mode.get() == "camera":
            try:
                int(self.camera_id.get())
            except:
                messagebox.showerror("Error", "Invalid camera ID!")
                return
        else:
            if not self.video_path.get() or not os.path.exists(self.video_path.get()):
                messagebox.showerror("Error", "Invalid video file!")
                return
            if self.save_output.get() and not self.output_path.get():
                messagebox.showerror("Error", "Specify output path!")
                return
        
        self.start_btn.config(state=tk.DISABLED, bg="#95a5a6")
        self.stop_btn.config(state=tk.NORMAL, bg=self.danger_color)
        self.root.withdraw()
        
        # Determine detection mode
        detection_mode = self.detection_mode.get()  # "none", "bbox", or "pose"
        
        # Always enable person detection (for counting), just control visualization
        enable_person_detection = True
        use_pose_mode = detection_mode == "pose"
        draw_mode = detection_mode  # "none", "bbox", or "pose"
        
        if self.mode.get() == "camera":
            self.detection_thread = threading.Thread(
                target=lambda: run_camera_detection(
                    self.model_path, int(self.camera_id.get()),
                    self.confidence_threshold.get(), self.performance_mode.get(),
                    use_pose_mode, enable_person_detection, draw_mode
                ), daemon=True
            )
        else:
            self.detection_thread = threading.Thread(
                target=lambda: run_video_detection(
                    self.model_path, self.video_path.get(),
                    self.output_path.get() if self.save_output.get() else None,
                    self.confidence_threshold.get(), self.performance_mode.get(),
                    use_pose_mode, enable_person_detection, draw_mode
                ), daemon=True
            )
        
        self.detection_thread.start()
        self.root.after(100, self.check_detection)
    
    def check_detection(self):
        """Check if detection is still running"""
        if self.detection_thread.is_alive():
            self.root.after(100, self.check_detection)
        else:
            self.detection_finished()
    
    def stop_detection(self):
        """Stop detection"""
        cv2.destroyAllWindows()
        self.detection_finished()
    
    def detection_finished(self):
        """Detection finished"""
        self.start_btn.config(state=tk.NORMAL, bg=self.success_color)
        self.stop_btn.config(state=tk.DISABLED, bg="#6c757d")
        self.root.deiconify()
    
    def run(self):
        """Run GUI"""
        self.root.mainloop()


def main():
    """Main function"""
    app = ViolenceDetectionGUI()
    app.run()


if __name__ == "__main__":
    main()
