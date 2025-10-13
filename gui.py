import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading
import time
from PIL import Image, ImageTk
import numpy as np
import os
from config import Config
from realtime_detector import RealtimeViolenceDetector

class ViolenceDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống phát hiện bạo lực realtime")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Detection variables
        self.detector = None
        self.cap = None
        self.detection_running = False
        self.current_frame = None
        
        # GUI variables
        self.model_path_var = tk.StringVar()
        self.confidence_threshold_var = tk.DoubleVar(value=Config.CONFIDENCE_THRESHOLD)
        self.model_type_var = tk.StringVar(value="resnet_lstm")
        
        # Statistics variables
        self.stats_vars = {
            'prediction': tk.StringVar(value="Chưa khởi động"),
            'confidence': tk.StringVar(value="0.00"),
            'fps': tk.StringVar(value="0.0"),
            'violence_percentage': tk.StringVar(value="0.0%"),
            'total_detections': tk.StringVar(value="0")
        }
        
        self.setup_gui()
        self.setup_styles()
        
    def setup_styles(self):
        """Setup custom styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#2c3e50', foreground='white')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#34495e', foreground='white')
        style.configure('Info.TLabel', font=('Arial', 10), background='#34495e', foreground='white')
        style.configure('Success.TLabel', font=('Arial', 10, 'bold'), background='#27ae60', foreground='white')
        style.configure('Warning.TLabel', font=('Arial', 10, 'bold'), background='#f39c12', foreground='white')
        style.configure('Danger.TLabel', font=('Arial', 10, 'bold'), background='#e74c3c', foreground='white')
        
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="HỆ THỐNG PHÁT HIỆN BẠO LỰC REALTIME", style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Detection tab
        self.setup_detection_tab(notebook)
        
        # Settings tab
        self.setup_settings_tab(notebook)
        
        # Statistics tab
        self.setup_statistics_tab(notebook)
        
    def setup_detection_tab(self, parent):
        """Setup detection tab"""
        detection_frame = ttk.Frame(parent)
        parent.add(detection_frame, text="Phát hiện")
        
        # Left panel - Video display
        left_panel = ttk.Frame(detection_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video display
        video_label = ttk.Label(left_panel, text="Video sẽ hiển thị ở đây", font=('Arial', 14))
        video_label.pack(fill=tk.BOTH, expand=True)
        self.video_label = video_label
        
        # Control buttons
        control_frame = ttk.Frame(left_panel)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_button = ttk.Button(control_frame, text="Bắt đầu", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Dừng", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.screenshot_button = ttk.Button(control_frame, text="Chụp ảnh", command=self.take_screenshot, state=tk.DISABLED)
        self.screenshot_button.pack(side=tk.LEFT)
        
        # Right panel - Status and controls
        right_panel = ttk.Frame(detection_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Status frame
        status_frame = ttk.LabelFrame(right_panel, text="Trạng thái", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Status indicators
        self.status_label = ttk.Label(status_frame, text="Chưa khởi động", style='Info.TLabel')
        self.status_label.pack(anchor=tk.W)
        
        # Model info
        model_frame = ttk.LabelFrame(right_panel, text="Thông tin model", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text="Model:").pack(anchor=tk.W)
        self.model_info_label = ttk.Label(model_frame, text="Chưa tải", style='Info.TLabel')
        self.model_info_label.pack(anchor=tk.W)
        
        ttk.Label(model_frame, text="Độ tin cậy:").pack(anchor=tk.W, pady=(5, 0))
        self.confidence_label = ttk.Label(model_frame, text="0.00", style='Info.TLabel')
        self.confidence_label.pack(anchor=tk.W)
        
        # Detection results
        results_frame = ttk.LabelFrame(right_panel, text="Kết quả phát hiện", padding=10)
        results_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(results_frame, text="Dự đoán:").pack(anchor=tk.W)
        self.prediction_label = ttk.Label(results_frame, text="Chưa có", style='Info.TLabel')
        self.prediction_label.pack(anchor=tk.W)
        
        ttk.Label(results_frame, text="FPS:").pack(anchor=tk.W, pady=(5, 0))
        self.fps_label = ttk.Label(results_frame, text="0.0", style='Info.TLabel')
        self.fps_label.pack(anchor=tk.W)
        
        # Alert frame
        alert_frame = ttk.LabelFrame(right_panel, text="Cảnh báo", padding=10)
        alert_frame.pack(fill=tk.X)
        
        self.alert_label = ttk.Label(alert_frame, text="Hệ thống sẵn sàng", style='Success.TLabel')
        self.alert_label.pack(anchor=tk.W)
        
    def setup_settings_tab(self, parent):
        """Setup settings tab"""
        settings_frame = ttk.Frame(parent)
        parent.add(settings_frame, text="Cài đặt")
        
        # Model settings
        model_frame = ttk.LabelFrame(settings_frame, text="Cài đặt Model", padding=20)
        model_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Model path
        ttk.Label(model_frame, text="Đường dẫn model:").pack(anchor=tk.W)
        path_frame = ttk.Frame(model_frame)
        path_frame.pack(fill=tk.X, pady=(5, 10))
        
        ttk.Entry(path_frame, textvariable=self.model_path_var, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_frame, text="Chọn file", command=self.browse_model).pack(side=tk.RIGHT, padx=(10, 0))
        
        # Model type
        ttk.Label(model_frame, text="Loại model:").pack(anchor=tk.W, pady=(10, 0))
        model_type_combo = ttk.Combobox(model_frame, textvariable=self.model_type_var, 
                                       values=["resnet_lstm", "convlstm3d", "efficientnet_lstm"], 
                                       state="readonly")
        model_type_combo.pack(anchor=tk.W, pady=(5, 10))
        
        # Confidence threshold
        ttk.Label(model_frame, text="Ngưỡng độ tin cậy:").pack(anchor=tk.W, pady=(10, 0))
        confidence_scale = ttk.Scale(model_frame, from_=0.1, to=1.0, variable=self.confidence_threshold_var, 
                                   orient=tk.HORIZONTAL)
        confidence_scale.pack(fill=tk.X, pady=(5, 0))
        
        confidence_value_label = ttk.Label(model_frame, textvariable=self.confidence_threshold_var)
        confidence_value_label.pack(anchor=tk.W)
        
        # Detection settings
        detection_frame = ttk.LabelFrame(settings_frame, text="Cài đặt phát hiện", padding=20)
        detection_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Camera selection
        ttk.Label(detection_frame, text="Camera:").pack(anchor=tk.W)
        self.camera_var = tk.StringVar(value="0")
        camera_combo = ttk.Combobox(detection_frame, textvariable=self.camera_var, 
                                   values=["0", "1", "2"], state="readonly")
        camera_combo.pack(anchor=tk.W, pady=(5, 10))
        
        # Detection interval
        ttk.Label(detection_frame, text="Khoảng thời gian phát hiện (giây):").pack(anchor=tk.W, pady=(10, 0))
        self.detection_interval_var = tk.DoubleVar(value=Config.DETECTION_INTERVAL)
        interval_scale = ttk.Scale(detection_frame, from_=0.1, to=5.0, variable=self.detection_interval_var, 
                                 orient=tk.HORIZONTAL)
        interval_scale.pack(fill=tk.X, pady=(5, 0))
        
        interval_value_label = ttk.Label(detection_frame, textvariable=self.detection_interval_var)
        interval_value_label.pack(anchor=tk.W)
        
        # Save settings button
        ttk.Button(settings_frame, text="Lưu cài đặt", command=self.save_settings).pack(pady=20)
        
    def setup_statistics_tab(self, parent):
        """Setup statistics tab"""
        stats_frame = ttk.Frame(parent)
        parent.add(stats_frame, text="Thống kê")
        
        # Real-time statistics
        realtime_frame = ttk.LabelFrame(stats_frame, text="Thống kê realtime", padding=20)
        realtime_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Create grid for statistics
        stats_grid = ttk.Frame(realtime_frame)
        stats_grid.pack(fill=tk.X)
        
        # Row 1
        ttk.Label(stats_grid, text="Dự đoán hiện tại:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Label(stats_grid, textvariable=self.stats_vars['prediction'], style='Info.TLabel').grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(stats_grid, text="Độ tin cậy:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        ttk.Label(stats_grid, textvariable=self.stats_vars['confidence'], style='Info.TLabel').grid(row=0, column=3, sticky=tk.W)
        
        # Row 2
        ttk.Label(stats_grid, text="FPS:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        ttk.Label(stats_grid, textvariable=self.stats_vars['fps'], style='Info.TLabel').grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        ttk.Label(stats_grid, text="% Bạo lực:").grid(row=1, column=2, sticky=tk.W, padx=(20, 10), pady=(10, 0))
        ttk.Label(stats_grid, textvariable=self.stats_vars['violence_percentage'], style='Info.TLabel').grid(row=1, column=3, sticky=tk.W, pady=(10, 0))
        
        # Row 3
        ttk.Label(stats_grid, text="Tổng phát hiện:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        ttk.Label(stats_grid, textvariable=self.stats_vars['total_detections'], style='Info.TLabel').grid(row=2, column=1, sticky=tk.W, pady=(10, 0))
        
        # Export statistics button
        ttk.Button(realtime_frame, text="Xuất thống kê", command=self.export_statistics).pack(pady=(20, 0))
        
        # Detection history plot (placeholder)
        history_frame = ttk.LabelFrame(stats_frame, text="Lịch sử phát hiện", padding=20)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        ttk.Label(history_frame, text="Biểu đồ lịch sử phát hiện sẽ được hiển thị ở đây", 
                 font=('Arial', 12)).pack(expand=True)
        
    def browse_model(self):
        """Browse for model file"""
        filename = filedialog.askopenfilename(
            title="Chọn file model",
            filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
            
    def save_settings(self):
        """Save current settings"""
        # Update detector settings if it exists
        if self.detector:
            self.detector.confidence_threshold = self.confidence_threshold_var.get()
        
        messagebox.showinfo("Thành công", "Cài đặt đã được lưu!")
        
    def start_detection(self):
        """Start realtime detection"""
        if not self.model_path_var.get():
            messagebox.showerror("Lỗi", "Vui lòng chọn file model!")
            return
        
        if not os.path.exists(self.model_path_var.get()):
            messagebox.showerror("Lỗi", "File model không tồn tại!")
            return
        
        try:
            # Initialize detector
            self.detector = RealtimeViolenceDetector(
                self.model_path_var.get(), 
                self.model_type_var.get()
            )
            self.detector.confidence_threshold = self.confidence_threshold_var.get()
            
            # Initialize camera
            camera_id = int(self.camera_var.get())
            self.cap = cv2.VideoCapture(camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", "Không thể mở camera!")
                return
            
            # Start detection
            self.detector.start_detection()
            self.detection_running = True
            
            # Update UI
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.screenshot_button.config(state=tk.NORMAL)
            self.status_label.config(text="Đang chạy", style='Success.TLabel')
            self.model_info_label.config(text=f"{self.model_type_var.get()}")
            
            # Start video update thread
            self.video_thread = threading.Thread(target=self.update_video)
            self.video_thread.daemon = True
            self.video_thread.start()
            
            # Start statistics update thread
            self.stats_thread = threading.Thread(target=self.update_statistics)
            self.stats_thread.daemon = True
            self.stats_thread.start()
            
            messagebox.showinfo("Thành công", "Phát hiện đã được khởi động!")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể khởi động phát hiện: {str(e)}")
            
    def stop_detection(self):
        """Stop realtime detection"""
        self.detection_running = False
        
        if self.detector:
            self.detector.stop_detection()
        
        if self.cap:
            self.cap.release()
        
        # Update UI
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.screenshot_button.config(state=tk.DISABLED)
        self.status_label.config(text="Đã dừng", style='Warning.TLabel')
        
        # Clear video display
        self.video_label.config(image='', text="Video đã dừng")
        
        messagebox.showinfo("Thông báo", "Phát hiện đã được dừng!")
        
    def update_video(self):
        """Update video display"""
        while self.detection_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Add frame to detector
            if self.detector:
                self.detector.add_frame(frame)
                frame = self.detector.draw_detection_overlay(frame)
            
            # Convert frame for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            # Update display
            self.video_label.config(image=frame_tk, text='')
            self.video_label.image = frame_tk  # Keep a reference
            
            time.sleep(1/30)  # 30 FPS
            
    def update_statistics(self):
        """Update statistics display"""
        while self.detection_running and self.detector:
            stats = self.detector.get_detection_stats()
            
            # Update GUI variables
            self.stats_vars['prediction'].set(stats['current_prediction'])
            self.stats_vars['confidence'].set(f"{stats['current_confidence']:.2f}")
            self.stats_vars['fps'].set(f"{stats['fps']:.1f}")
            self.stats_vars['violence_percentage'].set(f"{stats['violence_percentage']:.1f}%")
            self.stats_vars['total_detections'].set(str(stats['total_detections']))
            
            # Update labels
            self.prediction_label.config(text=stats['current_prediction'])
            self.confidence_label.config(text=f"{stats['current_confidence']:.2f}")
            self.fps_label.config(text=f"{stats['fps']:.1f}")
            
            # Update alert
            if stats['current_prediction'] == "Violence":
                self.alert_label.config(text="⚠️ CẢNH BÁO: Phát hiện bạo lực!", style='Danger.TLabel')
            elif stats['current_prediction'] == "Non-Violence":
                self.alert_label.config(text="✅ An toàn", style='Success.TLabel')
            else:
                self.alert_label.config(text="❓ Không chắc chắn", style='Warning.TLabel')
            
            time.sleep(1)  # Update every second
            
    def take_screenshot(self):
        """Take screenshot of current frame"""
        if self.current_frame is not None:
            timestamp = int(time.time())
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            messagebox.showinfo("Thành công", f"Ảnh đã được lưu: {filename}")
        else:
            messagebox.showwarning("Cảnh báo", "Không có frame nào để chụp!")
            
    def export_statistics(self):
        """Export detection statistics"""
        if not self.detector or not self.detector.detection_history:
            messagebox.showwarning("Cảnh báo", "Không có dữ liệu thống kê!")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Lưu thống kê",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("THỐNG KÊ PHÁT HIỆN BẠO LỰC\n")
                f.write("=" * 50 + "\n\n")
                
                stats = self.detector.get_detection_stats()
                f.write(f"Dự đoán hiện tại: {stats['current_prediction']}\n")
                f.write(f"Độ tin cậy: {stats['current_confidence']:.2f}\n")
                f.write(f"FPS: {stats['fps']:.1f}\n")
                f.write(f"% Bạo lực: {stats['violence_percentage']:.1f}%\n")
                f.write(f"Tổng phát hiện: {stats['total_detections']}\n\n")
                
                f.write("LỊCH SỬ PHÁT HIỆN:\n")
                f.write("-" * 30 + "\n")
                for i, detection in enumerate(self.detector.detection_history):
                    f.write(f"{i+1}. {detection['timestamp']}: {detection['prediction']} ({detection['confidence']:.2f})\n")
            
            messagebox.showinfo("Thành công", f"Thống kê đã được xuất: {filename}")

def main():
    """Main function"""
    root = tk.Tk()
    app = ViolenceDetectionGUI(root)
    
    # Set default model path if exists
    default_model = os.path.join(Config.SAVE_MODEL_PATH, "best_resnet_lstm_model.pth")
    if os.path.exists(default_model):
        app.model_path_var.set(default_model)
    
    root.mainloop()

if __name__ == "__main__":
    main()



