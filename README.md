# Violence Detection System

Hệ thống phát hiện bạo lực sử dụng deep learning với kiến trúc hybrid CNN-LSTM, có khả năng nhận diện bạo lực trong video real-time và phân tích file video.

## 🎯 Tính năng chính

- **Phát hiện bạo lực real-time** từ webcam
- **Phân tích video file** với batch processing
- **Pose estimation** với MediaPipe (33 landmarks)
- **Nhiều kiến trúc model** (ResNet18+LSTM, 3D CNN, EfficientNet+LSTM)
- **Transfer learning** với pre-trained models
- **Giao diện GUI** thân thiện
- **Đánh giá toàn diện** với metrics và visualizations

## 🏗️ Kiến trúc Model

Hệ thống sử dụng hybrid approach kết hợp:
- **CNN backbone** (ResNet18) để trích xuất spatial features
- **LSTM layers** để mô hình hóa temporal sequence
- **Bidirectional LSTM** để hiểu context tốt hơn
- **Pose estimation** với MediaPipe để phân tích chuyển động

## 📊 Hiệu suất Model

### Model chính (ResNet18 + LSTM):
- **Validation Accuracy**: **99.67%** 🏆
- **Test Accuracy**: **97.67%**
- **Precision**: **97.78%**
- **Recall**: **97.67%**
- **F1-Score**: **97.67%**

### So sánh với các model khác:
- **ResNet18+LSTM**: 97.67% (Best)
- **3D CNN**: ~95% (ước tính)
- **EfficientNet+LSTM**: ~96% (ước tính)

## 📁 Cấu trúc File và Công dụng

### 🧠 Core Files (Lõi hệ thống)
```
config.py              # ⚙️ Cấu hình toàn bộ hệ thống
├── Model parameters (batch size, learning rate, epochs)
├── Dataset paths và data split ratios
├── Training settings và device configuration
└── Detection thresholds và video processing settings

data_loader.py         # 📊 Xử lý và load dữ liệu
├── Video preprocessing và frame extraction
├── Data augmentation và normalization
├── Train/validation/test split
└── Custom dataset class cho PyTorch

model.py              # 🏗️ Định nghĩa các kiến trúc model
├── ViolenceDetectionModel (ResNet18 + LSTM) - 97.67%
├── ConvLSTM3D (3D CNN) - ~95%
├── EfficientNetLSTM (EfficientNet + LSTM) - ~96%
└── Model creation và parameter counting
```

### 🚀 Training & Evaluation
```
train.py              # 🎓 Script training chính
├── Trainer class với early stopping
├── Training loop với validation
├── Model checkpointing và best model saving
├── TensorBoard logging
├── Confusion matrix và training history plots
└── Final test evaluation

demo.py               # 🎬 Demo script đơn giản
├── Load trained model
├── Test trên sample video
└── Hiển thị kết quả cơ bản
```

### 📹 Detection & Analysis
```
realtime_detector.py  # 📷 Real-time detection từ webcam
├── Webcam capture và frame processing
├── Multi-threading cho performance
├── Real-time visualization
├── Detection statistics và FPS counter
└── Interactive controls (q=quit, s=screenshot)

video_test.py         # 🎥 Phân tích video file với GUI
├── File browser để chọn video
├── Output video generation
├── Progress tracking
├── Detection summary report
└── Screenshot và pause/resume controls

pose_realtime_detector.py  # 🦴 Real-time với pose estimation
├── MediaPipe pose detection (33 landmarks)
├── Movement analysis cho violence detection
├── Pose landmarks và connections overlay
├── Enhanced temporal smoothing
└── Interactive pose controls (p=toggle, c=clear)

pose_video_test.py    # 🎬 Video test với pose + GUI
├── GUI với advanced parameters
├── Pose analysis toggle
├── Confidence threshold adjustment
├── Violence count threshold setting
└── Complete pose-based video analysis
```

### 🖥️ User Interface
```
gui.py                # 🖱️ Giao diện GUI chính
├── Main application window
├── Video selection và model loading
├── Real-time detection controls
├── Settings và parameter adjustment
└── Results visualization

main.py               # 🚪 Entry point chính
├── Command line interface
├── Model selection
├── Input/output handling
└── System initialization
```

## 🛠️ Cài đặt và Sử dụng

### 1. Tạo và kích hoạt môi trường ảo

```bash
# Tạo môi trường ảo
python -m venv violence_detection_env

# Kích hoạt môi trường ảo
# Windows:
violence_detection_env\Scripts\activate

# Linux/Mac:
source violence_detection_env/bin/activate
```

### 2. Cài đặt dependencies

```bash
# Cài đặt PyTorch với CUDA support (cho GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Cài đặt các package khác
pip install opencv-python numpy matplotlib seaborn scikit-learn tqdm tensorboard

# Cài đặt MediaPipe cho pose estimation
pip install mediapipe

# Hoặc cài đặt từ requirements.txt
pip install -r requirements.txt
```

### 3. Chuẩn bị dữ liệu

```
Dataset/
├── Violence/          # 1000 video bạo lực
│   ├── V_1.mp4
│   ├── V_2.mp4
│   └── ...
└── NonViolence/       # 1000 video không bạo lực
    ├── NV_1.mp4
    ├── NV_2.mp4
    └── ...
```

## 🚀 Cách chạy từng file

### 🎓 Training Model
```bash
# Chạy training với GPU (khuyến nghị)
python train.py

# Kết quả: Model được lưu trong models/best_resnet_lstm_model.pth
# Thời gian: ~6-7 giờ với GPU RTX 3050
# Accuracy: 99.67% validation, 97.67% test
```

### 📷 Real-time Detection
```bash
# Real-time từ webcam (cơ bản)
python realtime_detector.py

# Real-time với pose estimation (nâng cao)
python realtime_detector.py

# Điều khiển:
# 'q' = Thoát
# 's' = Lưu screenshot
# 'p' = Pause/Resume (pose version)
# 'c' = Clear history (pose version)
```

### 🎥 Video Analysis
```bash
# Phân tích video file với GUI
python video_test.py

# Phân tích video với pose estimation + GUI
python pose_video_test.py

# Demo đơn giản
python demo.py
```

### 🖥️ GUI Interface
```bash
# Giao diện GUI chính
python gui.py

# Entry point
python main.py
```

## ⚙️ Cấu hình

### config.py - Các thông số quan trọng:
```python
# Model parameters
BATCH_SIZE = 16          # Tăng cho GPU
LEARNING_RATE = 0.0001   # Giảm để tránh overfitting
EPOCHS = 50             # Số epoch training
SEQUENCE_LENGTH = 16    # Số frame mỗi sequence

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.7  # Ngưỡng confidence
DETECTION_INTERVAL = 1.0    # Khoảng thời gian detection

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## 📊 So sánh hiệu suất

| Model | Validation Acc | Test Acc | Precision | Recall | F1-Score |
|-------|----------------|----------|-----------|--------|----------|
| **ResNet18+LSTM** | **99.67%** | **97.67%** | **97.78%** | **97.67%** | **97.67%** |
| 3D CNN | ~95% | ~94% | ~94% | ~94% | ~94% |
| EfficientNet+LSTM | ~96% | ~95% | ~95% | ~95% | ~95% |

## 🎯 Tính năng nổi bật

### Pose-based Detection:
- **33 landmarks** từ MediaPipe
- **Movement analysis** cho violence detection
- **Temporal smoothing** với 8 predictions
- **Enhanced accuracy** với pose information

### Real-time Performance:
- **GPU acceleration** với CUDA
- **Multi-threading** cho smooth performance
- **FPS counter** và performance metrics
- **Interactive controls** linh hoạt

### Advanced Features:
- **Early stopping** để tránh overfitting
- **Model checkpointing** và best model saving
- **TensorBoard logging** cho monitoring
- **Comprehensive evaluation** với multiple metrics

## 🔧 Troubleshooting

### Lỗi thường gặp:
1. **CUDA out of memory**: Giảm BATCH_SIZE trong config.py
2. **Model not found**: Chạy train.py trước
3. **Video not found**: Kiểm tra đường dẫn file
4. **Import error**: Cài đặt đầy đủ dependencies

### Tối ưu hóa:
- **GPU**: Sử dụng CUDA để tăng tốc 10-50x
- **Memory**: Điều chỉnh BATCH_SIZE phù hợp
- **Performance**: Sử dụng pose version cho độ chính xác cao hơn

## 📈 Kết quả thực tế

- **Training time**: 6-7 giờ (GPU) vs 100+ giờ (CPU)
- **Real-time FPS**: 15-30 FPS tùy hardware
- **Detection accuracy**: 97.67% trên test set
- **False positive rate**: <3%
- **Model size**: ~97MB (ResNet18+LSTM)

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Submit pull request

## 📄 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

## 🙏 Acknowledgments

- **PyTorch team** cho deep learning framework
- **OpenCV** cho computer vision capabilities  
- **MediaPipe** cho pose estimation
- **Research community** cho violence detection datasets