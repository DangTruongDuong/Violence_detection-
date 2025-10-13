import os
import torch

class Config:
    # Dataset paths
    DATASET_ROOT = "Dataset"
    VIOLENCE_DIR = os.path.join(DATASET_ROOT, "Violence")
    NON_VIOLENCE_DIR = os.path.join(DATASET_ROOT, "NonViolence")
    
    # Model parameters
    FRAME_SIZE = (224, 224)
    SEQUENCE_LENGTH = 16  # Number of frames per sequence
    BATCH_SIZE = 16  # Increased for GPU (RTX 3050 has 4GB VRAM)
    LEARNING_RATE = 0.0001  # Giảm từ 0.001 xuống 0.0001
    EPOCHS = 50
    
    # Data split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Model architecture
    HIDDEN_SIZE = 128
    NUM_CLASSES = 2
    
    # Training
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_MODEL_PATH = "models"
    LOG_DIR = "logs"
    
    # Realtime detection
    CONFIDENCE_THRESHOLD = 0.7
    DETECTION_INTERVAL = 1.0  # seconds
    
    # Video processing
    FPS = 30
    MAX_FRAMES_PER_VIDEO = 300  # Limit frames for training efficiency
