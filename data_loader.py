import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
from config import Config

class ViolenceDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, sequence_length=16):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract frames from video
        frames = self.extract_frames(video_path)
        
        if self.transform:
            # Apply transform to each frame
            transformed_frames = []
            for frame in frames:
                transformed_frame = self.transform(image=frame)['image']
                transformed_frames.append(transformed_frame)
            frames = np.array(transformed_frames)
        
        # Convert to float and normalize
        frames = frames.astype(np.float32) / 255.0
        
        return frames, torch.tensor(label, dtype=torch.long)
    
    def extract_frames(self, video_path):
        """Extract frames from video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to sample
        if total_frames <= self.sequence_length:
            # If video is shorter than sequence length, repeat frames
            frame_indices = list(range(total_frames))
            while len(frame_indices) < self.sequence_length:
                frame_indices.extend(list(range(total_frames)))
            frame_indices = frame_indices[:self.sequence_length]
        else:
            # Sample frames evenly across the video
            frame_indices = np.linspace(0, total_frames-1, self.sequence_length, dtype=int)
        
        # Extract frames
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, Config.FRAME_SIZE)
                frames.append(frame)
            else:
                # If frame reading fails, use the last successful frame
                if frames:
                    frames.append(frames[-1])
                else:
                    # Create a black frame if no frames were read
                    frames.append(np.zeros((Config.FRAME_SIZE[0], Config.FRAME_SIZE[1], 3), dtype=np.uint8))
        
        cap.release()
        return np.array(frames)

def get_data_transforms():
    """Get data augmentation transforms"""
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomGamma(p=0.3),
        A.GaussNoise(p=0.2),
        A.Blur(p=0.2),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        ToTensorV2()
    ])
    
    return train_transform, val_transform

def load_dataset():
    """Load and split dataset"""
    print("Loading dataset...")
    
    # Get all video files
    violence_files = []
    non_violence_files = []
    
    # Violence videos
    for file in os.listdir(Config.VIOLENCE_DIR):
        if file.endswith(('.mp4', '.avi')):
            violence_files.append(os.path.join(Config.VIOLENCE_DIR, file))
    
    # Non-violence videos
    for file in os.listdir(Config.NON_VIOLENCE_DIR):
        if file.endswith(('.mp4', '.avi')):
            non_violence_files.append(os.path.join(Config.NON_VIOLENCE_DIR, file))
    
    print(f"Found {len(violence_files)} violence videos")
    print(f"Found {len(non_violence_files)} non-violence videos")
    
    # Create labels (0: non-violence, 1: violence)
    violence_labels = [1] * len(violence_files)
    non_violence_labels = [0] * len(non_violence_files)
    
    # Combine all data
    all_files = violence_files + non_violence_files
    all_labels = violence_labels + non_violence_labels
    
    # Shuffle data
    combined = list(zip(all_files, all_labels))
    random.shuffle(combined)
    all_files, all_labels = zip(*combined)
    
    # Split dataset
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels, 
        test_size=1-Config.TRAIN_RATIO, 
        random_state=42, 
        stratify=all_labels
    )
    
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels,
        test_size=Config.TEST_RATIO/(Config.VAL_RATIO + Config.TEST_RATIO),
        random_state=42,
        stratify=temp_labels
    )
    
    print(f"Train: {len(train_files)} videos")
    print(f"Validation: {len(val_files)} videos")
    print(f"Test: {len(test_files)} videos")
    
    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)

def create_data_loaders():
    """Create data loaders for training"""
    # Load dataset
    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = load_dataset()
    
    # Get transforms
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets
    train_dataset = ViolenceDataset(train_files, train_labels, train_transform, Config.SEQUENCE_LENGTH)
    val_dataset = ViolenceDataset(val_files, val_labels, val_transform, Config.SEQUENCE_LENGTH)
    test_dataset = ViolenceDataset(test_files, test_labels, val_transform, Config.SEQUENCE_LENGTH)
    
    # Create data loaders (num_workers=0 for Windows compatibility)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader, test_loader = create_data_loaders()
    
    # Test a batch
    for frames, labels in train_loader:
        print(f"Batch shape: {frames.shape}")
        print(f"Labels: {labels}")
        break
