import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config import Config

class ViolenceDetectionModel(nn.Module):
    def __init__(self, num_classes=2, sequence_length=16, hidden_size=128):
        super(ViolenceDetectionModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        # Use pre-trained ResNet18 as feature extractor
        self.feature_extractor = models.resnet18(pretrained=True)
        
        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        
        # Freeze early layers for transfer learning
        for param in list(self.feature_extractor.parameters())[:-10]:
            param.requires_grad = False
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=512,  # ResNet18 output features
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.7),  # Tăng từ 0.5 lên 0.7
            nn.Linear(hidden_size * 2, 256),  # *2 for bidirectional LSTM
            nn.ReLU(),
            nn.Dropout(0.5),  # Tăng từ 0.3 lên 0.5
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),  # Tăng từ 0.2 lên 0.4
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape for feature extraction: (batch_size * seq_len, channels, height, width)
        x = x.view(-1, channels, height, width)
        
        # Extract features for each frame
        features = self.feature_extractor(x)  # (batch_size * seq_len, 512, 1, 1)
        features = features.view(batch_size, seq_len, -1)  # (batch_size, seq_len, 512)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # Use the last output from LSTM
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # Classification
        output = self.classifier(last_output)
        
        return output

class ConvLSTM3D(nn.Module):
    """Alternative 3D CNN model for video classification"""
    def __init__(self, num_classes=2):
        super(ConvLSTM3D, self).__init__()
        
        # 3D Convolutional layers
        self.conv3d_1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv3d_2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3d_3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length, height, width)
        x = F.relu(self.bn1(self.conv3d_1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv3d_2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3d_3(x)))
        x = self.pool3(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x

class EfficientNetLSTM(nn.Module):
    """Model using EfficientNet as feature extractor with LSTM"""
    def __init__(self, num_classes=2, sequence_length=16, hidden_size=128):
        super(EfficientNetLSTM, self).__init__()
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        # Use EfficientNet-B0 as feature extractor
        from torchvision.models import efficientnet_b0
        self.feature_extractor = efficientnet_b0(pretrained=True)
        
        # Remove the final classification layer
        self.feature_extractor.classifier = nn.Identity()
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=1280,  # EfficientNet-B0 output features
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape for feature extraction
        x = x.view(-1, channels, height, width)
        
        # Extract features for each frame
        features = self.feature_extractor(x)  # (batch_size * seq_len, 1280)
        features = features.view(batch_size, seq_len, -1)  # (batch_size, seq_len, 1280)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # Use the last output from LSTM
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # Classification
        output = self.classifier(last_output)
        
        return output

def create_model(model_type="resnet_lstm"):
    """Create model based on type"""
    if model_type == "resnet_lstm":
        return ViolenceDetectionModel(
            num_classes=Config.NUM_CLASSES,
            sequence_length=Config.SEQUENCE_LENGTH,
            hidden_size=Config.HIDDEN_SIZE
        )
    elif model_type == "convlstm3d":
        return ConvLSTM3D(num_classes=Config.NUM_CLASSES)
    elif model_type == "efficientnet_lstm":
        return EfficientNetLSTM(
            num_classes=Config.NUM_CLASSES,
            sequence_length=Config.SEQUENCE_LENGTH,
            hidden_size=Config.HIDDEN_SIZE
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Test model creation
    device = torch.device(Config.DEVICE)
    
    # Test different model architectures
    models_to_test = ["resnet_lstm", "convlstm3d", "efficientnet_lstm"]
    
    for model_type in models_to_test:
        print(f"\nTesting {model_type}...")
        model = create_model(model_type)
        model = model.to(device)
        
        # Create dummy input
        if model_type == "convlstm3d":
            dummy_input = torch.randn(2, 3, Config.SEQUENCE_LENGTH, Config.FRAME_SIZE[0], Config.FRAME_SIZE[1]).to(device)
        else:
            dummy_input = torch.randn(2, Config.SEQUENCE_LENGTH, 3, Config.FRAME_SIZE[0], Config.FRAME_SIZE[1]).to(device)
        
        # Forward pass
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
