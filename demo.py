#!/usr/bin/env python3
"""
Demo script để test hệ thống phát hiện bạo lực
"""

import os
import sys
import torch
import numpy as np
from config import Config
from data_loader import create_data_loaders
from model import create_model

def test_data_loading():
    """Test data loading functionality"""
    print("Testing data loading...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders()
        
        print(f"✓ Train loader: {len(train_loader)} batches")
        print(f"✓ Validation loader: {len(val_loader)} batches")
        print(f"✓ Test loader: {len(test_loader)} batches")
        
        # Test a batch
        for frames, labels in train_loader:
            print(f"✓ Batch shape: {frames.shape}")
            print(f"✓ Labels shape: {labels.shape}")
            print(f"✓ Label distribution: {torch.bincount(labels)}")
            break
            
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    
    models_to_test = ["resnet_lstm", "convlstm3d", "efficientnet_lstm"]
    device = torch.device(Config.DEVICE)
    
    for model_type in models_to_test:
        try:
            print(f"Testing {model_type}...")
            model = create_model(model_type)
            model = model.to(device)
            
            # Create dummy input
            if model_type == "convlstm3d":
                dummy_input = torch.randn(2, 3, Config.SEQUENCE_LENGTH, 
                                        Config.FRAME_SIZE[0], Config.FRAME_SIZE[1]).to(device)
            else:
                dummy_input = torch.randn(2, Config.SEQUENCE_LENGTH, 3, 
                                        Config.FRAME_SIZE[0], Config.FRAME_SIZE[1]).to(device)
            
            # Forward pass
            output = model(dummy_input)
            
            print(f"✓ {model_type}: Input {dummy_input.shape} -> Output {output.shape}")
            print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
        except Exception as e:
            print(f"✗ {model_type} failed: {e}")
            return False
    
    return True

def test_dataset_structure():
    """Test dataset structure"""
    print("\nTesting dataset structure...")
    
    violence_dir = Config.VIOLENCE_DIR
    non_violence_dir = Config.NON_VIOLENCE_DIR
    
    if not os.path.exists(violence_dir):
        print(f"✗ Violence directory not found: {violence_dir}")
        return False
    
    if not os.path.exists(non_violence_dir):
        print(f"✗ Non-violence directory not found: {non_violence_dir}")
        return False
    
    # Count files
    violence_files = [f for f in os.listdir(violence_dir) if f.endswith(('.mp4', '.avi'))]
    non_violence_files = [f for f in os.listdir(non_violence_dir) if f.endswith(('.mp4', '.avi'))]
    
    print(f"✓ Violence videos: {len(violence_files)}")
    print(f"✓ Non-violence videos: {len(non_violence_files)}")
    
    if len(violence_files) == 0:
        print("✗ No violence videos found")
        return False
    
    if len(non_violence_files) == 0:
        print("✗ No non-violence videos found")
        return False
    
    return True

def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    
    try:
        print(f"✓ Device: {Config.DEVICE}")
        print(f"✓ Frame size: {Config.FRAME_SIZE}")
        print(f"✓ Sequence length: {Config.SEQUENCE_LENGTH}")
        print(f"✓ Batch size: {Config.BATCH_SIZE}")
        print(f"✓ Learning rate: {Config.LEARNING_RATE}")
        print(f"✓ Epochs: {Config.EPOCHS}")
        print(f"✓ Confidence threshold: {Config.CONFIDENCE_THRESHOLD}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available, using CPU")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("DEMO HỆ THỐNG PHÁT HIỆN BẠO LỰC")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_config),
        ("Dataset Structure", test_dataset_structure),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"KẾT QUẢ: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 Tất cả tests đều PASSED! Hệ thống sẵn sàng sử dụng.")
        print("\nCác bước tiếp theo:")
        print("1. Chạy training: python main.py --mode train")
        print("2. Chạy GUI: python main.py --mode gui")
        print("3. Chạy detection: python main.py --mode detect")
    else:
        print("❌ Một số tests FAILED. Vui lòng kiểm tra lại cấu hình.")
        sys.exit(1)

if __name__ == "__main__":
    main()



