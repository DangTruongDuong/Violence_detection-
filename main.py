#!/usr/bin/env python3
"""
Main script for Violence Detection System
Hệ thống phát hiện bạo lực realtime
"""

import sys
import os
import argparse
from config import Config

def main():
    parser = argparse.ArgumentParser(description='Hệ thống phát hiện bạo lực realtime')
    parser.add_argument('--mode', choices=['train', 'gui', 'detect'], default='gui',
                       help='Chế độ chạy: train (training), gui (giao diện), detect (command line)')
    parser.add_argument('--model-type', choices=['resnet_lstm', 'convlstm3d', 'efficientnet_lstm'], 
                       default='resnet_lstm', help='Loại model')
    parser.add_argument('--model-path', type=str, 
                       default=os.path.join(Config.SAVE_MODEL_PATH, "best_resnet_lstm_model.pth"),
                       help='Đường dẫn model')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, help='Số epoch training')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=Config.LEARNING_RATE, help='Learning rate')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HỆ THỐNG PHÁT HIỆN BẠO LỰC REALTIME")
    print("=" * 60)
    print(f"Chế độ: {args.mode}")
    print(f"Loại model: {args.model_type}")
    print(f"Device: {Config.DEVICE}")
    print("=" * 60)
    
    if args.mode == 'train':
        print("Bắt đầu training...")
        from train import Trainer
        trainer = Trainer(args.model_type)
        trainer.train()
        
    elif args.mode == 'gui':
        print("Khởi động GUI...")
        try:
            import tkinter as tk
            from gui import ViolenceDetectionGUI
            
            root = tk.Tk()
            app = ViolenceDetectionGUI(root)
            
            # Set default model path if exists
            if os.path.exists(args.model_path):
                app.model_path_var.set(args.model_path)
                app.model_type_var.set(args.model_type)
            
            root.mainloop()
            
        except ImportError as e:
            print(f"Lỗi import GUI: {e}")
            print("Vui lòng cài đặt tkinter: sudo apt-get install python3-tk")
            sys.exit(1)
            
    elif args.mode == 'detect':
        print("Khởi động detection...")
        if not os.path.exists(args.model_path):
            print(f"Model không tồn tại: {args.model_path}")
            print("Vui lòng train model trước hoặc chỉ định đường dẫn model đúng")
            sys.exit(1)
            
        from realtime_detector import test_realtime_detection
        test_realtime_detection()

if __name__ == "__main__":
    main()



