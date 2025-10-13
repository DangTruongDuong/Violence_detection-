import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from model import create_model
from data_loader import create_data_loaders

class Trainer:
    def __init__(self, model_type="resnet_lstm"):
        self.device = torch.device(Config.DEVICE)
        self.model = create_model(model_type).to(self.device)
        self.model_type = model_type
        
        # Create directories
        os.makedirs(Config.SAVE_MODEL_PATH, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(Config.LOG_DIR)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        self.best_val_acc = 0.0
        self.best_model_path = os.path.join(Config.SAVE_MODEL_PATH, f"best_{model_type}_model.pth")
        
        # Early stopping
        self.patience = 7  # Dừng nếu val acc không cải thiện 7 epochs
        self.patience_counter = 0
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (frames, labels) in enumerate(pbar):
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(frames)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
            
            # Log to tensorboard
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), 
                                     len(self.train_losses) * len(train_loader) + batch_idx)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for frames, labels in pbar:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions for metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return avg_loss, accuracy, precision, recall, f1, all_predictions, all_labels
    
    def save_model(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'model_type': self.model_type,
            'config': {
                'FRAME_SIZE': Config.FRAME_SIZE,
                'SEQUENCE_LENGTH': Config.SEQUENCE_LENGTH,
                'BATCH_SIZE': Config.BATCH_SIZE,
                'LEARNING_RATE': Config.LEARNING_RATE,
                'EPOCHS': Config.EPOCHS,
                'HIDDEN_SIZE': Config.HIDDEN_SIZE,
                'NUM_CLASSES': Config.NUM_CLASSES
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(Config.SAVE_MODEL_PATH, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            print(f"New best model saved with validation accuracy: {self.best_val_acc:.2f}%")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.SAVE_MODEL_PATH, 'training_history.png'))
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, epoch):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Violence', 'Violence'],
                   yticklabels=['Non-Violence', 'Violence'])
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(Config.SAVE_MODEL_PATH, f'confusion_matrix_epoch_{epoch}.png'))
        plt.show()
    
    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model: {self.model_type}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Load data
        train_loader, val_loader, test_loader = create_data_loaders()
        
        start_time = time.time()
        
        for epoch in range(Config.EPOCHS):
            print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, precision, recall, f1, val_preds, val_labels = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Log to tensorboard
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/Train_Acc', train_acc, epoch)
            self.writer.add_scalar('Epoch/Val_Acc', val_acc, epoch)
            self.writer.add_scalar('Epoch/Precision', precision, epoch)
            self.writer.add_scalar('Epoch/Recall', recall, epoch)
            self.writer.add_scalar('Epoch/F1', f1, epoch)
            self.writer.add_scalar('Epoch/Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Save model if best
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.patience_counter = 0  # Reset counter
            else:
                self.patience_counter += 1
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_model(epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
                break
            
            # Plot confusion matrix every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.plot_confusion_matrix(val_labels, val_preds, epoch + 1)
        
        # Final evaluation on test set
        print("\n" + "="*50)
        print("FINAL TEST EVALUATION")
        print("="*50)
        
        test_loss, test_acc, test_precision, test_recall, test_f1, test_preds, test_labels = self.validate_epoch(test_loader)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")
        
        # Save final model
        self.save_model(Config.EPOCHS - 1, False)
        
        # Plot final results
        self.plot_training_history()
        self.plot_confusion_matrix(test_labels, test_preds, "Final")
        
        # Close tensorboard writer
        self.writer.close()
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")

def main():
    """Main training function"""
    # You can change model type here: "resnet_lstm", "convlstm3d", "efficientnet_lstm"
    model_type = "resnet_lstm"
    
    trainer = Trainer(model_type)
    trainer.train()

if __name__ == "__main__":
    main()

