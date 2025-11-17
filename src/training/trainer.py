import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import os
import time
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import wandb

from .losses import create_loss_function
from .metrics import WhiteBalanceMetrics, compute_competition_score

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = 'min', verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = lambda a, b: a < b - min_delta
            self.best_score = float('inf')
        else:
            self.monitor_op = lambda a, b: a > b + min_delta
            self.best_score = float('-inf')
    
    def __call__(self, score: float) -> bool:
        """Check if early stopping should be triggered"""
        
        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print('Early stopping triggered')
        
        return self.early_stop

class ModelCheckpoint:
    """Model checkpoint utility"""
    
    def __init__(self, checkpoint_dir: str, save_best: bool = True, 
                 mode: str = 'min', verbose: bool = True):
        self.checkpoint_dir = checkpoint_dir
        self.save_best = save_best
        self.mode = mode
        self.verbose = verbose
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if mode == 'min':
            self.monitor_op = lambda a, b: a < b
            self.best_score = float('inf')
        else:
            self.monitor_op = lambda a, b: a > b
            self.best_score = float('-inf')
    
    def save(self, model: nn.Module, optimizer: optim.Optimizer, 
             scheduler: Any, epoch: int, score: float, 
             is_best: bool = False, extra_info: Dict = None):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'score': score,
            'best_score': self.best_score
        }
        
        if extra_info:
            checkpoint.update(extra_info)
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if self.save_best and (is_best or self.monitor_op(score, self.best_score)):
            if self.monitor_op(score, self.best_score):
                self.best_score = score
                if self.verbose:
                    print(f'Best model saved with score: {score:.6f}')
            
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
        
        return checkpoint_path

class WhiteBalanceTrainer:
    """Trainer for White Balance prediction models"""
    
    def __init__(self, config, model: nn.Module, device: str = 'cuda'):
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        # Setup logging
        self.setup_logging()
        
        # Initialize loss function
        self.criterion = create_loss_function(config)
        
        # Initialize optimizer
        self.optimizer = self.create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self.create_scheduler()
        
        # Initialize utilities
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta,
            mode='max',  # We want to maximize the competition score
            verbose=True
        )
        
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=config.checkpoint_dir,
            save_best=config.training.save_best_model,
            mode='max',  # Maximize score
            verbose=True
        )
        
        # Initialize metrics
        self.train_metrics = WhiteBalanceMetrics()
        self.val_metrics = WhiteBalanceMetrics()
        
        # Initialize wandb if enabled
        if config.training.use_wandb:
            wandb.init(
                project=config.training.wandb_project,
                config=config.__dict__,
                name=f"wb_prediction_{int(time.time())}"
            )
    
    def setup_logging(self):
        """Setup logging"""
        os.makedirs(self.config.logs_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.logs_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
    
    def create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler"""
        if self.config.training.lr_scheduler == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=self.config.training.lr_min
            )
        elif self.config.training.lr_scheduler == 'step':
            return StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif self.config.training.lr_scheduler == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer, mode='max', patience=10, factor=0.5
            )
        else:
            return None
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='Training')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            batch = self.move_batch_to_device(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch)
            
            # Calculate loss
            targets = {
                'temperature': batch['temperature'],
                'tint': batch['tint']
            }
            
            loss_dict = self.criterion(outputs, targets)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            self.train_metrics.update(outputs, targets, loss.item())
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log batch metrics
            if batch_idx % self.config.training.log_freq == 0:
                self.logger.info(
                    f'Batch {batch_idx}/{len(train_loader)}: '
                    f'Loss = {loss.item():.6f}, '
                    f'LR = {self.optimizer.param_groups[0]["lr"]:.2e}'
                )
        
        # Compute epoch metrics
        metrics = self.train_metrics.compute_mae_score()
        metrics['avg_loss'] = total_loss / num_batches
        
        return metrics
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            
            for batch in pbar:
                # Move data to device
                batch = self.move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Calculate loss
                targets = {
                    'temperature': batch['temperature'],
                    'tint': batch['tint']
                }
                
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['total_loss']
                
                # Update metrics
                self.val_metrics.update(outputs, targets, loss.item())
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'val_loss': loss.item()})
        
        # Compute epoch metrics
        metrics = self.val_metrics.compute_mae_score()
        metrics['avg_loss'] = total_loss / num_batches
        
        return metrics
    
    def train(self, train_loader, val_loader) -> Dict[str, Any]:
        """Full training loop"""
        
        self.logger.info(f'Starting training for {self.config.training.epochs} epochs')
        
        train_history = []
        val_history = []
        best_score = 0.0
        
        for epoch in range(self.config.training.epochs):
            start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate epoch
            val_metrics = self.validate_epoch(val_loader)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['combined_score'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - start_time
            
            self.logger.info(
                f'Epoch {epoch + 1}/{self.config.training.epochs} '
                f'({epoch_time:.1f}s) - '
                f'Train Score: {train_metrics["combined_score"]:.6f}, '
                f'Val Score: {val_metrics["combined_score"]:.6f}, '
                f'LR: {self.optimizer.param_groups[0]["lr"]:.2e}'
            )
            
            # Log to wandb
            if self.config.training.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_score': train_metrics['combined_score'],
                    'val_score': val_metrics['combined_score'],
                    'train_loss': train_metrics['avg_loss'],
                    'val_loss': val_metrics['avg_loss'],
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time
                })
            
            # Save checkpoint
            is_best = val_metrics['combined_score'] > best_score
            if is_best:
                best_score = val_metrics['combined_score']
            
            if (epoch + 1) % self.config.training.save_checkpoint_freq == 0 or is_best:
                self.checkpoint.save(
                    self.model, self.optimizer, self.scheduler,
                    epoch + 1, val_metrics['combined_score'], is_best,
                    extra_info={
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics,
                        'config': self.config.__dict__
                    }
                )
            
            # Early stopping check
            if self.early_stopping(val_metrics['combined_score']):
                self.logger.info(f'Early stopping at epoch {epoch + 1}')
                break
            
            # Store history
            train_history.append(train_metrics)
            val_history.append(val_metrics)
        
        # Save final plots
        self.save_training_plots(train_history, val_history)
        
        return {
            'train_history': train_history,
            'val_history': val_history,
            'best_score': best_score,
            'final_epoch': epoch + 1
        }
    
    def move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch data to device"""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                device_batch[key] = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                device_batch[key] = value
        
        return device_batch
    
    def save_training_plots(self, train_history: list, val_history: list):
        """Save training visualization plots"""
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(train_history) + 1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Combined score
        ax1.plot(epochs, [h['combined_score'] for h in train_history], 'b-', label='Train')
        ax1.plot(epochs, [h['combined_score'] for h in val_history], 'r-', label='Validation')
        ax1.set_title('Competition Score')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(epochs, [h['avg_loss'] for h in train_history], 'b-', label='Train')
        ax2.plot(epochs, [h['avg_loss'] for h in val_history], 'r-', label='Validation')
        ax2.set_title('Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Temperature MAE
        ax3.plot(epochs, [h['temperature_mae'] for h in train_history], 'b-', label='Train')
        ax3.plot(epochs, [h['temperature_mae'] for h in val_history], 'r-', label='Validation')
        ax3.set_title('Temperature MAE')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('MAE')
        ax3.legend()
        ax3.grid(True)
        
        # Tint MAE
        ax4.plot(epochs, [h['tint_mae'] for h in train_history], 'b-', label='Train')
        ax4.plot(epochs, [h['tint_mae'] for h in val_history], 'r-', label='Validation')
        ax4.set_title('Tint MAE')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('MAE')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.logs_dir, 'training_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot final predictions
        self.val_metrics.plot_predictions(
            save_path=os.path.join(self.config.logs_dir, 'final_predictions.png'),
            show=False
        )
        
        self.val_metrics.plot_error_distribution(
            save_path=os.path.join(self.config.logs_dir, 'error_distribution.png'),
            show=False
        )

def load_checkpoint(checkpoint_path: str, model: nn.Module, 
                   optimizer: optim.Optimizer = None, 
                   scheduler: Any = None) -> Dict[str, Any]:
    """Load model from checkpoint"""
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint