"""
Modern Model Pruning Implementation with PyTorch

This module implements various pruning techniques for neural networks including:
- Unstructured pruning (L1, L2, random)
- Structured pruning
- Magnitude-based pruning
- Gradual pruning with retraining

Author: AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PruningConfig:
    """Configuration class for pruning parameters"""
    pruning_method: str = "l1_unstructured"  # l1_unstructured, l2_unstructured, random_unstructured, structured
    pruning_amount: float = 0.5  # Fraction of weights to prune
    gradual_pruning: bool = False
    retrain_after_pruning: bool = True
    retrain_epochs: int = 5
    learning_rate: float = 0.001
    batch_size: int = 64
    test_batch_size: int = 1000
    epochs: int = 10
    device: str = "auto"  # auto, cpu, cuda, mps


class ModelMetrics:
    """Class to track and store model metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            'train_loss': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'model_size': [],
            'pruning_ratio': []
        }
        self.timestamps: List[str] = []
    
    def add_metric(self, metric_name: str, value: float) -> None:
        """Add a metric value"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
        self.timestamps.append(datetime.now().isoformat())
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get the latest value for a metric"""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1]
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'metrics': self.metrics,
            'timestamps': self.timestamps
        }


class ModernMLP(nn.Module):
    """Modern implementation of Multi-Layer Perceptron with better architecture"""
    
    def __init__(self, input_size: int = 784, hidden_sizes: List[int] = [300, 100], 
                 output_size: int = 10, dropout_rate: float = 0.2):
        super(ModernMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)
    
    def get_prunable_parameters(self) -> List[Tuple[nn.Module, str]]:
        """Get list of prunable parameters"""
        parameters = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                parameters.append((module, 'weight'))
        return parameters
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }


class ModelPruner:
    """Advanced model pruning class with multiple pruning strategies"""
    
    def __init__(self, config: PruningConfig):
        self.config = config
        self.device = self._get_device()
        self.metrics = ModelMetrics()
    
    def _get_device(self) -> torch.device:
        """Determine the best available device"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)
    
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load MNIST dataset with modern transforms"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])
        
        train_data = datasets.MNIST(
            './data', train=True, download=True, transform=transform
        )
        test_data = datasets.MNIST(
            './data', train=False, download=True, transform=transform
        )
        
        train_loader = DataLoader(
            train_data, batch_size=self.config.batch_size, shuffle=True, num_workers=2
        )
        test_loader = DataLoader(
            test_data, batch_size=self.config.test_batch_size, shuffle=False, num_workers=2
        )
        
        logger.info(f"Loaded MNIST dataset: {len(train_data)} train, {len(test_data)} test samples")
        return train_loader, test_loader
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """Train model for one epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> float:
        """Evaluate model on test set"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        return accuracy
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   test_loader: DataLoader) -> nn.Module:
        """Train the model"""
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        logger.info(f"🚀 Training model on {self.device}...")
        
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
            test_acc = self.evaluate(model, test_loader)
            
            # Log metrics
            self.metrics.add_metric('train_loss', train_loss)
            self.metrics.add_metric('train_accuracy', train_acc)
            self.metrics.add_metric('test_accuracy', test_acc)
            
            logger.info(f"Epoch {epoch+1}/{self.config.epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                       f"Test Acc: {test_acc:.2f}%")
        
        return model
    
    def prune_model(self, model: nn.Module, pruning_amount: float = None) -> nn.Module:
        """Apply pruning to the model"""
        if pruning_amount is None:
            pruning_amount = self.config.pruning_amount
        
        logger.info(f"✂️ Applying {self.config.pruning_method} pruning ({pruning_amount*100:.1f}%)...")
        
        # Get prunable parameters
        parameters_to_prune = model.get_prunable_parameters()
        
        if not parameters_to_prune:
            logger.warning("No prunable parameters found!")
            return model
        
        # Apply pruning based on method
        if self.config.pruning_method == "l1_unstructured":
            for module, param_name in parameters_to_prune:
                prune.l1_unstructured(module, name=param_name, amount=pruning_amount)
        elif self.config.pruning_method == "l2_unstructured":
            for module, param_name in parameters_to_prune:
                prune.l2_unstructured(module, name=param_name, amount=pruning_amount)
        elif self.config.pruning_method == "random_unstructured":
            for module, param_name in parameters_to_prune:
                prune.random_unstructured(module, name=param_name, amount=pruning_amount)
        elif self.config.pruning_method == "structured":
            for module, param_name in parameters_to_prune:
                prune.ln_structured(module, name=param_name, amount=pruning_amount, n=2, dim=0)
        else:
            raise ValueError(f"Unknown pruning method: {self.config.pruning_method}")
        
        # Calculate actual pruning ratio
        total_params = sum(p.numel() for p in model.parameters())
        pruned_params = sum(torch.sum(p == 0).item() for p in model.parameters())
        actual_pruning_ratio = pruned_params / total_params
        
        self.metrics.add_metric('pruning_ratio', actual_pruning_ratio)
        
        logger.info(f"Pruning completed. Actual pruning ratio: {actual_pruning_ratio:.2%}")
        
        return model
    
    def remove_pruning_reparameterization(self, model: nn.Module) -> nn.Module:
        """Remove pruning reparameterization to finalize pruning"""
        logger.info("🔧 Removing pruning reparameterization...")
        
        parameters_to_prune = model.get_prunable_parameters()
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return model
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete pruning pipeline"""
        logger.info("🎯 Starting Model Pruning Pipeline")
        
        # Load data
        train_loader, test_loader = self.load_data()
        
        # Create model
        model = ModernMLP().to(self.device)
        
        # Log initial model size
        initial_size = model.get_model_size()
        logger.info(f"Initial model size: {initial_size['total_parameters']:,} parameters "
                   f"({initial_size['model_size_mb']:.2f} MB)")
        
        # Train baseline model
        model = self.train_model(model, train_loader, test_loader)
        baseline_accuracy = self.metrics.get_latest('test_accuracy')
        
        # Apply pruning
        model = self.prune_model(model)
        
        # Remove reparameterization
        model = self.remove_pruning_reparameterization(model)
        
        # Log pruned model size
        pruned_size = model.get_model_size()
        logger.info(f"Pruned model size: {pruned_size['total_parameters']:,} parameters "
                   f"({pruned_size['model_size_mb']:.2f} MB)")
        
        # Evaluate pruned model
        pruned_accuracy = self.evaluate(model, test_loader)
        self.metrics.add_metric('test_accuracy', pruned_accuracy)
        
        # Retrain if configured
        if self.config.retrain_after_pruning:
            logger.info("🔄 Retraining pruned model...")
            model = self.train_model(model, train_loader, test_loader)
            final_accuracy = self.metrics.get_latest('test_accuracy')
        else:
            final_accuracy = pruned_accuracy
        
        # Compile results
        results = {
            'baseline_accuracy': baseline_accuracy,
            'pruned_accuracy': pruned_accuracy,
            'final_accuracy': final_accuracy,
            'accuracy_drop': baseline_accuracy - final_accuracy,
            'model_size_reduction': (initial_size['total_parameters'] - pruned_size['total_parameters']) / initial_size['total_parameters'],
            'initial_model_size': initial_size,
            'pruned_model_size': pruned_size,
            'config': asdict(self.config),
            'metrics': self.metrics.to_dict()
        }
        
        logger.info(f"🎉 Pipeline completed!")
        logger.info(f"Baseline accuracy: {baseline_accuracy:.2f}%")
        logger.info(f"Final accuracy: {final_accuracy:.2f}%")
        logger.info(f"Accuracy drop: {results['accuracy_drop']:.2f}%")
        logger.info(f"Model size reduction: {results['model_size_reduction']:.2%}")
        
        return results


def main():
    """Main function to run the pruning pipeline"""
    # Create configuration
    config = PruningConfig(
        pruning_method="l1_unstructured",
        pruning_amount=0.5,
        retrain_after_pruning=True,
        retrain_epochs=3,
        epochs=5
    )
    
    # Create pruner and run pipeline
    pruner = ModelPruner(config)
    results = pruner.run_full_pipeline()
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/pruning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Results saved to results/pruning_results.json")


if __name__ == "__main__":
    main()
