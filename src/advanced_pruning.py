"""
Advanced Pruning Techniques Implementation

This module implements advanced pruning techniques including:
- Magnitude-based pruning
- Gradient-based pruning
- Lottery ticket hypothesis
- Gradual pruning
- Structured pruning variants
- Knowledge distillation for pruning
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from dataclasses import dataclass
from pathlib import Path
import copy

logger = logging.getLogger(__name__)


@dataclass
class PruningSchedule:
    """Configuration for gradual pruning schedule"""
    initial_sparsity: float = 0.0
    final_sparsity: float = 0.5
    start_epoch: int = 0
    end_epoch: int = 10
    frequency: int = 1


class MagnitudePruner:
    """Magnitude-based pruning implementation"""
    
    def __init__(self, model: nn.Module, sparsity: float = 0.5):
        self.model = model
        self.sparsity = sparsity
        self.pruned_modules = []
    
    def prune_by_magnitude(self, module: nn.Module, param_name: str = 'weight') -> None:
        """Prune weights based on magnitude"""
        if not hasattr(module, param_name):
            return
        
        param = getattr(module, param_name)
        if param is None:
            return
        
        # Calculate threshold
        flat_param = param.data.view(-1)
        threshold = torch.quantile(torch.abs(flat_param), self.sparsity)
        
        # Create mask
        mask = torch.abs(param.data) > threshold
        
        # Apply mask
        param.data *= mask.float()
        
        logger.info(f"Pruned {param_name} in {module.__class__.__name__} "
                   f"to {self.sparsity:.1%} sparsity")
    
    def prune_all_linear_layers(self) -> None:
        """Prune all linear layers in the model"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.prune_by_magnitude(module, 'weight')
                self.pruned_modules.append((name, module))


class GradientPruner:
    """Gradient-based pruning implementation"""
    
    def __init__(self, model: nn.Module, sparsity: float = 0.5):
        self.model = model
        self.sparsity = sparsity
        self.gradients = {}
    
    def compute_gradients(self, loss_fn: Callable, data_loader: torch.utils.data.DataLoader) -> None:
        """Compute gradients for pruning decision"""
        self.model.train()
        self.gradients = {}
        
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= 10:  # Limit to first 10 batches for efficiency
                break
            
            data, target = data.to(next(self.model.parameters()).device), target.to(next(self.model.parameters()).device)
            
            # Forward pass
            output = self.model(data)
            loss = loss_fn(output, target)
            
            # Backward pass
            loss.backward()
            
            # Store gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if name not in self.gradients:
                        self.gradients[name] = []
                    self.gradients[name].append(param.grad.data.clone())
            
            # Clear gradients
            self.model.zero_grad()
        
        # Average gradients
        for name in self.gradients:
            self.gradients[name] = torch.stack(self.gradients[name]).mean(0)
    
    def prune_by_gradient(self, module: nn.Module, param_name: str = 'weight') -> None:
        """Prune weights based on gradient magnitude"""
        if not hasattr(module, param_name):
            return
        
        param = getattr(module, param_name)
        if param is None:
            return
        
        # Find gradient for this parameter
        param_grad = None
        for name, grad in self.gradients.items():
            if param_name in name:
                param_grad = grad
                break
        
        if param_grad is None:
            logger.warning(f"No gradient found for {param_name}")
            return
        
        # Calculate importance score (gradient * weight)
        importance = torch.abs(param.data * param_grad)
        
        # Calculate threshold
        flat_importance = importance.view(-1)
        threshold = torch.quantile(flat_importance, self.sparsity)
        
        # Create mask
        mask = importance > threshold
        
        # Apply mask
        param.data *= mask.float()
        
        logger.info(f"Pruned {param_name} in {module.__class__.__name__} "
                   f"to {self.sparsity:.1%} sparsity using gradients")


class LotteryTicketPruner:
    """Lottery Ticket Hypothesis implementation"""
    
    def __init__(self, model: nn.Module, sparsity: float = 0.5):
        self.model = model
        self.sparsity = sparsity
        self.original_weights = {}
        self.masks = {}
    
    def save_original_weights(self) -> None:
        """Save original weights before any training"""
        for name, param in self.model.named_parameters():
            self.original_weights[name] = param.data.clone()
    
    def create_random_mask(self) -> None:
        """Create random pruning mask"""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                mask = torch.rand_like(param.data) > self.sparsity
                self.masks[name] = mask
    
    def apply_mask(self) -> None:
        """Apply pruning mask to weights"""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name].float()
    
    def reset_to_original(self) -> None:
        """Reset weights to original values"""
        for name, param in self.model.named_parameters():
            if name in self.original_weights:
                param.data = self.original_weights[name].clone()
    
    def prune_and_reset(self) -> None:
        """Prune weights and reset to original values"""
        self.apply_mask()
        self.reset_to_original()


class GradualPruner:
    """Gradual pruning implementation"""
    
    def __init__(self, model: nn.Module, schedule: PruningSchedule):
        self.model = model
        self.schedule = schedule
        self.current_sparsity = schedule.initial_sparsity
        self.pruned_modules = []
    
    def update_sparsity(self, epoch: int) -> float:
        """Update current sparsity based on schedule"""
        if epoch < self.schedule.start_epoch:
            self.current_sparsity = self.schedule.initial_sparsity
        elif epoch >= self.schedule.end_epoch:
            self.current_sparsity = self.schedule.final_sparsity
        else:
            # Linear interpolation
            progress = (epoch - self.schedule.start_epoch) / (self.schedule.end_epoch - self.schedule.start_epoch)
            self.current_sparsity = self.schedule.initial_sparsity + progress * (self.schedule.final_sparsity - self.schedule.initial_sparsity)
        
        return self.current_sparsity
    
    def prune_gradually(self, epoch: int) -> None:
        """Apply gradual pruning"""
        if epoch % self.schedule.frequency != 0:
            return
        
        current_sparsity = self.update_sparsity(epoch)
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
                # Apply L1 unstructured pruning
                prune.l1_unstructured(module, name='weight', amount=current_sparsity)
                self.pruned_modules.append((name, module))
        
        logger.info(f"Applied gradual pruning at epoch {epoch}, "
                   f"sparsity: {current_sparsity:.1%}")


class StructuredPruner:
    """Structured pruning implementation"""
    
    def __init__(self, model: nn.Module, sparsity: float = 0.5):
        self.model = model
        self.sparsity = sparsity
    
    def prune_channels(self, module: nn.Conv2d) -> None:
        """Prune entire channels from Conv2d layer"""
        if not isinstance(module, nn.Conv2d):
            return
        
        # Calculate channel importance (L2 norm of each channel)
        channel_norms = torch.norm(module.weight.data, dim=(1, 2, 3))
        
        # Calculate threshold
        threshold = torch.quantile(channel_norms, self.sparsity)
        
        # Create channel mask
        channel_mask = channel_norms > threshold
        
        # Apply mask to weights
        module.weight.data *= channel_mask.view(-1, 1, 1, 1).float()
        
        logger.info(f"Pruned {module.__class__.__name__} channels "
                   f"to {self.sparsity:.1%} sparsity")
    
    def prune_neurons(self, module: nn.Linear) -> None:
        """Prune entire neurons from Linear layer"""
        if not isinstance(module, nn.Linear):
            return
        
        # Calculate neuron importance (L2 norm of each neuron)
        neuron_norms = torch.norm(module.weight.data, dim=1)
        
        # Calculate threshold
        threshold = torch.quantile(neuron_norms, self.sparsity)
        
        # Create neuron mask
        neuron_mask = neuron_norms > threshold
        
        # Apply mask to weights
        module.weight.data *= neuron_mask.view(-1, 1).float()
        
        logger.info(f"Pruned {module.__class__.__name__} neurons "
                   f"to {self.sparsity:.1%} sparsity")
    
    def prune_all_structured(self) -> None:
        """Apply structured pruning to all applicable layers"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.prune_channels(module)
            elif isinstance(module, nn.Linear):
                self.prune_neurons(module)


class KnowledgeDistillationPruner:
    """Knowledge distillation for pruning"""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, 
                 temperature: float = 3.0, alpha: float = 0.7):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
    
    def distillation_loss(self, student_logits: torch.Tensor, 
                         teacher_logits: torch.Tensor, 
                         true_labels: torch.Tensor) -> torch.Tensor:
        """Compute knowledge distillation loss"""
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # Soft distillation loss
        soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        
        # Hard targets
        hard_loss = F.cross_entropy(student_logits, true_labels)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss
    
    def train_with_distillation(self, data_loader: torch.utils.data.DataLoader,
                               optimizer: torch.optim.Optimizer, epochs: int = 5) -> None:
        """Train student model with knowledge distillation"""
        self.teacher_model.eval()
        self.student_model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(next(self.student_model.parameters()).device), target.to(next(self.student_model.parameters()).device)
                
                optimizer.zero_grad()
                
                # Forward pass
                with torch.no_grad():
                    teacher_output = self.teacher_model(data)
                
                student_output = self.student_model(data)
                
                # Compute loss
                loss = self.distillation_loss(student_output, teacher_output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = student_output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            
            accuracy = 100. * correct / total
            avg_loss = total_loss / len(data_loader)
            
            logger.info(f"Distillation Epoch {epoch+1}/{epochs}, "
                       f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


class AdvancedPruningManager:
    """Manager for advanced pruning techniques"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.pruners = {}
        self.pruning_history = []
    
    def add_pruner(self, name: str, pruner: Any) -> None:
        """Add a pruner to the manager"""
        self.pruners[name] = pruner
    
    def apply_pruning(self, method: str, **kwargs) -> Dict[str, Any]:
        """Apply specified pruning method"""
        if method not in self.pruners:
            raise ValueError(f"Unknown pruning method: {method}")
        
        pruner = self.pruners[method]
        
        # Record pruning start
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        # Apply pruning
        if hasattr(pruner, 'prune_all_linear_layers'):
            pruner.prune_all_linear_layers()
        elif hasattr(pruner, 'prune_all_structured'):
            pruner.prune_all_structured()
        elif hasattr(pruner, 'prune_and_reset'):
            pruner.prune_and_reset()
        else:
            logger.warning(f"No pruning method found for {method}")
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            duration = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        else:
            duration = 0.0
        
        # Calculate sparsity
        total_params = sum(p.numel() for p in self.model.parameters())
        pruned_params = sum(torch.sum(p == 0).item() for p in self.model.parameters())
        sparsity = pruned_params / total_params
        
        # Record pruning result
        result = {
            "method": method,
            "sparsity": sparsity,
            "duration": duration,
            "total_params": total_params,
            "pruned_params": pruned_params,
            "timestamp": torch.cuda.Event(enable_timing=True).elapsed_time(start_time) / 1000.0 if start_time else 0.0
        }
        
        self.pruning_history.append(result)
        
        logger.info(f"Applied {method} pruning: {sparsity:.1%} sparsity, "
                   f"{duration:.4f}s duration")
        
        return result
    
    def get_pruning_summary(self) -> Dict[str, Any]:
        """Get summary of all pruning operations"""
        if not self.pruning_history:
            return {"message": "No pruning operations performed"}
        
        total_sparsity = self.pruning_history[-1]["sparsity"]
        total_duration = sum(op["duration"] for op in self.pruning_history)
        
        return {
            "total_operations": len(self.pruning_history),
            "final_sparsity": total_sparsity,
            "total_duration": total_duration,
            "operations": self.pruning_history
        }


def create_advanced_pruners(model: nn.Module, sparsity: float = 0.5) -> AdvancedPruningManager:
    """Create a manager with all advanced pruners"""
    manager = AdvancedPruningManager(model)
    
    # Add different pruners
    manager.add_pruner("magnitude", MagnitudePruner(model, sparsity))
    manager.add_pruner("structured", StructuredPruner(model, sparsity))
    manager.add_pruner("lottery_ticket", LotteryTicketPruner(model, sparsity))
    
    return manager


if __name__ == "__main__":
    # Test advanced pruning techniques
    logger.info("Testing Advanced Pruning Techniques")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(784, 300),
        nn.ReLU(),
        nn.Linear(300, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    
    # Test magnitude pruning
    magnitude_pruner = MagnitudePruner(model, sparsity=0.3)
    magnitude_pruner.prune_all_linear_layers()
    
    # Test structured pruning
    structured_pruner = StructuredPruner(model, sparsity=0.3)
    structured_pruner.prune_all_structured()
    
    # Test gradual pruning
    schedule = PruningSchedule(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        start_epoch=0,
        end_epoch=10,
        frequency=2
    )
    gradual_pruner = GradualPruner(model, schedule)
    
    # Test lottery ticket
    lottery_pruner = LotteryTicketPruner(model, sparsity=0.3)
    lottery_pruner.save_original_weights()
    lottery_pruner.create_random_mask()
    lottery_pruner.prune_and_reset()
    
    logger.info("✅ Advanced pruning techniques test completed!")
