"""
Visualization tools for Model Pruning Analysis

This module provides comprehensive visualization tools for analyzing
model pruning effects, performance metrics, and weight distributions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PruningVisualizer:
    """Comprehensive visualization tools for pruning analysis"""
    
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_metrics(self, metrics: Dict[str, List[float]], 
                            title: str = "Training Metrics", 
                            save_path: Optional[str] = None) -> None:
        """Plot training metrics over epochs"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Plot training loss
        if 'train_loss' in metrics:
            axes[0, 0].plot(metrics['train_loss'], 'b-', linewidth=2, marker='o')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot training accuracy
        if 'train_accuracy' in metrics:
            axes[0, 1].plot(metrics['train_accuracy'], 'g-', linewidth=2, marker='s')
            axes[0, 1].set_title('Training Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot test accuracy
        if 'test_accuracy' in metrics:
            axes[1, 0].plot(metrics['test_accuracy'], 'r-', linewidth=2, marker='^')
            axes[1, 0].set_title('Test Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot learning curve comparison
        if 'train_accuracy' in metrics and 'test_accuracy' in metrics:
            axes[1, 1].plot(metrics['train_accuracy'], 'g-', linewidth=2, label='Train', marker='s')
            axes[1, 1].plot(metrics['test_accuracy'], 'r-', linewidth=2, label='Test', marker='^')
            axes[1, 1].set_title('Train vs Test Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training metrics plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_weight_distribution(self, model: nn.Module, 
                               title: str = "Weight Distribution Analysis",
                               save_path: Optional[str] = None) -> None:
        """Plot weight distribution before and after pruning"""
        weights_data = []
        layer_names = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.data.cpu().numpy().flatten()
                weights_data.append(weights)
                layer_names.append(name)
        
        if not weights_data:
            logger.warning("No linear layers found for weight analysis")
            return
        
        fig, axes = plt.subplots(2, len(weights_data), figsize=(5*len(weights_data), 10))
        if len(weights_data) == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, (weights, layer_name) in enumerate(zip(weights_data, layer_names)):
            # Plot histogram
            axes[0, i].hist(weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, i].set_title(f'{layer_name}\nWeight Distribution')
            axes[0, i].set_xlabel('Weight Value')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].grid(True, alpha=0.3)
            
            # Plot box plot
            axes[1, i].boxplot(weights, patch_artist=True, 
                              boxprops=dict(facecolor='lightgreen', alpha=0.7))
            axes[1, i].set_title(f'{layer_name}\nWeight Statistics')
            axes[1, i].set_ylabel('Weight Value')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Weight distribution plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_pruning_effects(self, baseline_acc: float, pruned_acc: float,
                           model_size_reduction: float, pruning_ratio: float,
                           title: str = "Pruning Effects Analysis",
                           save_path: Optional[str] = None) -> None:
        """Plot the effects of pruning on model performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Accuracy comparison
        methods = ['Baseline', 'Pruned']
        accuracies = [baseline_acc, pruned_acc]
        colors = ['lightblue', 'lightcoral']
        
        bars1 = axes[0, 0].bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black')
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # Model size reduction
        axes[0, 1].pie([model_size_reduction, 1-model_size_reduction], 
                      labels=['Reduced', 'Remaining'], 
                      colors=['lightgreen', 'lightgray'],
                      autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Model Size Reduction')
        
        # Pruning ratio
        axes[1, 0].pie([pruning_ratio, 1-pruning_ratio], 
                      labels=['Pruned', 'Kept'], 
                      colors=['orange', 'lightblue'],
                      autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Weight Pruning Ratio')
        
        # Performance vs Size trade-off
        axes[1, 1].scatter([100-model_size_reduction*100], [pruned_acc], 
                         s=200, c='red', alpha=0.7, label='Pruned Model')
        axes[1, 1].scatter([100], [baseline_acc], 
                         s=200, c='blue', alpha=0.7, label='Baseline Model')
        axes[1, 1].set_xlabel('Model Size (%)')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Performance vs Size Trade-off')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pruning effects plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_experiment_comparison(self, experiments_data: pd.DataFrame,
                                 title: str = "Experiment Comparison",
                                 save_path: Optional[str] = None) -> None:
        """Plot comparison of multiple pruning experiments"""
        if experiments_data.empty:
            logger.warning("No experiment data provided")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Accuracy comparison
        sns.barplot(data=experiments_data, x='pruning_method', y='final_accuracy', 
                   ax=axes[0, 0], palette='viridis')
        axes[0, 0].set_title('Final Accuracy by Pruning Method')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Accuracy drop
        sns.barplot(data=experiments_data, x='pruning_method', y='accuracy_drop', 
                   ax=axes[0, 1], palette='Reds')
        axes[0, 1].set_title('Accuracy Drop by Pruning Method')
        axes[0, 1].set_ylabel('Accuracy Drop (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Model size reduction
        sns.barplot(data=experiments_data, x='pruning_method', y='model_size_reduction', 
                   ax=axes[1, 0], palette='Greens')
        axes[1, 0].set_title('Model Size Reduction by Pruning Method')
        axes[1, 0].set_ylabel('Size Reduction (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Scatter plot: Accuracy vs Size Reduction
        sns.scatterplot(data=experiments_data, x='model_size_reduction', y='final_accuracy',
                       hue='pruning_method', s=100, ax=axes[1, 1])
        axes[1, 1].set_title('Accuracy vs Size Reduction')
        axes[1, 1].set_xlabel('Model Size Reduction (%)')
        axes[1, 1].set_ylabel('Final Accuracy (%)')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Experiment comparison plot saved to {save_path}")
        else:
            plt.show()
    
    def create_interactive_dashboard(self, experiments_data: pd.DataFrame,
                                   save_path: Optional[str] = None) -> None:
        """Create an interactive Plotly dashboard"""
        if experiments_data.empty:
            logger.warning("No experiment data provided for dashboard")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Comparison', 'Accuracy Drop', 
                          'Model Size Reduction', 'Performance vs Size Trade-off'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Accuracy comparison
        fig.add_trace(
            go.Bar(x=experiments_data['pruning_method'], 
                  y=experiments_data['final_accuracy'],
                  name='Final Accuracy', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Accuracy drop
        fig.add_trace(
            go.Bar(x=experiments_data['pruning_method'], 
                  y=experiments_data['accuracy_drop'],
                  name='Accuracy Drop', marker_color='lightcoral'),
            row=1, col=2
        )
        
        # Model size reduction
        fig.add_trace(
            go.Bar(x=experiments_data['pruning_method'], 
                  y=experiments_data['model_size_reduction'],
                  name='Size Reduction', marker_color='lightgreen'),
            row=2, col=1
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(x=experiments_data['model_size_reduction'], 
                      y=experiments_data['final_accuracy'],
                      mode='markers+text',
                      text=experiments_data['pruning_method'],
                      textposition='top center',
                      name='Performance vs Size',
                      marker=dict(size=12, color='orange')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Model Pruning Analysis Dashboard",
            title_x=0.5,
            showlegend=False,
            height=800,
            template="plotly_white"
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Pruning Method", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
        
        fig.update_xaxes(title_text="Pruning Method", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy Drop (%)", row=1, col=2)
        
        fig.update_xaxes(title_text="Pruning Method", row=2, col=1)
        fig.update_yaxes(title_text="Size Reduction (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="Model Size Reduction (%)", row=2, col=2)
        fig.update_yaxes(title_text="Final Accuracy (%)", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        else:
            fig.show()
    
    def plot_layer_wise_analysis(self, model: nn.Module,
                               title: str = "Layer-wise Pruning Analysis",
                               save_path: Optional[str] = None) -> None:
        """Plot layer-wise pruning analysis"""
        layer_data = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.data
                total_params = weights.numel()
                pruned_params = torch.sum(weights == 0).item()
                pruning_ratio = pruned_params / total_params
                
                layer_data.append({
                    'layer': name,
                    'total_params': total_params,
                    'pruned_params': pruned_params,
                    'pruning_ratio': pruning_ratio,
                    'weight_mean': torch.mean(torch.abs(weights)).item(),
                    'weight_std': torch.std(weights).item()
                })
        
        if not layer_data:
            logger.warning("No linear layers found for analysis")
            return
        
        df = pd.DataFrame(layer_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Pruning ratio by layer
        axes[0, 0].bar(df['layer'], df['pruning_ratio'], color='lightcoral', alpha=0.8)
        axes[0, 0].set_title('Pruning Ratio by Layer')
        axes[0, 0].set_ylabel('Pruning Ratio')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Total parameters by layer
        axes[0, 1].bar(df['layer'], df['total_params'], color='lightblue', alpha=0.8)
        axes[0, 1].set_title('Total Parameters by Layer')
        axes[0, 1].set_ylabel('Parameter Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Weight statistics
        x = np.arange(len(df))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, df['weight_mean'], width, label='Mean', alpha=0.8)
        axes[1, 0].bar(x + width/2, df['weight_std'], width, label='Std', alpha=0.8)
        axes[1, 0].set_title('Weight Statistics by Layer')
        axes[1, 0].set_ylabel('Weight Value')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(df['layer'], rotation=45)
        axes[1, 0].legend()
        
        # Pruned vs Kept parameters
        axes[1, 1].bar(df['layer'], df['pruned_params'], label='Pruned', alpha=0.8)
        axes[1, 1].bar(df['layer'], df['total_params'] - df['pruned_params'], 
                      bottom=df['pruned_params'], label='Kept', alpha=0.8)
        axes[1, 1].set_title('Pruned vs Kept Parameters')
        axes[1, 1].set_ylabel('Parameter Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Layer-wise analysis plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(self, experiment_results: Dict[str, Any], 
                       model: nn.Module, save_dir: Optional[str] = None) -> None:
        """Generate a comprehensive visualization report"""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.save_dir
        
        logger.info(f"Generating visualization report in {save_dir}")
        
        # Plot training metrics
        if 'metrics' in experiment_results:
            self.plot_training_metrics(
                experiment_results['metrics'],
                save_path=save_dir / "training_metrics.png"
            )
        
        # Plot pruning effects
        if all(key in experiment_results for key in ['baseline_accuracy', 'final_accuracy', 
                                                    'model_size_reduction']):
            self.plot_pruning_effects(
                experiment_results['baseline_accuracy'],
                experiment_results['final_accuracy'],
                experiment_results['model_size_reduction'],
                experiment_results.get('pruning_ratio', 0.5),
                save_path=save_dir / "pruning_effects.png"
            )
        
        # Plot weight distribution
        self.plot_weight_distribution(
            model,
            save_path=save_dir / "weight_distribution.png"
        )
        
        # Plot layer-wise analysis
        self.plot_layer_wise_analysis(
            model,
            save_path=save_dir / "layer_wise_analysis.png"
        )
        
        logger.info("Visualization report generated successfully!")


if __name__ == "__main__":
    # Test the visualizer
    visualizer = PruningVisualizer()
    
    # Create sample data
    sample_metrics = {
        'train_loss': [0.5, 0.4, 0.3, 0.25, 0.2],
        'train_accuracy': [85.0, 87.0, 89.0, 91.0, 92.0],
        'test_accuracy': [84.0, 86.0, 88.0, 90.0, 91.0]
    }
    
    # Test plotting
    visualizer.plot_training_metrics(sample_metrics, "Sample Training Metrics")
    
    print("✅ Visualization tools test completed!")
