"""
Unit tests for Model Pruning Implementation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import os

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_pruning import ModernMLP, ModelPruner, ModelMetrics
from config import PruningConfig, ConfigManager, ExperimentConfig
from database import MockDatabase, ExperimentManager, ExperimentRecord


class TestModernMLP:
    """Test cases for ModernMLP class"""
    
    def test_model_creation(self):
        """Test model creation with default parameters"""
        model = ModernMLP()
        assert isinstance(model, nn.Module)
        assert model.input_size == 784
        assert model.output_size == 10
        assert model.hidden_sizes == [300, 100]
    
    def test_model_creation_custom(self):
        """Test model creation with custom parameters"""
        model = ModernMLP(
            input_size=256,
            hidden_sizes=[128, 64],
            output_size=5,
            dropout_rate=0.3
        )
        assert model.input_size == 256
        assert model.output_size == 5
        assert model.hidden_sizes == [128, 64]
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = ModernMLP()
        x = torch.randn(32, 28, 28)  # Batch of 32 MNIST images
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (32, 10)
        assert torch.all(torch.isfinite(output))
    
    def test_get_prunable_parameters(self):
        """Test getting prunable parameters"""
        model = ModernMLP()
        params = model.get_prunable_parameters()
        
        assert len(params) == 3  # fc1, fc2, fc3
        for module, param_name in params:
            assert isinstance(module, nn.Linear)
            assert param_name == 'weight'
    
    def test_get_model_size(self):
        """Test getting model size information"""
        model = ModernMLP()
        size_info = model.get_model_size()
        
        assert 'total_parameters' in size_info
        assert 'trainable_parameters' in size_info
        assert 'model_size_mb' in size_info
        assert size_info['total_parameters'] > 0
        assert size_info['model_size_mb'] > 0


class TestModelMetrics:
    """Test cases for ModelMetrics class"""
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = ModelMetrics()
        assert metrics.metrics == {
            'train_loss': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'model_size': [],
            'pruning_ratio': []
        }
        assert metrics.timestamps == []
    
    def test_add_metric(self):
        """Test adding metrics"""
        metrics = ModelMetrics()
        metrics.add_metric('train_loss', 0.5)
        metrics.add_metric('test_accuracy', 85.0)
        
        assert metrics.metrics['train_loss'] == [0.5]
        assert metrics.metrics['test_accuracy'] == [85.0]
        assert len(metrics.timestamps) == 2
    
    def test_get_latest(self):
        """Test getting latest metric value"""
        metrics = ModelMetrics()
        metrics.add_metric('train_loss', 0.5)
        metrics.add_metric('train_loss', 0.4)
        
        assert metrics.get_latest('train_loss') == 0.4
        assert metrics.get_latest('nonexistent') is None
    
    def test_to_dict(self):
        """Test converting metrics to dictionary"""
        metrics = ModelMetrics()
        metrics.add_metric('train_loss', 0.5)
        
        result = metrics.to_dict()
        assert 'metrics' in result
        assert 'timestamps' in result
        assert result['metrics']['train_loss'] == [0.5]


class TestPruningConfig:
    """Test cases for PruningConfig class"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = PruningConfig()
        assert config.pruning_method == "l1_unstructured"
        assert config.pruning_amount == 0.5
        assert config.gradual_pruning is False
        assert config.retrain_after_pruning is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = PruningConfig(
            pruning_method="l2_unstructured",
            pruning_amount=0.3,
            gradual_pruning=True,
            retrain_epochs=10
        )
        assert config.pruning_method == "l2_unstructured"
        assert config.pruning_amount == 0.3
        assert config.gradual_pruning is True
        assert config.retrain_epochs == 10


class TestConfigManager:
    """Test cases for ConfigManager class"""
    
    def test_create_default_config(self):
        """Test creating default configuration"""
        config = ConfigManager.create_default_config()
        assert isinstance(config, ExperimentConfig)
        assert config.model.input_size == 784
        assert config.training.learning_rate == 0.001
    
    def test_create_pruning_experiments(self):
        """Test creating pruning experiment configurations"""
        experiments = ConfigManager.create_pruning_experiments()
        
        assert len(experiments) == 5
        assert "l1_unstructured" in experiments
        assert "l2_unstructured" in experiments
        assert "random_unstructured" in experiments
        assert "structured" in experiments
        assert "gradual_pruning" in experiments
        
        # Check that each experiment has different pruning methods
        methods = [exp.pruning.pruning_method for exp in experiments.values()]
        assert len(set(methods)) == len(methods)  # All unique


class TestMockDatabase:
    """Test cases for MockDatabase class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.db = MockDatabase(self.db_path)
    
    def teardown_method(self):
        """Cleanup after each test method"""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_database_initialization(self):
        """Test database initialization"""
        assert os.path.exists(self.db_path)
    
    def test_save_and_get_experiment(self):
        """Test saving and retrieving experiments"""
        experiment = ExperimentRecord(
            experiment_id="test_001",
            timestamp="2024-01-01T00:00:00",
            config={"pruning_method": "l1_unstructured"},
            metrics={"accuracy": 85.0},
            model_info={"params": 1000000},
            results={"final_accuracy": 85.0}
        )
        
        # Save experiment
        self.db.save_experiment(experiment)
        
        # Retrieve experiment
        retrieved = self.db.get_experiment("test_001")
        assert retrieved is not None
        assert retrieved.experiment_id == "test_001"
        assert retrieved.config["pruning_method"] == "l1_unstructured"
    
    def test_get_all_experiments(self):
        """Test getting all experiments"""
        # Add multiple experiments
        for i in range(3):
            experiment = ExperimentRecord(
                experiment_id=f"test_{i:03d}",
                timestamp=f"2024-01-0{i+1}T00:00:00",
                config={"pruning_method": "l1_unstructured"},
                metrics={"accuracy": 85.0},
                model_info={"params": 1000000},
                results={"final_accuracy": 85.0}
            )
            self.db.save_experiment(experiment)
        
        # Get all experiments
        experiments = self.db.get_all_experiments()
        assert len(experiments) == 3
        assert all(exp.experiment_id.startswith("test_") for exp in experiments)
    
    def test_save_detailed_metrics(self):
        """Test saving detailed metrics"""
        experiment_id = "test_001"
        metrics_data = {
            'train_loss': [0.5, 0.4, 0.3],
            'test_accuracy': [85.0, 87.0, 89.0]
        }
        
        self.db.save_detailed_metrics(experiment_id, metrics_data)
        
        # Check that metrics were saved
        df = self.db.get_metrics_dataframe(experiment_id)
        assert not df.empty
        assert len(df) == 6  # 3 epochs * 2 metrics
        assert 'train_loss' in df['metric_name'].values
        assert 'test_accuracy' in df['metric_name'].values


class TestExperimentManager:
    """Test cases for ExperimentManager class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.manager = ExperimentManager(self.db_path)
    
    def teardown_method(self):
        """Cleanup after each test method"""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_create_experiment(self):
        """Test creating an experiment"""
        config = {"pruning_method": "l1_unstructured", "pruning_amount": 0.5}
        experiment_id = self.manager.create_experiment("test_001", config)
        
        assert experiment_id == "test_001"
        
        # Check that experiment was created
        experiment = self.manager.db.get_experiment("test_001")
        assert experiment is not None
        assert experiment.config == config
    
    def test_update_experiment_results(self):
        """Test updating experiment results"""
        # Create experiment first
        config = {"pruning_method": "l1_unstructured"}
        self.manager.create_experiment("test_001", config)
        
        # Update with results
        results = {"final_accuracy": 85.0, "model_size_reduction": 0.5}
        self.manager.update_experiment_results("test_001", results)
        
        # Check that results were updated
        experiment = self.manager.db.get_experiment("test_001")
        assert experiment.results == results
    
    def test_get_experiment_comparison(self):
        """Test getting experiment comparison"""
        # Create multiple experiments
        for i in range(3):
            config = {"pruning_method": f"method_{i}"}
            self.manager.create_experiment(f"test_{i:03d}", config)
            
            results = {
                "final_accuracy": 80.0 + i * 2,
                "model_size_reduction": 0.3 + i * 0.1,
                "pruning_method": f"method_{i}",
                "pruning_amount": 0.5
            }
            self.manager.update_experiment_results(f"test_{i:03d}", results)
        
        # Get comparison
        comparison_df = self.manager.get_experiment_comparison()
        assert len(comparison_df) == 3
        assert 'final_accuracy' in comparison_df.columns
        assert 'model_size_reduction' in comparison_df.columns


class TestModelPruner:
    """Test cases for ModelPruner class"""
    
    def test_pruner_initialization(self):
        """Test pruner initialization"""
        config = PruningConfig()
        pruner = ModelPruner(config)
        
        assert pruner.config == config
        assert isinstance(pruner.metrics, ModelMetrics)
        assert pruner.device in [torch.device("cpu"), torch.device("cuda"), torch.device("mps")]
    
    def test_get_device(self):
        """Test device selection"""
        config = PruningConfig(device="cpu")
        pruner = ModelPruner(config)
        assert pruner.device == torch.device("cpu")
        
        config = PruningConfig(device="auto")
        pruner = ModelPruner(config)
        assert pruner.device in [torch.device("cpu"), torch.device("cuda"), torch.device("mps")]
    
    def test_evaluate(self):
        """Test model evaluation"""
        config = PruningConfig()
        pruner = ModelPruner(config)
        
        # Create a simple model
        model = ModernMLP()
        
        # Create dummy data
        test_data = [(torch.randn(10, 28, 28), torch.randint(0, 10, (10,))) for _ in range(5)]
        
        # Mock DataLoader
        class MockDataLoader:
            def __init__(self, data):
                self.data = data
            
            def __iter__(self):
                return iter(self.data)
        
        test_loader = MockDataLoader(test_data)
        
        # Evaluate model
        accuracy = pruner.evaluate(model, test_loader)
        
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 100


# Integration tests
class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    def test_basic_pruning_pipeline(self):
        """Test basic pruning pipeline"""
        config = PruningConfig(
            pruning_method="l1_unstructured",
            pruning_amount=0.1,  # Small amount for quick test
            epochs=1,  # Minimal training
            retrain_after_pruning=False
        )
        
        pruner = ModelPruner(config)
        
        # This would normally load real data, but for testing we'll mock it
        # In a real test, you'd want to use actual MNIST data or a smaller dataset
        
        # Create model
        model = ModernMLP()
        initial_size = model.get_model_size()
        
        # Apply pruning
        pruned_model = pruner.prune_model(model, config.pruning_amount)
        
        # Remove reparameterization
        final_model = pruner.remove_pruning_reparameterization(pruned_model)
        
        # Check that pruning was applied
        final_size = final_model.get_model_size()
        assert final_size['total_parameters'] < initial_size['total_parameters']


if __name__ == "__main__":
    pytest.main([__file__])
