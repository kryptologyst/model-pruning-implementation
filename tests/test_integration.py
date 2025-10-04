"""
Integration tests for Model Pruning Implementation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import os
import json

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_pruning import ModelPruner, ModernMLP
from config import ConfigManager, ExperimentConfig
from database import ExperimentManager
from visualization import PruningVisualizer


class TestFullPipeline:
    """Integration tests for the complete pruning pipeline"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup after each test method"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_to_experiment_pipeline(self):
        """Test configuration to experiment pipeline"""
        # Create configuration
        config = ConfigManager.create_default_config()
        config.training.epochs = 1  # Minimal training for test
        config.pruning.pruning_amount = 0.1  # Small pruning amount
        
        # Create experiment manager
        db_path = os.path.join(self.temp_dir, "test_experiments.db")
        exp_manager = ExperimentManager(db_path)
        
        # Create experiment
        experiment_id = exp_manager.create_experiment("integration_test", config.__dict__)
        assert experiment_id == "integration_test"
        
        # Update with mock results
        mock_results = {
            "baseline_accuracy": 90.0,
            "final_accuracy": 89.0,
            "accuracy_drop": 1.0,
            "model_size_reduction": 0.1,
            "initial_model_size": {"total_parameters": 1000000},
            "pruned_model_size": {"total_parameters": 900000}
        }
        exp_manager.update_experiment_results(experiment_id, mock_results)
        
        # Verify experiment was saved
        comparison_df = exp_manager.get_experiment_comparison()
        assert len(comparison_df) == 1
        assert comparison_df.iloc[0]['experiment_id'] == "integration_test"
    
    def test_multiple_experiments_comparison(self):
        """Test comparing multiple experiments"""
        db_path = os.path.join(self.temp_dir, "comparison_test.db")
        exp_manager = ExperimentManager(db_path)
        
        # Create multiple experiments with different configurations
        experiments = [
            ("l1_test", {"pruning_method": "l1_unstructured", "pruning_amount": 0.3}),
            ("l2_test", {"pruning_method": "l2_unstructured", "pruning_amount": 0.3}),
            ("random_test", {"pruning_method": "random_unstructured", "pruning_amount": 0.3})
        ]
        
        for exp_id, config in experiments:
            exp_manager.create_experiment(exp_id, config)
            
            # Add mock results
            results = {
                "baseline_accuracy": 90.0,
                "final_accuracy": 89.0 + np.random.uniform(-2, 2),
                "accuracy_drop": 1.0 + np.random.uniform(-0.5, 0.5),
                "model_size_reduction": 0.3,
                "pruning_method": config["pruning_method"],
                "pruning_amount": config["pruning_amount"],
                "initial_model_size": {"total_parameters": 1000000},
                "pruned_model_size": {"total_parameters": 700000}
            }
            exp_manager.update_experiment_results(exp_id, results)
        
        # Get comparison
        comparison_df = exp_manager.get_experiment_comparison()
        assert len(comparison_df) == 3
        assert set(comparison_df['pruning_method'].values) == {"l1_unstructured", "l2_unstructured", "random_unstructured"}
    
    def test_model_pruning_and_evaluation(self):
        """Test model pruning and evaluation workflow"""
        # Create a simple model
        model = ModernMLP(hidden_sizes=[100, 50])  # Smaller model for testing
        
        # Get initial model info
        initial_size = model.get_model_size()
        initial_params = initial_size['total_parameters']
        
        # Apply pruning manually
        parameters_to_prune = model.get_prunable_parameters()
        
        # Apply L1 unstructured pruning
        import torch.nn.utils.prune as prune
        for module, param_name in parameters_to_prune:
            prune.l1_unstructured(module, name=param_name, amount=0.2)
        
        # Count pruned parameters
        total_params = sum(p.numel() for p in model.parameters())
        pruned_params = sum(torch.sum(p == 0).item() for p in model.parameters())
        actual_pruning_ratio = pruned_params / total_params
        
        # Remove reparameterization
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        # Verify pruning was applied
        final_size = model.get_model_size()
        assert final_size['total_parameters'] < initial_params
        assert actual_pruning_ratio > 0.1  # Should have pruned at least 10%
    
    def test_visualization_components(self):
        """Test visualization components"""
        visualizer = PruningVisualizer(save_dir=self.temp_dir)
        
        # Test with mock data
        mock_metrics = {
            'train_loss': [0.5, 0.4, 0.3, 0.25],
            'train_accuracy': [85.0, 87.0, 89.0, 91.0],
            'test_accuracy': [84.0, 86.0, 88.0, 90.0]
        }
        
        # Test plotting functions (they should not raise exceptions)
        try:
            visualizer.plot_training_metrics(mock_metrics, save_path=os.path.join(self.temp_dir, "test_metrics.png"))
            visualizer.plot_pruning_effects(
                baseline_acc=90.0,
                pruned_acc=89.0,
                model_size_reduction=0.3,
                pruning_ratio=0.3,
                save_path=os.path.join(self.temp_dir, "test_effects.png")
            )
        except Exception as e:
            pytest.fail(f"Visualization failed: {e}")
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization"""
        # Create configuration
        config = ConfigManager.create_default_config()
        
        # Test YAML serialization
        yaml_path = os.path.join(self.temp_dir, "test_config.yaml")
        ConfigManager.save_yaml(config, yaml_path)
        
        # Test JSON serialization
        json_path = os.path.join(self.temp_dir, "test_config.json")
        ConfigManager.save_json(config, json_path)
        
        # Verify files were created
        assert os.path.exists(yaml_path)
        assert os.path.exists(json_path)
        
        # Test loading from YAML
        loaded_yaml_config = ConfigManager.load_yaml(yaml_path)
        assert loaded_yaml_config.model.input_size == config.model.input_size
        assert loaded_yaml_config.training.learning_rate == config.training.learning_rate
        
        # Test loading from JSON
        loaded_json_config = ConfigManager.load_json(json_path)
        assert loaded_json_config.model.input_size == config.model.input_size
        assert loaded_json_config.training.learning_rate == config.training.learning_rate
    
    def test_database_export_import(self):
        """Test database export and import functionality"""
        db_path = os.path.join(self.temp_dir, "export_test.db")
        db = ExperimentManager(db_path).db
        
        # Add test data
        from database import ExperimentRecord
        test_record = ExperimentRecord(
            experiment_id="export_test",
            timestamp="2024-01-01T00:00:00",
            config={"pruning_method": "l1_unstructured"},
            metrics={"accuracy": 85.0},
            model_info={"params": 1000000},
            results={"final_accuracy": 85.0}
        )
        db.save_experiment(test_record)
        
        # Export to JSON
        export_path = os.path.join(self.temp_dir, "exported_data.json")
        db.export_to_json(export_path)
        
        # Verify export file was created and contains data
        assert os.path.exists(export_path)
        
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        assert exported_data['total_experiments'] == 1
        assert len(exported_data['experiments']) == 1
        assert exported_data['experiments'][0]['experiment_id'] == "export_test"


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_pruning_amount(self):
        """Test handling of invalid pruning amounts"""
        config = ConfigManager.create_default_config()
        config.pruning.pruning_amount = 1.5  # Invalid: > 1.0
        
        # This should be handled gracefully
        pruner = ModelPruner(config)
        assert pruner.config.pruning.pruning_amount == 1.5  # Config should accept it
    
    def test_empty_model_pruning(self):
        """Test pruning an empty model"""
        # Create model with no prunable layers
        class EmptyModel(nn.Module):
            def forward(self, x):
                return x
        
        model = EmptyModel()
        
        # This should not crash
        prunable_params = model.get_prunable_parameters() if hasattr(model, 'get_prunable_parameters') else []
        assert len(prunable_params) == 0
    
    def test_database_error_handling(self):
        """Test database error handling"""
        # Try to access non-existent experiment
        db_path = os.path.join(tempfile.mkdtemp(), "error_test.db")
        db = ExperimentManager(db_path).db
        
        non_existent = db.get_experiment("non_existent_id")
        assert non_existent is None
        
        # Try to get metrics for non-existent experiment
        metrics_df = db.get_metrics_dataframe("non_existent_id")
        assert metrics_df.empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
