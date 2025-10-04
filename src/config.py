"""
Configuration management for Model Pruning project
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    input_size: int = 784
    hidden_sizes: list = None
    output_size: int = 10
    dropout_rate: float = 0.2
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [300, 100]


@dataclass
class TrainingConfig:
    """Training configuration"""
    learning_rate: float = 0.001
    batch_size: int = 64
    test_batch_size: int = 1000
    epochs: int = 10
    optimizer: str = "adam"  # adam, sgd, rmsprop
    scheduler: Optional[str] = None  # cosine, step, exponential
    weight_decay: float = 1e-4


@dataclass
class PruningConfig:
    """Pruning configuration"""
    pruning_method: str = "l1_unstructured"  # l1_unstructured, l2_unstructured, random_unstructured, structured
    pruning_amount: float = 0.5
    gradual_pruning: bool = False
    gradual_steps: int = 5
    retrain_after_pruning: bool = True
    retrain_epochs: int = 5
    fine_tune_lr: float = 0.0001


@dataclass
class DataConfig:
    """Data configuration"""
    dataset: str = "mnist"  # mnist, cifar10, fashion_mnist
    data_dir: str = "./data"
    normalize: bool = True
    augment: bool = False
    num_workers: int = 2


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    model: ModelConfig
    training: TrainingConfig
    pruning: PruningConfig
    data: DataConfig
    device: str = "auto"  # auto, cpu, cuda, mps
    seed: int = 42
    save_dir: str = "./results"
    log_level: str = "INFO"
    
    def __post_init__(self):
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.pruning, dict):
            self.pruning = PruningConfig(**self.pruning)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)


class ConfigManager:
    """Configuration manager for loading and saving configurations"""
    
    @staticmethod
    def load_yaml(config_path: str) -> ExperimentConfig:
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return ExperimentConfig(**config_dict)
    
    @staticmethod
    def load_json(config_path: str) -> ExperimentConfig:
        """Load configuration from JSON file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return ExperimentConfig(**config_dict)
    
    @staticmethod
    def save_yaml(config: ExperimentConfig, save_path: str) -> None:
        """Save configuration to YAML file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(config)
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {save_path}")
    
    @staticmethod
    def save_json(config: ExperimentConfig, save_path: str) -> None:
        """Save configuration to JSON file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(config)
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {save_path}")
    
    @staticmethod
    def create_default_config() -> ExperimentConfig:
        """Create default configuration"""
        return ExperimentConfig(
            model=ModelConfig(),
            training=TrainingConfig(),
            pruning=PruningConfig(),
            data=DataConfig()
        )
    
    @staticmethod
    def create_pruning_experiments() -> Dict[str, ExperimentConfig]:
        """Create multiple configurations for different pruning experiments"""
        base_config = ConfigManager.create_default_config()
        
        experiments = {}
        
        # L1 Unstructured Pruning
        l1_config = ExperimentConfig(
            model=ModelConfig(),
            training=TrainingConfig(epochs=5),
            pruning=PruningConfig(
                pruning_method="l1_unstructured",
                pruning_amount=0.5,
                retrain_after_pruning=True
            ),
            data=DataConfig()
        )
        experiments["l1_unstructured"] = l1_config
        
        # L2 Unstructured Pruning
        l2_config = ExperimentConfig(
            model=ModelConfig(),
            training=TrainingConfig(epochs=5),
            pruning=PruningConfig(
                pruning_method="l2_unstructured",
                pruning_amount=0.5,
                retrain_after_pruning=True
            ),
            data=DataConfig()
        )
        experiments["l2_unstructured"] = l2_config
        
        # Random Unstructured Pruning
        random_config = ExperimentConfig(
            model=ModelConfig(),
            training=TrainingConfig(epochs=5),
            pruning=PruningConfig(
                pruning_method="random_unstructured",
                pruning_amount=0.5,
                retrain_after_pruning=True
            ),
            data=DataConfig()
        )
        experiments["random_unstructured"] = random_config
        
        # Structured Pruning
        structured_config = ExperimentConfig(
            model=ModelConfig(),
            training=TrainingConfig(epochs=5),
            pruning=PruningConfig(
                pruning_method="structured",
                pruning_amount=0.3,  # Lower amount for structured
                retrain_after_pruning=True
            ),
            data=DataConfig()
        )
        experiments["structured"] = structured_config
        
        # Gradual Pruning
        gradual_config = ExperimentConfig(
            model=ModelConfig(),
            training=TrainingConfig(epochs=5),
            pruning=PruningConfig(
                pruning_method="l1_unstructured",
                pruning_amount=0.5,
                gradual_pruning=True,
                gradual_steps=5,
                retrain_after_pruning=True
            ),
            data=DataConfig()
        )
        experiments["gradual_pruning"] = gradual_config
        
        return experiments


if __name__ == "__main__":
    # Create and save default configurations
    config_manager = ConfigManager()
    
    # Save default config
    default_config = config_manager.create_default_config()
    config_manager.save_yaml(default_config, "configs/default.yaml")
    config_manager.save_json(default_config, "configs/default.json")
    
    # Save experiment configs
    experiments = config_manager.create_pruning_experiments()
    for name, config in experiments.items():
        config_manager.save_yaml(config, f"configs/{name}.yaml")
        config_manager.save_json(config, f"configs/{name}.json")
    
    print("Configuration files created successfully!")
