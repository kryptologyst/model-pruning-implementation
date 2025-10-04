# Model Pruning Implementation

A comprehensive implementation of neural network pruning techniques with modern tools, visualization, and interactive web interface.

## Features

- **Multiple Pruning Techniques**: L1 unstructured, L2 unstructured, random unstructured, and structured pruning
- **Modern PyTorch Implementation**: Latest PyTorch practices with type hints and proper architecture
- **Interactive Web UI**: Streamlit-based dashboard for running experiments and visualizing results
- **Comprehensive Visualization**: Weight distributions, layer-wise analysis, and performance metrics
- **Mock Database**: SQLite-based storage for experiment results and metrics
- **Configuration Management**: YAML/JSON configuration files for easy experiment setup
- **Advanced Metrics Tracking**: Detailed logging and metrics collection
- **Experiment Comparison**: Compare multiple pruning strategies side-by-side

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0 or higher

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/kryptologyst/model-pruning-implementation.git
cd model-pruning-implementation

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Optional: Install with extras

```bash
# For development tools
pip install -e .[dev]

# For visualization tools
pip install -e .[viz]

# For web interface
pip install -e .[web]
```

## Quick Start

### 1. Run the Web Interface

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` to access the interactive dashboard.

### 2. Run from Command Line

```bash
# Run with default configuration
python src/model_pruning.py

# Run with custom configuration
python src/model_pruning.py --config configs/l1_unstructured.yaml
```

### 3. Run Experiments Programmatically

```python
from src.model_pruning import ModelPruner
from src.config import PruningConfig

# Create configuration
config = PruningConfig(
    pruning_method="l1_unstructured",
    pruning_amount=0.5,
    retrain_after_pruning=True
)

# Run experiment
pruner = ModelPruner(config)
results = pruner.run_full_pipeline()
print(f"Final accuracy: {results['final_accuracy']:.2f}%")
```

## Usage Examples

### Basic Pruning Experiment

```python
import torch
from src.model_pruning import ModelPruner, ModernMLP
from src.config import PruningConfig

# Create model
model = ModernMLP(hidden_sizes=[300, 100], dropout_rate=0.2)

# Configure pruning
config = PruningConfig(
    pruning_method="l1_unstructured",
    pruning_amount=0.5,
    epochs=5,
    retrain_after_pruning=True
)

# Run experiment
pruner = ModelPruner(config)
results = pruner.run_full_pipeline()

# Print results
print(f"Baseline accuracy: {results['baseline_accuracy']:.2f}%")
print(f"Final accuracy: {results['final_accuracy']:.2f}%")
print(f"Model size reduction: {results['model_size_reduction']:.1%}")
```

### Comparing Different Pruning Methods

```python
from src.config import ConfigManager

# Create different experiment configurations
config_manager = ConfigManager()
experiments = config_manager.create_pruning_experiments()

# Run all experiments
results = {}
for name, config in experiments.items():
    pruner = ModelPruner(config)
    results[name] = pruner.run_full_pipeline()

# Compare results
for name, result in results.items():
    print(f"{name}: {result['final_accuracy']:.2f}% accuracy, "
          f"{result['model_size_reduction']:.1%} size reduction")
```

### Using the Database

```python
from src.database import ExperimentManager

# Initialize experiment manager
exp_manager = ExperimentManager()

# Create and run experiment
experiment_id = exp_manager.create_experiment("my_experiment", config_dict)
exp_manager.update_experiment_results(experiment_id, results_dict)

# Get experiment comparison
comparison_df = exp_manager.get_experiment_comparison()
print(comparison_df)
```

## 🔧 Configuration

The project uses YAML/JSON configuration files for easy experiment setup. Example configurations are provided in the `configs/` directory:

- `default.yaml`: Default configuration
- `l1_unstructured.yaml`: L1 unstructured pruning
- `l2_unstructured.yaml`: L2 unstructured pruning
- `random_unstructured.yaml`: Random unstructured pruning
- `structured.yaml`: Structured pruning
- `gradual_pruning.yaml`: Gradual pruning with retraining

### Configuration Structure

```yaml
model:
  input_size: 784
  hidden_sizes: [300, 100]
  output_size: 10
  dropout_rate: 0.2

training:
  learning_rate: 0.001
  batch_size: 64
  epochs: 10
  optimizer: "adam"

pruning:
  pruning_method: "l1_unstructured"
  pruning_amount: 0.5
  gradual_pruning: false
  retrain_after_pruning: true
  retrain_epochs: 5

data:
  dataset: "mnist"
  data_dir: "./data"
  normalize: true

device: "auto"
seed: 42
```

## Visualization

The project includes comprehensive visualization tools:

### Weight Distribution Analysis
```python
from src.visualization import PruningVisualizer

visualizer = PruningVisualizer()
visualizer.plot_weight_distribution(model, "Weight Analysis")
```

### Pruning Effects Visualization
```python
visualizer.plot_pruning_effects(
    baseline_acc=90.0,
    pruned_acc=89.0,
    model_size_reduction=0.5,
    pruning_ratio=0.5
)
```

### Experiment Comparison
```python
visualizer.plot_experiment_comparison(experiments_df)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_model_pruning.py
```

## 📁 Project Structure

```
model-pruning-implementation/
├── src/                          # Source code
│   ├── model_pruning.py         # Main pruning implementation
│   ├── config.py                # Configuration management
│   ├── database.py              # Mock database implementation
│   └── visualization.py         # Visualization tools
├── configs/                     # Configuration files
│   ├── default.yaml
│   ├── l1_unstructured.yaml
│   └── ...
├── tests/                       # Test files
│   ├── test_model_pruning.py
│   ├── test_config.py
│   └── ...
├── app.py                       # Streamlit web interface
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── README.md                    # This file
└── .gitignore                   # Git ignore rules
```

## Pruning Methods

### Unstructured Pruning
- **L1 Unstructured**: Removes weights with smallest L1 norms
- **L2 Unstructured**: Removes weights with smallest L2 norms  
- **Random Unstructured**: Randomly removes weights

### Structured Pruning
- **L2 Structured**: Removes entire neurons/channels based on L2 norms

### Advanced Techniques
- **Gradual Pruning**: Gradually increases pruning ratio over epochs
- **Magnitude-based Pruning**: Prunes based on weight magnitudes
- **Retraining**: Fine-tunes pruned models to recover accuracy

## Results and Metrics

The system tracks comprehensive metrics:

- **Accuracy**: Baseline, pruned, and final accuracy
- **Model Size**: Parameter count and memory usage
- **Pruning Ratio**: Actual vs target pruning ratios
- **Training Metrics**: Loss and accuracy curves
- **Efficiency Score**: Size reduction vs accuracy drop ratio

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Streamlit team for the amazing web interface framework
- The machine learning community for pruning research and techniques

## References

1. Han, S., et al. "Learning both weights and connections for efficient neural network." NIPS 2015.
2. LeCun, Y., et al. "Optimal brain damage." NIPS 1989.
3. Hassibi, B., et al. "Second order derivatives for network pruning: Optimal brain surgeon." NIPS 1992.


# model-pruning-implementation
