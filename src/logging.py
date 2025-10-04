"""
Advanced logging and metrics tracking system for Model Pruning

This module provides comprehensive logging, metrics tracking, and experiment
monitoring capabilities with support for multiple backends.
"""

import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import time
import threading
from contextlib import contextmanager
import numpy as np
import torch

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pruning_experiments.log')
    ]
)

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Advanced metrics tracking with multiple backends"""
    
    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.metrics: Dict[str, List[Any]] = {}
        self.timestamps: List[str] = []
        self.metadata: Dict[str, Any] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize experiment log file
        self.experiment_log_file = self.log_dir / f"{experiment_name}.jsonl"
        
        # Log experiment start
        self._log_event("experiment_started", {"timestamp": datetime.now().isoformat()})
    
    def log_metric(self, metric_name: str, value: Union[float, int, str], 
                   step: Optional[int] = None, metadata: Optional[Dict] = None) -> None:
        """Log a metric value"""
        with self._lock:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            
            self.metrics[metric_name].append(value)
            
            if step is None:
                step = len(self.metrics[metric_name]) - 1
            
            timestamp = datetime.now().isoformat()
            self.timestamps.append(timestamp)
            
            # Log to file
            log_entry = {
                "timestamp": timestamp,
                "step": step,
                "metric_name": metric_name,
                "value": value,
                "metadata": metadata or {}
            }
            
            with open(self.experiment_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    def log_metrics_batch(self, metrics_dict: Dict[str, Any], 
                         step: Optional[int] = None) -> None:
        """Log multiple metrics at once"""
        for metric_name, value in metrics_dict.items():
            self.log_metric(metric_name, value, step)
    
    def log_model_info(self, model: torch.nn.Module) -> None:
        """Log model architecture and parameter information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            "architecture": str(model)
        }
        
        self.log_metric("model_info", model_info)
        self.metadata["model_info"] = model_info
    
    def log_pruning_info(self, pruning_method: str, pruning_amount: float,
                        actual_pruning_ratio: float, layer_stats: List[Dict]) -> None:
        """Log pruning information"""
        pruning_info = {
            "pruning_method": pruning_method,
            "target_pruning_amount": pruning_amount,
            "actual_pruning_ratio": actual_pruning_ratio,
            "layer_statistics": layer_stats
        }
        
        self.log_metric("pruning_info", pruning_info)
        self.metadata["pruning_info"] = pruning_info
    
    def log_training_step(self, epoch: int, step: int, loss: float, 
                         accuracy: Optional[float] = None) -> None:
        """Log training step information"""
        step_info = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "accuracy": accuracy
        }
        
        self.log_metric("training_step", step_info, step=step)
    
    def log_evaluation(self, accuracy: float, loss: Optional[float] = None,
                      dataset: str = "test") -> None:
        """Log evaluation results"""
        eval_info = {
            "dataset": dataset,
            "accuracy": accuracy,
            "loss": loss,
            "timestamp": datetime.now().isoformat()
        }
        
        self.log_metric(f"{dataset}_evaluation", eval_info)
    
    def log_experiment_result(self, results: Dict[str, Any]) -> None:
        """Log final experiment results"""
        self.log_metric("experiment_results", results)
        self.metadata["final_results"] = results
    
    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event"""
        event_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        
        with open(self.experiment_log_file, 'a') as f:
            f.write(json.dumps(event_entry) + '\n')
    
    def get_metric_history(self, metric_name: str) -> List[Any]:
        """Get history of a specific metric"""
        with self._lock:
            return self.metrics.get(metric_name, []).copy()
    
    def get_latest_metric(self, metric_name: str) -> Optional[Any]:
        """Get the latest value of a metric"""
        history = self.get_metric_history(metric_name)
        return history[-1] if history else None
    
    def export_metrics(self, output_path: Optional[str] = None) -> str:
        """Export all metrics to JSON file"""
        if output_path is None:
            output_path = self.log_dir / f"{self.experiment_name}_metrics.json"
        
        export_data = {
            "experiment_name": self.experiment_name,
            "export_timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "timestamps": self.timestamps,
            "metadata": self.metadata
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {output_path}")
        return str(output_path)
    
    def close(self) -> None:
        """Close the metrics tracker and log experiment end"""
        self._log_event("experiment_ended", {"timestamp": datetime.now().isoformat()})
        logger.info(f"Experiment '{self.experiment_name}' completed")


class ExperimentLogger:
    """High-level experiment logging interface"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.active_trackers: Dict[str, MetricsTracker] = {}
    
    @contextmanager
    def experiment(self, experiment_name: str):
        """Context manager for experiment tracking"""
        tracker = MetricsTracker(experiment_name, str(self.log_dir))
        self.active_trackers[experiment_name] = tracker
        
        try:
            yield tracker
        finally:
            tracker.close()
            if experiment_name in self.active_trackers:
                del self.active_trackers[experiment_name]
    
    def get_experiment_summary(self, experiment_name: str) -> Dict[str, Any]:
        """Get summary of an experiment"""
        log_file = self.log_dir / f"{experiment_name}.jsonl"
        
        if not log_file.exists():
            return {}
        
        metrics = {}
        events = []
        
        with open(log_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                
                if "metric_name" in entry:
                    metric_name = entry["metric_name"]
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append(entry["value"])
                elif "event_type" in entry:
                    events.append(entry)
        
        return {
            "experiment_name": experiment_name,
            "metrics": metrics,
            "events": events,
            "log_file": str(log_file)
        }
    
    def list_experiments(self) -> List[str]:
        """List all available experiments"""
        experiments = []
        for log_file in self.log_dir.glob("*.jsonl"):
            if log_file.name.endswith("_metrics.json"):
                continue
            experiments.append(log_file.stem)
        return experiments


class PerformanceProfiler:
    """Performance profiling for model pruning experiments"""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.memory_usage: Dict[str, List[float]] = {}
        self.gpu_usage: Dict[str, List[float]] = {}
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            if operation_name not in self.timings:
                self.timings[operation_name] = []
            self.timings[operation_name].append(duration)
            
            logger.info(f"Operation '{operation_name}' took {duration:.4f} seconds")
    
    def log_memory_usage(self, operation_name: str, memory_mb: float) -> None:
        """Log memory usage for an operation"""
        if operation_name not in self.memory_usage:
            self.memory_usage[operation_name] = []
        self.memory_usage[operation_name].append(memory_mb)
    
    def log_gpu_usage(self, operation_name: str, gpu_memory_mb: float) -> None:
        """Log GPU memory usage"""
        if operation_name not in self.gpu_usage:
            self.gpu_usage[operation_name] = []
        self.gpu_usage[operation_name].append(gpu_memory_mb)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        
        for operation, times in self.timings.items():
            summary[operation] = {
                "total_time": sum(times),
                "average_time": np.mean(times),
                "min_time": min(times),
                "max_time": max(times),
                "count": len(times)
            }
        
        return summary


class LoggingConfig:
    """Configuration for logging system"""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_to_file: bool = True,
                 log_to_console: bool = True,
                 log_dir: str = "logs",
                 experiment_logging: bool = True,
                 performance_profiling: bool = True):
        
        self.log_level = getattr(logging, log_level.upper())
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.log_dir = log_dir
        self.experiment_logging = experiment_logging
        self.performance_profiling = performance_profiling
    
    def setup_logging(self) -> None:
        """Setup logging configuration"""
        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if self.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if self.log_to_file:
            log_dir = Path(self.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_dir / "pruning_experiments.log")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Set level
        root_logger.setLevel(self.log_level)
        
        logger.info("Logging system initialized")


# Global instances
experiment_logger = ExperimentLogger()
performance_profiler = PerformanceProfiler()


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """Setup logging system"""
    if config is None:
        config = LoggingConfig()
    
    config.setup_logging()


def get_experiment_logger() -> ExperimentLogger:
    """Get the global experiment logger"""
    return experiment_logger


def get_performance_profiler() -> PerformanceProfiler:
    """Get the global performance profiler"""
    return performance_profiler


# Convenience functions
def log_experiment_start(experiment_name: str, config: Dict[str, Any]) -> MetricsTracker:
    """Log experiment start"""
    tracker = MetricsTracker(experiment_name)
    tracker.log_metric("experiment_config", config)
    return tracker


def log_model_training(tracker: MetricsTracker, epoch: int, loss: float, 
                      accuracy: Optional[float] = None) -> None:
    """Log model training metrics"""
    tracker.log_metric("train_loss", loss, step=epoch)
    if accuracy is not None:
        tracker.log_metric("train_accuracy", accuracy, step=epoch)


def log_model_evaluation(tracker: MetricsTracker, accuracy: float, 
                        loss: Optional[float] = None, dataset: str = "test") -> None:
    """Log model evaluation metrics"""
    tracker.log_evaluation(accuracy, loss, dataset)


def log_pruning_results(tracker: MetricsTracker, baseline_acc: float, 
                        pruned_acc: float, size_reduction: float) -> None:
    """Log pruning results"""
    results = {
        "baseline_accuracy": baseline_acc,
        "pruned_accuracy": pruned_acc,
        "accuracy_drop": baseline_acc - pruned_acc,
        "model_size_reduction": size_reduction
    }
    tracker.log_experiment_result(results)


if __name__ == "__main__":
    # Test the logging system
    setup_logging()
    
    # Test experiment tracking
    with experiment_logger.experiment("test_experiment") as tracker:
        tracker.log_metric("test_metric", 42.0)
        tracker.log_metrics_batch({"metric1": 1.0, "metric2": 2.0})
        
        # Test performance profiling
        with performance_profiler.time_operation("test_operation"):
            time.sleep(0.1)
        
        tracker.log_experiment_result({"final_score": 95.0})
    
    print("✅ Logging system test completed!")
    print(f"Available experiments: {experiment_logger.list_experiments()}")
