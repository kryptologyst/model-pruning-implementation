"""
Mock Database for Model Pruning Results

This module provides a simple in-memory database for storing and retrieving
model pruning experiment results, metrics, and configurations.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRecord:
    """Record for storing experiment data"""
    experiment_id: str
    timestamp: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    model_info: Dict[str, Any]
    results: Dict[str, Any]


class MockDatabase:
    """Mock database for storing pruning experiment results"""
    
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create experiments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                config TEXT NOT NULL,
                metrics TEXT NOT NULL,
                model_info TEXT NOT NULL,
                results TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create metrics table for detailed metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                epoch INTEGER,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        ''')
        
        # Create model_weights table for storing weight statistics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                layer_name TEXT NOT NULL,
                weight_count INTEGER NOT NULL,
                pruned_count INTEGER NOT NULL,
                pruning_ratio REAL NOT NULL,
                weight_mean REAL,
                weight_std REAL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def save_experiment(self, experiment_record: ExperimentRecord) -> None:
        """Save an experiment record to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO experiments 
                (experiment_id, timestamp, config, metrics, model_info, results)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                experiment_record.experiment_id,
                experiment_record.timestamp,
                json.dumps(experiment_record.config),
                json.dumps(experiment_record.metrics),
                json.dumps(experiment_record.model_info),
                json.dumps(experiment_record.results)
            ))
            
            conn.commit()
            logger.info(f"Experiment {experiment_record.experiment_id} saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving experiment: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Retrieve an experiment by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT experiment_id, timestamp, config, metrics, model_info, results
                FROM experiments WHERE experiment_id = ?
            ''', (experiment_id,))
            
            row = cursor.fetchone()
            if row:
                return ExperimentRecord(
                    experiment_id=row[0],
                    timestamp=row[1],
                    config=json.loads(row[2]),
                    metrics=json.loads(row[3]),
                    model_info=json.loads(row[4]),
                    results=json.loads(row[5])
                )
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving experiment: {e}")
            return None
        finally:
            conn.close()
    
    def get_all_experiments(self) -> List[ExperimentRecord]:
        """Retrieve all experiments"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT experiment_id, timestamp, config, metrics, model_info, results
                FROM experiments ORDER BY created_at DESC
            ''')
            
            experiments = []
            for row in cursor.fetchall():
                experiments.append(ExperimentRecord(
                    experiment_id=row[0],
                    timestamp=row[1],
                    config=json.loads(row[2]),
                    metrics=json.loads(row[3]),
                    model_info=json.loads(row[4]),
                    results=json.loads(row[5])
                ))
            
            return experiments
            
        except Exception as e:
            logger.error(f"Error retrieving experiments: {e}")
            return []
        finally:
            conn.close()
    
    def save_detailed_metrics(self, experiment_id: str, metrics_data: Dict[str, List[float]]) -> None:
        """Save detailed metrics for an experiment"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for metric_name, values in metrics_data.items():
                for epoch, value in enumerate(values):
                    cursor.execute('''
                        INSERT INTO metrics 
                        (experiment_id, metric_name, metric_value, epoch, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        experiment_id,
                        metric_name,
                        value,
                        epoch,
                        datetime.now().isoformat()
                    ))
            
            conn.commit()
            logger.info(f"Detailed metrics saved for experiment {experiment_id}")
            
        except Exception as e:
            logger.error(f"Error saving detailed metrics: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def save_weight_statistics(self, experiment_id: str, weight_stats: List[Dict[str, Any]]) -> None:
        """Save weight statistics for an experiment"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for stats in weight_stats:
                cursor.execute('''
                    INSERT INTO model_weights 
                    (experiment_id, layer_name, weight_count, pruned_count, 
                     pruning_ratio, weight_mean, weight_std)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    experiment_id,
                    stats['layer_name'],
                    stats['weight_count'],
                    stats['pruned_count'],
                    stats['pruning_ratio'],
                    stats.get('weight_mean'),
                    stats.get('weight_std')
                ))
            
            conn.commit()
            logger.info(f"Weight statistics saved for experiment {experiment_id}")
            
        except Exception as e:
            logger.error(f"Error saving weight statistics: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_experiments_summary(self) -> pd.DataFrame:
        """Get a summary of all experiments as a DataFrame"""
        experiments = self.get_all_experiments()
        
        if not experiments:
            return pd.DataFrame()
        
        summary_data = []
        for exp in experiments:
            summary_data.append({
                'experiment_id': exp.experiment_id,
                'timestamp': exp.timestamp,
                'pruning_method': exp.config.get('pruning', {}).get('pruning_method', 'unknown'),
                'pruning_amount': exp.config.get('pruning', {}).get('pruning_amount', 0),
                'baseline_accuracy': exp.results.get('baseline_accuracy', 0),
                'final_accuracy': exp.results.get('final_accuracy', 0),
                'accuracy_drop': exp.results.get('accuracy_drop', 0),
                'model_size_reduction': exp.results.get('model_size_reduction', 0),
                'total_parameters': exp.results.get('pruned_model_size', {}).get('total_parameters', 0)
            })
        
        return pd.DataFrame(summary_data)
    
    def get_metrics_dataframe(self, experiment_id: str) -> pd.DataFrame:
        """Get detailed metrics for an experiment as a DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            query = '''
                SELECT metric_name, metric_value, epoch, timestamp
                FROM metrics 
                WHERE experiment_id = ?
                ORDER BY epoch, metric_name
            '''
            
            df = pd.read_sql_query(query, conn, params=(experiment_id,))
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving metrics DataFrame: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_weight_statistics_dataframe(self, experiment_id: str) -> pd.DataFrame:
        """Get weight statistics for an experiment as a DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            query = '''
                SELECT layer_name, weight_count, pruned_count, pruning_ratio, 
                       weight_mean, weight_std
                FROM model_weights 
                WHERE experiment_id = ?
                ORDER BY layer_name
            '''
            
            df = pd.read_sql_query(query, conn, params=(experiment_id,))
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving weight statistics DataFrame: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and all related data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Delete from all tables
            cursor.execute('DELETE FROM metrics WHERE experiment_id = ?', (experiment_id,))
            cursor.execute('DELETE FROM model_weights WHERE experiment_id = ?', (experiment_id,))
            cursor.execute('DELETE FROM experiments WHERE experiment_id = ?', (experiment_id,))
            
            conn.commit()
            logger.info(f"Experiment {experiment_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting experiment: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def clear_database(self) -> None:
        """Clear all data from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM metrics')
            cursor.execute('DELETE FROM model_weights')
            cursor.execute('DELETE FROM experiments')
            conn.commit()
            logger.info("Database cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def export_to_json(self, output_path: str) -> None:
        """Export all experiments to JSON file"""
        experiments = self.get_all_experiments()
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_experiments': len(experiments),
            'experiments': [asdict(exp) for exp in experiments]
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Database exported to {output_path}")


class ExperimentManager:
    """High-level interface for managing pruning experiments"""
    
    def __init__(self, db_path: str = "experiments.db"):
        self.db = MockDatabase(db_path)
    
    def create_experiment(self, experiment_id: str, config: Dict[str, Any]) -> str:
        """Create a new experiment record"""
        experiment_record = ExperimentRecord(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            config=config,
            metrics={},
            model_info={},
            results={}
        )
        
        self.db.save_experiment(experiment_record)
        return experiment_id
    
    def update_experiment_results(self, experiment_id: str, results: Dict[str, Any]) -> None:
        """Update experiment with results"""
        experiment = self.db.get_experiment(experiment_id)
        if experiment:
            experiment.results.update(results)
            self.db.save_experiment(experiment)
        else:
            logger.warning(f"Experiment {experiment_id} not found")
    
    def get_experiment_comparison(self) -> pd.DataFrame:
        """Get comparison of all experiments"""
        return self.db.get_experiments_summary()
    
    def get_best_experiment(self, metric: str = 'final_accuracy') -> Optional[ExperimentRecord]:
        """Get the best experiment based on a metric"""
        experiments = self.db.get_all_experiments()
        if not experiments:
            return None
        
        best_exp = max(experiments, key=lambda x: x.results.get(metric, 0))
        return best_exp


if __name__ == "__main__":
    # Test the mock database
    db = MockDatabase("test_experiments.db")
    
    # Create a test experiment
    test_record = ExperimentRecord(
        experiment_id="test_001",
        timestamp=datetime.now().isoformat(),
        config={"pruning_method": "l1_unstructured", "pruning_amount": 0.5},
        metrics={"train_loss": [0.5, 0.4, 0.3], "test_accuracy": [85.0, 87.0, 89.0]},
        model_info={"total_parameters": 1000000, "model_size_mb": 4.0},
        results={"baseline_accuracy": 90.0, "final_accuracy": 89.0, "accuracy_drop": 1.0}
    )
    
    # Save and retrieve
    db.save_experiment(test_record)
    retrieved = db.get_experiment("test_001")
    
    if retrieved:
        print("✅ Database test successful!")
        print(f"Retrieved experiment: {retrieved.experiment_id}")
        print(f"Final accuracy: {retrieved.results['final_accuracy']}%")
    else:
        print("❌ Database test failed!")
    
    # Clean up
    db.clear_database()
    Path("test_experiments.db").unlink(missing_ok=True)
