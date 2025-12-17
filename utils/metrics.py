# decentralized_control/utils/metrics.py

import numpy as np
import pandas as pd
import os
import pickle

class MetricsTracker:
    """Utility class for tracking and analyzing metrics during experiments"""
    
    def __init__(self, save_dir='results'):
        """Initialize metrics tracker
        
        Args:
            save_dir: Directory to save metrics data
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize metrics
        self.reset()
    
    def reset(self):
        """Reset metrics"""
        self.metrics_history = []
        self.current_metrics = {}
        self.experiment_results = []
    
    def update(self, metrics, time=None):
        """Update metrics with new values
        
        Args:
            metrics: Dictionary of metric values
            time: Simulation time (optional)
        """
        # Add time if provided
        if time is not None:
            metrics['time'] = time
        
        # Update current metrics
        self.current_metrics.update(metrics)
        
        # Add to history
        self.metrics_history.append(metrics.copy())
    
    def add_experiment_result(self, params, results):
        """Add results from a single experiment
        
        Args:
            params: Dictionary of experiment parameters
            results: Dictionary of results
        """
        # Combine parameters and results
        entry = {**params, **results}
        
        # Add to results list
        self.experiment_results.append(entry)
    
    def get_metrics_dataframe(self):
        """Get metrics history as a DataFrame
        
        Returns:
            pd.DataFrame: Metrics history
        """
        return pd.DataFrame(self.metrics_history)
    
    def get_experiment_results_dataframe(self):
        """Get experiment results as a DataFrame
        
        Returns:
            pd.DataFrame: Experiment results
        """
        return pd.DataFrame(self.experiment_results)
    
    def save_metrics(self, filename='metrics_history.csv'):
        """Save metrics history to CSV
        
        Args:
            filename: Output filename
        """
        filepath = os.path.join(self.save_dir, filename)
        self.get_metrics_dataframe().to_csv(filepath, index=False)
        print(f"Metrics saved to {filepath}")
    
    def save_experiment_results(self, filename='experiment_results.csv'):
        """Save experiment results to CSV
        
        Args:
            filename: Output filename
        """
        filepath = os.path.join(self.save_dir, filename)
        self.get_experiment_results_dataframe().to_csv(filepath, index=False)
        print(f"Experiment results saved to {filepath}")
    
    def save_all(self, prefix=''):
        """Save all data with optional prefix
        
        Args:
            prefix: Prefix for filenames
        """
        # Save CSV files
        self.save_metrics(f"{prefix}metrics_history.csv")
        self.save_experiment_results(f"{prefix}experiment_results.csv")
        
        # Save pickle files for easier reloading
        with open(os.path.join(self.save_dir, f"{prefix}all_data.pkl"), 'wb') as f:
            pickle.dump({
                'metrics_history': self.metrics_history,
                'experiment_results': self.experiment_results
            }, f)
        
        print(f"All data saved with prefix '{prefix}'")
    
    def load(self, filename):
        """Load saved data
        
        Args:
            filename: Pickle file to load
            
        Returns:
            bool: True if loading was successful
        """
        filepath = os.path.join(self.save_dir, filename)
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            self.metrics_history = data.get('metrics_history', [])
            self.experiment_results = data.get('experiment_results', [])
            
            if self.metrics_history:
                self.current_metrics = self.metrics_history[-1].copy()
            
            print(f"Data loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def calculate_statistics(self, group_by=None):
        """Calculate statistics on experiment results
        
        Args:
            group_by: List of parameter names to group by
            
        Returns:
            pd.DataFrame: Statistics dataframe
        """
        df = self.get_experiment_results_dataframe()
        
        if not df.empty:
            # Identify metrics columns (non-parameter columns)
            if group_by:
                metrics_cols = [col for col in df.columns if col not in group_by]
                
                # Group by parameters and calculate statistics
                grouped = df.groupby(group_by)[metrics_cols]
                stats = grouped.agg(['mean', 'std', 'min', 'max']).reset_index()
                
                return stats
            else:
                # Calculate overall statistics
                return df.describe()
        
        return pd.DataFrame()