# utils/data_collector.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle

class AuctionDataCollector:
    """Data collector for auction algorithm metrics"""
    
    def __init__(self, output_dir='results/auction_data'):
        """Initialize data collector
        
        Args:
            output_dir: Directory to save collected data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data structures
        self.reset()
        
        # Create a timestamp for this session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def reset(self):
        """Reset all collected data"""
        self.runs = []
        self.current_run = None
        self.current_run_index = -1
        self.current_iteration = 0
        self.current_run_metrics = {
            'iterations': [],
            'price_history': {},
            'bid_history': {},
            'utility_history': {},
            'assignment_history': {},
            'metrics': {}
        }
    
    def start_run(self, config):
        """Start a new run with given configuration
        
        Args:
            config: Dictionary of run configuration parameters
        """
        # If there's an active run, end it first
        if self.current_run is not None:
            self.end_run()
        
        # Create new run
        self.current_run = config
        self.current_run_index += 1
        self.current_iteration = 0
        
        # Initialize metrics for this run
        self.current_run_metrics = {
            'config': config,
            'iterations': [],
            'price_history': {},
            'bid_history': {},
            'utility_history': {},
            'assignment_history': {},
            'metrics': {}
        }
        
        # Log run start
        print(f"Data collection started for run {self.current_run_index}: epsilon={config.get('epsilon', 'N/A')}, tasks={config.get('num_tasks', 'N/A')}")
    
    def end_run(self, final_metrics=None):
        """End the current run and save metrics
        
        Args:
            final_metrics: Dictionary of final metrics for the run
        """
        if self.current_run is None:
            return
        
        # Add final metrics if provided
        if final_metrics is not None:
            self.current_run_metrics['metrics'].update(final_metrics)
        
        # Add run to list of runs
        self.runs.append(self.current_run_metrics)
        
        # Reset current run
        self.current_run = None
        
        # Save run data
        self.save_run_data(self.current_run_index)
        
        print(f"Data collection completed for run {self.current_run_index}")
    
    def record_iteration(self, iteration, data):
        """Record data for an iteration
        
        Args:
            iteration: Iteration number
            data: Dictionary of iteration data
        """
        if self.current_run is None:
            return
        
        # Ensure iteration number is up to date
        self.current_iteration = max(self.current_iteration, iteration)
        
        # Add iteration data
        self.current_run_metrics['iterations'].append({
            'iteration': iteration,
            **data
        })
    
    def record_price_update(self, task_id, old_price, new_price, epsilon, utility):
        """Record price update for a task
        
        Args:
            task_id: Task ID
            old_price: Previous price
            new_price: New price
            epsilon: Epsilon value used
            utility: Utility value
        """
        if self.current_run is None:
            return
        
        # Initialize price history for this task if needed
        if task_id not in self.current_run_metrics['price_history']:
            self.current_run_metrics['price_history'][task_id] = []
        
        # Add price update
        self.current_run_metrics['price_history'][task_id].append({
            'iteration': self.current_iteration,
            'old_price': old_price,
            'new_price': new_price,
            'epsilon': epsilon,
            'utility': utility,
            'price_increase': new_price - old_price
        })
    
    def record_bid(self, robot_id, task_id, bid_value, utility_value):
        """Record bid from a robot for a task
        
        Args:
            robot_id: Robot ID
            task_id: Task ID
            bid_value: Bid value
            utility_value: Utility value (bid - price)
        """
        if self.current_run is None:
            return
        
        # Initialize bid history for this robot-task pair if needed
        key = (robot_id, task_id)
        if key not in self.current_run_metrics['bid_history']:
            self.current_run_metrics['bid_history'][key] = []
        
        # Add bid
        self.current_run_metrics['bid_history'][key].append({
            'iteration': self.current_iteration,
            'bid': bid_value,
            'utility': utility_value
        })
        
        # Add to utility history as well (for easier analysis)
        if task_id not in self.current_run_metrics['utility_history']:
            self.current_run_metrics['utility_history'][task_id] = []
        
        self.current_run_metrics['utility_history'][task_id].append({
            'iteration': self.current_iteration,
            'robot_id': robot_id,
            'utility': utility_value
        })
    
    def record_assignment(self, task_id, robot_id, iteration):
        """Record task assignment
        
        Args:
            task_id: Task ID
            robot_id: Robot ID
            iteration: Iteration when assignment occurred
        """
        if self.current_run is None:
            return
        
        # Add assignment
        self.current_run_metrics['assignment_history'][task_id] = {
            'robot_id': robot_id,
            'iteration': iteration
        }
    
    def save_run_data(self, run_index):
        """Save data for a specific run
        
        Args:
            run_index: Index of the run to save
        """
        if run_index >= len(self.runs):
            return
        
        # Create run directory
        run_dir = os.path.join(self.output_dir, f"run_{self.session_id}_{run_index}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Save run data
        run_data = self.runs[run_index]
        
        # Save as pickle for easy loading
        with open(os.path.join(run_dir, 'run_data.pkl'), 'wb') as f:
            pickle.dump(run_data, f)
        
        # Save config as JSON
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(run_data['config'], f, indent=2)
        
        # Save iterations as CSV
        if run_data['iterations']:
            pd.DataFrame(run_data['iterations']).to_csv(
                os.path.join(run_dir, 'iterations.csv'), index=False)
        
        # Save price history as CSV
        if run_data['price_history']:
            # Flatten price history
            price_data = []
            for task_id, history in run_data['price_history'].items():
                for update in history:
                    price_data.append({
                        'task_id': task_id,
                        **update
                    })
            
            pd.DataFrame(price_data).to_csv(
                os.path.join(run_dir, 'price_history.csv'), index=False)
        
        # Save assignment history as CSV
        if run_data['assignment_history']:
            # Convert to list of dicts
            assignment_data = []
            for task_id, data in run_data['assignment_history'].items():
                assignment_data.append({
                    'task_id': task_id,
                    **data
                })
            
            pd.DataFrame(assignment_data).to_csv(
                os.path.join(run_dir, 'assignments.csv'), index=False)
        
        # Save metrics as JSON
        with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
            json.dump(run_data['metrics'], f, indent=2)
    
    def save_session_summary(self):
        """Save summary data for the entire session"""
        if not self.runs:
            return
        
        # Create summary for all runs
        summary_data = []
        for i, run in enumerate(self.runs):
            # Extract key metrics
            config = run['config']
            metrics = run['metrics']
            
            # Count iterations
            total_iterations = max([item['iteration'] for item in run['iterations']]) if run['iterations'] else 0
            
            # Create summary row
            summary_row = {
                'run_index': i,
                'epsilon': config.get('epsilon', None),
                'num_tasks': config.get('num_tasks', None),
                'comm_delay': config.get('comm_delay', None),
                'packet_loss': config.get('packet_loss', None),
                'total_iterations': total_iterations,
                'message_count': metrics.get('message_count', None),
                'makespan': metrics.get('makespan', None),
                'optimality_gap': metrics.get('optimality_gap', None),
                'completion_rate': metrics.get('completion_rate', None),
                'workload_balance': metrics.get('workload_balance', None)
            }
            
            summary_data.append(summary_row)
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_path = os.path.join(self.output_dir, f"summary_{self.session_id}.csv")
        summary_df.to_csv(summary_path, index=False)
        
        # Also save full data
        full_data_path = os.path.join(self.output_dir, f"full_data_{self.session_id}.pkl")
        with open(full_data_path, 'wb') as f:
            pickle.dump(self.runs, f)
        
        print(f"Session summary saved to {summary_path}")
    
    def analyze_epsilon_effects(self, output_dir=None):
        """Analyze the effects of epsilon on algorithm metrics
        
        Args:
            output_dir: Directory to save analysis results
        """
        if not self.runs:
            return
        
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, f"analysis_{self.session_id}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create summary DataFrame
        summary_data = []
        for i, run in enumerate(self.runs):
            # Extract key metrics
            config = run['config']
            metrics = run['metrics']
            
            # Count iterations
            total_iterations = max([item['iteration'] for item in run['iterations']]) if run['iterations'] else 0
            
            # Create summary row
            summary_row = {
                'run_index': i,
                'epsilon': config.get('epsilon', None),
                'num_tasks': config.get('num_tasks', None),
                'comm_delay': config.get('comm_delay', None),
                'packet_loss': config.get('packet_loss', None),
                'total_iterations': total_iterations,
                'message_count': metrics.get('message_count', None),
                'makespan': metrics.get('makespan', None),
                'optimality_gap': metrics.get('optimality_gap', None),
                'completion_rate': metrics.get('completion_rate', None),
                'workload_balance': metrics.get('workload_balance', None)
            }
            
            summary_data.append(summary_row)
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_path = os.path.join(output_dir, "analysis_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        # Generate plots if epsilon varies
        if len(summary_df['epsilon'].unique()) > 1:
            # Plot Message Count vs Epsilon
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=summary_df, x='epsilon', y='message_count')
            
            # Add 1/epsilon curve fit if possible
            if len(summary_df) > 2:
                try:
                    # Fit curve y = a/x + b
                    x = summary_df['epsilon'].values
                    y = summary_df['message_count'].values
                    
                    # Use only finite values
                    mask = np.isfinite(x) & np.isfinite(y)
                    x = x[mask]
                    y = y[mask]
                    
                    # Fit curve
                    from scipy.optimize import curve_fit
                    
                    def func(x, a, b):
                        return a / x + b
                    
                    popt, _ = curve_fit(func, x, y)
                    
                    # Plot fitted curve
                    x_fit = np.linspace(min(x), max(x), 100)
                    plt.plot(x_fit, func(x_fit, *popt), 'r-', 
                             label=f'Fit: y = {popt[0]:.2f}/x + {popt[1]:.2f}')
                    plt.legend()
                except:
                    pass
            
            plt.title('Message Count vs Epsilon')
            plt.xlabel('Epsilon')
            plt.ylabel('Message Count')
            plt.xscale('log')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'message_count_vs_epsilon.png'))
            plt.close()
            
            # Plot Optimality Gap vs Epsilon
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=summary_df, x='epsilon', y='optimality_gap')
            
            # Add 2*epsilon line
            x_range = np.linspace(min(summary_df['epsilon']), max(summary_df['epsilon']), 100)
            plt.plot(x_range, 2 * x_range, 'r--', label='2*epsilon bound')
            plt.legend()
            
            plt.title('Optimality Gap vs Epsilon')
            plt.xlabel('Epsilon')
            plt.ylabel('Optimality Gap')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'optimality_gap_vs_epsilon.png'))
            plt.close()
            
            # Plot Iteration Count vs Epsilon
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=summary_df, x='epsilon', y='total_iterations')
            
            # Add 1/epsilon curve fit if possible
            if len(summary_df) > 2:
                try:
                    # Fit curve y = a/x + b
                    x = summary_df['epsilon'].values
                    y = summary_df['total_iterations'].values
                    
                    # Use only finite values
                    mask = np.isfinite(x) & np.isfinite(y)
                    x = x[mask]
                    y = y[mask]
                    
                    # Fit curve
                    from scipy.optimize import curve_fit
                    
                    def func(x, a, b):
                        return a / x + b
                    
                    popt, _ = curve_fit(func, x, y)
                    
                    # Plot fitted curve
                    x_fit = np.linspace(min(x), max(x), 100)
                    plt.plot(x_fit, func(x_fit, *popt), 'r-', 
                             label=f'Fit: y = {popt[0]:.2f}/x + {popt[1]:.2f}')
                    plt.legend()
                except:
                    pass
            
            plt.title('Iteration Count vs Epsilon')
            plt.xlabel('Epsilon')
            plt.ylabel('Iteration Count')
            plt.xscale('log')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'iteration_count_vs_epsilon.png'))
            plt.close()
            
            # Plot Price Evolution for different epsilon values
            # Select a few representative epsilon values
            epsilon_values = sorted(summary_df['epsilon'].unique())
            if len(epsilon_values) > 3:
                epsilon_values = [epsilon_values[0], epsilon_values[len(epsilon_values)//2], epsilon_values[-1]]
            
            plt.figure(figsize=(12, 8))
            
            for eps in epsilon_values:
                # Find run with this epsilon
                run_index = summary_df[summary_df['epsilon'] == eps]['run_index'].iloc[0]
                run = self.runs[run_index]
                
                # Get price history for first task (if exists)
                if run['price_history'] and len(run['price_history']) > 0:
                    task_id = list(run['price_history'].keys())[0]
                    history = run['price_history'][task_id]
                    
                    # Extract price evolution
                    iterations = [update['iteration'] for update in history]
                    prices = [update['new_price'] for update in history]
                    
                    # Plot
                    plt.plot(iterations, prices, 'o-', label=f'Epsilon = {eps}')
            
            plt.title('Price Evolution by Epsilon Value')
            plt.xlabel('Iteration')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'price_evolution_by_epsilon.png'))
            plt.close()
        
        print(f"Analysis completed. Results saved to {output_dir}")
        
        return summary_df