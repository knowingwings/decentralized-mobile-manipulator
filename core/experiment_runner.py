# core/experiment_runner.py
import numpy as np
import pandas as pd
import os
import time
import datetime
import pickle
import random
from itertools import product
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path

from core.simulator import Simulator
from core.centralized_solver import CentralizedSolver


class ExperimentRunner:
    def __init__(self, config):
        """Initialize experiment runner
        
        Args:
            config: Experiment configuration dictionary
        """
        self.config = config
        
        # Create time-stamped directory with config name
        config_name = Path(config.get('config_file', 'default')).stem
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.results_dir = os.path.join('results', f"{config_name}_{timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Save the configuration for reference
        self._save_config()
        
        print(f"Results will be saved to: {self.results_dir}")
    
    def _save_config(self):
        """Save the configuration to the results directory"""
        import json
        import yaml
        
        # Try to save as YAML first (more readable)
        try:
            with open(os.path.join(self.results_dir, 'config.yaml'), 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception:
            # Fall back to JSON if YAML fails
            try:
                with open(os.path.join(self.results_dir, 'config.json'), 'w') as f:
                    json.dump(self.config, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save configuration: {e}")
    
    def run_factorial_experiment(self, num_processes=None):
        """Run full factorial experiment based on control variables
        
        Args:
            num_processes: Number of processes for parallel execution (None for auto)
            
        Returns:
            pd.DataFrame: Results dataframe
        """
        params = self.config['parameters']
        
        # Ensure all parameter values are lists, even if they are single values in config
        num_tasks = params['num_tasks']
        comm_delay = params['comm_delay']
        packet_loss = params['packet_loss']
        epsilon = params['epsilon']
        
        # Convert any single values to lists
        if not isinstance(num_tasks, list):
            num_tasks = [num_tasks]
        if not isinstance(comm_delay, list):
            comm_delay = [comm_delay]
        if not isinstance(packet_loss, list):
            packet_loss = [packet_loss]
        if not isinstance(epsilon, list):
            epsilon = [epsilon]
        
        # Include termination mode in parameter combinations if present
        termination_modes = params.get('termination_mode', ["assignment-complete"])
        if isinstance(termination_modes, str):
            termination_modes = [termination_modes]  # Convert single string to list
        
        # Create all parameter combinations
        if 'termination_mode' in params and len(termination_modes) > 1:
            # Include termination mode in combinations
            param_combinations = list(product(
                num_tasks,
                comm_delay,
                packet_loss,
                epsilon,
                termination_modes
            ))
        else:
            # Original combinations without termination mode
            param_combinations = list(product(
                num_tasks,
                comm_delay,
                packet_loss,
                epsilon
            ))
        
        # Number of repetitions per combination
        num_runs = self.config['experiment'].get('num_runs', 1)
        
        # Expand combinations with run index
        full_combinations = []
        for combo in param_combinations:
            for run in range(num_runs):
                # Add run index to the end
                full_combinations.append(combo + (run,))
        
        print(f"Running {len(full_combinations)} experiments "
            f"({len(param_combinations)} parameter combinations x {num_runs} runs)...")
        
        # When using AMD GPU via CUDA, we need to be careful with parallel processes
        # as they might compete for GPU memory
        amd_via_cuda = False
        if 'use_gpu' in self.config and self.config['use_gpu']:
            try:
                import torch
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0).lower()
                    if 'amd' in device_name or 'radeon' in device_name:
                        amd_via_cuda = True
            except:
                pass
        
        # Adjust process count based on GPU detection
        if num_processes is None:
            if amd_via_cuda:
                # With AMD GPU via CUDA, limit to fewer processes to avoid memory issues
                num_processes = min(4, os.cpu_count() - 1)
                print("AMD GPU detected via CUDA - limiting parallel processes to avoid memory issues")
            else:
                # Standard CPU case
                num_processes = os.cpu_count()
                if num_processes >= 8:
                    num_processes -= 1
        
        print(f"Using {num_processes} parallel processes")
        
        # Important: use 'spawn' method for better GPU compatibility
        ctx = mp.get_context('spawn')
        
        # When using AMD GPU via CUDA, try a small batch first
        if amd_via_cuda:
            test_size = min(num_processes, 4)
            print(f"Testing with {test_size} processes first to verify GPU stability...")
            test_combinations = full_combinations[:test_size]
            
            try:
                with ctx.Pool(processes=test_size) as pool:
                    test_results = list(pool.map(self._run_experiment_config, test_combinations))
                print("GPU test successful - proceeding with full experiment")
            except Exception as e:
                print(f"GPU test failed: {e}")
                print("Falling back to CPU-only execution")
                # Modify config to disable GPU
                self.config['use_gpu'] = False
        
        # Run the full experiment
        try:
            with ctx.Pool(processes=num_processes) as pool:
                # Use larger chunksize for better utilization
                chunksize = max(1, len(full_combinations) // (num_processes * 5))
                
                results = list(tqdm(pool.imap(self._run_experiment_config, 
                                            full_combinations,
                                            chunksize=chunksize), 
                                total=len(full_combinations)))
        except Exception as e:
            print(f"Parallel execution failed: {e}")
            print("Falling back to sequential execution")
            results = []
            for combo in tqdm(full_combinations):
                results.append(self._run_experiment_config(combo))
        
        # Combine results into DataFrame
        results_df = pd.DataFrame(results)
        
        # If multiple runs, add statistics
        if num_runs > 1:
            # Group by parameter combinations
            grouped_results = results_df.groupby(['num_tasks', 'comm_delay', 'packet_loss', 'epsilon'])
            
            # Calculate statistics
            stats_df = grouped_results.agg({
                'makespan': ['mean', 'std', 'min', 'max'],
                'message_count': ['mean', 'std', 'min', 'max'],
                'optimality_gap': ['mean', 'std', 'min', 'max'],
                'recovery_time': ['mean', 'std', 'min', 'max'],
                'completion_rate': ['mean', 'std', 'min', 'max'],
                'workload_balance': ['mean', 'std', 'min', 'max']
            }).reset_index()
            
            # Flatten column names
            stats_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                              for col in stats_df.columns.values]
            
            # Save both detailed and statistical results
            self.save_results(results_df, 'factorial_experiment_detailed.csv')
            self.save_results(stats_df, 'factorial_experiment_stats.csv')
            
            return stats_df
        
        return results_df
    
    def _run_experiment_config(self, param_combo):
        """Run experiment with specific parameter configuration
        
        Args:
            param_combo: Parameter combination tuple (num_tasks, comm_delay, packet_loss, epsilon, run_index)
                or with termination mode: (num_tasks, comm_delay, packet_loss, epsilon, termination_mode, run_index)
            
        Returns:
            dict: Experiment results
        """
        # Handle param_combo with or without termination_mode
        if len(param_combo) == 6:
            num_tasks, comm_delay, packet_loss, epsilon, termination_mode, run_index = param_combo
        else:
            num_tasks, comm_delay, packet_loss, epsilon, run_index = param_combo
            # Use default termination mode from config or fallback to assignment-complete
            termination_mode = self.config.get('parameters', {}).get('termination_mode', "assignment-complete")
        
        # Get price stability threshold from config or use default
        price_stability_threshold = self.config.get('parameters', {}).get('price_stability_threshold', 0.01)
        
        # Ensure price_stability_threshold is a scalar
        if isinstance(price_stability_threshold, (list, tuple)):
            price_stability_threshold = price_stability_threshold[0]
            
        # Set random seed for reproducibility
        random_seed = self.config['experiment']['random_seed'] + run_index
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Initialize simulator
        simulator = Simulator(
            num_robots=self.config['experiment']['robot_count'],
            workspace_size=self.config['experiment']['workspace_size'],
            comm_delay=comm_delay,
            packet_loss=packet_loss,
            epsilon=epsilon,
            use_gpu=self.config.get('use_gpu', False),
            termination_mode=termination_mode,
            price_stability_threshold=price_stability_threshold
        )
        
        # Generate random tasks
        simulator.generate_random_tasks(num_tasks)
        
        # Get centralized solution for comparison
        centralized_solver = CentralizedSolver()
        optimal_solution = centralized_solver.solve(simulator.robots, simulator.tasks)
        optimal_makespan = optimal_solution['makespan']
        
        # Run simulation with decentralized control
        start_time = time.time()
        simulation_results = simulator.run_simulation(
            self.config['experiment']['simulation_time'],
            inject_failure=self.config['experiment'].get('failure_probability', 0) > 0,
            failure_time_fraction=0.3,
            visualize=self.config.get('visualize', False)
        )
        execution_time = time.time() - start_time
        
        # Extract metrics
        makespan = simulation_results['makespan']
        message_count = simulation_results['message_count']
        recovery_time = simulation_results.get('recovery_time', 0)
        
        # Calculate optimality gap
        optimality_gap = (makespan - optimal_makespan) / optimal_makespan if optimal_makespan > 0 else 0
        
        # Create result record
        result = {
            'num_tasks': num_tasks,
            'comm_delay': comm_delay,
            'packet_loss': packet_loss,
            'epsilon': epsilon,
            'termination_mode': termination_mode,
            'run_index': run_index,
            'makespan': makespan,
            'message_count': message_count,
            'optimality_gap': optimality_gap,
            'recovery_time': recovery_time,
            'execution_time': execution_time,
            'completion_rate': simulation_results['completion_rate'],
            'workload_balance': simulation_results['workload_balance'],
            'optimal_makespan': optimal_makespan
        }
        
        return result
    
    def save_results(self, results_df, filename=None):
        """Save experiment results to CSV
        
        Args:
            results_df: Results DataFrame
            filename: Output filename (default: 'experiment_results.csv')
        """
        if filename is None:
            filename = 'experiment_results.csv'
        
        # Use just the filename without path, as we'll add our timestamped path
        base_filename = os.path.basename(filename)
        filepath = os.path.join(self.results_dir, base_filename)
        
        # Save CSV
        results_df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
        
        # Also save a pickle version for easier reloading
        pickle_path = filepath.replace('.csv', '.pkl')
        results_df.to_pickle(pickle_path)
        print(f"Pickle version saved to {pickle_path}")
        
        # Save detailed and stats versions
        if 'run_index' in results_df.columns:
            # This is detailed data - also create a stats version
            try:
                import pandas as pd
                # Group by parameters and calculate statistics
                grouped = results_df.groupby(['num_tasks', 'comm_delay', 'packet_loss', 'epsilon'])
                stats_df = grouped.agg({
                    'makespan': ['mean', 'std', 'min', 'max'],
                    'message_count': ['mean', 'std', 'min', 'max'],
                    'optimality_gap': ['mean', 'std', 'min', 'max'],
                    'recovery_time': ['mean', 'std', 'min', 'max'],
                    'completion_rate': ['mean', 'std', 'min', 'max'],
                    'workload_balance': ['mean', 'std', 'min', 'max']
                }).reset_index()
                
                # Flatten column names
                stats_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                  for col in stats_df.columns.values]
                
                # Save stats
                stats_filepath = os.path.join(self.results_dir, 'stats_' + base_filename)
                stats_df.to_csv(stats_filepath, index=False)
                print(f"Statistics saved to {stats_filepath}")
                
                # Save stats pickle
                stats_pickle_path = stats_filepath.replace('.csv', '.pkl')
                stats_df.to_pickle(stats_pickle_path)
                print(f"Statistics pickle saved to {stats_pickle_path}")
            except Exception as e:
                print(f"Warning: Could not create statistics: {e}")
        
        return filepath  # Return the path where results were saved