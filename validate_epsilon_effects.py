#!/usr/bin/env python3
# validate_epsilon_effects.py

import os
import sys
import argparse
import numpy as np
import random
import time
import pandas as pd

# Force matplotlib to use non-interactive Agg backend
import matplotlib
matplotlib.use('Agg')  # This must be set before importing pyplot
import matplotlib.pyplot as plt

from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simulator import Simulator
from core.auction import DistributedAuction
from core.task import Task, TaskDependencyGraph
from core.robot import Robot
from utils.data_collector import AuctionDataCollector

def run_epsilon_validation(output_dir=None, use_gpu=False):
    """Run validation tests for epsilon effects
    
    Args:
        output_dir: Directory to save validation results
        use_gpu: Whether to use GPU acceleration
    """
    if output_dir is None:
        output_dir = 'results/epsilon_validation'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Define epsilon values to test (wide range)
    epsilon_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # Define different task counts to test
    task_counts = [8, 16, 32]
    
    # Define communication parameters
    comm_delay = 0  # No delay for isolation test
    packet_loss = 0  # No packet loss for isolation test
    
    # Initialize data collector
    data_collector = AuctionDataCollector(output_dir=output_dir)
    
    # Create results DataFrame
    results = []
    
    print("Starting epsilon validation tests...")
    
    # Run tests for each task count and epsilon value
    for num_tasks in task_counts:
        for epsilon in tqdm(epsilon_values, desc=f"Testing {num_tasks} tasks"):
            # Create simulator
            simulator = Simulator(
                num_robots=2,
                workspace_size=(10, 8),
                comm_delay=comm_delay,
                packet_loss=packet_loss,
                epsilon=epsilon,
                use_gpu=use_gpu
            )
            
            # Generate random tasks
            simulator.generate_random_tasks(num_tasks)
            
            # Get robots and tasks
            robots = simulator.robots
            tasks = simulator.tasks
            task_graph = simulator.task_graph
            
            # Create auction algorithm
            auction = DistributedAuction(
                epsilon=epsilon,
                communication_delay=comm_delay,
                packet_loss_prob=packet_loss,
                use_gpu=use_gpu
            )
            
            # Set data collector
            auction.data_collector = data_collector
            
            # Run centralized solver for optimal solution
            from core.centralized_solver import CentralizedSolver
            centralized_solver = CentralizedSolver()
            optimal_solution = centralized_solver.solve(robots, tasks)
            auction.centralized_solution = optimal_solution
            
            # Run auction
            start_time = time.time()
            assignments, messages = auction.run_auction(robots, tasks, task_graph)
            runtime = time.time() - start_time
            
            # Calculate makespan
            makespan = 0
            for robot in robots:
                robot_tasks = [task for task in tasks if task.assigned_to == robot.id]
                robot_makespan = sum(task.completion_time for task in robot_tasks)
                makespan = max(makespan, robot_makespan)
            
            # Calculate optimality gap
            optimal_makespan = optimal_solution.get('makespan', 0)
            optimality_gap = (makespan - optimal_makespan) / optimal_makespan if optimal_makespan > 0 else 0
            
            # Get iteration count from data collector
            iteration_count = 0
            if auction.data_collector and auction.data_collector.runs:
                run = auction.data_collector.runs[-1]
                if run['iterations']:
                    iteration_count = max([item['iteration'] for item in run['iterations']])
            
            # Store results
            results.append({
                'num_tasks': num_tasks,
                'epsilon': epsilon,
                'message_count': messages,
                'makespan': makespan,
                'optimal_makespan': optimal_makespan,
                'optimality_gap': optimality_gap,
                'iteration_count': iteration_count,
                'runtime': runtime,
                'theoretical_bound': 2 * epsilon,
                'below_bound': optimality_gap <= 2 * epsilon
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'epsilon_validation_results.csv'), index=False)
    
    print(f"Validation tests completed. Results saved to {output_dir}")
    
    # Generate basic analysis
    print("Generating analysis...")
    
    # Save data collector session summary
    data_collector.save_session_summary()
    
    # Run analysis script
    try:
        from analyze_detailed_data import analyze_epsilon_effects
        analyze_epsilon_effects(output_dir, os.path.join(output_dir, 'analysis'))
    except Exception as e:
        print(f"Warning: Could not run analysis: {e}")
        
        # Generate basic plots
        for num_tasks in task_counts:
            task_results = results_df[results_df['num_tasks'] == num_tasks]
            
            # Plot message count vs epsilon
            plt.figure(figsize=(10, 6))
            plt.plot(task_results['epsilon'], task_results['message_count'], 'o-')
            plt.title(f'Message Count vs Epsilon ({num_tasks} tasks)')
            plt.xlabel('Epsilon')
            plt.ylabel('Message Count')
            plt.xscale('log')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'message_count_vs_epsilon_{num_tasks}_tasks.png'))
            plt.close()
            
            # Plot optimality gap vs epsilon
            plt.figure(figsize=(10, 6))
            plt.plot(task_results['epsilon'], task_results['optimality_gap'], 'o-')
            plt.plot(task_results['epsilon'], 2 * task_results['epsilon'], 'r--', label='2*epsilon bound')
            plt.title(f'Optimality Gap vs Epsilon ({num_tasks} tasks)')
            plt.xlabel('Epsilon')
            plt.ylabel('Optimality Gap')
            plt.xscale('log')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'optimality_gap_vs_epsilon_{num_tasks}_tasks.png'))
            plt.close()
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Validate epsilon effects on auction algorithm')
    parser.add_argument('--output', type=str, default=None, help='Output directory for validation results')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--analysis-only', action='store_true', 
                       help='Only analyze existing results without running new tests')
    args = parser.parse_args()
    
    if args.analysis_only:
        if args.output is None:
            print("Error: Must specify output directory with --output for analysis-only mode")
            return
        
        try:
            from analyze_detailed_data import analyze_epsilon_effects
            analyze_epsilon_effects(args.output, os.path.join(args.output, 'analysis'))
        except Exception as e:
            print(f"Error running analysis: {e}")
    else:
        run_epsilon_validation(args.output, args.use_gpu)

if __name__ == "__main__":
    main()