#!/usr/bin/env python3
# visualize_results.py
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('data_dir', type=str, help='Path to experiment result directory')
    parser.add_argument('--interactive', action='store_true', help='Force interactive mode')
    parser.add_argument('--output', type=str, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    # Configure matplotlib based on interactive setting
    if args.interactive:
        import matplotlib
        # Try different interactive backends
        backends = ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'WXAgg']
        
        for backend in backends:
            try:
                matplotlib.use(backend)
                break
            except:
                continue
        else:
            print("Warning: Could not set interactive backend, using Agg")
            matplotlib.use('Agg')
    else:
        import matplotlib
        matplotlib.use('Agg')
    
    # Import visualization libraries after backend is set
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.backends.backend_pdf import PdfPages
    
    # Handle path to plot data
    if os.path.isdir(args.data_dir):
        # Look for plot_data.pkl in standard locations
        potential_paths = [
            os.path.join(args.data_dir, 'plot_data', 'plot_data.pkl'),
            os.path.join(args.data_dir, 'plot_data.pkl')
        ]
        
        data_path = None
        for path in potential_paths:
            if os.path.exists(path):
                data_path = path
                break
                
        if data_path is None:
            # Search recursively for plot_data.pkl
            for root, dirs, files in os.walk(args.data_dir):
                if 'plot_data.pkl' in files:
                    data_path = os.path.join(root, 'plot_data.pkl')
                    break
    else:
        # Assume direct path to pickle file
        data_path = args.data_dir
    
    if not data_path or not os.path.exists(data_path):
        print(f"Error: Could not find plot data in {args.data_dir}")
        sys.exit(1)
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        # Use a 'visualizations' directory in the same location as the data
        output_dir = os.path.join(os.path.dirname(data_path), 'visualizations')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading plot data from: {data_path}")
    
    # Load plot data
    try:
        with open(data_path, 'rb') as f:
            plot_data = pickle.load(f)
        
        results = plot_data.get('results')
        agg_results = plot_data.get('agg_results')
        is_detailed = plot_data.get('is_detailed', False)
        factor_vars = plot_data.get('factor_vars', [])
        response_vars = plot_data.get('response_vars', [])
        
        print(f"Loaded plot data with {len(response_vars)} response variables")
        
        if not results is not None or agg_results is None or len(response_vars) == 0:
            # Try to reconstruct if data structure is different
            print("Warning: Plot data incomplete. Attempting to reconstruct.")
            
            # Find all DataFrames in the plot_data
            for key, value in plot_data.items():
                if isinstance(value, pd.DataFrame):
                    if 'results' not in locals() or results is None:
                        results = value
                    elif 'agg_results' not in locals() or agg_results is None:
                        agg_results = value
            
            # If we have results, try to reconstruct other info
            if results is not None:
                # Try to determine response vars
                if not response_vars:
                    potential_response_vars = ['makespan', 'message_count', 'optimality_gap', 'recovery_time']
                    response_vars = [v for v in potential_response_vars if v in results.columns]
                
                # Try to determine factor vars
                if not factor_vars:
                    potential_factor_vars = ['num_tasks', 'comm_delay', 'packet_loss', 'epsilon']
                    factor_vars = [v for v in potential_factor_vars if v in results.columns]
                
                # Check if detailed
                is_detailed = 'run_index' in results.columns
                
                # Create aggregated results if needed
                if agg_results is None and is_detailed:
                    grouped = results.groupby(factor_vars)
                    agg_results = grouped.mean().reset_index()
                elif agg_results is None:
                    agg_results = results.copy()
        
    except Exception as e:
        print(f"Error loading plot data: {e}")
        sys.exit(1)
    
    if not response_vars:
        print("No response variables found in plot data!")
        sys.exit(1)
    
    print(f"Generating visualizations in {'interactive' if args.interactive else 'headless'} mode")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate plots
    try:
        # 1. Main effects plots
        for response in response_vars:
            fig, axs = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Main Effects on {response}', fontsize=16)
            axs = axs.flatten()
            
            for i, factor in enumerate(factor_vars):
                if i >= len(axs):
                    break
                    
                if factor not in agg_results.columns or response not in agg_results.columns:
                    continue
                    
                means = agg_results.groupby(factor)[response].mean().reset_index()
                axs[i].plot(means[factor], means[response], marker='o', linewidth=2, markersize=8)
                axs[i].set_xlabel(factor, fontsize=12)
                axs[i].set_ylabel(response, fontsize=12)
                axs[i].set_title(f'Main effect of {factor}', fontsize=14)
                axs[i].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'main_effects_{response}.png'), dpi=300)
            plt.close()
        
        # 2. Correlation matrix of response variables
        if len(response_vars) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = agg_results[response_vars].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=.5, square=True)
            plt.title('Correlation Matrix of Response Variables', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'), dpi=300)
            plt.close()
        
        # 3. Performance across epsilon values
        if 'epsilon' in factor_vars:
            plt.figure(figsize=(12, 8))
            for response in response_vars:
                eps_values = sorted(agg_results['epsilon'].unique())
                response_means = []
                
                for eps in eps_values:
                    filtered = agg_results[agg_results['epsilon'] == eps]
                    response_means.append(filtered[response].mean())
                
                # Normalize for comparison
                if max(response_means) > 0:
                    response_means = [x/max(response_means) for x in response_means]
                    
                plt.plot(eps_values, response_means, marker='o', linewidth=2, label=response)
            
            plt.xlabel('Epsilon (minimum bid increment)', fontsize=12)
            plt.ylabel('Normalized Performance', fontsize=12)
            plt.title('Effect of Epsilon on Performance Metrics', fontsize=14)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'epsilon_performance.png'), dpi=300)
            plt.close()
        
        # 4. Additional plot: Makespan vs Task Count with comparison to theoretical bound
        if 'makespan' in response_vars and 'num_tasks' in factor_vars:
            plt.figure(figsize=(10, 6))
            task_counts = sorted(agg_results['num_tasks'].unique())
            makespan_avg = [agg_results[agg_results['num_tasks'] == tc]['makespan'].mean() for tc in task_counts]
            
            # Plot actual makespan
            plt.plot(task_counts, makespan_avg, 'bo-', linewidth=2, label='Actual Makespan')
            
            # Plot theoretical bound (O(K²)) - scaled to match the actual data
            if len(task_counts) > 0 and len(makespan_avg) > 0 and task_counts[0] > 0 and makespan_avg[0] > 0:
                scale_factor = makespan_avg[0] / (task_counts[0]**2)
                theoretical = [tc**2 * scale_factor for tc in task_counts]
                plt.plot(task_counts, theoretical, 'r--', linewidth=2, label='Theoretical O(K²)')
            
            plt.xlabel('Number of Tasks (K)', fontsize=12)
            plt.ylabel('Makespan (seconds)', fontsize=12)
            plt.title('Makespan vs Task Count with Theoretical Bound', fontsize=14)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'makespan_vs_tasks.png'), dpi=300)
            plt.close()
        
        print(f"Visualizations generated in {output_dir}")
        
        # If interactive mode, show a sample plot
        if args.interactive:
            try:
                plt.figure(figsize=(10, 8))
                plt.title("Interactive Mode Test")
                
                # Create a plot that works with whatever data we have
                if 'makespan' in response_vars and 'num_tasks' in factor_vars:
                    task_counts = sorted(agg_results['num_tasks'].unique())
                    makespan_avg = [agg_results[agg_results['num_tasks'] == tc]['makespan'].mean() for tc in task_counts]
                    plt.plot(task_counts, makespan_avg, 'bo-', linewidth=2, label='Makespan')
                    plt.xlabel('Number of Tasks (K)', fontsize=12)
                    plt.ylabel('Makespan (seconds)', fontsize=12)
                else:
                    # Just plot the first response variable in some way
                    response = response_vars[0]
                    plt.hist(agg_results[response], bins=10)
                    plt.xlabel(response, fontsize=12)
                    plt.ylabel('Frequency', fontsize=12)
                
                plt.grid(True)
                plt.show()
            except Exception as e:
                print(f"Could not display interactive plot: {e}")
    
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()