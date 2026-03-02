#!/usr/bin/env python3
# basic_viz.py
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def basic_visualize(results_dir, output_dir=None, interactive=False):
    """Create visualizations directly from the CSV results"""
    # Configure matplotlib
    if interactive:
        try:
            import matplotlib
            matplotlib.use('TkAgg')
        except:
            try:
                import matplotlib
                matplotlib.use('Qt5Agg')
            except:
                import matplotlib
                matplotlib.use('Agg')
                print("Warning: Could not set interactive backend")
    
    # Find the CSV files
    potential_files = [
        os.path.join(results_dir, 'experiment_results.csv'),
        os.path.join(results_dir, 'factorial_experiment_stats.csv'),
        os.path.join(results_dir, 'stats_factorial_experiment_detailed.csv'),
        os.path.join(results_dir, 'factorial_experiment_detailed.csv')
    ]
    
    csv_file = None
    for file_path in potential_files:
        if os.path.exists(file_path):
            csv_file = file_path
            break
    
    if csv_file is None:
        # Search for any CSV file
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_file = os.path.join(root, file)
                    break
            if csv_file:
                break
    
    if csv_file is None:
        print(f"Error: No CSV result files found in {results_dir}")
        return
    
    print(f"Using results file: {csv_file}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.join(results_dir, 'visualizations')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    try:
        data = pd.read_csv(csv_file)
        print(f"Loaded data with {len(data)} rows and {len(data.columns)} columns")
        print(f"Column names: {list(data.columns)}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    # IMPROVED: Identify factor variables more reliably
    factor_vars = []
    for column in data.columns:
        col_lower = column.lower()
        if any(param in col_lower for param in ['task', 'delay', 'loss', 'epsilon', 'eps']):
            factor_vars.append(column)
    
    # Manually identify the epsilon column - this is critical for our analysis
    epsilon_column = None
    for column in data.columns:
        col_lower = column.lower()
        if 'epsilon' in col_lower or 'eps' in col_lower:
            epsilon_column = column
            break
    
    # If we couldn't find epsilon, look for columns with values that match expected epsilon ranges
    if epsilon_column is None:
        for column in data.columns:
            if data[column].dtype in [np.float64, np.float32]:
                values = data[column].unique()
                # Typical epsilon values are small (0.01 to 10)
                if len(values) > 1 and all(0 < v < 20 for v in values if v > 0):
                    epsilon_column = column
                    print(f"Auto-detected epsilon column: {column} with values {values}")
                    break
    
    # IMPROVED: Identify response variables more reliably
    response_vars = []
    response_patterns = ['makespan', 'message', 'count', 'gap', 'recovery', 'completion']
    
    for column in data.columns:
        col_lower = column.lower()
        if any(pattern in col_lower for pattern in response_patterns) and column not in factor_vars:
            response_vars.append(column)
    
    # Add any columns with 'mean' suffix
    for column in data.columns:
        if '_mean' in column.lower() and column not in response_vars:
            response_vars.append(column)
    
    if not factor_vars:
        print("Error: Could not identify factor variables")
        print("Available columns:", list(data.columns))
        return
    
    if not response_vars:
        print("Error: Could not identify response variables")
        print("Available columns:", list(data.columns))
        return
    
    print(f"Factor variables: {factor_vars}")
    print(f"Response variables: {response_vars}")
    print(f"Epsilon column: {epsilon_column}")
    
    # Generate visualizations
    
    # 1. Main effects plots
    for response in response_vars:
        if response not in data.columns:
            continue
            
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Main Effects on {response}', fontsize=16)
        axs = axs.flatten()
        
        for i, factor in enumerate(factor_vars):
            if i >= len(axs) or factor not in data.columns:
                continue
                
            means = data.groupby(factor)[response].mean().reset_index()
            axs[i].plot(means[factor], means[response], marker='o', linewidth=2, markersize=8)
            axs[i].set_xlabel(factor, fontsize=12)
            axs[i].set_ylabel(response, fontsize=12)
            axs[i].set_title(f'Main effect of {factor}', fontsize=14)
            axs[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'main_effects_{response}.png'), dpi=300)
        plt.close()
    
    # 2. Correlation matrix
    if len(response_vars) > 1:
        valid_response_vars = [r for r in response_vars if r in data.columns]
        if len(valid_response_vars) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = data[valid_response_vars].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                      linewidths=.5, square=True)
            plt.title('Correlation Matrix of Response Variables', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300)
            plt.close()
    
    # 3. IMPROVED: Dedicated non-normalized epsilon plots to show raw effects
    if epsilon_column is not None:
        # Create dedicated directory for epsilon analysis
        epsilon_dir = os.path.join(output_dir, 'epsilon_analysis')
        os.makedirs(epsilon_dir, exist_ok=True)
        
        # For each response variable, create a non-normalized plot to show raw effects
        for response in response_vars:
            if response not in data.columns:
                continue
                
            # Create a basic line plot without normalization
            plt.figure(figsize=(10, 6))
            
            # Group by epsilon only
            means = data.groupby(epsilon_column)[response].mean().reset_index()
            
            # Sort by epsilon value
            means = means.sort_values(by=epsilon_column)
            
            # Plot the raw data
            plt.plot(means[epsilon_column], means[response], 'bo-', linewidth=2, markersize=8)
            plt.xlabel(f'Epsilon', fontsize=12)
            plt.ylabel(response, fontsize=12)
            plt.title(f'Raw Effect of Epsilon on {response}', fontsize=14)
            plt.grid(True)
            
            # Add trendline
            try:
                z = np.polyfit(means[epsilon_column], means[response], 1)
                p = np.poly1d(z)
                plt.plot(means[epsilon_column], p(means[epsilon_column]), "r--", 
                        linewidth=1, label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
                plt.legend()
            except:
                pass
                
            plt.tight_layout()
            plt.savefig(os.path.join(epsilon_dir, f'epsilon_raw_effect_{response}.png'), dpi=300)
            plt.close()
            
            # 3b. Create a separate plot controlling for other factors
            # Group by epsilon and other factors to see interactions
            for factor in factor_vars:
                if factor == epsilon_column or factor not in data.columns:
                    continue
                    
                plt.figure(figsize=(10, 6))
                
                # Prepare data for multi-line plot
                factor_values = sorted(data[factor].unique())
                
                for value in factor_values:
                    # Filter data for this factor value
                    filtered = data[data[factor] == value]
                    
                    # Group by epsilon and calculate mean
                    means = filtered.groupby(epsilon_column)[response].mean().reset_index()
                    
                    # Sort by epsilon
                    means = means.sort_values(by=epsilon_column)
                    
                    if not means.empty:
                        plt.plot(means[epsilon_column], means[response], 'o-', 
                                linewidth=2, markersize=6, label=f"{factor}={value}")
                
                plt.xlabel(f'Epsilon', fontsize=12)
                plt.ylabel(response, fontsize=12)
                plt.title(f'Effect of Epsilon on {response} by {factor}', fontsize=14)
                plt.grid(True)
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(epsilon_dir, f'epsilon_by_{factor}_{response}.png'), dpi=300)
                plt.close()
        
        # 3c. Create a summary plot specifically for optimality gap vs epsilon
        optimality_gap_col = next((col for col in response_vars if 'gap' in col.lower()), None)
        if optimality_gap_col:
            plt.figure(figsize=(10, 6))
            
            # Focus on the relationship between epsilon and optimality gap
            means = data.groupby(epsilon_column)[optimality_gap_col].mean().reset_index()
            means = means.sort_values(by=epsilon_column)
            
            # Plot theoretical vs actual
            plt.plot(means[epsilon_column], means[optimality_gap_col], 'bo-', 
                    linewidth=2, label='Actual Gap')
            
            # Add theoretical bound line (2*epsilon)
            epsilon_values = means[epsilon_column].values
            theoretical = 2 * epsilon_values
            plt.plot(epsilon_values, theoretical, 'r--', linewidth=2, 
                    label='Theoretical Bound (2ε)')
            
            plt.xlabel('Epsilon (ε)', fontsize=12)
            plt.ylabel('Optimality Gap', fontsize=12)
            plt.title('Optimality Gap vs Epsilon with Theoretical Bound', fontsize=14)
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(epsilon_dir, 'optimality_gap_vs_epsilon.png'), dpi=300)
            plt.close()
    
    # 4. Message count vs epsilon (should show inverse relationship)
    message_count_col = next((col for col in response_vars if 'message' in col.lower() or 'count' in col.lower()), None)
    if message_count_col and epsilon_column:
        plt.figure(figsize=(10, 6))
        
        # Group by epsilon only for message count
        means = data.groupby(epsilon_column)[message_count_col].mean().reset_index()
        means = means.sort_values(by=epsilon_column)
        
        # Plot the data
        plt.plot(means[epsilon_column], means[message_count_col], 'go-', 
                linewidth=2, markersize=8)
        
        # Try to add a 1/x curve fit to show inverse relationship
        try:
            # Fit 1/x model: y = a/x + b
            def func(x, a, b):
                return a/x + b
                
            from scipy.optimize import curve_fit
            popt, pcov = curve_fit(func, means[epsilon_column], means[message_count_col])
            
            x_fit = np.linspace(min(means[epsilon_column]), max(means[epsilon_column]), 100)
            y_fit = func(x_fit, *popt)
            
            plt.plot(x_fit, y_fit, 'r--', linewidth=1, 
                    label=f"Fit: y={popt[0]:.1f}/x+{popt[1]:.1f}")
            plt.legend()
        except:
            pass
        
        plt.xlabel('Epsilon (ε)', fontsize=12)
        plt.ylabel('Message Count', fontsize=12)
        plt.title('Message Count vs Epsilon', fontsize=14)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'message_count_vs_epsilon.png'), dpi=300)
        plt.close()
    
    # 5. Performance across epsilon values (normalized)
    if epsilon_column:
        plt.figure(figsize=(12, 8))
        
        valid_responses = [r for r in response_vars if r in data.columns]
        for response in valid_responses:
            # Group by epsilon and calculate mean
            eps_means = data.groupby(epsilon_column)[response].mean().reset_index()
            eps_means = eps_means.sort_values(by=epsilon_column)
            
            # Normalize for easier comparison across metrics
            max_value = eps_means[response].max()
            if max_value > 0:
                eps_means['normalized'] = eps_means[response] / max_value
                plt.plot(eps_means[epsilon_column], eps_means['normalized'], 
                        marker='o', linewidth=2, label=response)
        
        plt.xlabel('Epsilon (ε)', fontsize=12)
        plt.ylabel('Normalized Performance', fontsize=12)
        plt.title('Normalized Effect of Epsilon on Performance Metrics', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'normalized_epsilon_effects.png'), dpi=300)
        plt.close()
    
    # 6. Statistical analysis summary
    if epsilon_column:
        # Create a statistical summary table
        stat_summary = pd.DataFrame(columns=['Response', 'R²', 'p-value', 'Direction'])
        
        for response in response_vars:
            if response not in data.columns:
                continue
                
            try:
                # Simple linear regression: response ~ epsilon
                from scipy import stats
                
                # Group by epsilon to reduce noise
                means = data.groupby(epsilon_column)[response].mean().reset_index()
                
                # Calculate linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    means[epsilon_column], means[response])
                
                # Add to summary table
                stat_summary = stat_summary.append({
                    'Response': response,
                    'R²': r_value**2,
                    'p-value': p_value,
                    'Direction': 'Positive' if slope > 0 else 'Negative'
                }, ignore_index=True)
            except:
                pass
        
        # Save to file
        if not stat_summary.empty:
            stat_summary.to_csv(os.path.join(output_dir, 'epsilon_statistical_summary.csv'), index=False)
            
            # Create a visual table
            plt.figure(figsize=(10, 6))
            plt.axis('off')
            plt.title('Statistical Analysis of Epsilon Effects', fontsize=16)
            
            # Create a table
            table_data = []
            for _, row in stat_summary.iterrows():
                table_data.append([
                    row['Response'],
                    f"{row['R²']:.3f}",
                    f"{row['p-value']:.3f}",
                    row['Direction']
                ])
            
            table = plt.table(
                cellText=table_data,
                colLabels=['Response', 'R²', 'p-value', 'Direction'],
                loc='center',
                cellLoc='center',
                bbox=[0.2, 0.2, 0.6, 0.6]
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            
            plt.savefig(os.path.join(output_dir, 'epsilon_statistical_table.png'), dpi=300)
            plt.close()
    
    print(f"Visualizations generated in {output_dir}")
    
    # Show interactive visualization if requested
    if interactive:
        plt.figure(figsize=(10, 8))
        plt.title("Your Experiment Results")
        
        if epsilon_column and message_count_col:
            # Show epsilon vs message count
            means = data.groupby(epsilon_column)[message_count_col].mean().reset_index()
            means = means.sort_values(by=epsilon_column)
            
            plt.plot(means[epsilon_column], means[message_count_col], 'go-', 
                    linewidth=2, markersize=8)
            plt.xlabel('Epsilon (ε)', fontsize=12)
            plt.ylabel('Message Count', fontsize=12)
            plt.title('Message Count vs Epsilon', fontsize=14)
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, "No suitable data for interactive plot", 
                    ha='center', va='center', fontsize=14)
        
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display interactive plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Basic visualization of experiment results')
    parser.add_argument('results_dir', help='Directory containing experiment results')
    parser.add_argument('--output', help='Output directory for visualizations')
    parser.add_argument('--interactive', action='store_true', help='Show interactive plots')
    args = parser.parse_args()
    
    basic_visualize(args.results_dir, args.output, args.interactive)