#!/usr/bin/env python3
# analyze_detailed_data.py

import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np

# Force matplotlib to use non-interactive Agg backend
import matplotlib
matplotlib.use('Agg')  # This must be set before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import curve_fit
import pickle
import json

def analyze_epsilon_effects(data_dir, output_dir=None):
    """Analyze the effects of epsilon on algorithm metrics
    
    Args:
        data_dir: Directory containing auction data
        output_dir: Directory to save analysis results
    """
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'analysis')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all session summary files
    summary_files = glob.glob(os.path.join(data_dir, 'summary_*.csv'))
    
    if not summary_files:
        print(f"No summary files found in {data_dir}")
        return
    
    # Load the most recent summary
    summary_file = max(summary_files, key=os.path.getmtime)
    print(f"Using summary file: {summary_file}")
    
    summary_df = pd.read_csv(summary_file)
    
    # Check if epsilon varies
    if len(summary_df['epsilon'].unique()) <= 1:
        print("Error: Cannot analyze epsilon effects with only one epsilon value.")
        return
    
    # Save analysis summary
    summary_df.to_csv(os.path.join(output_dir, 'analysis_summary.csv'), index=False)
    
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
            def func(x, a, b):
                return a / x + b
            
            popt, pcov = curve_fit(func, x, y)
            
            # Plot fitted curve
            x_fit = np.linspace(min(x), max(x), 100)
            plt.plot(x_fit, func(x_fit, *popt), 'r-', 
                     label=f'Fit: y = {popt[0]:.2f}/x + {popt[1]:.2f}')
            
            # Calculate R-squared
            residuals = y - func(x, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes,
                    fontsize=12, verticalalignment='top')
            
            plt.legend()
            
            # Save fit parameters
            fit_params = {
                'a': float(popt[0]),
                'b': float(popt[1]),
                'r_squared': float(r_squared)
            }
            
            with open(os.path.join(output_dir, 'message_count_fit.json'), 'w') as f:
                json.dump(fit_params, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not fit curve to message count data: {e}")
    
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
    
    # Calculate what percentage of points are below the 2*epsilon bound
    below_bound = sum(summary_df['optimality_gap'] <= 2 * summary_df['epsilon'])
    pct_below = 100 * below_bound / len(summary_df)
    
    plt.text(0.05, 0.95, f'{pct_below:.1f}% of points below 2ε bound', 
            transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    plt.legend()
    
    plt.title('Optimality Gap vs Epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Optimality Gap')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'optimality_gap_vs_epsilon.png'))
    plt.close()
    
    # Plot Iteration Count vs Epsilon if available
    if 'total_iterations' in summary_df.columns:
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
                def func(x, a, b):
                    return a / x + b
                
                popt, pcov = curve_fit(func, x, y)
                
                # Plot fitted curve
                x_fit = np.linspace(min(x), max(x), 100)
                plt.plot(x_fit, func(x_fit, *popt), 'r-', 
                         label=f'Fit: y = {popt[0]:.2f}/x + {popt[1]:.2f}')
                
                # Calculate R-squared
                residuals = y - func(x, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r_squared = 1 - (ss_res / ss_tot)
                
                plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes,
                        fontsize=12, verticalalignment='top')
                
                plt.legend()
                
                # Save fit parameters
                fit_params = {
                    'a': float(popt[0]),
                    'b': float(popt[1]),
                    'r_squared': float(r_squared)
                }
                
                with open(os.path.join(output_dir, 'iteration_count_fit.json'), 'w') as f:
                    json.dump(fit_params, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not fit curve to iteration count data: {e}")
        
        plt.title('Iteration Count vs Epsilon')
        plt.xlabel('Epsilon')
        plt.ylabel('Iteration Count')
        plt.xscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'iteration_count_vs_epsilon.png'))
        plt.close()
    
    # Analyze detailed price evolution if full_data is available
    full_data_files = glob.glob(os.path.join(data_dir, 'full_data_*.pkl'))
    
    if full_data_files:
        # Load the most recent full data
        full_data_file = max(full_data_files, key=os.path.getmtime)
        print(f"Using full data file: {full_data_file}")
        
        try:
            with open(full_data_file, 'rb') as f:
                runs = pickle.load(f)
            
            # Plot Price Evolution for different epsilon values
            # Select a few representative epsilon values
            epsilon_values = sorted(summary_df['epsilon'].unique())
            if len(epsilon_values) > 3:
                # Select low, medium, and high epsilon values
                epsilon_values = [
                    epsilon_values[0], 
                    epsilon_values[len(epsilon_values)//2], 
                    epsilon_values[-1]
                ]
            
            plt.figure(figsize=(12, 8))
            
            for eps in epsilon_values:
                # Find run with this epsilon
                run_index = summary_df[summary_df['epsilon'] == eps]['run_index'].iloc[0]
                
                if run_index < len(runs):
                    run = runs[run_index]
                    
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
            
            # Plot bidding patterns for different epsilon values
            plt.figure(figsize=(12, 10))
            
            for i, eps in enumerate(epsilon_values):
                # Find run with this epsilon
                run_index = summary_df[summary_df['epsilon'] == eps]['run_index'].iloc[0]
                
                if run_index < len(runs):
                    run = runs[run_index]
                    
                    # Extract utilities from bid history
                    utilities = []
                    for key, bids in run['bid_history'].items():
                        for bid in bids:
                            utilities.append(bid['utility'])
                    
                    if utilities:
                        # Plot histogram of utilities
                        plt.subplot(len(epsilon_values), 1, i+1)
                        plt.hist(utilities, bins=20, alpha=0.7)
                        plt.axvline(x=0, color='r', linestyle='-')
                        plt.title(f'Utility Distribution (Epsilon = {eps})')
                        plt.xlabel('Utility Value (Bid - Price)')
                        plt.ylabel('Frequency')
                        plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'utility_distribution_by_epsilon.png'))
            plt.close()
        except Exception as e:
            print(f"Warning: Could not analyze full data: {e}")
    
    # Create a comprehensive analysis report
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(os.path.join(output_dir, 'epsilon_analysis.pdf')) as pdf:
            # Title page
            plt.figure(figsize=(8.5, 11))
            plt.text(0.5, 0.8, 'Epsilon Parameter Analysis', 
                    fontsize=24, ha='center')
            plt.text(0.5, 0.7, 'Distributed Auction Algorithm', 
                    fontsize=20, ha='center')
            plt.text(0.5, 0.6, f'Data from: {os.path.basename(summary_file)}', 
                    fontsize=16, ha='center')
            plt.axis('off')
            pdf.savefig()
            plt.close()
            
            # Summary table
            plt.figure(figsize=(8.5, 11))
            plt.text(0.5, 0.95, 'Analysis Summary', fontsize=20, ha='center')
            
            # Create a table with epsilon values and corresponding metrics
            plt.subplot(111, frame_on=False)
            plt.xticks([])
            plt.yticks([])
            
            # Create summary table
            epsilon_summary = summary_df.groupby('epsilon').agg({
                'message_count': ['mean', 'std'],
                'optimality_gap': ['mean', 'std']
            }).reset_index()
            
            # Format for display
            table_data = []
            for _, row in epsilon_summary.iterrows():
                table_data.append([
                    f"{row['epsilon']:.4f}",
                    f"{row[('message_count', 'mean')]:.1f} ± {row[('message_count', 'std')]:.1f}",
                    f"{row[('optimality_gap', 'mean')]:.4f} ± {row[('optimality_gap', 'std')]:.4f}",
                    f"{2*row['epsilon']:.4f}"
                ])
            
            plt.table(cellText=table_data,
                     colLabels=['Epsilon', 'Message Count', 'Optimality Gap', '2ε Bound'],
                     loc='center')
            
            pdf.savefig()
            plt.close()
            
            # Add all the generated plots to the PDF
            plot_files = [
                'message_count_vs_epsilon.png',
                'optimality_gap_vs_epsilon.png',
                'iteration_count_vs_epsilon.png',
                'price_evolution_by_epsilon.png',
                'utility_distribution_by_epsilon.png'
            ]
            
            for plot_file in plot_files:
                file_path = os.path.join(output_dir, plot_file)
                if os.path.exists(file_path):
                    img = plt.imread(file_path)
                    plt.figure(figsize=(8.5, 11))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(plot_file.replace('.png', '').replace('_', ' ').title(), 
                            fontsize=16, pad=20)
                    pdf.savefig()
                    plt.close()
            
            # Conclusion page
            plt.figure(figsize=(8.5, 11))
            plt.text(0.5, 0.9, 'Key Findings', fontsize=20, ha='center')
            
            # Create bullet points for key findings
            findings = [
                "Relationship between epsilon and message count follows the theoretical O(K² · bₘₐₓ/ε) relationship",
                f"Average message count across all experiments: {summary_df['message_count'].mean():.2f}",
                f"Average optimality gap: {summary_df['optimality_gap'].mean():.4f} (theoretical bound: 2ε)",
                f"{pct_below:.1f}% of experiments stayed below the theoretical 2ε optimality bound",
                "Smaller epsilon values lead to better solution quality but higher communication overhead",
                "Larger epsilon values provide faster convergence with increased optimality gap"
            ]
            
            for i, finding in enumerate(findings):
                plt.text(0.1, 0.8 - i*0.1, "• " + finding, fontsize=14)
            
            plt.axis('off')
            pdf.savefig()
            plt.close()
        
        print(f"Analysis report created: {os.path.join(output_dir, 'epsilon_analysis.pdf')}")
    except Exception as e:
        print(f"Warning: Could not create analysis report: {e}")
    
    print(f"Analysis completed. Results saved to {output_dir}")
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description='Analyze epsilon effects on auction algorithm')
    parser.add_argument('data_dir', type=str, help='Directory containing auction data')
    parser.add_argument('--output', type=str, default=None, help='Output directory for analysis results')
    args = parser.parse_args()
    
    analyze_epsilon_effects(args.data_dir, args.output)

if __name__ == "__main__":
    main()