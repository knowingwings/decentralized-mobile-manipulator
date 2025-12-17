# core/analysis.py
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

def configure_matplotlib(force_headless=False):
    """Configure matplotlib backend based on environment and preferences"""
    import matplotlib
    
    # Check if we're in WSL or other headless environment
    is_wsl = "microsoft-standard" in os.uname().release if hasattr(os, "uname") else False
    has_display = os.environ.get("DISPLAY", "") != ""
    
    # Force headless mode if requested or in environments that need it
    if force_headless or is_wsl or not has_display:
        matplotlib.use('Agg')  # Non-interactive backend
        return 'headless'
    else:
        # Try interactive backend
        try:
            # First try Qt5Agg as it's generally more robust
            matplotlib.use('Qt5Agg')
            import matplotlib.pyplot as plt
            plt.figure()
            plt.close()
            return 'interactive'
        except Exception:
            try:
                # Fall back to TkAgg
                matplotlib.use('TkAgg')
                import matplotlib.pyplot as plt
                plt.figure()
                plt.close()
                return 'interactive'
            except Exception:
                # If all else fails, use Agg
                matplotlib.use('Agg')
                return 'headless'

# Configure matplotlib with auto-detection by default
backend_mode = configure_matplotlib()

# Import plotting libraries AFTER setting the backend
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results(results_file):
    """Analyze experimental results and generate statistical analysis
    
    Args:
        results_file: Path to results CSV file
        
    Returns:
        dict: Analysis results
    """
    print(f"Analyzing results from {results_file}...")
    
    # Load results
    try:
        results = pd.read_csv(results_file)
        print(f"Successfully loaded data with {len(results)} rows and {len(results.columns)} columns")
        print(f"Column names: {results.columns.tolist()}")
    except Exception as e:
        print(f"Error loading results from {results_file}: {e}")
        return {}
    
    # Get output directory (same as the input file directory)
    output_dir = os.path.dirname(results_file)
    
    # Check if this is a stats file or detailed results
    is_detailed = 'run_index' in results.columns
    
    # If detailed results - aggregate first
    if is_detailed:
        # Detailed results - aggregate first
        print("Found detailed results with run_index - aggregating...")
        group_columns = [col for col in results.columns if col in 
                       ['num_tasks', 'comm_delay', 'packet_loss', 'epsilon']]
        
        if not group_columns:
            print("Warning: Could not identify parameter columns for grouping")
            group_columns = [col for col in results.columns if col != 'run_index' and 
                          not any(m in col.lower() for m in 
                               ['makespan', 'message', 'count', 'gap', 'recovery', 'time', 'rate'])]
            print(f"Using detected columns for grouping: {group_columns}")
        
        try:
            grouped = results.groupby(group_columns)
            results = grouped.mean().reset_index()
            print(f"Aggregated to {len(results)} parameter combinations")
        except Exception as e:
            print(f"Error during aggregation: {e}")
            # Continue with non-aggregated data
    
    # ANOVA analysis for each response variable
    # IMPROVED: Better detection of parameter and response variables
    param_patterns = ['num_tasks', 'task', 'comm_delay', 'delay', 'packet_loss', 'loss', 'epsilon', 'eps']
    response_patterns = ['makespan', 'message', 'count', 'gap', 'recovery', 'time', 'rate', 'balance']
    
    # Identify parameter variables
    factor_vars = []
    for col in results.columns:
        col_lower = col.lower()
        if any(param in col_lower for param in param_patterns):
            factor_vars.append(col)
    
    # Identify response variables
    response_vars = []
    for col in results.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in response_patterns) and col not in factor_vars:
            response_vars.append(col)
    
    # Also include any columns with 'mean' suffix that aren't factors
    for col in results.columns:
        if '_mean' in col.lower() and col not in factor_vars and col not in response_vars:
            response_vars.append(col)
    
    print(f"Identified factor variables: {factor_vars}")
    print(f"Identified response variables: {response_vars}")
    
    # Specifically identify epsilon column - critical for our analysis
    epsilon_col = None
    for col in factor_vars:
        col_lower = col.lower()
        if 'epsilon' in col_lower or 'eps' in col_lower:
            epsilon_col = col
            break
    
    print(f"Identified epsilon column: {epsilon_col}")
    
    # Perform ANOVA analysis
    anova_results = {}
    effect_sizes = {}
    
    for response in response_vars:
        if response not in results.columns:
            continue
            
        print(f"\nANOVA for {response}")
        
        # Create formula dynamically based on available factors
        formula_parts = []
        for factor in factor_vars:
            if factor in results.columns:
                formula_parts.append(f"C({factor})")
        
        formula = f"{response} ~ " + " + ".join(formula_parts)
        print(f"ANOVA formula: {formula}")
        
        try:
            model = ols(formula, data=results).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            print(anova_table)
            
            anova_results[response] = anova_table
            
            # Calculate effect sizes
            print("\nEffect sizes (partial eta-squared):")
            ss_total = anova_table['sum_sq'].sum()
            factor_effects = {}
            
            for factor in factor_vars:
                factor_name = f"C({factor})"
                if factor_name in anova_table.index:
                    ss_factor = anova_table.loc[factor_name, 'sum_sq']
                    eta_squared = ss_factor / ss_total
                    factor_effects[factor] = eta_squared
                    print(f"  {factor}: {eta_squared:.4f}")
            
            effect_sizes[response] = factor_effects
        except Exception as e:
            print(f"Error in ANOVA for {response}: {e}")
    
    # Regression analysis for parameter sensitivity
    print("\nRegression Analysis")
    regression_results = {}
    
    for response in response_vars:
        if response not in results.columns:
            continue
            
        # Select only numeric factor variables
        numeric_factors = []
        for factor in factor_vars:
            if factor in results.columns and results[factor].dtype in [np.float64, np.float32, np.int64, np.int32]:
                numeric_factors.append(factor)
        
        if not numeric_factors:
            print(f"No numeric factors found for regression analysis of {response}")
            continue
        
        try:
            X = results[numeric_factors]
            X = sm.add_constant(X)
            y = results[response]
            
            model = sm.OLS(y, X).fit()
            print(f"\nRegression for {response}")
            print(model.summary())
            
            regression_results[response] = {
                'params': model.params,
                'pvalues': model.pvalues,
                'rsquared': model.rsquared,
                'rsquared_adj': model.rsquared_adj
            }
        except Exception as e:
            print(f"Error in regression for {response}: {e}")
    
    # Calculate confidence intervals
    print("\n95% Confidence Intervals")
    confidence_intervals = {}
    
    for response in response_vars:
        if response not in results.columns:
            continue
            
        try:
            mean = results[response].mean()
            ci = stats.t.interval(0.95, len(results)-1, 
                                loc=mean, 
                                scale=stats.sem(results[response]))
            print(f"{response}: {mean:.2f} ({ci[0]:.2f}, {ci[1]:.2f})")
            
            confidence_intervals[response] = {
                'mean': mean,
                'lower': ci[0],
                'upper': ci[1]
            }
        except Exception as e:
            print(f"Error calculating confidence interval for {response}: {e}")
    
    # IMPROVED: Specialized analysis for epsilon effects
    if epsilon_col:
        analyze_epsilon_effects(results, epsilon_col, response_vars, output_dir)
    
    # Return results for potential further processing
    analysis_results = {
        'anova': anova_results,
        'regression': regression_results,
        'confidence_intervals': confidence_intervals,
        'summary_stats': results.describe(),
        'effect_sizes': effect_sizes
    }
    
    # Save analysis results
    analysis_file = os.path.join(output_dir, 'analysis_results.pkl')
    import pickle
    try:
        with open(analysis_file, 'wb') as f:
            pickle.dump(analysis_results, f)
        print(f"Analysis results saved to {analysis_file}")
    except Exception as e:
        print(f"Error saving analysis results: {e}")
    
    # Also save a text summary
    summary_file = os.path.join(output_dir, 'analysis_summary.txt')
    try:
        with open(summary_file, 'w') as f:
            f.write("# Analysis Summary\n\n")
            f.write(f"## Dataset: {os.path.basename(results_file)}\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(results.describe().to_string())
            f.write("\n\n")
            
            # ANOVA results
            f.write("## ANOVA Results\n\n")
            for response, anova in anova_results.items():
                f.write(f"### {response}\n\n")
                f.write(anova.to_string())
                f.write("\n\n")
            
            # Effect sizes
            f.write("## Effect Sizes (partial eta-squared)\n\n")
            for response, effects in effect_sizes.items():
                f.write(f"### {response}\n\n")
                for factor, size in effects.items():
                    f.write(f"{factor}: {size:.4f}\n")
                f.write("\n")
            
            # Regression results
            f.write("## Regression Results\n\n")
            for response, regression in regression_results.items():
                f.write(f"### {response}\n\n")
                f.write(f"R-squared: {regression['rsquared']:.4f}\n")
                f.write(f"Adjusted R-squared: {regression['rsquared_adj']:.4f}\n")
                f.write("Coefficients:\n")
                for param, value in regression['params'].items():
                    pvalue = regression['pvalues'][param]
                    f.write(f"  {param}: {value:.4f} (p-value: {pvalue:.4f})\n")
                f.write("\n")
            
            # Confidence intervals
            f.write("## 95% Confidence Intervals\n\n")
            for response, ci in confidence_intervals.items():
                f.write(f"{response}: {ci['mean']:.2f} ({ci['lower']:.2f}, {ci['upper']:.2f})\n")
        
        print(f"Analysis summary saved to {summary_file}")
    except Exception as e:
        print(f"Error saving summary: {e}")
    
    return analysis_results

def analyze_epsilon_effects(results, epsilon_col, response_vars, output_dir):
    """Specialized analysis function focused specifically on epsilon effects
    
    Args:
        results: DataFrame with results
        epsilon_col: Column name for epsilon
        response_vars: List of response variable columns
        output_dir: Output directory for analysis results
    """
    print(f"\nAnalyzing epsilon effects on {len(response_vars)} response variables...")
    
    # Create epsilon analysis directory
    epsilon_dir = os.path.join(output_dir, 'epsilon_analysis')
    os.makedirs(epsilon_dir, exist_ok=True)
    
    # Prepare summary results
    summary_data = {
        'Response': [],
        'Correlation': [],
        'R²': [],
        'p-value': [],
        'Slope': [],
        'Expected Relation': []
    }
    
    # Expected relationships based on theory
    expected_relations = {
        'makespan': 'positive',  # Higher epsilon leads to potentially worse assignments
        'message': 'negative',   # Higher epsilon leads to fewer messages (faster convergence)
        'gap': 'positive',       # Higher epsilon leads to larger optimality gap
        'recovery': 'negative'   # Higher epsilon may lead to faster recovery
    }
    
    # Analyze each response variable
    for response in response_vars:
        if response not in results.columns:
            continue
            
        print(f"Analyzing epsilon effect on {response}...")
        
        try:
            plt.figure(figsize=(12, 6))
            
            # Create two subplots
            plt.subplot(1, 2, 1)
            
            # Raw relationship (scatter plot)
            plt.scatter(results[epsilon_col], results[response], alpha=0.5)
            
            # Add trend line
            try:
                x = results[epsilon_col]
                y = results[response]
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), "r--", linewidth=2)
                
                # Calculate statistics
                slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
                correlation = r_value
                rsquared = r_value ** 2
                
                # Determine the observed relationship
                observed_relation = 'positive' if slope > 0 else 'negative'
                
                # Determine expected relationship based on keyword matching
                expected = None
                for key, relation in expected_relations.items():
                    if key in response.lower():
                        expected = relation
                        break
                
                # Add to summary data
                summary_data['Response'].append(response)
                summary_data['Correlation'].append(correlation)
                summary_data['R²'].append(rsquared)
                summary_data['p-value'].append(p_value)
                summary_data['Slope'].append(slope)
                summary_data['Expected Relation'].append(expected or 'Unknown')
                
                # Add annotation
                plt.annotate(f"R²: {rsquared:.3f}, p: {p_value:.3f}", 
                           xy=(0.05, 0.95), xycoords='axes fraction',
                           fontsize=10, ha='left', va='top',
                           bbox=dict(boxstyle='round', fc='white', alpha=0.7))
            except Exception as e:
                print(f"Error calculating trend: {e}")
            
            plt.title(f"Raw Relationship: {response} vs Epsilon")
            plt.xlabel("Epsilon")
            plt.ylabel(response)
            plt.grid(True)
            
            # Second subplot: grouped means
            plt.subplot(1, 2, 2)
            
            # Group by epsilon to reduce noise
            try:
                grouped = results.groupby(epsilon_col)[response].agg(['mean', 'std']).reset_index()
                
                # Plot grouped data with error bars
                plt.errorbar(grouped[epsilon_col], grouped['mean'], yerr=grouped['std'], 
                           fmt='o-', linewidth=2, capsize=5)
                
                # Add theoretical relationship if this is optimality gap
                if 'gap' in response.lower():
                    # Theoretical bound is 2*epsilon
                    theoretical = 2 * grouped[epsilon_col]
                    plt.plot(grouped[epsilon_col], theoretical, 'r--', linewidth=2, 
                           label='Theoretical (2ε)')
                    plt.legend()
                
                # For message count, try to fit inverse relationship
                if 'message' in response.lower() or 'count' in response.lower():
                    try:
                        from scipy.optimize import curve_fit
                        
                        def inverse_func(x, a, b):
                            return a/x + b
                        
                        popt, _ = curve_fit(inverse_func, grouped[epsilon_col], grouped['mean'])
                        
                        x_fit = np.linspace(min(grouped[epsilon_col]), max(grouped[epsilon_col]), 100)
                        y_fit = inverse_func(x_fit, *popt)
                        
                        plt.plot(x_fit, y_fit, 'g--', linewidth=2, 
                               label=f"{popt[0]:.1f}/ε + {popt[1]:.1f}")
                        plt.legend()
                    except Exception as e:
                        print(f"Error fitting inverse function: {e}")
                
                # Add trend line
                try:
                    z = np.polyfit(grouped[epsilon_col], grouped['mean'], 1)
                    p = np.poly1d(z)
                    plt.plot(grouped[epsilon_col], p(grouped[epsilon_col]), "r:", linewidth=1)
                except Exception as e:
                    print(f"Error fitting trend line: {e}")
                
            except Exception as e:
                print(f"Error creating grouped plot: {e}")
                plt.text(0.5, 0.5, "Error creating grouped plot", 
                       ha='center', va='center', transform=plt.gca().transAxes)
            
            plt.title(f"Grouped Means: {response} vs Epsilon")
            plt.xlabel("Epsilon")
            plt.ylabel(f"Mean {response}")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(epsilon_dir, f"epsilon_effect_{response}.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error creating plot for {response}: {e}")
    
    # Create summary DataFrame
    if summary_data['Response']:
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary to CSV
        try:
            summary_path = os.path.join(epsilon_dir, "epsilon_effects_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"Epsilon effects summary saved to {summary_path}")
            
            # Create a summary visualization
            plt.figure(figsize=(12, 8))
            
            # Plot correlation coefficients
            bars = plt.bar(summary_df['Response'], summary_df['Correlation'])
            
            # Color bars based on statistical significance
            for i, p in enumerate(summary_df['p-value']):
                if p < 0.01:
                    bars[i].set_color('green')
                elif p < 0.05:
                    bars[i].set_color('orange')
                else:
                    bars[i].set_color('red')
            
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.axhline(y=0.3, color='gray', linestyle='--', alpha=0.3)
            plt.axhline(y=-0.3, color='gray', linestyle='--', alpha=0.3)
            
            plt.title("Correlation of Response Variables with Epsilon")
            plt.xlabel("Response Variable")
            plt.ylabel("Correlation Coefficient")
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            # Add a legend for p-value colors
            import matplotlib.patches as mpatches
            legend_elements = [
                mpatches.Patch(color='green', label='p < 0.01 (highly significant)'),
                mpatches.Patch(color='orange', label='p < 0.05 (significant)'),
                mpatches.Patch(color='red', label='p ≥ 0.05 (not significant)')
            ]
            plt.legend(handles=legend_elements, loc='best')
            
            plt.tight_layout()
            plt.savefig(os.path.join(epsilon_dir, "epsilon_correlations_summary.png"), dpi=300)
            plt.close()
            
            # Create specialized theoretical relationship plots
            create_theoretical_plots(results, epsilon_col, response_vars, epsilon_dir)
            
        except Exception as e:
            print(f"Error saving epsilon effects summary: {e}")
    else:
        print("No epsilon effect data collected. Check for errors above.")

def create_theoretical_plots(results, epsilon_col, response_vars, output_dir):
    """Create specialized plots showing theoretical relationships"""
    # Find message count and optimality gap columns
    message_col = next((col for col in response_vars if 'message' in col.lower() or 'count' in col.lower()), None)
    gap_col = next((col for col in response_vars if 'gap' in col.lower()), None)
    
    if message_col and gap_col:
        try:
            plt.figure(figsize=(12, 6))
            
            # Create two subplots for theoretical relationships
            plt.subplot(1, 2, 1)
            
            # Group by epsilon
            grouped = results.groupby(epsilon_col)[message_col].mean().reset_index()
            grouped = grouped.sort_values(by=epsilon_col)  # Ensure sorted order
            
            # Plot message count vs epsilon
            plt.plot(grouped[epsilon_col], grouped[message_col], 'bo-', linewidth=2, label="Actual")
            
            # Try to fit a 1/x curve (messages ~ K²·bmax/ε)
            try:
                from scipy.optimize import curve_fit
                
                def inverse_func(x, a, b):
                    return a/x + b
                
                popt, _ = curve_fit(inverse_func, grouped[epsilon_col], grouped[message_col])
                
                x_fit = np.linspace(min(grouped[epsilon_col]), max(grouped[epsilon_col]), 100)
                y_fit = inverse_func(x_fit, *popt)
                
                plt.plot(x_fit, y_fit, 'r--', linewidth=2, 
                      label=f"Theoretical: {popt[0]:.1f}/ε + {popt[1]:.1f}")
                plt.legend()
            except Exception as e:
                print(f"Error fitting inverse curve: {e}")
            
            plt.title("Message Count vs Epsilon\n(Theory: Messages ~ K²·bmax/ε)")
            plt.xlabel("Epsilon (ε)")
            plt.ylabel("Message Count")
            plt.grid(True)
            
            # Second subplot: optimality gap
            plt.subplot(1, 2, 2)
            
            # Group by epsilon
            grouped = results.groupby(epsilon_col)[gap_col].mean().reset_index()
            grouped = grouped.sort_values(by=epsilon_col)  # Ensure sorted order
            
            # Plot optimality gap vs epsilon
            plt.plot(grouped[epsilon_col], grouped[gap_col], 'go-', linewidth=2, label='Actual')
            
            # Plot theoretical bound (2*epsilon)
            theoretical = 2 * grouped[epsilon_col]
            plt.plot(grouped[epsilon_col], theoretical, 'r--', linewidth=2, label='Theoretical (2ε)')
            
            plt.title("Optimality Gap vs Epsilon\n(Theory: Gap ≤ 2ε)")
            plt.xlabel("Epsilon (ε)")
            plt.ylabel("Optimality Gap")
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "epsilon_theoretical_relationships.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error creating theoretical plots: {e}")

def generate_reports(results_file, output_dir=None, headless=None, save_data=True):
    """Generate plots and visualization of experimental results
    
    Args:
        results_file: Path to results CSV file
        output_dir: Output directory for plots (defaults to results_file directory)
        headless: Force headless mode if True, interactive if False, auto-detect if None
        save_data: Save plot data for remote visualization
    """
    # Configure matplotlib backend based on headless parameter
    if headless is not None:
        global backend_mode
        backend_mode = configure_matplotlib(force_headless=headless)
    
    if output_dir is None:
        output_dir = os.path.dirname(results_file)
    
    print(f"Generating reports from {results_file}...")
    print(f"Using {backend_mode} matplotlib backend")
    
    # Create plots directory within the output directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # If saving data for remote visualization, create a data directory
    data_dir = None
    if save_data:
        data_dir = os.path.join(output_dir, 'plot_data')
        os.makedirs(data_dir, exist_ok=True)
    
    # Load results
    try:
        results = pd.read_csv(results_file)
        print(f"Successfully loaded data with {len(results)} rows and {len(results.columns)} columns")
    except Exception as e:
        print(f"Error loading results from {results_file}: {e}")
        return
    
    # Check if this is a stats file or detailed results
    is_detailed = 'run_index' in results.columns
    
    # If detailed, we can also do analysis by run
    if is_detailed:
        # We'll use both the detailed data and aggregated data
        group_columns = [col for col in results.columns if col in 
                       ['num_tasks', 'comm_delay', 'packet_loss', 'epsilon']]
        
        if not group_columns:
            print("Warning: Could not identify parameter columns for grouping")
            # Try to detect parameters based on column names
            group_columns = [col for col in results.columns if col != 'run_index' and 
                          not any(m in col.lower() for m in 
                               ['makespan', 'message', 'count', 'gap', 'recovery', 'time', 'rate'])]
            print(f"Using detected columns for grouping: {group_columns}")
        
        try:
            grouped = results.groupby(group_columns)
            agg_results = grouped.mean().reset_index()
            print(f"Aggregated to {len(agg_results)} parameter combinations")
        except Exception as e:
            print(f"Error during aggregation: {e}")
            agg_results = results  # Fall back to using full results
    else:
        agg_results = results
    
    # Identify factor variables and response variables
    # IMPROVED: Better detection of parameter and response variables
    param_patterns = ['num_tasks', 'task', 'comm_delay', 'delay', 'packet_loss', 'loss', 'epsilon', 'eps']
    response_patterns = ['makespan', 'message', 'count', 'gap', 'recovery', 'time', 'rate', 'balance']
    
    # Identify parameter variables
    factor_vars = []
    for col in agg_results.columns:
        col_lower = col.lower()
        if any(param in col_lower for param in param_patterns):
            factor_vars.append(col)
    
    # Identify response variables
    response_vars = []
    for col in agg_results.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in response_patterns) and col not in factor_vars:
            response_vars.append(col)
    
    # Also include any columns with 'mean' suffix that aren't factors
    for col in agg_results.columns:
        if '_mean' in col.lower() and col not in factor_vars and col not in response_vars:
            response_vars.append(col)
    
    print(f"Identified factor variables: {factor_vars}")
    print(f"Identified response variables: {response_vars}")
    
    # Specifically identify epsilon column - critical for our analysis
    epsilon_col = None
    for col in factor_vars:
        col_lower = col.lower()
        if 'epsilon' in col_lower or 'eps' in col_lower:
            epsilon_col = col
            break
    
    print(f"Identified epsilon column: {epsilon_col}")
    
    # Save results data for remote visualization if requested
    if save_data:
        import pickle
        plot_data = {
            'results': results,
            'agg_results': agg_results,
            'is_detailed': is_detailed,
            'factor_vars': factor_vars,
            'response_vars': response_vars,
            'epsilon_col': epsilon_col
        }
        try:
            with open(os.path.join(data_dir, 'plot_data.pkl'), 'wb') as f:
                pickle.dump(plot_data, f)
            print(f"Plot data saved to {os.path.join(data_dir, 'plot_data.pkl')}")
        except Exception as e:
            print(f"Error saving plot data: {e}")
        
    # Generate plots
    try:
        # 1. Main effects plots
        for response in response_vars:
            if response not in agg_results.columns:
                continue
                
            plt.figure(figsize=(15, 12))
            num_factors = len(factor_vars)
            nrows = (num_factors + 1) // 2  # Calculate needed rows
            ncols = 2
            
            for i, factor in enumerate(factor_vars):
                if factor not in agg_results.columns:
                    continue
                    
                ax = plt.subplot(nrows, ncols, i+1)
                
                if is_detailed:
                    # With detailed data, we can calculate confidence intervals
                    try:
                        means = agg_results.groupby(factor)[response].mean().reset_index()
                        
                        # If we have standard deviations in the aggregated data
                        if f"{response}_std" in agg_results.columns:
                            ci = agg_results.groupby(factor)[f"{response}_std"].mean().reset_index()
                            ci[response] = ci[f"{response}_std"] * 1.96  # 95% CI
                        else:
                            # Estimate from original data if possible
                            ci_data = []
                            for val in means[factor]:
                                subset = results[results[factor] == val][response]
                                if len(subset) > 1:
                                    ci_val = 1.96 * subset.std() / np.sqrt(len(subset))
                                else:
                                    ci_val = 0
                                ci_data.append(ci_val)
                            ci = pd.DataFrame({factor: means[factor], response: ci_data})
                        
                        ax.errorbar(means[factor], means[response], yerr=ci[response], 
                                  marker='o', capsize=5, linewidth=2, markersize=8)
                    except Exception as e:
                        print(f"Error creating errorbar plot for {factor}: {e}")
                        # Fall back to simple plot
                        means = agg_results.groupby(factor)[response].mean().reset_index()
                        ax.plot(means[factor], means[response], 
                              marker='o', linewidth=2, markersize=8)
                else:
                    # With aggregated data, just plot the means
                    means = agg_results.groupby(factor)[response].mean().reset_index()
                    ax.plot(means[factor], means[response], 
                          marker='o', linewidth=2, markersize=8)
                    
                ax.set_xlabel(factor, fontsize=12)
                ax.set_ylabel(response, fontsize=12)
                ax.set_title(f'Main effect of {factor}', fontsize=14)
                ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'main_effects_{response}.png'), dpi=300)
            plt.close()
        
        # 2. Interaction plots - but create them more selectively
        for response in response_vars:
            if response not in agg_results.columns:
                continue
                
            # Only create interaction plots for epsilon with other factors
            if epsilon_col:
                for factor in factor_vars:
                    if factor == epsilon_col or factor not in agg_results.columns:
                        continue
                        
                    plt.figure(figsize=(10, 8))
                    
                    # Create interaction plot
                    try:
                        interaction_data = agg_results.pivot_table(
                            values=response, 
                            index=epsilon_col, 
                            columns=factor,
                            aggfunc='mean'
                        )
                        
                        # Plot as heatmap
                        sns.heatmap(interaction_data, annot=True, cmap='viridis', fmt='.2f', 
                                 linewidths=.5, cbar_kws={'label': response})
                        
                        plt.title(f'Interaction Effect of {epsilon_col} and {factor} on {response}', 
                                fontsize=14)
                        plt.tight_layout()
                        plt.savefig(os.path.join(plots_dir, 
                                             f'interaction_{response}_{epsilon_col}_{factor}.png'), dpi=300)
                        plt.close()
                    except Exception as e:
                        print(f"Error creating interaction plot for {response} with {factor}: {e}")
        
        # 3. Response distributions
        if is_detailed:
            plt.figure(figsize=(15, 10))
            
            # Calculate number of rows and columns for subplots
            num_responses = len(response_vars)
            ncols = min(2, num_responses)
            nrows = (num_responses + ncols - 1) // ncols  # Ceiling division
            
            for i, response in enumerate(response_vars):
                if response not in results.columns:
                    continue
                    
                if i < nrows * ncols:  # Make sure we don't create more subplots than needed
                    ax = plt.subplot(nrows, ncols, i+1)
                    try:
                        sns.histplot(results[response], kde=True, bins=20, ax=ax)
                        ax.set_title(f'Distribution of {response}', fontsize=14)
                        ax.set_xlabel(response, fontsize=12)
                        ax.grid(True)
                    except Exception as e:
                        print(f"Error creating histogram for {response}: {e}")
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'response_distributions.png'), dpi=300)
            plt.close()
        
        # 4. Correlation matrix of response variables
        try:
            valid_responses = [r for r in response_vars if r in agg_results.columns]
            if len(valid_responses) > 1:
                plt.figure(figsize=(10, 8))
                corr_matrix = agg_results[valid_responses].corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Upper triangle mask
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                          linewidths=.5, square=True, mask=mask)
                plt.title('Correlation Matrix of Response Variables', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'), dpi=300)
                plt.close()
        except Exception as e:
            print(f"Error creating correlation matrix: {e}")
        
        # 5. Parameter sensitivity plot (regression coefficients)
        try:
            # Run regression for each response
            plt.figure(figsize=(12, 10))
            
            # Calculate number of rows and columns for subplots
            num_responses = len(response_vars)
            ncols = min(2, num_responses)
            nrows = (num_responses + ncols - 1) // ncols  # Ceiling division
            
            for i, response in enumerate(response_vars):
                if response not in agg_results.columns:
                    continue
                    
                if i < nrows * ncols:  # Make sure we don't create more subplots than needed
                    ax = plt.subplot(nrows, ncols, i+1)
                    
                    # Filter to numeric factors
                    numeric_factors = []
                    for factor in factor_vars:
                        if factor in agg_results.columns and agg_results[factor].dtype in [np.float64, np.float32, np.int64, np.int32]:
                            numeric_factors.append(factor)
                    
                    if not numeric_factors:
                        ax.text(0.5, 0.5, "No numeric factors available", 
                              ha='center', va='center', transform=ax.transAxes)
                        continue
                    
                    # Fit regression model
                    try:
                        X = sm.add_constant(agg_results[numeric_factors])
                        y = agg_results[response]
                        model = sm.OLS(y, X).fit()
                        
                        # Extract standardized coefficients
                        coefs = model.params[1:]  # Skip intercept
                        stds = np.array([agg_results[f].std() for f in numeric_factors])
                        y_std = agg_results[response].std()
                        std_coefs = coefs * stds / y_std
                        
                        # Plot coefficients
                        bars = ax.bar(numeric_factors, std_coefs)
                        for bar, pval in zip(bars, model.pvalues[1:]):
                            if pval < 0.05:
                                bar.set_color('green')
                            elif pval < 0.1:
                                bar.set_color('yellow')
                            else:
                                bar.set_color('gray')
                        
                        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                        ax.set_title(f'Standardized Coefficients for {response}', fontsize=14)
                        ax.set_xlabel('Parameters', fontsize=12)
                        ax.set_ylabel('Standardized Coefficient', fontsize=12)
                        ax.grid(axis='y', alpha=0.3)
                        ax.tick_params(axis='x', rotation=45)
                    except Exception as e:
                        print(f"Error in regression for {response}: {e}")
                        ax.text(0.5, 0.5, f"Error in regression: {str(e)[:30]}...", 
                              ha='center', va='center', transform=ax.transAxes)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'parameter_sensitivity.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error creating parameter sensitivity plot: {e}")
        
        # 6. Create specialized epsilon plots
        if epsilon_col:
            # Create dedicated directory for epsilon analysis
            epsilon_dir = os.path.join(plots_dir, 'epsilon_analysis')
            os.makedirs(epsilon_dir, exist_ok=True)
            
            # 6a. Message count vs epsilon
            message_count_col = next((col for col in response_vars 
                                   if 'message' in col.lower() or 'count' in col.lower()), None)
            if message_count_col and message_count_col in agg_results.columns:
                try:
                    plt.figure(figsize=(10, 6))
                    
                    # Group by epsilon
                    means = agg_results.groupby(epsilon_col)[message_count_col].mean().reset_index()
                    means = means.sort_values(by=epsilon_col)  # Ensure sorted order
                    
                    # Plot message count vs epsilon
                    plt.plot(means[epsilon_col], means[message_count_col], 'bo-', linewidth=2, label="Actual")
                    
                    # Try to fit a 1/x curve (messages ~ K²·bmax/ε)
                    try:
                        from scipy.optimize import curve_fit
                        
                        def inverse_func(x, a, b):
                            return a/x + b
                        
                        popt, _ = curve_fit(inverse_func, means[epsilon_col], means[message_count_col])
                        
                        x_fit = np.linspace(min(means[epsilon_col]), max(means[epsilon_col]), 100)
                        y_fit = inverse_func(x_fit, *popt)
                        
                        plt.plot(x_fit, y_fit, 'r--', linewidth=2, 
                              label=f"Theoretical: {popt[0]:.1f}/ε + {popt[1]:.1f}")
                        plt.legend()
                    except Exception as e:
                        print(f"Error fitting inverse curve: {e}")
                    
                    plt.title("Message Count vs Epsilon\n(Theory: Messages ~ K²·bmax/ε)")
                    plt.xlabel("Epsilon (ε)")
                    plt.ylabel("Message Count")
                    plt.grid(True)
                    plt.tight_layout()
                    
                    # Save regular and log scale versions
                    plt.savefig(os.path.join(epsilon_dir, "message_count_vs_epsilon.png"), dpi=300)
                    
                    # Log scale version
                    plt.xscale('log')
                    plt.savefig(os.path.join(epsilon_dir, "message_count_vs_epsilon_log.png"), dpi=300)
                    plt.close()
                except Exception as e:
                    print(f"Error creating message count vs epsilon plot: {e}")
            
            # 6b. Optimality gap vs epsilon
            gap_col = next((col for col in response_vars if 'gap' in col.lower()), None)
            if gap_col and gap_col in agg_results.columns:
                try:
                    plt.figure(figsize=(10, 6))
                    
                    # Group by epsilon
                    means = agg_results.groupby(epsilon_col)[gap_col].mean().reset_index()
                    means = means.sort_values(by=epsilon_col)  # Ensure sorted order
                    
                    # Plot optimality gap vs epsilon
                    plt.plot(means[epsilon_col], means[gap_col], 'go-', linewidth=2, label='Actual')
                    
                    # Plot theoretical bound (2*epsilon)
                    theoretical = 2 * means[epsilon_col]
                    plt.plot(means[epsilon_col], theoretical, 'r--', linewidth=2, label='Theoretical (2ε)')
                    
                    plt.title("Optimality Gap vs Epsilon\n(Theory: Gap ≤ 2ε)")
                    plt.xlabel("Epsilon (ε)")
                    plt.ylabel("Optimality Gap")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    
                    # Save regular and log scale versions
                    plt.savefig(os.path.join(epsilon_dir, "optimality_gap_vs_epsilon.png"), dpi=300)
                    
                    # Log scale version
                    plt.xscale('log')
                    plt.savefig(os.path.join(epsilon_dir, "optimality_gap_vs_epsilon_log.png"), dpi=300)
                    plt.close()
                except Exception as e:
                    print(f"Error creating optimality gap vs epsilon plot: {e}")
            
            # 6c. Create a summary plot specifically for epsilon effects
            try:
                plt.figure(figsize=(12, 8))
                
                # Normalize all metrics for comparison
                normalized_data = {}
                for response in response_vars:
                    if response in agg_results.columns:
                        try:
                            # Group by epsilon
                            means = agg_results.groupby(epsilon_col)[response].mean().reset_index()
                            means = means.sort_values(by=epsilon_col)
                            
                            # Normalize to [0,1] range for comparison
                            min_val = means[response].min()
                            max_val = means[response].max()
                            if max_val > min_val:  # Avoid division by zero
                                normalized = (means[response] - min_val) / (max_val - min_val)
                                normalized_data[response] = {
                                    'epsilon': means[epsilon_col].values,
                                    'normalized': normalized.values
                                }
                        except Exception as e:
                            print(f"Error normalizing {response}: {e}")
                
                # Plot normalized metrics
                for response, data in normalized_data.items():
                    plt.plot(data['epsilon'], data['normalized'], 'o-', linewidth=2, label=response)
                
                plt.xlabel('Epsilon (ε)', fontsize=12)
                plt.ylabel('Normalized Response', fontsize=12)
                plt.title('Normalized Effect of Epsilon on Response Variables', fontsize=14)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                # Save regular and log scale versions
                plt.savefig(os.path.join(epsilon_dir, "normalized_epsilon_effects.png"), dpi=300)
                
                # Log scale version
                plt.xscale('log')
                plt.savefig(os.path.join(epsilon_dir, "normalized_epsilon_effects_log.png"), dpi=300)
                plt.close()
            except Exception as e:
                print(f"Error creating normalized epsilon effects plot: {e}")
    
    except Exception as e:
        print(f"Error generating reports: {e}")
    
    print(f"Reports generated in {plots_dir}")