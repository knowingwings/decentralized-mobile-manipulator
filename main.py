# main.py
import os
import sys
import argparse
import yaml
import numpy as np
import random
import torch

def main():
    parser = argparse.ArgumentParser(description='Decentralized Control System for Dual Mobile Manipulators')
    parser.add_argument('--mode', choices=['gui', 'experiment', 'analysis'], default='gui',
                       help='Run mode: gui (interactive), experiment (batch), or analysis')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Configuration file path')
    parser.add_argument('--results', type=str, default='results/factorial_experiment.csv',
                       help='Results file path for analysis mode')
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of processes for parallel experiment execution')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Enable GPU acceleration')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization during simulations (slower)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--headless', action='store_true',
                   help='Force headless mode for visualization')
    parser.add_argument('--save-plot-data', action='store_true', default=True,
                   help='Save plot data for remote visualization')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    
    if args.mode == 'gui':
        # Run interactive GUI
        from PyQt5.QtWidgets import QApplication
        from gui.visualization import VisualizationApp
        
        app = QApplication(sys.argv)
        config = load_config(args.config)
        config['use_gpu'] = args.use_gpu
        
        window = VisualizationApp(config)
        window.show()
        sys.exit(app.exec_())
    
    elif args.mode == 'experiment':
        # Run factorial experiment
        from core.experiment_runner import ExperimentRunner
        
        config = load_config(args.config)
        config['use_gpu'] = args.use_gpu
        config['visualize'] = args.visualize
        config['headless'] = args.headless
        config['save_plot_data'] = args.save_plot_data
        # Add config file name to the configuration
        config['config_file'] = args.config
        
        # Create experiment runner
        runner = ExperimentRunner(config)
        
        # Run experiment
        results = runner.run_factorial_experiment(num_processes=args.processes)
        
        # Save results (runner will handle it)
        results_path = runner.save_results(results)
        
        # Generate basic analysis automatically
        from core.analysis import analyze_results, generate_reports
        analyze_results(results_path)
        generate_reports(results_path, output_dir=runner.results_dir, 
                        headless=config.get('headless', None),
                        save_data=config.get('save_plot_data', True))
    
    elif args.mode == 'analysis':
        # Run analysis on existing results
        from core.analysis import analyze_results, generate_reports
        
        if not os.path.exists(args.results):
            print(f"Error: Results file not found: {args.results}")
            return
        
        analysis_results = analyze_results(args.results)
        generate_reports(args.results)
        
        # Return results to allow further analysis in interactive mode
        return analysis_results

def load_config(config_path):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

if __name__ == '__main__':
    main()