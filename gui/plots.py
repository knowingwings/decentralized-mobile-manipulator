# decentralized_control/gui/plots.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTabWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class MetricsPlotDialog(QDialog):
    def __init__(self, metrics_data, parent=None):
        super().__init__(parent)
        self.metrics_data = metrics_data
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI components"""
        self.setWindowTitle('Performance Metrics')
        self.setGeometry(100, 100, 900, 700)
        
        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create tab widget
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Time series tab
        time_series_tab = self._create_time_series_tab()
        tab_widget.addTab(time_series_tab, "Time Series")
        
        # Distribution tab
        distribution_tab = self._create_distribution_tab()
        tab_widget.addTab(distribution_tab, "Distributions")
        
        # Correlation tab
        correlation_tab = self._create_correlation_tab()
        tab_widget.addTab(correlation_tab, "Correlations")
    
    def _create_time_series_tab(self):
        """Create time series plots tab"""
        tab = QVBoxLayout()
        widget = QTabWidget()
        tab.addWidget(widget)
        
        # Convert data to DataFrame if needed
        if isinstance(self.metrics_data, list):
            df = pd.DataFrame(self.metrics_data)
        else:
            df = self.metrics_data
        
        # Create figure for time series
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        # Plot time-based metrics if 'time' column exists
        if 'time' in df.columns:
            # Select numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'time']
            
            # Plot each metric
            for col in numeric_cols:
                if df[col].nunique() > 1:  # Only plot if there's variance
                    ax.plot(df['time'], df[col], label=col)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Metric Value')
            ax.set_title('Metrics Over Time')
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "No time-series data available", 
                   ha='center', va='center', fontsize=14)
        
        # Create canvas
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, widget)
        
        # Create widget
        time_widget = QVBoxLayout()
        time_widget.addWidget(toolbar)
        time_widget.addWidget(canvas)
        
        time_tab = QTabWidget()
        time_tab.setLayout(time_widget)
        
        return time_tab
    
    def _create_distribution_tab(self):
        """Create distributions plots tab"""
        tab = QVBoxLayout()
        widget = QTabWidget()
        tab.addWidget(widget)
        
        # Convert data to DataFrame if needed
        if isinstance(self.metrics_data, list):
            df = pd.DataFrame(self.metrics_data)
        else:
            df = self.metrics_data
        
        # Create figure for distributions
        fig = plt.figure(figsize=(8, 6))
        
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # If we have more than time column, create subplots
        if len(numeric_cols) > 0:
            # Create a grid of subplots
            n_cols = min(2, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            for i, col in enumerate(numeric_cols):
                ax = fig.add_subplot(n_rows, n_cols, i+1)
                if df[col].nunique() > 1:  # Only plot if there's variance
                    sns.histplot(df[col], kde=True, ax=ax)
                    ax.set_title(f'Distribution of {col}')
                else:
                    ax.text(0.5, 0.5, f"{col} has only one value: {df[col].iloc[0]}", 
                           ha='center', va='center')
        else:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No numeric data available for distribution analysis", 
                   ha='center', va='center', fontsize=14)
        
        fig.tight_layout()
        
        # Create canvas
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, widget)
        
        # Create widget
        dist_widget = QVBoxLayout()
        dist_widget.addWidget(toolbar)
        dist_widget.addWidget(canvas)
        
        dist_tab = QTabWidget()
        dist_tab.setLayout(dist_widget)
        
        return dist_tab
    
    def _create_correlation_tab(self):
        """Create correlation plots tab"""
        tab = QVBoxLayout()
        widget = QTabWidget()
        tab.addWidget(widget)
        
        # Convert data to DataFrame if needed
        if isinstance(self.metrics_data, list):
            df = pd.DataFrame(self.metrics_data)
        else:
            df = self.metrics_data
        
        # Create figure for correlations
        fig = plt.figure(figsize=(8, 6))
        
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            # Create correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Plot heatmap
            ax = fig.add_subplot(111)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
            ax.set_title('Correlation Matrix')
        else:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Not enough numeric columns for correlation analysis", 
                   ha='center', va='center', fontsize=14)
        
        # Create canvas
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, widget)
        
        # Create widget
        corr_widget = QVBoxLayout()
        corr_widget.addWidget(toolbar)
        corr_widget.addWidget(canvas)
        
        corr_tab = QTabWidget()
        corr_tab.setLayout(corr_widget)
        
        return corr_tab