# decentralized_control/gui/visualization.py

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QTabWidget, 
                           QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
                           QSpinBox, QDoubleSpinBox, QSlider, QCheckBox, QTextEdit,
                           QListWidget, QGroupBox, QGridLayout, QFileDialog)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot

from core.simulator import Simulator
from core.task import TaskDependencyGraph
from core.auction import DistributedAuction
from gui.control_panel import ControlPanel

class VisualizationApp(QMainWindow):
    def __init__(self, config=None):
        super().__init__()
        
        # Initialize simulator with default settings
        self.simulator = Simulator(
            num_robots=2, 
            workspace_size=(10, 8),
            comm_delay=0,
            packet_loss=0,
            epsilon=0.01
        )
        
        # Initialize GUI
        self.init_ui()
        
        # Initialize with random tasks
        self.simulator.generate_random_tasks(8)
        self.update_visualization()
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.animation_speed = 1.0
        self.running = False
    
    def init_ui(self):
        """Initialize UI components"""
        self.setWindowTitle('Decentralized Control System')
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Left side: Control panel
        control_panel = ControlPanel(self)
        main_layout.addWidget(control_panel, 1)
        
        # Right side: Visualization
        viz_widget = QWidget()
        viz_layout = QVBoxLayout()
        viz_widget.setLayout(viz_layout)
        main_layout.addWidget(viz_widget, 3)
        
        # Matplotlib figure for simulation visualization
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        viz_layout.addWidget(self.toolbar)
        viz_layout.addWidget(self.canvas)
        
        # Bottom row: Status and controls
        status_widget = QWidget()
        status_layout = QHBoxLayout()
        status_widget.setLayout(status_layout)
        
        # Status label
        self.status_label = QLabel('Status: Ready')
        status_layout.addWidget(self.status_label)
        
        # Simulation time label
        self.time_label = QLabel('Time: 0.0s')
        status_layout.addWidget(self.time_label)
        
        # Play/Pause button
        self.play_button = QPushButton('▶ Play')
        self.play_button.clicked.connect(self.toggle_simulation)
        status_layout.addWidget(self.play_button)
        
        # Step button
        self.step_button = QPushButton('Step')
        self.step_button.clicked.connect(self.step_simulation)
        status_layout.addWidget(self.step_button)
        
        # Reset button
        self.reset_button = QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset_simulation)
        status_layout.addWidget(self.reset_button)
        
        # Speed slider
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(20)
        self.speed_slider.setValue(10)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(5)
        self.speed_slider.valueChanged.connect(self.update_speed)
        status_layout.addWidget(self.speed_slider)
        
        viz_layout.addWidget(status_widget)
        
        # Tabs for additional visualizations
        self.tab_widget = QTabWidget()
        
        # Task dependency graph tab
        self.dependency_tab = QWidget()
        dependency_layout = QVBoxLayout()
        self.dependency_tab.setLayout(dependency_layout)
        
        self.dependency_figure = plt.figure(figsize=(8, 6))
        self.dependency_canvas = FigureCanvas(self.dependency_figure)
        dependency_layout.addWidget(self.dependency_canvas)
        
        # Metrics tab
        self.metrics_tab = QWidget()
        metrics_layout = QVBoxLayout()
        self.metrics_tab.setLayout(metrics_layout)
        
        self.metrics_figure = plt.figure(figsize=(8, 6))
        self.metrics_canvas = FigureCanvas(self.metrics_figure)
        metrics_layout.addWidget(self.metrics_canvas)
        
        # Add tabs
        self.tab_widget.addTab(self.dependency_tab, 'Task Dependencies')
        self.tab_widget.addTab(self.metrics_tab, 'Metrics')
        
        viz_layout.addWidget(self.tab_widget)
    
    def update_visualization(self):
        """Update the visualization with current simulator state"""
        # Clear the figure and draw simulation
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.simulator.visualize(ax, show_trajectories=True)
        
        # Draw task dependency graph
        self.dependency_figure.clear()
        ax2 = self.dependency_figure.add_subplot(111)
        if self.simulator.task_graph:
            self.simulator.task_graph.visualize(ax2)
        
        # Draw metrics
        self.metrics_figure.clear()
        ax3 = self.metrics_figure.add_subplot(111)
        
        # Plot metrics over time (if available)
        if hasattr(self.simulator, 'metrics_history'):
            metrics_df = pd.DataFrame(self.simulator.metrics_history)
            if not metrics_df.empty:
                metrics_df.plot(x='time', y=['completion_rate', 'workload_balance'], ax=ax3)
                ax3.set_title('Performance Metrics')
                ax3.set_xlabel('Simulation Time')
                ax3.set_ylabel('Metric Value')
                ax3.legend()
                ax3.grid(True)
        
        # Refresh canvases
        self.canvas.draw()
        self.dependency_canvas.draw()
        self.metrics_canvas.draw()
        
        # Update status labels
        self.time_label.setText(f'Time: {self.simulator.sim_time:.1f}s')
        
        # Update task completion
        completed = sum(1 for task in self.simulator.tasks if task.status == 'completed')
        total = len(self.simulator.tasks)
        self.status_label.setText(f'Status: {completed}/{total} tasks completed')
    
    def toggle_simulation(self):
        """Toggle between play and pause"""
        if self.running:
            # Pause simulation
            self.timer.stop()
            self.running = False
            self.play_button.setText('▶ Play')
        else:
            # Start simulation
            self.timer.start(50)  # 50ms refresh rate
            self.running = True
            self.play_button.setText('⏸ Pause')
    
    def step_simulation(self):
        """Run a single simulation step"""
        # Make sure simulation is paused
        if self.running:
            self.toggle_simulation()
        
        # Run one step
        self.update_simulation()
    
    def reset_simulation(self):
        """Reset the simulation to initial state"""
        # Stop simulation if running
        if self.running:
            self.toggle_simulation()
        
        # Reset simulator
        num_tasks = len(self.simulator.tasks)
        self.simulator = Simulator(
            num_robots=2,
            workspace_size=(10, 8),
            comm_delay=self.simulator.comm_delay,
            packet_loss=self.simulator.packet_loss,
            epsilon=self.simulator.epsilon
        )
        self.simulator.generate_random_tasks(num_tasks)
        
        # Update visualization
        self.update_visualization()
    
    def update_speed(self):
        """Update simulation speed based on slider value"""
        # Map slider value (1-20) to speed multiplier (0.1-3.0)
        self.animation_speed = self.speed_slider.value() / 10.0
    
    def update_simulation(self):
        """Update simulation state"""
        # Run the simulation for dt * animation_speed
        dt = self.simulator.dt
        steps = max(1, int(round(self.animation_speed)))
        
        for _ in range(steps):
            # Run auction if needed
            unassigned_exist = any(task.assigned_to == 0 and task.status == 'pending' 
                                 for task in self.simulator.tasks)
            
            if unassigned_exist:
                _, messages = self.simulator.auction.run_auction(
                    self.simulator.robots, 
                    self.simulator.tasks, 
                    self.simulator.task_graph
                )
                self.simulator.metrics['message_count'] += messages
            
            # Update task progress
            for task in self.simulator.tasks:
                if task.status == 'in_progress':
                    task.progress += dt / task.completion_time
                    
                    # Check if task completed
                    if task.progress >= 1.0:
                        task.progress = 1.0
                        task.update_status('completed')
                        task.completion_time_actual = self.simulator.sim_time - task.start_time
            
            # Move robots toward assigned tasks
            for robot in self.simulator.robots:
                if robot.status != 'operational':
                    continue
                    
                # Find assigned tasks for this robot
                assigned_tasks = [task for task in self.simulator.tasks 
                                if task.assigned_to == robot.id and 
                                task.status == 'pending']
                
                if assigned_tasks:
                    # Move toward first pending task
                    target_task = assigned_tasks[0]
                    reached = robot.update_position(target_task.position, dt)
                    
                    # If reached task, start executing
                    if reached and target_task.status == 'pending':
                        target_task.update_status('in_progress')
                        target_task.start_time = self.simulator.sim_time
            
            # Update simulation time
            self.simulator.sim_time += dt
            
            # Check if all tasks are completed
            if all(task.status == 'completed' for task in self.simulator.tasks):
                self.toggle_simulation()  # Pause when done
                break
        
        # Update visualization
        self.update_visualization()