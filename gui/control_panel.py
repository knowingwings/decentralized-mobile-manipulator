# decentralized_control/gui/control_panel.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
                           QSpinBox, QDoubleSpinBox, QSlider, QCheckBox, QTextEdit,
                           QListWidget, QGroupBox, QGridLayout, QScrollArea)
from PyQt5.QtCore import Qt
from core.auction import DistributedAuction

class ControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
    
    def init_ui(self):
        """Initialize the control panel UI"""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Make this a scrollable area for small screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)
        
        # 1. Robot controls
        robot_group = QGroupBox("Robot Control")
        robot_layout = QGridLayout()
        robot_group.setLayout(robot_layout)
        
        # Robot selector
        robot_layout.addWidget(QLabel("Select Robot:"), 0, 0)
        self.robot_selector = QComboBox()
        self.robot_selector.addItems([f"Robot {i+1}" for i in range(len(self.parent.simulator.robots))])
        self.robot_selector.currentIndexChanged.connect(self.update_robot_display)
        robot_layout.addWidget(self.robot_selector, 0, 1)
        
        # Robot status
        robot_layout.addWidget(QLabel("Status:"), 1, 0)
        self.robot_status = QComboBox()
        self.robot_status.addItems(["Operational", "Failed", "Partial Failure"])
        self.robot_status.currentIndexChanged.connect(self.update_robot_status)
        robot_layout.addWidget(self.robot_status, 1, 1)
        
        # Inject failure button
        self.inject_failure_btn = QPushButton("Inject Failure")
        self.inject_failure_btn.clicked.connect(self.inject_robot_failure)
        robot_layout.addWidget(self.inject_failure_btn, 2, 0, 1, 2)
        
        # Assigned tasks list
        robot_layout.addWidget(QLabel("Assigned Tasks:"), 3, 0, 1, 2)
        self.assigned_tasks_list = QListWidget()
        robot_layout.addWidget(self.assigned_tasks_list, 4, 0, 1, 2)
        
        scroll_layout.addWidget(robot_group)
        
        # 2. Task controls
        task_group = QGroupBox("Task Management")
        task_layout = QGridLayout()
        task_group.setLayout(task_layout)
        
        # Number of tasks
        task_layout.addWidget(QLabel("Number of Tasks:"), 0, 0)
        self.num_tasks_spinner = QSpinBox()
        self.num_tasks_spinner.setRange(1, 50)
        self.num_tasks_spinner.setValue(len(self.parent.simulator.tasks))
        task_layout.addWidget(self.num_tasks_spinner, 0, 1)
        
        # Generate tasks button
        self.generate_tasks_btn = QPushButton("Generate Random Tasks")
        self.generate_tasks_btn.clicked.connect(self.generate_random_tasks)
        task_layout.addWidget(self.generate_tasks_btn, 1, 0, 1, 2)
        
        # Task list
        task_layout.addWidget(QLabel("Task Status:"), 2, 0, 1, 2)
        self.task_list = QListWidget()
        self.update_task_list()
        task_layout.addWidget(self.task_list, 3, 0, 1, 2)
        
        scroll_layout.addWidget(task_group)
        
        # 3. Algorithm parameters
        param_group = QGroupBox("Algorithm Parameters")
        param_layout = QGridLayout()
        param_group.setLayout(param_layout)
        
        # Communication delay
        param_layout.addWidget(QLabel("Comm Delay (ms):"), 0, 0)
        self.comm_delay_spinner = QSpinBox()
        self.comm_delay_spinner.setRange(0, 1000)
        self.comm_delay_spinner.setValue(self.parent.simulator.comm_delay)
        param_layout.addWidget(self.comm_delay_spinner, 0, 1)
        
        # Packet loss probability
        param_layout.addWidget(QLabel("Packet Loss:"), 1, 0)
        self.packet_loss_spinner = QDoubleSpinBox()
        self.packet_loss_spinner.setRange(0, 1)
        self.packet_loss_spinner.setSingleStep(0.05)
        self.packet_loss_spinner.setValue(self.parent.simulator.packet_loss)
        param_layout.addWidget(self.packet_loss_spinner, 1, 1)
        
        # Epsilon
        param_layout.addWidget(QLabel("Epsilon:"), 2, 0)
        self.epsilon_spinner = QDoubleSpinBox()
        self.epsilon_spinner.setRange(0.001, 1)
        self.epsilon_spinner.setSingleStep(0.01)
        self.epsilon_spinner.setValue(self.parent.simulator.epsilon)
        param_layout.addWidget(self.epsilon_spinner, 2, 1)
        
        # Update parameters button
        self.update_params_btn = QPushButton("Update Parameters")
        self.update_params_btn.clicked.connect(self.update_parameters)
        param_layout.addWidget(self.update_params_btn, 3, 0, 1, 2)
        
        scroll_layout.addWidget(param_group)
        
        # 4. Experiment controls
        exp_group = QGroupBox("Experiment")
        exp_layout = QGridLayout()
        exp_group.setLayout(exp_layout)
        
        # Run experiment button
        self.run_exp_btn = QPushButton("Run Single Experiment")
        self.run_exp_btn.clicked.connect(self.run_single_experiment)
        exp_layout.addWidget(self.run_exp_btn, 0, 0, 1, 2)
        
        # Output area
        exp_layout.addWidget(QLabel("Results:"), 1, 0, 1, 2)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        exp_layout.addWidget(self.results_text, 2, 0, 1, 2)
        
        scroll_layout.addWidget(exp_group)
        
        # Initialize displays
        self.update_robot_display(0)
        self.update_task_list()
    
    def update_robot_display(self, index):
        """Update the robot information display"""
        if index < 0 or index >= len(self.parent.simulator.robots):
            return
            
        robot = self.parent.simulator.robots[index]
        
        # Update status dropdown
        if robot.status == 'operational':
            self.robot_status.setCurrentIndex(0)
        elif robot.status == 'failed':
            self.robot_status.setCurrentIndex(1)
        else:  # partial_failure
            self.robot_status.setCurrentIndex(2)
        
        # Update assigned tasks list
        self.assigned_tasks_list.clear()
        for task in self.parent.simulator.tasks:
            if task.assigned_to == robot.id:
                self.assigned_tasks_list.addItem(f"Task {task.id} ({task.status})")
    
    def update_robot_status(self, index):
        """Update robot status based on dropdown selection"""
        robot_idx = self.robot_selector.currentIndex()
        if robot_idx < 0 or robot_idx >= len(self.parent.simulator.robots):
            return
            
        robot = self.parent.simulator.robots[robot_idx]
        
        # Update status based on selection
        status_map = ['operational', 'failed', 'partial_failure']
        new_status = status_map[index]
        
        if new_status != robot.status:
            robot.update_status(new_status)
            
            # If changing to failure, trigger failure handling
            if new_status != 'operational':
                failed_tasks = [task for task in self.parent.simulator.tasks 
                               if task.assigned_to == robot.id and 
                               task.status != 'completed']
                
                # Mark tasks as unassigned
                for task in failed_tasks:
                    task.assigned_to = 0
                
                # Update visualization
                self.parent.update_visualization()
    
    def inject_robot_failure(self):
        """Inject a failure in the selected robot"""
        robot_idx = self.robot_selector.currentIndex()
        if robot_idx < 0 or robot_idx >= len(self.parent.simulator.robots):
            return
            
        robot_id = self.parent.simulator.robots[robot_idx].id
        
        # Inject failure and get affected tasks
        failed_robot, failed_tasks = self.parent.simulator.inject_robot_failure(
            robot_id=robot_id,
            failure_type='complete' if self.robot_status.currentIndex() == 1 else 'partial'
        )
        
        # Update robot status dropdown
        if failed_robot.status == 'failed':
            self.robot_status.setCurrentIndex(1)
        else:  # partial_failure
            self.robot_status.setCurrentIndex(2)
        
        # Run recovery if needed
        if failed_tasks:
            self.parent.simulator.run_recovery(failed_tasks)
        
        # Update displays
        self.update_robot_display(robot_idx)
        self.update_task_list()
        self.parent.update_visualization()
    
    def update_task_list(self):
        """Update the task status list"""
        self.task_list.clear()
        
        for task in self.parent.simulator.tasks:
            status_text = {
                'pending': 'Pending',
                'in_progress': f'In Progress ({task.progress:.0%})',
                'completed': 'Completed',
                'failed': 'Failed'
            }.get(task.status, task.status)
            
            assigned_to = f"R{task.assigned_to}" if task.assigned_to > 0 else "None"
            
            self.task_list.addItem(
                f"Task {task.id}: {status_text} - Assigned to: {assigned_to}" +
                (" (Collaborative)" if task.collaborative else "")
            )
    
    def generate_random_tasks(self):
        """Generate new random tasks"""
        num_tasks = self.num_tasks_spinner.value()
        
        # Generate tasks
        self.parent.simulator.generate_random_tasks(num_tasks)
        
        # Update displays
        self.update_task_list()
        self.parent.update_visualization()
    
    def update_parameters(self):
        """Update algorithm parameters"""
        # Get values from UI
        comm_delay = self.comm_delay_spinner.value()
        packet_loss = self.packet_loss_spinner.value()
        epsilon = self.epsilon_spinner.value()
        
        # Update simulator
        self.parent.simulator.comm_delay = comm_delay
        self.parent.simulator.packet_loss = packet_loss
        self.parent.simulator.epsilon = epsilon
        
        # Update auction algorithm
        self.parent.simulator.auction = DistributedAuction(
            epsilon=epsilon,
            communication_delay=comm_delay, 
            packet_loss_prob=packet_loss
        )
        
        self.results_text.append(f"Parameters updated:\n"
                               f"- Communication Delay: {comm_delay} ms\n"
                               f"- Packet Loss: {packet_loss:.2f}\n"
                               f"- Epsilon: {epsilon:.4f}\n")
    
    def run_single_experiment(self):
        """Run a single experiment with current settings"""
        # Pause the simulation if running
        if self.parent.running:
            self.parent.toggle_simulation()
        
        # Reset the simulator
        self.parent.reset_simulation()
        
        # Update parameters
        self.update_parameters()
        
        # Run the simulation until completion
        self.results_text.append("Running experiment...")
        
        # Run simulation with current settings
        metrics = self.parent.simulator.run_simulation(
            max_time=300,
            inject_failure=True,
            failure_time_fraction=0.3
        )
        
        # Display results
        self.results_text.append("Experiment completed!\n")
        self.results_text.append(f"Results:\n"
                               f"- Makespan: {metrics['makespan']:.2f}s\n"
                               f"- Message Count: {metrics['message_count']}\n"
                               f"- Completion Rate: {metrics['completion_rate']:.2%}\n"
                               f"- Workload Balance: {metrics['workload_balance']:.2%}\n"
                               f"- Recovery Time: {metrics['recovery_time']:.2f}s\n")
        
        # Update visualization
        self.parent.update_visualization()