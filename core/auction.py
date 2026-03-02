# core/auction.py
import numpy as np
import time
import random
import torch

from core.gpu_accelerator import GPUAccelerator

class DistributedAuction:
    def __init__(self, epsilon=0.01, communication_delay=0, packet_loss_prob=0, use_gpu=True):
        """Initialize distributed auction algorithm
        
        Args:
            epsilon: Minimum bid increment (controls optimality gap and convergence)
            communication_delay: Communication delay in ms
            packet_loss_prob: Probability of packet loss (0-1)
            use_gpu: Whether to use GPU acceleration when available
        """
        self.epsilon = epsilon
        self.communication_delay = communication_delay / 1000.0  # Convert ms to seconds
        self.packet_loss_prob = packet_loss_prob
        self.weights = {
            'alpha1': 0.3,  # Distance weight
            'alpha2': 0.2,  # Configuration cost weight
            'alpha3': 0.3,  # Capability similarity weight
            'alpha4': 0.1,  # Workload weight
            'alpha5': 0.1,  # Energy consumption weight
            'W': np.eye(6)  # Weight matrix for configuration
        }
        self.beta_weights = {
            'beta1': 0.5,  # Progress weight
            'beta2': 0.3,  # Criticality weight
            'beta3': 0.2   # Urgency weight
        }
        
        # Initialize GPU accelerator if requested
        self.use_gpu = use_gpu
        if use_gpu:
            try:
                self.gpu = GPUAccelerator(communication_delay, packet_loss_prob)
                self.use_gpu = self.gpu.using_gpu
            except Exception as e:
                print(f"Could not initialize GPU acceleration: {e}")
                self.use_gpu = False
        
        # Add data collector initialization
        from utils.data_collector import AuctionDataCollector
        self.data_collector = AuctionDataCollector()
        self.centralized_solver = None
        self.centralized_solution = None
    
    def run_auction(self, robots, tasks, task_graph, termination_mode="assignment-complete", price_stability_threshold=0.01):
        """Run distributed auction algorithm for task allocation
        
        Args:
            robots: List of Robot objects
            tasks: List of Task objects
            task_graph: TaskDependencyGraph object
            termination_mode: "assignment-complete" or "price-stabilized"
            price_stability_threshold: Threshold for price stability detection
            
        Returns:
            tuple: (assignments dict, message count)
        """
        # Ensure price_stability_threshold is a scalar, not a list/array
        if isinstance(price_stability_threshold, (list, tuple, np.ndarray)):
            price_stability_threshold = float(price_stability_threshold[0])
        else:
            price_stability_threshold = float(price_stability_threshold)
            
        # Start a new run with configuration
        if hasattr(self, 'data_collector'):
            self.data_collector.start_run({
                'epsilon': self.epsilon,
                'num_tasks': len(tasks),
                'comm_delay': self.communication_delay * 1000.0,  # Convert to ms
                'packet_loss': self.packet_loss_prob,
                'termination_mode': termination_mode,
                'price_stability_threshold': price_stability_threshold
            })
        
        # Print debug information about parameters
        print(f"Running auction with epsilon={self.epsilon}, delay={self.communication_delay*1000}ms, " 
            f"packet_loss={self.packet_loss_prob}, termination_mode={termination_mode}")
        
        # Run centralized solver for comparison if available
        if hasattr(self, 'data_collector') and self.centralized_solver is None:
            from core.centralized_solver import CentralizedSolver
            self.centralized_solver = CentralizedSolver()
            self.centralized_solution = self.centralized_solver.solve(robots, tasks)
            
            # Record optimal solution
            if self.centralized_solution:
                self.data_collector.current_run_metrics['centralized_solution'] = self.centralized_solution
        
        # Find unassigned tasks that are ready (prerequisites completed)
        unassigned_tasks = [task for task in tasks 
                        if task.assigned_to == 0 and 
                        task.status == 'pending' and 
                        task_graph.is_available(task.id)]
        
        if not unassigned_tasks:
            # Record run completion with data collector
            if hasattr(self, 'data_collector'):
                self.data_collector.record_iteration(0, {
                    'phase': 'no_tasks',
                    'message_count': 0
                })
                self.data_collector.end_run()
            
            return {}, 0  # No tasks to assign
        
        # Use GPU implementation if enabled and possible
        if self.use_gpu and len(robots) > 0 and len(unassigned_tasks) > 0:
            # Update GPU accelerator communication parameters
            self.gpu.set_communication_params(self.communication_delay, self.packet_loss_prob)
            assignments, messages = self._run_auction_gpu(robots, unassigned_tasks, tasks, 
                                                    termination_mode, price_stability_threshold)
        else:
            assignments, messages = self._run_auction_cpu(robots, unassigned_tasks, tasks, 
                                                    termination_mode, price_stability_threshold)
        
        # Calculate optimality gap if centralized solution is available
        optimality_gap = None
        if hasattr(self, 'centralized_solution') and self.centralized_solution:
            # Find makespan based on current assignment
            current_makespan = 0
            for robot in robots:
                if robot.status != 'operational':
                    continue
                    
                robot_tasks = [task for task in tasks if task.assigned_to == robot.id]
                robot_makespan = sum(task.completion_time for task in robot_tasks)
                current_makespan = max(current_makespan, robot_makespan)
            
            # Calculate optimality gap
            optimal_makespan = self.centralized_solution.get('makespan', 0)
            if optimal_makespan > 0:
                optimality_gap = (current_makespan - optimal_makespan) / optimal_makespan
        
        # Record run completion with data collector
        if hasattr(self, 'data_collector'):
            completion_data = {
                'phase': 'auction_complete',
                'assignments': len(assignments),
                'messages': messages,
                'termination_mode': termination_mode
            }
            
            if optimality_gap is not None:
                completion_data['optimality_gap'] = optimality_gap
            
            self.data_collector.record_iteration(
                self.data_collector.current_iteration,
                completion_data
            )
            
            # End the run
            self.data_collector.end_run({
                'message_count': messages,
                'optimality_gap': optimality_gap
            })
        
        return assignments, messages
    
    def _run_auction_gpu(self, robots, unassigned_tasks, all_tasks, termination_mode="assignment-complete", 
                   price_stability_threshold=0.01):
        """GPU-accelerated auction implementation with enhanced termination conditions"""
        # Debug logging to track epsilon's effect
        print(f"DEBUG: Using epsilon={self.epsilon} in GPU auction with termination_mode={termination_mode}")
        
        # Prepare data for GPU processing
        robot_positions = np.array([robot.position for robot in robots])
        robot_capabilities = np.array([robot.capabilities for robot in robots])
        robot_statuses = [robot.status for robot in robots]
        
        task_positions = np.array([task.position for task in unassigned_tasks])
        task_capabilities = np.array([task.capabilities for task in unassigned_tasks])
        
        # Map unassigned tasks to their indices in all_tasks
        task_id_to_idx = {task.id: i for i, task in enumerate(unassigned_tasks)}
        
        # Current assignments and prices
        task_assignments = np.zeros(len(unassigned_tasks), dtype=np.int32)
        prices = np.zeros(len(unassigned_tasks), dtype=np.float32)
        
        # Record initial state for data collection
        if hasattr(self, 'data_collector'):
            # Create price dictionary for all tasks
            price_dict = {task.id: 0.0 for task in all_tasks}
            
            self.data_collector.record_iteration(0, {
                'phase': 'gpu_auction_start',
                'num_tasks': len(unassigned_tasks),
                'epsilon': self.epsilon,
                'prices': price_dict.copy(),
                'termination_mode': termination_mode
            })
        
        # Create weights dictionary with epsilon explicitly included
        weights = self.weights.copy()
        weights['epsilon'] = self.epsilon  # Critical: ensure epsilon is passed to GPU
        
        # Run GPU auction with specified termination mode
        new_assignments, new_prices, messages = self.gpu.run_auction_gpu(
            robot_positions, robot_capabilities, robot_statuses,
            task_positions, task_capabilities, task_assignments,
            self.epsilon, prices,
            weights=weights,  # Pass weights including epsilon
            data_collector=self.data_collector if hasattr(self, 'data_collector') else None,
            unassigned_task_ids=[task.id for task in unassigned_tasks],
            termination_mode=termination_mode,
            price_stability_threshold=price_stability_threshold
        )
        
        # Convert results back to dictionary format
        assignments = {}
        
        # Record final auction state for data collection
        if hasattr(self, 'data_collector'):
            # Create final price and assignment dictionaries
            final_prices = {}
            final_assignments = {}
            
            for i, task in enumerate(unassigned_tasks):
                robot_idx = new_assignments[i]
                price = new_prices[i]
                
                final_prices[task.id] = price
                
                if robot_idx > 0:  # If assigned
                    robot_id = robots[robot_idx-1].id
                    final_assignments[task.id] = robot_id
            
            self.data_collector.record_iteration(1, {  # GPU uses a single "iteration"
                'phase': 'gpu_auction_complete',
                'final_prices': final_prices,
                'final_assignments': final_assignments,
                'total_messages': messages,
                'termination_mode': termination_mode
            })
        
        # Process task assignments
        for i, task in enumerate(unassigned_tasks):
            robot_idx = new_assignments[i]
            if robot_idx > 0:  # If assigned
                robot_id = robots[robot_idx-1].id
                task.assigned_to = robot_id
                assignments[task.id] = robot_id
                
                # Record assignment for data collection
                if hasattr(self, 'data_collector'):
                    self.data_collector.record_iteration_status = 1  # GPU uses a single "iteration"
        
        return assignments, messages
    
    def _run_auction_cpu(self, robots, unassigned_tasks, tasks, termination_mode="assignment-complete", price_stability_threshold=0.01):
        """Original CPU implementation with enhanced termination conditions
        
        Args:
            robots: List of Robot objects
            unassigned_tasks: List of unassigned Task objects
            tasks: Complete list of Task objects
            termination_mode: "assignment-complete" or "price-stabilized"
            price_stability_threshold: Threshold for price stability detection
            
        Returns:
            tuple: (assignments dict, message count)
        """
        
        #Ensure price_stability_threshold is a scalar, not a list/array
        if isinstance(price_stability_threshold, (list, tuple, np.ndarray)):
            price_stability_threshold = float(price_stability_threshold[0])
        else:
            price_stability_threshold = float(price_stability_threshold)
        
        # Track assignments and prices
        prices = {task.id: 0.0 for task in tasks}
        assignments = {task.id: 0 for task in tasks}
        messages_sent = 0
        
        # For price stability detection
        price_changes_history = []
        max_stable_count = 3  # Number of iterations with stable prices to confirm convergence
        stable_count = 0
        
        # Set max_iterations based on theoretical bound: K² · bₘₐₓ/ε 
        # where K is task count and bₘₐₓ is maximum possible bid value
        K = len(unassigned_tasks)
        b_max = 100.0  # Estimate of maximum possible bid value
        theoretical_max_iter = int(K**2 * b_max / self.epsilon)
        
        # Set a practical upper limit to prevent excessive iterations
        max_iterations = min(theoretical_max_iter, 1000)
        
        # Debug logging for data collection
        if hasattr(self, 'data_collector'):
            self.data_collector.record_iteration(0, {
                'phase': 'auction_start',
                'num_tasks': len(unassigned_tasks),
                'max_iterations': max_iterations,
                'epsilon': self.epsilon,
                'theoretical_max_iter': theoretical_max_iter,
                'termination_mode': termination_mode,
                'price_stability_threshold': price_stability_threshold
            })
        
        # Run auction algorithm
        iter_count = 0
        
        while iter_count < max_iterations:
            iter_count += 1
            tasks_assigned_this_iter = 0
            
            # Start iteration with fresh price change tracker
            price_changes = []
            
            # Create a snapshot of prices at the start of this iteration
            # for data collection and correct utility calculations
            current_prices = prices.copy()
            
            # If using data collection, record iteration start
            if hasattr(self, 'data_collector'):
                self.data_collector.record_iteration(iter_count, {
                    'phase': 'iteration_start',
                    'unassigned_count': len(unassigned_tasks),
                    'prices': current_prices.copy()
                })
            
            # For each robot, calculate bids for unassigned tasks
            for robot in robots:
                # Skip failed robots
                if robot.status != 'operational':
                    continue
                
                # Calculate current workload
                workload = sum(task.completion_time for task in tasks 
                            if task.assigned_to == robot.id and
                            task.status != 'completed')
                robot.workload = workload
                
                # Find best task for this robot
                best_utility = float('-inf')
                best_task = None
                best_bid = 0
                
                for task in unassigned_tasks:
                    # Skip collaborative tasks for simplicity
                    if task.collaborative:
                        continue
                    
                    # Calculate bid
                    bid = robot.calculate_bid(task, self.weights, workload)
                    
                    # Apply communication delay
                    if self.communication_delay > 0:
                        time.sleep(self.communication_delay)
                    
                    # Check for packet loss
                    if random.random() < self.packet_loss_prob:
                        continue  # Simulate packet loss
                    
                    messages_sent += 1
                    
                    # Calculate utility (bid - price)
                    task_price = current_prices[task.id]  # Use snapshot price
                    utility = bid - task_price
                    
                    # Record bid and utility for data collection
                    if hasattr(self, 'data_collector'):
                        self.data_collector.record_bid(robot.id, task.id, bid, utility)
                    
                    if utility > best_utility:
                        best_utility = utility
                        best_task = task
                        best_bid = bid
                
                # If found a task with positive utility, propose assignment
                if best_task and best_utility > 0:
                    # Apply communication delay for assignment message
                    if self.communication_delay > 0:
                        time.sleep(self.communication_delay)
                    
                    # Check for packet loss for assignment message  
                    if random.random() < self.packet_loss_prob:
                        continue  # Packet loss on assignment message
                        
                    # Calculate old and new price
                    old_price = prices[best_task.id]
                    
                    # CRITICAL: Apply epsilon in price update formula - clearly separate components
                    price_increase = self.epsilon + best_utility
                    new_price = old_price + price_increase
                    
                    # Track price change for stability detection
                    price_changes.append(abs(new_price - old_price))
                    
                    # Debug logging for price update
                    print(f"DEBUG: Task {best_task.id} - Old price: {old_price:.4f}, "
                        f"Increase: {price_increase:.4f} (epsilon={self.epsilon:.4f}, "
                        f"utility={best_utility:.4f}), New price: {new_price:.4f}")
                    
                    # Record price update for data collection
                    if hasattr(self, 'data_collector'):
                        self.data_collector.record_price_update(
                            best_task.id, old_price, new_price, self.epsilon, best_utility)
                    
                    # Update price
                    prices[best_task.id] = new_price
                    
                    # Assign task to robot
                    best_task.assigned_to = robot.id
                    assignments[best_task.id] = robot.id
                    
                    # Record assignment for data collection
                    if hasattr(self, 'data_collector'):
                        self.data_collector.record_assignment(
                            best_task.id, robot.id, iter_count)
                    
                    # Remove from unassigned tasks
                    unassigned_tasks.remove(best_task)
                    
                    messages_sent += 1  # Assignment message
                    tasks_assigned_this_iter += 1
            
            # Track price changes for stability detection
            avg_price_change = sum(price_changes) / len(price_changes) if price_changes else 0
            price_changes_history.append(avg_price_change)
            
            # Record iteration end with data collector
            if hasattr(self, 'data_collector'):
                self.data_collector.record_iteration(iter_count, {
                    'phase': 'iteration_end',
                    'tasks_assigned': tasks_assigned_this_iter,
                    'unassigned_remaining': len(unassigned_tasks),
                    'messages_this_iter': messages_sent,
                    'current_prices': prices.copy(),
                    'avg_price_change': avg_price_change
                })
            
            # Check termination conditions based on mode
            if termination_mode == "assignment-complete":
                # Original termination logic
                if not unassigned_tasks:
                    # All tasks assigned
                    break
                if tasks_assigned_this_iter == 0:
                    # No progress in this iteration
                    break
                    
            elif termination_mode == "price-stabilized":
                # Check price stability
                if avg_price_change < price_stability_threshold:
                    stable_count += 1
                    if stable_count >= max_stable_count:
                        # Prices have stabilized for several consecutive iterations
                        break
                else:
                    stable_count = 0
                    
                # Also break if all tasks assigned (common sense)
                if not unassigned_tasks:
                    break
                    
        # Log if we hit the iteration limit (useful for debugging)
        if iter_count >= max_iterations and unassigned_tasks:
            print(f"Warning: CPU Auction reached maximum iterations ({max_iterations}) with {len(unassigned_tasks)} tasks still unassigned.")
        
        # Record final iteration status with data collector
        if hasattr(self, 'data_collector'):
            self.data_collector.record_iteration(iter_count + 1, {
                'phase': 'auction_complete',
                'total_iterations': iter_count,
                'unassigned_remaining': len(unassigned_tasks),
                'total_messages': messages_sent,
                'final_prices': prices.copy(),
                'final_assignments': assignments.copy(),
                'termination_reason': ('max_iterations' if iter_count >= max_iterations else
                                    ('all_assigned' if not unassigned_tasks else
                                    ('no_progress' if tasks_assigned_this_iter == 0 else
                                    'price_stabilized')))
            })
        
        # Handle collaborative tasks (simplified)
        collaborative_tasks = [task for task in tasks 
                            if task.collaborative and task.assigned_to == 0 and
                            task.status == 'pending']
        
        for task in collaborative_tasks:
            # Check if at least two robots are operational
            operational_robots = [r for r in robots if r.status == 'operational']
            if len(operational_robots) >= 2:
                # For simplicity, assign to the first operational robot
                # In a real system, this would require coordination
                task.assigned_to = operational_robots[0].id
                assignments[task.id] = operational_robots[0].id
                
                # Record assignment for data collection
                if hasattr(self, 'data_collector'):
                    self.data_collector.record_assignment(
                        task.id, operational_robots[0].id, iter_count)
                    
                messages_sent += 1
        
        return assignments, messages_sent
    
    def run_recovery_auction(self, operational_robots, failed_tasks, task_graph):
        """Special auction for task reallocation after failure
        
        Args:
            operational_robots: List of operational Robot objects
            failed_tasks: List of Task objects that need reallocation
            task_graph: TaskDependencyGraph object
            
        Returns:
            tuple: (assignments dict, message count)
        """
        # Print debug info
        print(f"Running recovery auction for {len(failed_tasks)} tasks with {len(operational_robots)} robots")
        # ADDED: Debug logging for epsilon in recovery auction
        print(f"DEBUG: Using epsilon={self.epsilon} in recovery auction")
        
        assignments = {}
        messages_sent = 0
        
        # Use GPU acceleration for recovery when possible
        if self.use_gpu and len(operational_robots) > 0 and len(failed_tasks) > 0:
            # Prepare data structures for GPU processing
            robot_positions = np.array([robot.position for robot in operational_robots])
            robot_capabilities = np.array([robot.capabilities for robot in operational_robots])
            robot_statuses = ['operational'] * len(operational_robots)
            
            task_positions = np.array([task.position for task in failed_tasks])
            task_capabilities = np.array([task.capabilities for task in failed_tasks])
            
            # Current assignments and prices
            task_assignments = np.zeros(len(failed_tasks), dtype=np.int32)
            prices = np.zeros(len(failed_tasks), dtype=np.float32)
            
            # Update GPU accelerator communication parameters
            self.gpu.set_communication_params(self.communication_delay, self.packet_loss_prob)
            
            # IMPROVED: Explicit epsilon handling for recovery
            # Run GPU auction with parameters adjusted for recovery
            new_assignments, _, messages = self.gpu.run_auction_gpu(
                robot_positions, robot_capabilities, robot_statuses,
                task_positions, task_capabilities, task_assignments,
                self.epsilon * 2,  # Higher epsilon for faster convergence in recovery
                prices,
                weights={'epsilon': self.epsilon * 2}  # Explicit epsilon for recovery
            )
            
            # Process results
            for i, task in enumerate(failed_tasks):
                robot_idx = new_assignments[i]
                if robot_idx > 0:  # If assigned
                    robot_id = operational_robots[robot_idx-1].id
                    task.assigned_to = robot_id
                    assignments[task.id] = robot_id
            
            return assignments, messages
        
        # Fall back to CPU implementation
        for task in failed_tasks:
            best_bid = float('-inf')
            best_robot = None
            
            for robot in operational_robots:
                # Apply communication delay for bid calculation
                if self.communication_delay > 0:
                    time.sleep(self.communication_delay)
                
                # Check for packet loss
                if random.random() < self.packet_loss_prob:
                    continue  # Simulate packet loss
                    
                # Calculate standard bid
                standard_bid = robot.calculate_bid(task, self.weights, robot.workload)
                
                # Calculate task criticality (number of dependent tasks)
                criticality = task_graph.get_task_criticality(task.id)
                
                # Calculate urgency
                urgency = task.progress if task.status == 'in_progress' else 0
                
                # Calculate recovery bid
                recovery_bid = robot.calculate_recovery_bid(standard_bid, task.progress, 
                                                          criticality, urgency, 
                                                          self.beta_weights)
                
                messages_sent += 1
                
                if recovery_bid > best_bid:
                    best_bid = recovery_bid
                    best_robot = robot
            
            if best_robot:
                # Apply communication delay for assignment
                if self.communication_delay > 0:
                    time.sleep(self.communication_delay)
                
                # Check for packet loss on assignment
                if random.random() < self.packet_loss_prob:
                    continue  # Packet loss on assignment message
                    
                task.assigned_to = best_robot.id
                assignments[task.id] = best_robot.id
                messages_sent += 1
        
        return assignments, messages_sent