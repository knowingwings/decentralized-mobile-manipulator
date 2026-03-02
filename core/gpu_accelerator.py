# core/gpu_accelerator.py
import torch
import numpy as np
import os
import time
import random

class GPUAccelerator:
    """GPU acceleration with support for AMD GPU detected through CUDA compatibility"""
    
    def __init__(self, communication_delay=0, packet_loss_prob=0):
        """Initialize GPU accelerator for the unique configuration detected"""
        self.using_gpu = False
        self.device = torch.device('cpu')
        self.is_amd_via_cuda = False
        
        # Communication parameters (added)
        self.communication_delay = communication_delay
        self.packet_loss_prob = packet_loss_prob
        
        try:
            if torch.cuda.is_available():
                # Check if this is actually an AMD GPU detected through CUDA
                device_name = torch.cuda.get_device_name(0).lower()
                if 'amd' in device_name or 'radeon' in device_name:
                    self.is_amd_via_cuda = True
                    print(f"AMD GPU detected through CUDA: {torch.cuda.get_device_name(0)}")
                else:
                    print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
                
                self.device = torch.device('cuda')
                self.using_gpu = True
                
                # Set specific settings for AMD GPUs detected through CUDA
                if self.is_amd_via_cuda:
                    # Limit memory usage to avoid crashes
                    torch.cuda.set_per_process_memory_fraction(0.8)
            else:
                print("No GPU detected, using CPU")
                # Optimize CPU settings
                torch.set_num_threads(torch.get_num_threads())
                print(f"Using {torch.get_num_threads()} CPU threads")
                
        except Exception as e:
            print(f"Error during GPU initialization: {e}")
            print("Falling back to CPU")
            self.device = torch.device('cpu')
            self.using_gpu = False
        
        # Pre-allocate tensors for common operations
        self.eye4 = torch.eye(4, device=self.device)
        
        # Verify GPU is working
        if self.using_gpu:
            try:
                # Simple test tensor operation
                test_tensor = torch.ones(10, device=self.device)
                test_result = test_tensor + 1
                # Check for numeric results (avoiding NaNs)
                if torch.isnan(test_result).any():
                    raise Exception("GPU produced NaN values")
                print("GPU test successful")
            except Exception as e:
                print(f"GPU test failed: {e}")
                self.device = torch.device('cpu')
                self.using_gpu = False
                print("Falling back to CPU")
    
    def set_communication_params(self, communication_delay, packet_loss_prob):
        """Update communication parameters"""
        self.communication_delay = communication_delay
        self.packet_loss_prob = packet_loss_prob
    
    def to_tensor(self, array):
        """Convert numpy array to tensor on correct device"""
        if isinstance(array, torch.Tensor):
            return array.to(self.device)
        return torch.tensor(array, dtype=torch.float32, device=self.device)
    
    def to_numpy(self, tensor):
        """Convert tensor to numpy array"""
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.cpu().numpy()
        
    def calculate_bids_batch(self, robot_positions, robot_capabilities, task_positions, 
                task_capabilities, workloads, weights):
        """Calculate bids for all robot-task combinations in a single batch operation"""
        # Convert inputs to tensors
        r_pos = self.to_tensor(robot_positions)
        r_cap = self.to_tensor(robot_capabilities)
        t_pos = self.to_tensor(task_positions)
        t_cap = self.to_tensor(task_capabilities)
        work = self.to_tensor(workloads)
        
        # Extract weights
        alpha1 = weights['alpha1']
        alpha2 = weights['alpha2']
        alpha3 = weights['alpha3']
        alpha4 = weights['alpha4']
        alpha5 = weights.get('alpha5', 0.1)
        
        # CRITICAL FIX: Extract epsilon properly from weights
        epsilon = weights.get('epsilon', 0.01)
        
        # Get dimensions
        num_robots = r_pos.shape[0]
        num_tasks = t_pos.shape[0]
        
        # Calculate distances (batch operation)
        r_pos_expanded = r_pos.unsqueeze(1)
        t_pos_expanded = t_pos.unsqueeze(0)
        distances = torch.sqrt(torch.sum((r_pos_expanded - t_pos_expanded)**2, dim=2) + 1e-10)
        
        # Calculate terms
        distance_term = alpha1 / (distances + 1e-10)
        
        # Calculate proper configuration cost
        r_cap_expanded = r_cap.unsqueeze(1)
        t_cap_expanded = t_cap.unsqueeze(0)
        config_diffs = r_cap_expanded - t_cap_expanded
        config_costs = torch.sqrt(torch.sum(config_diffs**2, dim=2) + 1e-6)
        config_term = alpha2 / (config_costs + 1e-10)
        
        # Calculate capability similarity
        r_cap_norm = torch.norm(r_cap, dim=1, keepdim=True)
        t_cap_norm = torch.norm(t_cap, dim=1, keepdim=True)
        dot_products = torch.sum(r_cap_expanded * t_cap_expanded, dim=2)
        norms_product = r_cap_norm * t_cap_norm.t()
        similarity = dot_products / (norms_product + 1e-10)
        capability_term = alpha3 * similarity
        
        # Calculate workload term
        workload_term = alpha4 * work.unsqueeze(1).expand(-1, num_tasks)
        
        # CRITICAL FIX: Calculate bids WITHOUT epsilon scaling
        # The original formula should not scale by epsilon, as epsilon is used in price updates
        bids = distance_term + config_term + capability_term - workload_term
        
        return bids
            
    def run_auction_gpu(self, robot_positions, robot_capabilities, robot_statuses, 
               task_positions, task_capabilities, task_assignments, 
               epsilon, prices, weights=None, data_collector=None,
               unassigned_task_ids=None, termination_mode="assignment-complete",
               price_stability_threshold=0.01):
        """Run distributed auction algorithm on GPU
        
        Args:
            robot_positions: Robot positions as numpy array
            robot_capabilities: Robot capabilities as numpy array
            robot_statuses: List of robot status strings
            task_positions: Task positions as numpy array
            task_capabilities: Task capabilities as numpy array
            task_assignments: Current task assignments as numpy array
            epsilon: Minimum bid increment
            prices: Current prices as numpy array
            weights: Dictionary of weights for bid calculation
            data_collector: Optional data collector for metrics
            unassigned_task_ids: Optional list of original task IDs for data collection
            termination_mode: "assignment-complete" or "price-stabilized"
            price_stability_threshold: Threshold for price stability detection
            
        Returns:
            tuple: (new_assignments, new_prices, message_count)
        """
        # Apply communication delay
        if self.communication_delay > 0:
            time.sleep(self.communication_delay)
        
        # Apply packet loss simulation
        if random.random() < self.packet_loss_prob:
            # Return unchanged assignments with no messages to simulate packet loss
            return task_assignments, prices, 0
            
        # Default weights (can be passed as parameter)
        if weights is None:
            weights = {
                'alpha1': 0.3,
                'alpha2': 0.2,
                'alpha3': 0.3,
                'alpha4': 0.1,
                'alpha5': 0.1
            }
        
        # CRITICAL: Ensure epsilon is explicitly included in weights
        weights['epsilon'] = epsilon
        
        # Print debug info about epsilon
        print(f"\nDEBUG: Starting GPU auction with epsilon={epsilon}")
        
        # Convert inputs to tensors
        r_pos = self.to_tensor(robot_positions)
        r_cap = self.to_tensor(robot_capabilities)
        t_pos = self.to_tensor(task_positions)
        t_cap = self.to_tensor(task_capabilities)
        t_assign = self.to_tensor(task_assignments).long()
        prices_tensor = self.to_tensor(prices)
        
        # Create mask for operational robots
        op_robots = torch.tensor([i for i, status in enumerate(robot_statuses) 
                                if status == 'operational'], device=self.device)
        
        if len(op_robots) == 0:
            # No operational robots
            return t_assign.cpu().numpy(), prices_tensor.cpu().numpy(), 0
        
        # Get dimensions
        num_tasks = t_pos.shape[0]
        
        # Calculate workloads of robots
        workloads = torch.zeros(len(robot_statuses), device=self.device)
        
        # Set max_iterations based on theoretical bound: K² · bₘₐₓ/ε 
        # where K is task count and bₘₐₓ is maximum possible bid value
        K = num_tasks
        b_max = 100.0  # Estimate of maximum possible bid value
        theoretical_max_iter = int(K**2 * b_max / max(epsilon, 0.001))  # Avoid division by zero
        
        # Set a practical upper limit to prevent excessive iterations
        max_iterations = min(theoretical_max_iter, 1000)
        
        # Ensure price_stability_threshold is a scalar, not a list/array
        if isinstance(price_stability_threshold, (list, tuple, np.ndarray)):
            price_stability_threshold = float(price_stability_threshold[0])
        else:
            price_stability_threshold = float(price_stability_threshold)

        # For price stability detection
        price_changes_history = []
        max_stable_count = 3  # Number of iterations with stable prices to confirm convergence
        stable_count = 0
            
        # Record auction start with data collector
        if data_collector is not None:
            data_collector.record_iteration(0, {
                'phase': 'gpu_auction_start',
                'num_tasks': num_tasks,
                'epsilon': epsilon,
                'theoretical_max_iter': theoretical_max_iter,
                'max_iterations': max_iterations,
                'termination_mode': termination_mode,
                'price_stability_threshold': price_stability_threshold
            })
        
        # Initialize message counter and iteration counter
        messages = 0
        iterations_used = 0
        
        # Track bid statistics for debugging
        max_bid = 0.0
        min_bid = float('inf')
        
        # Store history of price updates for data collection
        price_updates = []
        bid_history = []
        
        for iteration in range(max_iterations):
            iterations_used += 1
            
            # Find unassigned tasks
            unassigned = (t_assign == 0).nonzero().flatten()
            
            # Record iteration start with data collector
            if data_collector is not None:
                # Convert prices to dictionary for data collector
                price_dict = {}
                for i, price in enumerate(prices_tensor.cpu().numpy()):
                    task_id = unassigned_task_ids[i] if unassigned_task_ids else i+1
                    price_dict[task_id] = float(price)
                
                data_collector.record_iteration(iteration, {
                    'phase': 'gpu_iteration_start',
                    'unassigned_count': len(unassigned),
                    'prices': price_dict
                })
            
            # Track tasks assigned in this iteration
            tasks_assigned_this_iter = 0
            
            # Track price changes in this iteration
            price_changes = []
            
            # For each operational robot
            for r_idx in op_robots:
                # Apply communication delay again for each robot's bidding process
                if self.communication_delay > 0:
                    time.sleep(self.communication_delay * 0.1)  # Scale down for per-robot delay
                
                # Check for packet loss per robot's bidding
                if random.random() < self.packet_loss_prob:
                    continue  # Skip this robot's bidding due to packet loss
                    
                # Calculate bids for all unassigned tasks
                task_indices = unassigned.cpu().numpy()
                
                if len(task_indices) == 0:
                    continue
                
                # Select only unassigned tasks
                t_pos_unassigned = t_pos[task_indices]
                t_cap_unassigned = t_cap[task_indices]
                
                # Calculate bids for this robot with unassigned tasks
                bids = self.calculate_bids_batch(
                    r_pos[r_idx:r_idx+1], 
                    r_cap[r_idx:r_idx+1],
                    t_pos_unassigned,
                    t_cap_unassigned,
                    workloads[r_idx:r_idx+1],
                    weights
                )
                
                # Track bid statistics
                if bids.numel() > 0:
                    current_max = float(torch.max(bids))
                    current_min = float(torch.min(bids))
                    max_bid = max(max_bid, current_max)
                    min_bid = min(min_bid, current_min) if current_min > 0 else min_bid
                
                # Count messages - one message per unassigned task (bid request)
                messages += len(task_indices)
                
                # Calculate utilities (bid - price)
                prices_unassigned = prices_tensor[task_indices]
                utilities = bids[0] - prices_unassigned
                
                # Record bids with data collector
                if data_collector is not None and unassigned_task_ids is not None:
                    for j, task_idx in enumerate(task_indices):
                        task_id = unassigned_task_ids[task_idx] if task_idx < len(unassigned_task_ids) else task_idx+1
                        robot_id = int(r_idx.cpu().numpy()) + 1
                        bid_value = float(bids[0, j].cpu().numpy())
                        utility_value = float(utilities[j].cpu().numpy())
                        
                        data_collector.record_bid(robot_id, task_id, bid_value, utility_value)
                        
                        # Store for history
                        bid_history.append({
                            'iteration': iteration,
                            'robot_id': robot_id,
                            'task_id': task_id,
                            'bid': bid_value,
                            'utility': utility_value
                        })
                
                # Find best task for this robot
                if len(utilities) > 0:
                    best_idx = torch.argmax(utilities).item()
                    best_utility = utilities[best_idx].item()
                    
                    if best_utility > 0:
                        # Get original task index
                        task_idx = task_indices[best_idx]
                        
                        # KEY FIX: The critical price update formula for auction algorithms
                        # This is where epsilon directly affects the algorithm output
                        old_price = float(prices_tensor[task_idx].item())
                        price_increase = epsilon + best_utility  # Key formula includes epsilon
                        new_price = old_price + price_increase
                        
                        # Track price change for stability detection
                        price_changes.append(float(price_increase))
                        
                        print(f"DEBUG: Task {task_idx} - Old price: {old_price:.4f}, " 
                            f"Increase: {price_increase:.4f} (epsilon={epsilon:.4f}, utility={best_utility:.4f}), "
                            f"New price: {new_price:.4f}")
                        
                        # Record price update with data collector
                        if data_collector is not None and unassigned_task_ids is not None:
                            task_id = unassigned_task_ids[task_idx] if task_idx < len(unassigned_task_ids) else task_idx+1
                            data_collector.record_price_update(
                                task_id, old_price, new_price, epsilon, best_utility)
                            
                            # Store for history
                            price_updates.append({
                                'iteration': iteration,
                                'task_id': task_id,
                                'old_price': old_price,
                                'new_price': new_price,
                                'epsilon': epsilon,
                                'utility': best_utility,
                                'price_increase': price_increase
                            })
                        
                        # Update price
                        prices_tensor[task_idx] = new_price
                        
                        # Assign task to robot
                        t_assign[task_idx] = r_idx + 1  # +1 because assignment 0 means unassigned
                        
                        # Record assignment with data collector
                        if data_collector is not None and unassigned_task_ids is not None:
                            task_id = unassigned_task_ids[task_idx] if task_idx < len(unassigned_task_ids) else task_idx+1
                            robot_id = int(r_idx.cpu().numpy()) + 1
                            data_collector.record_assignment(task_id, robot_id, iteration)
                        
                        # Update unassigned tasks
                        unassigned = (t_assign == 0).nonzero().flatten()
                        
                        # Message for assignment notification
                        messages += 1
                        
                        # Track assignments in this iteration
                        tasks_assigned_this_iter += 1
            
            # Calculate average price change
            avg_price_change = sum(price_changes) / len(price_changes) if price_changes else 0
            price_changes_history.append(avg_price_change)
            
            # Record iteration end with data collector
            if data_collector is not None:
                data_collector.record_iteration(iteration, {
                    'phase': 'gpu_iteration_end',
                    'tasks_assigned': tasks_assigned_this_iter,
                    'unassigned_remaining': len(unassigned),
                    'messages_sent': messages,
                    'avg_price_change': avg_price_change
                })
            
            # Check termination conditions based on mode
            if termination_mode == "assignment-complete":
                # Original termination logic
                if len(unassigned) == 0:
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
                if len(unassigned) == 0:
                    break
        
        # Debug logging for bid statistics
        print(f"DEBUG: Auction completed in {iterations_used}/{max_iterations} iterations")
        print(f"DEBUG: Final bid stats - Min: {min_bid:.4f}, Max: {max_bid:.4f}")
        print(f"DEBUG: Messages sent: {messages}")
        
        # Determine termination reason
        if iterations_used >= max_iterations:
            termination_reason = 'max_iterations'
        elif len(unassigned) == 0:
            termination_reason = 'all_assigned'
        elif tasks_assigned_this_iter == 0:
            termination_reason = 'no_progress'
        else:
            termination_reason = 'price_stabilized'
        
        # Record auction completion with data collector
        if data_collector is not None:
            # Convert final prices and assignments to dictionaries
            final_prices = {}
            final_assignments = {}
            
            for i, (price, assignment) in enumerate(zip(prices_tensor.cpu().numpy(), t_assign.cpu().numpy())):
                task_id = unassigned_task_ids[i] if unassigned_task_ids and i < len(unassigned_task_ids) else i+1
                final_prices[task_id] = float(price)
                
                if assignment > 0:  # If assigned
                    robot_id = int(assignment)
                    final_assignments[task_id] = robot_id
            
            data_collector.record_iteration(iterations_used, {
                'phase': 'gpu_auction_complete',
                'total_iterations': iterations_used,
                'final_prices': final_prices,
                'final_assignments': final_assignments,
                'total_messages': messages,
                'max_bid': max_bid,
                'min_bid': min_bid,
                'price_update_history': price_updates,
                'bid_history': bid_history,
                'termination_reason': termination_reason,
                'price_changes_history': price_changes_history
            })
        
        return t_assign.cpu().numpy(), prices_tensor.cpu().numpy(), messages