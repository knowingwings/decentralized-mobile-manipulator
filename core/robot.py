# core/robot.py
import numpy as np

class Robot:
    def __init__(self, robot_id, position, orientation, config=None, capabilities=None):
        """Initialize a robot with parameters matching TurtleBot3 Waffle Pi with OpenMANIPULATOR-X
        
        Args:
            robot_id: Unique identifier for the robot
            position: Initial position as [x, y] array
            orientation: Initial orientation in radians
            config: Initial joint configuration (4-DOF manipulator + gripper)
            capabilities: Capability vector describing robot abilities
        """
        self.id = robot_id
        self.position = np.array(position, dtype=float)
        self.orientation = float(orientation)
        # Reduced from 6-DOF to 4-DOF + gripper to match OpenMANIPULATOR-X
        self.config = np.zeros(5) if config is None else np.array(config)
        self.capabilities = np.random.rand(5) if capabilities is None else np.array(capabilities)
        self.status = 'operational'  # 'operational', 'failed', 'partial_failure'
        self.workload = 0
        self.assigned_tasks = []
        self.trajectory = [np.copy(position)]  # Store trajectory for visualization
        
        # Kinematics parameters - Updated to match TurtleBot3 Waffle Pi
        self.max_linear_velocity = 0.26  # m/s (reduced from 0.5 to match platform)
        self.max_angular_velocity = 1.82  # rad/s (updated from 1.0 to match platform)
        self.base_dimensions = [0.281, 0.306]  # width, length in meters
        self.manipulator_reach = 0.380  # meters (reduced from 0.85m)
        self.manipulator_payload = 0.5  # kg (reduced from 5kg)
        
        # Virtual F/T sensor for force-controlled operations
        self.ft_sensor = {'force': np.zeros(3), 'torque': np.zeros(3)}
        
        # For visualization
        self.color = 'blue'
    
    def update_status(self, status):
        """Update robot operational status"""
        self.status = status
        if status != 'operational':
            self.color = 'red' if status == 'failed' else 'orange'
        else:
            self.color = 'blue'
    
    def calculate_bid(self, task, weights, current_workload, energy_consumption=0):
        """Calculate bid for a task based on the formula from the paper
        
        b_ij = (α₁/d_ij) + (α₂/c_ij) + α₃·s_ij - α₄·l_i - α₅·e_ij
        
        Args:
            task: Task object to calculate bid for
            weights: Dictionary of weight parameters
            current_workload: Current workload value
            energy_consumption: Estimated energy consumption (optional)
            
        Returns:
            float: Bid value
        """
        # Extract weights
        alpha1 = weights['alpha1']
        alpha2 = weights['alpha2']
        alpha3 = weights['alpha3']
        alpha4 = weights['alpha4']
        alpha5 = weights['alpha5']
        W = weights['W']
        
        # Calculate distance to task
        # Use faster Euclidean distance calculation for 2D points
        dx = self.position[0] - task.position[0]
        dy = self.position[1] - task.position[1]
        distance = (dx*dx + dy*dy)**0.5
        
        # Calculate configuration transition cost - adapted for 4-DOF manipulator
        # Use only first 4 components for configuration to match OpenMANIPULATOR-X
        config_diff = self.config[:4] - task.required_config[:4]
        # Ensure W matrix is properly sized for 4-DOF
        W_sub = W[:4,:4] if W.shape[0] > 4 else W
        config_cost = np.sqrt(config_diff.T @ W_sub @ config_diff)
        
        # Calculate capability similarity
        robot_cap_norm = np.linalg.norm(self.capabilities)
        task_cap_norm = np.linalg.norm(task.capabilities)
        
        if robot_cap_norm > 0 and task_cap_norm > 0:
            # Use dot product for faster calculation
            capability_similarity = np.dot(self.capabilities, task.capabilities) / (robot_cap_norm * task_cap_norm)
        else:
            capability_similarity = 0
        
        # Calculate bid
        term1 = alpha1 / distance if distance > 0 else float('inf')
        term2 = alpha2 / config_cost if config_cost > 0 else float('inf')
        
        bid = term1 + term2 + (alpha3 * capability_similarity) - (alpha4 * current_workload) - (alpha5 * energy_consumption)
        return bid
    
    def calculate_recovery_bid(self, standard_bid, task_progress, task_criticality, task_urgency, beta_weights):
        """Calculate recovery bid for task reallocation after robot failure
        
        b^r_ij = b_ij + β₁(1 - p_j) + β₂·criticality(j) + β₃·urgency(j)
        
        Args:
            standard_bid: Standard bid value
            task_progress: Current progress on the task (0-1)
            task_criticality: Criticality measure of the task
            task_urgency: Urgency measure of the task
            beta_weights: Dictionary of beta weight parameters
            
        Returns:
            float: Recovery bid value
        """
        beta1 = beta_weights['beta1']
        beta2 = beta_weights['beta2']
        beta3 = beta_weights['beta3']
        
        recovery_bid = standard_bid + \
                      beta1 * (1 - task_progress) + \
                      beta2 * task_criticality + \
                      beta3 * task_urgency
        return recovery_bid
    
    def update_position(self, target_position, dt, max_speed=None):
        """Move robot toward target position
        
        Args:
            target_position: Target position to move toward
            dt: Time step
            max_speed: Maximum speed (or None to use robot's max_linear_velocity)
            
        Returns:
            bool: True if target reached, False otherwise
        """
        if max_speed is None:
            max_speed = self.max_linear_velocity
            
        # Get direction vector
        direction = target_position - self.position
        distance = np.linalg.norm(direction)
        
        # If already at target (or very close)
        if distance < 0.1:
            return True
            
        # Normalize direction
        direction = direction / distance
        
        # Calculate movement distance for this time step
        move_distance = min(max_speed * dt, distance)
        
        # Update position
        new_position = self.position + move_distance * direction
        self.position = new_position
        
        # Store trajectory point
        self.trajectory.append(np.copy(new_position))
        
        # Update orientation based on movement direction
        target_orientation = np.arctan2(direction[1], direction[0])
        orientation_diff = target_orientation - self.orientation
        
        # Normalize to [-pi, pi]
        while orientation_diff > np.pi:
            orientation_diff -= 2*np.pi
        while orientation_diff < -np.pi:
            orientation_diff += 2*np.pi
        
        # Apply angular velocity constraint
        max_rotation = self.max_angular_velocity * dt
        if abs(orientation_diff) > max_rotation:
            orientation_diff = np.sign(orientation_diff) * max_rotation
            
        # Update orientation
        self.orientation += orientation_diff
        
        # Return whether target is reached
        return distance < 0.1
    
    def estimate_travel_time(self, target_position):
        """Estimate travel time to target position"""
        distance = np.linalg.norm(target_position - self.position)
        return distance / self.max_linear_velocity
    
    def apply_partial_failure(self, capability_reduction_factor=0.5):
        """Apply partial failure to robot by reducing capabilities"""
        # Create random mask with values between capability_reduction_factor and 1
        mask = capability_reduction_factor + (1-capability_reduction_factor) * np.random.rand(len(self.capabilities))
        self.capabilities = self.capabilities * mask
        self.update_status('partial_failure')
        
    def reset_trajectory(self):
        """Reset stored trajectory"""
        self.trajectory = [np.copy(self.position)]
        
    def update_ft_sensor(self, force, torque):
        """Update virtual force/torque sensor readings
        
        Args:
            force: 3D force vector [fx, fy, fz]
            torque: 3D torque vector [tx, ty, tz]
        """
        self.ft_sensor['force'] = np.array(force)
        self.ft_sensor['torque'] = np.array(torque)