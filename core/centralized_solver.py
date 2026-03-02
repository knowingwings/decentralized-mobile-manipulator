# core/centralized_solver.py
import numpy as np
import itertools
from tqdm import tqdm

class CentralizedSolver:
    """Centralized solver for task allocation (for comparison with decentralized)"""
    
    def __init__(self):
        self.weights = {
            'alpha1': 0.3,  # Distance weight
            'alpha2': 0.2,  # Configuration cost weight
            'alpha3': 0.3,  # Capability similarity weight
            'alpha4': 0.1,  # Workload weight
            'alpha5': 0.1,  # Energy consumption weight
            'W': np.eye(6)  # Weight matrix for configuration
        }
    
    def solve(self, robots, tasks):
        """Find optimal solution using centralized approach
        
        For small problems, we can enumerate all possible assignments.
        For larger problems, we use a heuristic approach.
        
        Args:
            robots: List of Robot objects
            tasks: List of Task objects
            
        Returns:
            dict: Optimal solution details
        """
        num_robots = len(robots)
        num_tasks = len(tasks)
        
        # For small problems, we can enumerate all assignments
        if num_tasks <= 8:
            return self._solve_exhaustive(robots, tasks)
        else:
            # For larger problems, use a heuristic approach
            return self._solve_heuristic(robots, tasks)
    
    def _solve_exhaustive(self, robots, tasks):
        """Find optimal solution by exhaustive search"""
        num_robots = len(robots)
        num_tasks = len(tasks)
        
        # Copy tasks and create dependency graph
        from core.task import TaskDependencyGraph
        task_graph = TaskDependencyGraph(tasks)
        
        # Find tasks that can be executed immediately (no prerequisites)
        available_tasks = [task for task in tasks 
                          if task_graph.is_available(task.id)]
        
        # Generate all possible assignments of available tasks to robots
        best_makespan = float('inf')
        best_assignment = {}
        
        # All possible assignments of tasks to robots
        robot_ids = [robot.id for robot in robots]
        
        # Handle small problems efficiently
        if len(available_tasks) <= 6:
            # All possible assignments
            for assignment in itertools.product(robot_ids, repeat=len(available_tasks)):
                # Create assignment dict
                task_assignment = {available_tasks[i].id: assignment[i] 
                                  for i in range(len(available_tasks))}
                
                # Evaluate makespan
                makespan = self._evaluate_makespan(robots, tasks, task_assignment, task_graph)
                
                # Update best assignment
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_assignment = task_assignment
        else:
            # Heuristic for larger problems
            return self._solve_heuristic(robots, tasks)
        
        return {
            'assignment': best_assignment,
            'makespan': best_makespan
        }
    
    def _solve_heuristic(self, robots, tasks):
        """Find approximate solution using a greedy heuristic"""
        num_robots = len(robots)
        num_tasks = len(tasks)
        
        # Copy tasks and create dependency graph
        from core.task import TaskDependencyGraph
        task_graph = TaskDependencyGraph(tasks)
        
        # Initialize assignment and robot workloads
        assignment = {}
        workloads = {robot.id: 0 for robot in robots}
        completion_times = {robot.id: 0 for robot in robots}
        
        # Sort tasks by dependencies (topological sort)
        sorted_tasks = list(tasks)
        try:
            import networkx as nx
            G = nx.DiGraph()
            
            # Add nodes
            for task in tasks:
                G.add_node(task.id)
            
            # Add edges
            for task in tasks:
                for prereq in task.prerequisites:
                    G.add_edge(prereq, task.id)
            
            # Topological sort
            task_order = list(nx.topological_sort(G))
            sorted_tasks = [task for task in tasks if task.id in task_order]
            sorted_tasks.sort(key=lambda t: task_order.index(t.id))
        except:
            # If topological sort fails, use simple sort by ID
            sorted_tasks.sort(key=lambda t: t.id)
        
        # Assign tasks one by one
        for task in sorted_tasks:
            # Skip tasks with unassigned prerequisites
            prereqs_assigned = all(prereq in assignment for prereq in task.prerequisites)
            if not prereqs_assigned:
                continue
            
            # Find the best robot for this task
            best_robot = None
            best_finish_time = float('inf')
            
            for robot in robots:
                # Calculate earliest start time based on prerequisites
                earliest_start = 0
                for prereq_id in task.prerequisites:
                    prereq_robot = assignment.get(prereq_id)
                    if prereq_robot is not None:
                        prereq_finish = completion_times[prereq_robot]
                        earliest_start = max(earliest_start, prereq_finish)
                
                # Calculate finish time if this robot is assigned the task
                travel_time = np.linalg.norm(robot.position - task.position) / robot.max_linear_velocity
                start_time = max(completion_times[robot.id], earliest_start)
                finish_time = start_time + travel_time + task.completion_time
                
                # Update best robot
                if finish_time < best_finish_time:
                    best_finish_time = finish_time
                    best_robot = robot.id
            
            # Assign task to best robot
            if best_robot is not None:
                assignment[task.id] = best_robot
                completion_times[best_robot] = best_finish_time
                workloads[best_robot] += task.completion_time
        
        # Calculate makespan
        makespan = max(completion_times.values())
        
        return {
            'assignment': assignment,
            'makespan': makespan,
            'workloads': workloads
        }
    
    def _evaluate_makespan(self, robots, tasks, assignment, task_graph):
        """Evaluate makespan for a given assignment"""
        # Initialize robot positions and completion times
        robot_positions = {robot.id: robot.position.copy() for robot in robots}
        robot_completion_times = {robot.id: 0 for robot in robots}
        task_completion_times = {}
        
        # Topologically sort tasks based on prerequisites
        sorted_tasks = []
        remaining_tasks = list(tasks)
        
        while remaining_tasks:
            # Find tasks with all prerequisites satisfied
            for task in remaining_tasks:
                prereqs_done = all(prereq_id in task_completion_times 
                                  for prereq_id in task.prerequisites)
                if prereqs_done:
                    sorted_tasks.append(task)
                    remaining_tasks.remove(task)
                    break
            else:
                # If no tasks were added, there's a cycle or missing prerequisite
                break
        
        # Process tasks in order
        for task in sorted_tasks:
            # Skip tasks not in the assignment
            if task.id not in assignment:
                continue
                
            robot_id = assignment[task.id]
            
            # Calculate earliest start time (after prerequisites)
            earliest_start = 0
            for prereq_id in task.prerequisites:
                if prereq_id in task_completion_times:
                    earliest_start = max(earliest_start, task_completion_times[prereq_id])
            
            # Calculate robot travel time
            travel_time = np.linalg.norm(robot_positions[robot_id] - task.position) / 0.5  # Assuming 0.5 m/s speed
            
            # Calculate task start time (after robot is free and prerequisites are done)
            start_time = max(robot_completion_times[robot_id], earliest_start)
            
            # Calculate task completion time
            completion_time = start_time + travel_time + task.completion_time
            
            # Update robot state
            robot_positions[robot_id] = task.position.copy()
            robot_completion_times[robot_id] = completion_time
            
            # Record task completion time
            task_completion_times[task.id] = completion_time
        
        # Makespan is the maximum completion time
        if task_completion_times:
            return max(task_completion_times.values())
        else:
            return 0