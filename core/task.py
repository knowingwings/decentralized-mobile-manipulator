# core/task.py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Task:
    def __init__(self, task_id, position, capabilities=None, completion_time=None, 
                 prerequisites=None, collaborative=False):
        """Initialize a task with given parameters
        
        Args:
            task_id: Unique identifier for the task
            position: Position as [x, y] array
            capabilities: Required capabilities for this task
            completion_time: Time to complete the task (seconds)
            prerequisites: List of task IDs that must be completed before this task
            collaborative: Whether this task requires collaboration
        """
        self.id = task_id
        self.position = np.array(position, dtype=float)
        self.capabilities = np.random.rand(5) if capabilities is None else np.array(capabilities)
        self.completion_time = 5.0 if completion_time is None else float(completion_time)
        self.prerequisites = [] if prerequisites is None else list(prerequisites)
        self.collaborative = bool(collaborative)
        self.assigned_to = 0  # 0 = unassigned, otherwise robot ID
        self.progress = 0.0
        self.status = 'pending'  # 'pending', 'in_progress', 'completed', 'failed'
        self.required_config = np.random.rand(6)  # Random required configuration
        self.start_time = None
        self.completion_time_actual = None
        self.color = 'black'  # For visualization
        
    def update_status(self, status):
        """Update task status and color for visualization"""
        self.status = status
        
        # Update color based on status
        if status == 'pending':
            self.color = 'black'
        elif status == 'in_progress':
            self.color = 'green'
        elif status == 'completed':
            self.color = 'blue'
        else:  # failed
            self.color = 'red'
    
    def get_marker(self):
        """Get marker shape based on task status and type"""
        if self.collaborative:
            return '*'  # Star for collaborative tasks
        
        if self.status == 'pending':
            return 's'  # Square for pending
        elif self.status == 'in_progress':
            return 'd'  # Diamond for in progress
        elif self.status == 'completed':
            return 'o'  # Circle for completed
        else:  # failed
            return 'x'  # X for failed


class TaskDependencyGraph:
    def __init__(self, tasks=None):
        """Initialize task dependency graph
        
        Args:
            tasks: List of Task objects to initialize the graph
        """
        self.graph = nx.DiGraph()
        if tasks:
            self.update_graph(tasks)
    
    def update_graph(self, tasks):
        """Update dependency graph based on task prerequisites"""
        self.graph.clear()
        
        # Add all tasks as nodes
        for task in tasks:
            self.graph.add_node(task.id, task=task)
        
        # Add dependencies as edges
        for task in tasks:
            for prereq_id in task.prerequisites:
                if prereq_id in self.graph:  # Ensure prerequisite exists
                    self.graph.add_edge(prereq_id, task.id)
    
    def is_available(self, task_id):
        """Check if task is available (all prerequisites completed)"""
        if task_id not in self.graph:
            return False
            
        task = self.graph.nodes[task_id]['task']
        
        if not task.prerequisites:
            return True
            
        for prereq_id in task.prerequisites:
            if prereq_id in self.graph:
                prereq_task = self.graph.nodes[prereq_id]['task']
                if prereq_task.status != 'completed':
                    return False
            else:
                return False  # Prerequisite doesn't exist
        
        return True
    
    def get_critical_path(self):
        """Identify critical path through task dependency graph
        
        Returns:
            List of task IDs representing the critical path
        """
        if not self.graph:
            return []
            
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Task graph contains cycles")
        
        # Use negated task completion times as weights to find longest path
        for task_id in self.graph.nodes:
            self.graph.nodes[task_id]['weight'] = -self.graph.nodes[task_id]['task'].completion_time
        
        # Find all terminal nodes (no outgoing edges)
        terminal_nodes = [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]
        
        # Find the critical path by finding the longest path to each terminal node
        critical_path = []
        for end_node in terminal_nodes:
            # Find all possible start nodes (no incoming edges)
            start_nodes = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
            
            for start_node in start_nodes:
                try:
                    path = nx.dag_longest_path(self.graph, 
                                              weight='weight', 
                                              source=start_node, 
                                              target=end_node)
                    if len(path) > len(critical_path):
                        critical_path = path
                except nx.NetworkXNoPath:
                    continue
        
        return critical_path
    
    def get_task_criticality(self, task_id):
        """Calculate task criticality (number of dependent tasks)"""
        if task_id not in self.graph:
            return 0
            
        # Get all descendants of this task
        descendants = list(nx.descendants(self.graph, task_id))
        return len(descendants)
    
    def visualize(self, ax=None):
        """Visualize the task dependency graph"""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
            
        # Position nodes using spring layout
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes with colors based on status
        node_colors = []
        for node in self.graph.nodes:
            task = self.graph.nodes[node]['task']
            if task.status == 'pending':
                node_colors.append('lightgray')
            elif task.status == 'in_progress':
                node_colors.append('lightgreen')
            elif task.status == 'completed':
                node_colors.append('lightblue')
            else:  # failed
                node_colors.append('salmon')
                
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                              node_size=800, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, arrows=True, 
                              arrowstyle='->', arrowsize=20, ax=ax)
        
        # Draw labels
        labels = {node: f"T{node}" for node in self.graph.nodes}
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=10, ax=ax)
        
        # Highlight critical path if exists
        try:
            critical_path = self.get_critical_path()
            if critical_path:
                critical_edges = [(critical_path[i], critical_path[i+1]) 
                                for i in range(len(critical_path)-1)]
                nx.draw_networkx_edges(self.graph, pos, edgelist=critical_edges,
                                      edge_color='red', width=3, arrows=True,
                                      arrowstyle='->', arrowsize=20, ax=ax)
        except:
            pass  # Ignore errors in critical path calculation
            
        ax.set_title("Task Dependency Graph")
        ax.set_axis_off()
        
        return ax