# Distributed Auction Algorithm for Multi-Robot Task Allocation

GPU-accelerated simulation framework implementing distributed auction-based task allocation for decentralized control of dual mobile manipulators in collaborative assembly tasks.

## Overview

This repository contains a Python/PyTorch implementation of a distributed auction algorithm for decentralized multi-robot coordination. The system enables mobile manipulators to autonomously allocate tasks among themselves without central coordination, while maintaining provable convergence properties and bounded optimality gaps.

**Key features:**
- Decentralized task allocation using distributed auction algorithm
- GPU-accelerated computation with PyTorch/CUDA for batch bid calculations
- Robust failure recovery with time-weighted consensus protocol
- Collaborative task handling with leader-follower coordination
- Task dependency management with topological ordering
- Interactive GUI for visualization and parameter tuning
- Comprehensive experiment framework for parameter optimization

This implementation is based on theoretical foundations from Zavlanos et al. (2008) with significant extensions for communication constraints, task dependencies, collaborative tasks, and failure recovery.

## Research Context

Developed as part of BEng dissertation research in Mechatronics Engineering at University of Gloucestershire (First Class Honours). This simulation framework validates the algorithmic approach before implementation on real robot systems.

**Mathematical guarantees:**
- Convergence time: O(K² · b_max/ε) iterations
- Optimality gap: ≤ 2ε from centralized optimal
- Recovery time: O(|T_f|) + O(b_max/ε) after robot failure

## Quick Start

### Installation

\`\`\`bash
# Clone the repository
git clone https://github.com/knowingwings/decentralized-mobile-manipulator.git
cd decentralized-mobile-manipulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
\`\`\`

### Running the Interactive GUI

\`\`\`bash
python main.py --mode gui --use-gpu
\`\`\`

The GUI provides:
- Real-time visualization of robot positions and task allocations
- Interactive parameter tuning (epsilon, alpha weights, etc.)
- Live metrics (convergence time, optimality gap, communication overhead)
- Failure injection and recovery testing

### Running Batch Experiments

\`\`\`bash
# Run factorial experiment with default config
python main.py --mode experiment --config config/factorial_design.yaml --use-gpu
\`\`\`

### Analyzing Results

\`\`\`bash
python main.py --mode analysis --results results/factorial_experiment.csv
\`\`\`

## Repository Structure

\`\`\`
.
├── core/                       # Core algorithm implementation
│   ├── auction.py              # Auction algorithm and bid calculation
│   ├── robot.py                # Robot representation and state
│   ├── task.py                 # Task definition and dependencies
│   ├── simulator.py            # Simulation engine
│   ├── gpu_accelerator.py      # GPU acceleration with PyTorch
│   ├── experiment_runner.py    # Batch experiment execution
│   ├── analysis.py             # Statistical analysis
│   └── centralized_solver.py   # Optimal baseline (Hungarian algorithm)
│
├── gui/                        # Interactive visualization
│   ├── visualization.py        # Main GUI application
│   ├── control_panel.py        # Parameter controls
│   └── plots.py                # Real-time plotting
│
├── config/                     # Experiment configurations
│   ├── default.yaml            # Default parameters
│   ├── factorial_design.yaml   # Full factorial experiment
│   ├── gpu_test.yaml           # GPU performance testing
│   └── price_stability.yaml    # Price convergence analysis
│
├── utils/                      # Utility functions
│   ├── data_collector.py       # Metrics collection
│   ├── metrics.py              # Performance metrics
│   └── check_gpu.py            # GPU availability check
│
├── main.py                     # Entry point
├── requirements.txt            # Python dependencies
└── README.md                   # This file
\`\`\`

## Algorithm Details

### Distributed Auction Algorithm

Each robot maintains:
- Bid values for each task based on capability match, distance, workload, priority
- Price estimates for all tasks (synchronized via consensus)
- Task assignments (locally determined, globally coordinated)

**Bid calculation:**
\`\`\`
bid_ij = α₁·capability - α₂·distance - α₃·workload + α₄·priority - α₅·price_j
\`\`\`

**Price update with consensus:**
\`\`\`
price_j = max(bids for task_j)
price_j ← (1-γ)·price_j + γ·Σ(neighbor_prices)  # Time-weighted consensus
\`\`\`

### GPU Acceleration

Batch computation of all robot-task bid pairs using PyTorch:
- ~10x speedup on CUDA-enabled GPUs
- Automatic CPU fallback
- Support for both NVIDIA and AMD GPUs

### Failure Recovery

When a robot fails:
1. Detect failure via missed heartbeats
2. Identify orphaned tasks
3. Run recovery auction with adjusted weights (β parameters)
4. Reallocate tasks to operational robots

## Configuration

### Key Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| epsilon | Minimum bid increment (controls optimality vs speed) | 0.05 | 0.01-1.0 |
| alpha1 | Weight for capability matching | 0.8 | 0.0-2.0 |
| alpha2 | Weight for distance cost | 0.3 | 0.0-2.0 |
| alpha3 | Weight for workload balancing | 1.0 | 0.0-2.0 |
| alpha4 | Weight for task priority | 1.2 | 0.0-2.0 |
| alpha5 | Weight for current price | 0.2 | 0.0-1.0 |
| gamma | Consensus protocol weight | 0.5 | 0.0-1.0 |
| lambda | Information decay rate | 0.1 | 0.0-1.0 |

See config/default.yaml for full configuration options.

## Performance

Benchmarks on AMD Radeon RX 7900 XTX (24GB):

| Scenario | CPU Time | GPU Time | Speedup |
|----------|----------|----------|---------|
| 2 robots, 10 tasks | 45ms | 4ms | 11.2x |
| 2 robots, 50 tasks | 210ms | 18ms | 11.7x |
| 4 robots, 100 tasks | 820ms | 65ms | 12.6x |

Algorithm performance (2 robots, 10 tasks, ε=0.05):
- Convergence: 12-18 iterations (typical)
- Optimality gap: < 2% from centralized optimal
- Recovery time: 3-5 iterations after failure

## Related Work

**Theoretical Foundation:**
- Zavlanos, M.M., Spesivtsev, L. and Pappas, G.J. (2008). "A distributed auction algorithm for the assignment problem." IEEE CDC, pp. 1212-1217.

**Consensus Protocols:**
- Olfati-Saber, R., Fax, J.A. and Murray, R.M. (2007). "Consensus and Cooperation in Networked Multi-Agent Systems." Proceedings of the IEEE, 95(1), pp. 215-233.

**Multi-Robot Coordination:**
- Shorinwa, O., et al. (2023). "Distributed Optimization Methods for Multi-Robot Systems: Part 2—A Survey." IEEE RAM.

## Future Work

- ROS2 implementation for deployment on real mobile manipulators
- Multi-team coordination with hierarchical auction structure
- Dynamic task arrival during execution
- Hardware validation with sim-to-real transfer

## Citation

If you use this code in your research, please cite:

\`\`\`bibtex
@software{lehuray2024distributed,
  author = {Le Huray, Thomas},
  title = {Distributed Auction Algorithm for Multi-Robot Task Allocation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/knowingwings/decentralized-mobile-manipulator}
}
\`\`\`

## License

MIT License - see LICENSE for details.

## Author

**Thomas Le Huray**
- GitHub: @knowingwings
- LinkedIn: /in/tom-le-huray

MSc Robotics and Autonomous Systems @ University of Bath  
BEng Mechatronics (First Class Honours) @ University of Gloucestershire

---

*Part of ongoing research in decentralized multi-robot coordination.*
