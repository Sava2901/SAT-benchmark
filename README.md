# SAT Benchmark: Comparative Analysis of SAT Solving Algorithms

A comprehensive implementation and analysis of modern SAT solving techniques, focusing on Conflict-Driven Clause Learning (CDCL) and various branching heuristics.

## Overview

This project implements and benchmarks multiple SAT solving algorithms, including:
- Resolution Method
- Davis-Putnam (DP)
- Davis-Putnam-Logemann-Loveland (DPLL)
- Conflict-Driven Clause Learning (CDCL)

The CDCL implementation includes seven different branching heuristics:
- ORDERED
- RANDOM
- VSIDS
- MiniSat-style activity
- JEROSLOW-WANG
- BERKMIN
- Clause-size-based

And 3 restarting strategies:
- Luby
- Geometric
- No Restarter

## Features

- **Multiple Algorithm Implementations**: Compare classical and modern SAT solving approaches
- **Diverse Heuristic Support**: Test various branching strategies within CDCL framework
- **Comprehensive Benchmarking**: Evaluate performance across different problem types
- **Detailed Analysis**: Track execution time, decision counts, and memory usage
- **Visualization Tools**: Generate performance comparison charts
- **Modular Architecture**: Easy to extend with new algorithms or heuristics
- **Caching System**: Results are cached for faster subsequent runs
- **Reset Option**: Force fresh benchmark runs when needed

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Sava2901/SAT-benchmark.git
cd SAT-benchmark
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Benchmarks

To run the complete benchmark suite:

```bash
python benchmark.py
```

### Caching System

The benchmark system implements a caching mechanism to improve performance:
- Results are stored in a temporary directory
- Subsequent runs with the same parameters will use cached results
- Use the `--reset` flag to force fresh benchmark runs
- Cache is automatically invalidated when parameters change

To rerun the complete benchmark suite:

```bash
python benchmark.py --reset
```

### Custom Benchmarks

You can add your own benchmark instances by placing them in the `benchmarks` directory:

```
benchmarks/
├── uf/           # Uniform random 3-SAT
├── uuf/          # Unsatisfiable uniform random 3-SAT
├── flat/         # Flat graph coloring
└── my_benchmarks/ # Your custom benchmarks
```

Requirements for custom benchmarks:
- Each benchmark category should be in its own subdirectory
- All files must be in DIMACS CNF format
- Files should be named consistently (e.g., `instance_001.cnf`, `instance_002.cnf`)
- Include a README.md in your benchmark directory describing the instances

## Project Structure

```
SAT-benchmark/
├── solvers/
│   ├── cdcl/          # CDCL implementation with various heuristics
│   ├── dp.py          # Davis-Putnam algorithm
│   ├── dpll.py        # DPLL algorithm
│   └── resolution.py  # Resolution-based solver
├── utils/
│   ├── parser.py      # CNF file parser
│   ├── memory.py      # Memory usage tracking
│   ├── timer.py       # Execution time measurement
│   └── summariser/    # Benchmark results analysis
├── benchmarks/        # Test instances
│   ├── uf/           # Uniform random 3-SAT
│   ├── uuf/          # Unsatisfiable uniform random 3-SAT
│   └── flat/         # Flat graph coloring
├── benchmark.py      # Main benchmarking script
├── plot.py          # Visualization tools
└── requirements.txt  # Project dependencies
```

## Input Format

The solver accepts CNF files in DIMACS format:

```
p cnf <variables> <clauses>
<clause1>
<clause2>
...
```

Example:
```
p cnf 3 2
1 -2 3 0
-1 2 0
```

## Results

Benchmark results include:
- Execution time
- Number of decisions
- Memory usage
- Success rate
- Timeout statistics

Results are saved in CSV format and can be visualized using the included plotting tools.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SATLIB for benchmark instances
- MiniSat for inspiration in CDCL implementation
- The SAT solving community for valuable insights

## Contact

Sergiu Sava
- Email: sergiu.sava06@e-uvt.ro
- Department of Computer Science, West University Timișoara

## Future Work

- Parallel solving support
- Machine learning-based heuristic selection
- Additional benchmark categories
- Web interface for result visualization