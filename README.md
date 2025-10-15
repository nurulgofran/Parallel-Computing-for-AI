# PARMAI Sheet 1 Submission

## Author
[Your Name - YOURLASTNAME]

## Contents
This submission contains solutions for PARMAI Exercise Sheet 1.

## Structure
```
.
├── README.md           # This file
├── src/               # Source code files
│   ├── task1/        # Code for Task 1
│   ├── task2/        # Code for Task 2
│   └── ...           # Additional tasks
├── plots/            # Generated plots (PNG/JPG format)
└── results/          # Output files and results
```

## Requirements
### For C/C++ Code
- Compiler: GCC/G++ (version X.X or higher)
- Make (optional, if using Makefile)

### For Python Code
- Python 3.x
- Required packages:
  ```bash
  pip install numpy matplotlib scipy
  ```

### For Java Code
- JDK (version X or higher)

## How to Compile & Run

### C/C++ Code
```bash
# Navigate to the specific task directory
cd src/task1

# Compile
gcc -o program program.c -lm
# Or for C++
g++ -o program program.cpp

# Run
./program
```

### Python Code
```bash
# Navigate to the specific task directory
cd src/task1

# Run directly
python script.py

# Or if using python3
python3 script.py
```

### Java Code
```bash
# Navigate to the specific task directory
cd src/task1

# Compile
javac Program.java

# Run
java Program
```

## Tasks Description

### Task 1: Prefix Sum Algorithms (Think Parallel - 20 pt.)
- **File(s)**: `src/task1/`
  - `task1_1_prefix_sum_sequential.py` - Task 1.1 Sequential O(n) algorithm
  - `task1_1_prefix_sum_parallel.py` - Task 1.1 Parallel O(log n) Blelloch algorithm
  - `task1_1_comparison.py` - Task 1.1 Combined analysis and comparison
  - `task1_2_complexity_analysis.py` - Task 1.2 Detailed complexity analysis
- **Description**: 
  - **Task 1.1**: Implementation of sequential and parallel prefix sum algorithms for array [2,4,6,8,1,3,5,7]
  - **Task 1.2**: Calculate time steps, operations, and CPU requirements for both approaches
- **Reference**: Guy E. Blelloch, "Prefix Sums and Their Applications", Carnegie Mellon University
- **Plots**: 
  - `plots/task1_1_sequential.png` - Task 1.1 Sequential algorithm visualization
  - `plots/task1_1_parallel.png` - Task 1.1 Parallel algorithm (Blelloch) visualization
  - `plots/task1_1_comparison.png` - Task 1.1 Side-by-side comparison
  - `plots/task1_2_complexity_analysis.png` - Task 1.2 Detailed complexity table
  - `plots/task1_2_scaling_analysis.png` - Task 1.2 Scaling analysis charts
- **How to run**:
  ```bash
  cd src/task1
  python task1_1_comparison.py              # Task 1.1: Complete algorithm comparison
  python task1_2_complexity_analysis.py     # Task 1.2: Detailed complexity analysis
  # OR run individually:
  python task1_1_prefix_sum_sequential.py
  python task1_1_prefix_sum_parallel.py
  ```
- **Results (Task 1.1 & 1.2)**:
  - **Sequential**: 7 time steps, 7 operations, 1 CPU, O(n) complexity
  - **Parallel**: 6 time steps, 14 operations, 4 CPUs max, O(log n) complexity
  - **Speedup**: 1.17x faster (for n=8)
  - **Efficiency**: 29.2% (parallel resources utilization)
  - **Cost**: Parallel is 3.43x more expensive in processor-time units
  - **Conclusion**: Parallel scales better with larger arrays (O(log n) vs O(n))

### Task 2: [Task Name]
- **File(s)**: `src/task2/`
- **Description**: [Brief description of what this task does]
- **Plots**: `plots/task2_*.png`
- **How to run**: [Specific instructions]

## Notes
- All plots are provided in PNG or JPG format as required
- [Any additional notes or considerations]

## Results
[Brief summary of key findings or results]
