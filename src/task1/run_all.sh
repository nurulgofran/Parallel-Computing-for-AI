#!/bin/bash

# Task 1 Complete Solution Runner
# Executes both Task 1.1 and Task 1.2

echo "=============================================================================="
echo "                    TASK 1: PREFIX SUM ALGORITHMS"
echo "                   Complete Solution (Task 1.1 & 1.2)"
echo "=============================================================================="
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
VENV_PYTHON="/Users/nurulgofran/Projects/Parallel-Computing-for-AI/.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the following commands first:"
    echo "  python3 -m venv ../../.venv"
    echo "  source ../../.venv/bin/activate"
    echo "  pip install matplotlib numpy"
    exit 1
fi

echo "✓ Using Python from virtual environment"
echo ""

# Create plots directory if it doesn't exist
mkdir -p plots
echo "✓ Plots directory ready"
echo ""

echo "------------------------------------------------------------------------------"
echo "Running Task 1.1: Algorithm Implementation and Comparison"
echo "------------------------------------------------------------------------------"
echo ""
$VENV_PYTHON task1_1_comparison.py
echo ""

echo "------------------------------------------------------------------------------"
echo "Running Task 1.2: Detailed Complexity Analysis"
echo "------------------------------------------------------------------------------"
echo ""
$VENV_PYTHON task1_2_complexity_analysis.py
echo ""

# Copy plots to main plots directory
echo "------------------------------------------------------------------------------"
echo "Organizing output files..."
echo "------------------------------------------------------------------------------"
cp plots/*.png ../../plots/ 2>/dev/null || true
echo "✓ Plots copied to main plots directory"
echo ""

echo "=============================================================================="
echo "                         TASK 1 COMPLETE!"
echo "=============================================================================="
echo ""
echo "Generated files:"
echo "  • Code implementations:"
echo "    - task1_1_prefix_sum_sequential.py"
echo "    - task1_1_prefix_sum_parallel.py"
echo "    - task1_1_comparison.py"
echo "    - task1_2_complexity_analysis.py"
echo ""
echo "  • Visualizations (PNG):"
echo "    - task1_1_sequential.png"
echo "    - task1_1_parallel.png"
echo "    - task1_1_comparison.png"
echo "    - task1_2_complexity_analysis.png"
echo "    - task1_2_scaling_analysis.png"
echo ""
echo "  • Documentation:"
echo "    - TASK1_SOLUTION_SUMMARY.md"
echo "    - README_TASK1.md"
echo ""
echo "All files are ready for submission!"
echo "=============================================================================="
