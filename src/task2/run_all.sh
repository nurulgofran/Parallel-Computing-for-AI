#!/bin/bash

echo "Running all Task 2 scripts..."
echo ""

echo "1. Sequential Scalar Product"
python task2_1_scalar_product_sequential.py
echo ""

echo "2. Parallel Scalar Product"
python task2_2_scalar_product_parallel.py
echo ""

echo "3. Time Steps Analysis & Speedup/Efficiency"
python task2_3_comparison.py
echo ""

echo "4. Generalization"
python task2_5_generalization.py
echo ""

echo "All Task 2 scripts completed!"
