# Parallel Computing for AI — FAU Exercises

> University exercise solutions for Parallel Computing for AI course at FAU Erlangen-Nürnberg. Covers MPI, OpenMP, CUDA, and parallel algorithms.

## Architecture

**Stack**: C (MPI/OpenMP), Python (Jupyter notebooks), CUDA

```
Exercise 1/          → First exercise set
Exercise 3/          → Third exercise set
Doc/                 → Course documentation
*.c                  → C source files (MPI, OpenMP)
*.py                 → Python utility scripts
*.ipynb              → Jupyter notebooks for exercises
```

### Key Files
- `ompi_helloworld.c` — MPI Hello World example
- `ompi_colcomm.c` — MPI collective communication exercise
- `PARMAI_Exercise5.ipynb` — Jupyter notebook exercise
- `create_matmul_notebook.py` — Matrix multiplication notebook generator
- `modify_notebook.py` — Notebook processing utility

## Developer Workflows

```bash
# Compile MPI programs
mpicc -o helloworld ompi_helloworld.c
mpirun -np 4 ./helloworld

# Compile OpenMP programs
gcc -fopenmp -o program program.c

# Jupyter notebooks
jupyter notebook

# Python scripts
python create_matmul_notebook.py
```

## Conventions

- **C code**: MPI uses `MPI_Init`/`MPI_Finalize` pattern — always pair them
- **Notebooks**: Exercise solutions in Jupyter — include explanations in markdown cells
- **Naming**: Exercises numbered sequentially in directories

## Gotchas

1. **MPI installation required** — needs OpenMPI or MPICH installed on the system
2. **Cluster-oriented** — some exercises are designed for cluster execution, not local
3. **Exercise PDFs** — included for reference, don't modify
