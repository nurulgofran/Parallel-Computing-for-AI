# Parallel Computing for AI

This repository contains implementations and analysis of parallel algorithms and distributed computing concepts, with a focus on performance optimization and scalability.

## 🎯 What This Repository Contains

Implementations of various parallel computing algorithms including:
- Sequential vs. parallel algorithm comparisons
- PRAM (Parallel Random Access Machine) models
- Performance benchmarking and analysis
- Speedup and efficiency metrics
- Visualization of parallel computing concepts

## 🔬 Key Concepts Explored

### Parallel Computing Models
- **EREW-PRAM** (Exclusive Read Exclusive Write)
- **CREW-PRAM** (Concurrent Read Exclusive Write)
- **CRCW-PRAM** (Concurrent Read Concurrent Write)

### Performance Analysis
- Time complexity analysis: O(n) vs O(n/p)
- Speedup calculations: Sp = T₁/Tp
- Efficiency metrics: Ep = Sp/p
- Scalability testing across different data sizes and processor counts

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries:**
  - `numpy` - Numerical computations
  - `pandas` - Data analysis
  - `matplotlib` - Visualization
  - `seaborn` - Statistical plotting
  - `concurrent.futures` - Parallel execution
  - `multiprocessing` - Process management

## 🚀 Getting Started

### Prerequisites

Install required Python packages:

```bash
pip install numpy pandas matplotlib seaborn
```

### Running the Notebooks

1. Open any Jupyter Notebook in the repository
2. Run cells sequentially or use: Kernel → Restart Kernel and Run All Cells
3. Check generated outputs for results and visualizations

## 🎓 Learning Outcomes

This repository demonstrates:
- Parallel algorithm design and analysis
- PRAM computational models (EREW, CREW, CRCW)
- Performance metrics (speedup, efficiency)
- Time complexity analysis (O(n) vs O(n/p))
- Practical parallel programming implementation
- Benchmarking and performance visualization

## 📖 References

- Blelloch, Guy E. "Prefix Sums and Their Applications" - Carnegie Mellon University
- Parallel Random Access Machine (PRAM) models

## � About

This repository explores parallel computing concepts through practical implementations, focusing on algorithm optimization and performance analysis for AI and computational tasks.

---

