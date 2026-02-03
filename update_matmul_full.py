import json

notebook_path = "Exercise 3/opencl_matmul.ipynb"

# --- C++ Code with Hybrid Benchmark ---
cpp_code = r"""%%writefile ocl_matmul.cpp
/**
 * OpenCL Matrix Multiplication - Hybrid CPU + GPU
 * Exercise 3, Problem 2
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

const char* kernelSource = 
"__kernel void matMul(__global const float* A, \n"
"                     __global const float* B, \n"
"                     __global float* C, \n"
"                     const int width, \n"
"                     const int rowOffset) { \n"
"    int col = get_global_id(0); \n" 
"    int row = get_global_id(1) + rowOffset; \n" // Apply offset
"    \n"
"    if (col < width && row < width) { \n"
"        float sum = 0.0f; \n"
"        for (int k = 0; k < width; k++) { \n"
"            sum += A[row * width + k] * B[k * width + col]; \n"
"        } \n"
"        C[row * width + col] = sum; \n"
"    } \n"
"}\n";

void checkError(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        printf("Error during %s: %d\n", operation, error);
        exit(1);
    }
}

double getCurrentTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// CPU Matrix Multiplication for a range of rows [startRow, endRow)
void matrixMulCPU_Partial(float* A, float* B, float* C, int width, int startRow, int endRow) {
    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += A[i * width + k] * B[k * width + j];
            }
            C[i * width + j] = sum;
        }
    }
}

void verify_result(float* h_C, float* h_C_Ref, int width) {
    int errors = 0;
    for (int i = 0; i < width * width; i++) {
        if (fabs(h_C[i] - h_C_Ref[i]) > 1e-3) {
            if (errors < 5) printf("Mismatch at %d: %.4f vs %.4f\n", i, h_C[i], h_C_Ref[i]);
            errors++;
        }
    }
    if (errors == 0) printf("Verification PASSED\n");
    else printf("Verification FAILED (Errors: %d)\n", errors);
}

void run_benchmark(int width, float gpu_portion, double *totalTimes, int T, int verify) {
    size_t size = width * width * sizeof(float);
    
    // Calculate split
    int gpu_rows = (int)(width * gpu_portion);
    int cpu_rows = width - gpu_rows;
    // CPU does rows [0, cpu_rows)
    // GPU does rows [cpu_rows, width)
    
    if (!verify) {
        printf("Config: GPU=%.0f%% (Rows %d-%d), CPU=%.0f%% (Rows 0-%d)\n", 
               gpu_portion*100, cpu_rows, width, (1.0-gpu_portion)*100, cpu_rows);
    }

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_C_Ref = verify ? (float*)malloc(size) : NULL;

    srand(2025);
    for(int i=0; i<width*width; i++) {
        h_A[i] = (float)rand()/RAND_MAX;
        h_B[i] = (float)rand()/RAND_MAX;
    }
    
    // Verify: compute full CPU reference first
    if (verify) {
        printf("Computing CPU Reference...\n");
        matrixMulCPU_Partial(h_A, h_B, h_C_Ref, width, 0, width);
    }

    // OpenCL Setup
    cl_platform_id platform; cl_device_id device; cl_context context; cl_command_queue queue;
    cl_program program; cl_kernel kernel; cl_int ret; cl_uint n_plat, n_dev;
    
    clGetPlatformIDs(1, &platform, &n_plat);
    if(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &n_dev) != CL_SUCCESS) {
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &n_dev);
    }
    
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &ret);
    queue = clCreateCommandQueue(context, device, 0, &ret);
    
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, h_A, &ret);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, h_B, &ret);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, &ret); // Full buffer for simplicity
    
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &ret);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "matMul", &ret);
    
    // Set fixed args
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &width);
    int rowOffset = cpu_rows;
    clSetKernelArg(kernel, 4, sizeof(int), &rowOffset);
    
    // GPU Work Size
    // We only need to compute 'gpu_rows' rows. 
    // Global Size Y = gpu_rows.
    size_t globalSize[2] = { (size_t)width, (size_t)(gpu_rows > 0 ? gpu_rows : 1) };
    size_t localSize[2] = { 16, 16 };
    // Pad global size if needed... reusing naive logic for now
    
    int iterations = verify ? 1 : T;
    
    for(int iter=0; iter<iterations; iter++) {
        double start = getCurrentTime();
        
        // 1. CPU Part
        if (cpu_rows > 0) {
            matrixMulCPU_Partial(h_A, h_B, h_C, width, 0, cpu_rows);
        }
        
        // 2. GPU Part
        if (gpu_rows > 0) {
            ret = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
            clFinish(queue);
            // Read back ONLY the GPU part
            size_t offset = cpu_rows * width * sizeof(float);
            size_t cb = gpu_rows * width * sizeof(float);
            clEnqueueReadBuffer(queue, d_C, CL_TRUE, offset, cb, h_C + (cpu_rows * width), 0, NULL, NULL);
        }
        
        double end = getCurrentTime();
        if (totalTimes) totalTimes[iter] = end - start;
    }
    
    if (verify) {
        verify_result(h_C, h_C_Ref, width);
        free(h_C_Ref);
    }
    
    clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);
    free(h_A); free(h_B); free(h_C);
}

int main(int argc, char* argv[]) {
    // Usage: ./ocl_matmul [N]
    // Default N=1024
    int N = 1024;
    if (argc > 1) N = atoi(argv[1]);

    // 1. Correctness Check (50/50 split)
    printf("=== Correctness Check (N=%d) ===\n", N);
    run_benchmark(N, 0.5, NULL, 1, 1);
    
    // 2. Stress Test Mode (special large inputs)
    // If N > 8000, we assume it's a manual stress test run and skip full benchmark loop
    if (N > 8000) {
        printf("\n=== Stress Test Run (N=%d) ===\n", N);
        run_benchmark(N, 1.0, NULL, 1, 0);
        return 0;
    }

    // 3. Benchmark Suite
    const int T = 3;
    double times[T];
    FILE *csv = fopen("matmul_results.csv", "w");
    if (csv) {
        fprintf(csv, "Portion");
        for(int i=0; i<T; i++) fprintf(csv, ",Run_%d", i+1);
        fprintf(csv, "\n");
    }
    
    float portions[] = {0.0, 0.25, 0.5, 0.75, 1.0};
    printf("\n=== Starting Benchmark (N=%d) ===\n", N);
    for(int p=0; p<5; p++) {
        run_benchmark(N, portions[p], times, T, 0);
        if (csv) {
            fprintf(csv, "%.2f", portions[p]);
            for(int i=0; i<T; i++) fprintf(csv, ",%.6f", times[i]);
            fprintf(csv, "\n");
        }
    }
    if (csv) fclose(csv);
    printf("Benchmark Complete. Results in matmul_results.csv\n");
    return 0;
}
"""

# --- Python Plotting Code ---
plot_code = """import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    df = pd.read_csv('matmul_results.csv')
    # Calculate average time
    time_cols = [c for c in df.columns if 'Run_' in c]
    df['AvgTime'] = df[time_cols].mean(axis=1)
    
    # Calculate Speedup relative to CPU only (Portion 0.0)
    cpu_time = df.loc[df['Portion'] == 0.0, 'AvgTime'].values[0]
    df['Speedup'] = cpu_time / df['AvgTime']
    
    print(df[['Portion', 'AvgTime', 'Speedup']])
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Portion'], df['Speedup'], marker='o', linewidth=2)
    plt.title("Amdahl's Law - Matrix Multiplication Speedup")
    plt.xlabel("Parallel Portion (GPU Workload)")
    plt.ylabel("Speedup (vs Pure CPU)")
    plt.grid(True)
    plt.savefig('amdahls_law_matmul.png')
    plt.show()
    print("Plot saved to amdahls_law_matmul.png")
except Exception as e:
    print(f"Could not plot: {e}")
"""

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Check if installation cell exists
install_exists = False
for cell in nb['cells']:
    if "!apt-get install -y --no-install-recommends ocl-icd-opencl-dev" in "".join(cell['source']):
        install_exists = True
        break

if not install_exists:
    install_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!apt-get update -qq\n",
            "!apt-get install -y --no-install-recommends software-properties-common build-essential\n",
            "!apt-get install -y --no-install-recommends ocl-icd-opencl-dev opencl-headers\n",
            "!mkdir -p /etc/OpenCL/vendors && echo \"libnvidia-opencl.so.1\" > /etc/OpenCL/vendors/nvidia.icd"
        ]
    }
    # Insert after header (index 1)
    nb['cells'].insert(1, install_cell)

# Replace the C++ cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if src.startswith('%%writefile ocl_matmul.cpp'):
            cell['source'] = cpp_code.splitlines(keepends=True)
            # Compilation Command Cell (assumed to be next, or we can find it)
            # Actually, we need to ensure the cells Following this one are correct.
            # 1. Compile
            # 2. Small Test
            # 3. Large Test
            # 4. Benchmark

# Logic to Insert/Update the Runtime Cells
# We'll remove existing run cells and inject the explicit sequence.
# Find index of C++ cell
cpp_cell_idx = -1
for i, cell in enumerate(nb['cells']):
    if "%%writefile ocl_matmul.cpp" in "".join(cell['source']):
        cpp_cell_idx = i
        break

if cpp_cell_idx != -1:
    # Remove everything after cpp_cell_idx
    nb['cells'] = nb['cells'][:cpp_cell_idx+1]
    
    # 1. Compile
    nb['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["!g++ -O3 ocl_matmul.cpp -o ocl_matmul -lOpenCL"]
    })
    
    # 2. Small Input Verification (Rule 3)
    nb['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Rule 3: Small input verification\n",
            "!./ocl_matmul 64"
        ]
    })

    # 3. Large Input Test (Rule 5)
    nb['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Rule 5: Large input test before benchmark\n",
            "!./ocl_matmul 2048"
        ]
    })
    
    # 4. Full Benchmark
    nb['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Full Benchmark Loop (N=1024)\n",
            "!./ocl_matmul"
        ]
    })
    
    # 5. VRAM Stress Test (Task 4)
    nb['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Task 4: VRAM Stress Test (N=10240 ~ 3GB VRAM usage)\n",
            "# Run this cell and capture a screenshot of system/GPU utilization.\n",
            "!./ocl_matmul 10240"
        ]
    })

# Append a new Python plotting cell
nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": plot_code.splitlines(keepends=True)
})

# Append Final Answers Cell (Rule 2 + Task 2)
nb['cells'].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Final Tasks Answers & Discussion\n",
        "1. **Estimation vs Reality (Task 2)**: [Discuss how your rough estimation of improvements matched (or didn't) the actual results. Don't aim for a perfect match, show your learning process.]\n",
        "2. **Speedup Observed**: [Fill in after running benchmark]\n",
        "3. **Amdahl's Law validation**: [Comment on the plot based on the results]\n",
        "4. **VRAM Stress Test**: [Attach your screenshot here showing hardware utilization]"
    ]
})

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)
