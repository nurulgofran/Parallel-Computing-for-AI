import json

notebook_path = "Exercise 3/opencl_vadd.ipynb"

# New C++ code that accepts command line arguments for N and LocalWorkSize
cpp_code = r"""%%writefile ocl_vadd.cpp
/**
 * OpenCL Vector Addition
 * Exercise 3
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

// Default: ~256 MB (1024*1024*64 floats)
#define DEFAULT_N (1024*1024*64)

const char* kernelSource = 
"__kernel void vectorAdd(__global const float* A,\n"
"                       __global const float* B,\n"
"                       __global float* C,\n"
"                       const int numElements) {\n"
"    int i = get_global_id(0);\n"
"    if (i < numElements) {\n"
"        C[i] = A[i] + B[i];\n"
"    }\n"
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

void verify_result(float *h_A, float *h_B, float *h_C, int numElements) {
    printf("Verifying results (checking sample)...\n");
    int errors = 0;
    // Check strided sample to be fast
    int step = (numElements > 10000) ? numElements / 10000 : 1;
    for (int i = 0; i < numElements; i+=step) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-4) {
            if (errors < 5) printf("Mismatch at %d: %.4f vs %.4f\n", i, h_C[i], expected);
            errors++;
        }
    }
    if (errors == 0) printf("Verification PASSED\n");
    else printf("Verification FAILED (Errors: %d found in sample)\n", errors);
}

void run_benchmark(int numElements, int local_work_size, float gpu_portion, double *totalTimes, int T, int verify) {
    const size_t dataSize = numElements * sizeof(float);
    
    const int gpuElements = (int)(numElements * gpu_portion);
    const int cpuElements = numElements - gpuElements;
    const size_t gpuDataSize = gpuElements * sizeof(float);

    if(!verify) {
        printf("Config: N=%d, GPU=%.0f%%, LWS=%d\n", numElements, gpu_portion * 100, local_work_size);
        printf("Memory per vector: %.2f MB\n", (float)dataSize / (1024*1024));
    }

    float *h_A = (float*)malloc(dataSize);
    float *h_B = (float*)malloc(dataSize);
    float *h_C = (float*)malloc(dataSize);
    
    if (!h_A || !h_B || !h_C) { printf("Failed to allocate host memory\n"); return; }
    
    srand(2025);
    for (int i = 0; i < numElements; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }
    
    // OpenCL Setup
    cl_platform_id platform_id = NULL; cl_device_id device_id = NULL; 
    cl_uint n_plat, n_dev; cl_int ret;
    
    clGetPlatformIDs(1, &platform_id, &n_plat);
    if(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &n_dev) != CL_SUCCESS) {
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &n_dev);
    }
    
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    
    // Allocate only GPU portion on device
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, gpuDataSize > 0 ? gpuDataSize : 4, NULL, &ret);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, gpuDataSize > 0 ? gpuDataSize : 4, NULL, &ret);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, gpuDataSize > 0 ? gpuDataSize : 4, NULL, &ret);
    
    if (gpuElements > 0) {
        clEnqueueWriteBuffer(command_queue, d_A, CL_TRUE, 0, gpuDataSize, &h_A[cpuElements], 0, NULL, NULL);
        clEnqueueWriteBuffer(command_queue, d_B, CL_TRUE, 0, gpuDataSize, &h_B[cpuElements], 0, NULL, NULL);
    }
    
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &ret);
    clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "vectorAdd", &ret);
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_C);
    clSetKernelArg(kernel, 3, sizeof(int), (void*)&gpuElements);

    size_t localWorkSize = local_work_size;
    size_t globalWorkSize = gpuElements;
    // Pad
    if (localWorkSize > 0 && globalWorkSize % localWorkSize != 0) {
        globalWorkSize = ((globalWorkSize / localWorkSize) + 1) * localWorkSize;
    }
    
    int iterations = verify ? 1 : T;

    for (int iter = 0; iter < iterations; iter++) {
        double start = getCurrentTime();

        // 1. CPU
        for (int i = 0; i < cpuElements; i++) h_C[i] = h_A[i] + h_B[i];
        
        // 2. GPU
        if (gpuElements > 0) {
            ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &globalWorkSize, 
                                         localWorkSize > 0 ? &localWorkSize : NULL, 0, NULL, NULL);
            clFinish(command_queue);
            clEnqueueReadBuffer(command_queue, d_C, CL_TRUE, 0, gpuDataSize, &h_C[cpuElements], 0, NULL, NULL);
        }
        
        double end = getCurrentTime();
        if (totalTimes) totalTimes[iter] = end - start;
    }
    
    if (verify) {
        verify_result(h_A, h_B, h_C, numElements);
    }
    
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C);
    clReleaseCommandQueue(command_queue); clReleaseContext(context);
    free(h_A); free(h_B); free(h_C);
}

int main(int argc, char* argv[]) {
    // Usage: ./ocl_vadd [N] [LocalWorkSize] [verify_only]
    // If verify_only=1, run 50/50 check once
    // Else run full benchmark
    
    int N = DEFAULT_N;
    if (argc > 1) N = atoi(argv[1]);
    
    int LWS = 256; // Default optimization
    if (argc > 2) LWS = atoi(argv[2]);
    
    int mode_verify = 0; // 0=Benchmark, 1=VerifyOnly, 2=StressTest(Just run once large)
    if (argc > 3) mode_verify = atoi(argv[3]);
    
    if (mode_verify == 1) {
        printf("=== Verification Run (N=%d, LWS=%d) ===\n", N, LWS);
        run_benchmark(N, LWS, 0.5, NULL, 1, 1);
        return 0;
    }
    
    if (mode_verify == 2) {
        printf("=== Stress Test Run (N=%d) ===\n", N);
        // Run once, 100% GPU to max VRAM
        run_benchmark(N, LWS, 1.0, NULL, 1, 0);
        return 0;
    }

    // Benchmark Suite
    // Task 1: Exclude compute unit variation. Only vary workload portion.
    const int T = 5;
    double times[T];
    FILE *csv = fopen("vadd_benchmark.csv", "w");
    if(csv) {
        fprintf(csv, "Portion");
        for(int i=0; i<T; i++) fprintf(csv, ",Run_%d", i+1);
        fprintf(csv, "\n");
    }
    
    float portions[] = {0.0, 0.25, 0.5, 0.75, 1.0};
    printf("\n=== Starting Benchmark (N=%d, LWS=%d) ===\n", N, LWS);
    for(int p=0; p<5; p++) {
        run_benchmark(N, LWS, portions[p], times, T, 0);
        if(csv) {
            fprintf(csv, "%.2f", portions[p]);
            for(int i=0; i<T; i++) fprintf(csv, ",%.6f", times[i]);
            fprintf(csv, "\n");
        }
    }
    if(csv) fclose(csv);
    printf("Benchmark Complete. Results in vadd_benchmark.csv\n");
    return 0;
}
"""

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Update Cells
new_cells = []

# 1. Header & Install
# Preserve header if exists, or recreate
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["# OpenCL Vector Addition\n", "Exercise 3 - Problem 1"]
})

new_cells.append({
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
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!lscpu | head -n 10\n",
        "!nvidia-smi\n",
        "!clinfo | grep \"Platform Name\""
    ]
})

# 2. C++ Code
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": cpp_code.splitlines(keepends=True)
})

# 3. Compile
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["!g++ -O3 ocl_vadd.cpp -o ocl_vadd -lOpenCL"]
})

# 4. Verification Check
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Small Verification Run\n",
        "!./ocl_vadd 10000 256 1"
    ]
})

# 5. Task 2: Rough Estimation & Update
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Task 2: Design Improvement & Discussion\n",
        "Make a rough estimation of actions to improve the design (e.g. adjust Global/Local Work Size).\n",
        "The code assumes LWS=256 by default. You can change the second argument below to test different sizes (e.g. 32, 64, 128, 512)."
    ]
})
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Run with default N, LWS=256, Benchmark Mode (0)\n",
        "!./ocl_vadd 67108864 256 0"
    ]
})
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Experiment: Try LWS=64\n",
        "!./ocl_vadd 67108864 64 0"
    ]
})

# 6. Task 3: VRAM Stress Test
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Task 3: VRAM Stress Test\n",
        "# N = 400,000,000 floats * 4 bytes * 3 arrays = ~4.8 GB\n",
        "# N = 800,000,000 floats = ~9.6 GB\n",
        "# Run and capture screenshot.\n",
        "!./ocl_vadd 800000000 256 2"
    ]
})

# 7. Plotting & Final Answers
plot_code = """import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('vadd_benchmark.csv')
    time_cols = [c for c in df.columns if 'Run_' in c]
    df['AvgTime'] = df[time_cols].mean(axis=1)
    
    cpu_time = df.loc[df['Portion'] == 0.0, 'AvgTime'].values[0]
    df['Speedup'] = cpu_time / df['AvgTime']
    
    print(df[['Portion', 'AvgTime', 'Speedup']])
    
    plt.figure(figsize=(10,6))
    plt.plot(df['Portion'], df['Speedup'], marker='o')
    plt.title("Vector Addition Speedup (Amdahl's Law)")
    plt.xlabel("GPU Portion")
    plt.ylabel("Speedup")
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Plotting skipped/failed: {e}")
"""

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": plot_code.splitlines(keepends=True)
})

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Final Observations\n",
        "1. **Task 2 Discussion**: [Discuss how changing Local Work Size affected performance relative to your estimation.]\n",
        "2. **Task 3 Screenshot**: [Attach VRAM stress test screenshot here]"
    ]
})

nb['cells'] = new_cells

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Updated opencl_vadd.ipynb successfully.")
