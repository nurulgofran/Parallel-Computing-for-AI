import json

notebook_path = "Exercise 3/opencl_matmul.ipynb"

# Define the new content
cpp_code = r"""%%writefile ocl_matmul.cpp
/**
 * OpenCL Matrix Multiplication
 * Problem 2
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

// Matrix Size (Square Matrix N x N)
// Start small for verification, increase for stress test
#define N 1024 

// OpenCL Kernel: Naive Matrix Multiplication
const char* kernelSource = 
"__kernel void matMul(__global const float* A, \n"
"                     __global const float* B, \n"
"                     __global float* C, \n"
"                     const int width) { \n"
"    int col = get_global_id(0); \n" // x
"    int row = get_global_id(1); \n" // y
"    \n"
"    if (col < width && row < width) { \n"
"        float sum = 0.0f; \n"
"        for (int k = 0; k < width; k++) { \n"
"            sum += A[row * width + k] * B[k * width + col]; \n"
"        } \n"
"        C[row * width + col] = sum; \n"
"    } \n"
"}\n";

// Helper Functions (Error checking, Time)
const char* getErrorString(cl_int error) {
    switch(error) {
        case CL_SUCCESS: return "Success";
        case CL_DEVICE_NOT_FOUND: return "Device not found";
        case CL_DEVICE_NOT_AVAILABLE: return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE: return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES: return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY: return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "Profiling info not available";
        case CL_MEM_COPY_OVERLAP: return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH: return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE: return "Build program failure";
        case CL_MAP_FAILURE: return "Map failure";
        case CL_INVALID_VALUE: return "Invalid value";
        case CL_INVALID_DEVICE_TYPE: return "Invalid device type";
        case CL_INVALID_PLATFORM: return "Invalid platform";
        case CL_INVALID_DEVICE: return "Invalid device";
        case CL_INVALID_CONTEXT: return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES: return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE: return "Invalid command queue";
        case CL_INVALID_HOST_PTR: return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT: return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE: return "Invalid image size";
        case CL_INVALID_SAMPLER: return "Invalid sampler";
        case CL_INVALID_BINARY: return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS: return "Invalid build options";
        case CL_INVALID_PROGRAM: return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME: return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION: return "Invalid kernel definition";
        case CL_INVALID_KERNEL: return "Invalid kernel";
        case CL_INVALID_ARG_INDEX: return "Invalid argument index";
        case CL_INVALID_ARG_VALUE: return "Invalid argument value";
        case CL_INVALID_ARG_SIZE: return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS: return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION: return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE: return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE: return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET: return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST: return "Invalid event wait list";
        case CL_INVALID_EVENT: return "Invalid event";
        case CL_INVALID_OPERATION: return "Invalid operation";
        case CL_INVALID_GL_OBJECT: return "Invalid GL object";
        case CL_INVALID_BUFFER_SIZE: return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL: return "Invalid mip level";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "Invalid global work size";
        case -1001: return "Platform not found (CL_PLATFORM_NOT_FOUND_KHR)";
        default: return "Unknown error";
    }
}

void checkError(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        printf("Error during %s: %d (%s)\n", operation, error, getErrorString(error));
        exit(1);
    }
}

double getCurrentTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// Sequential implementation for verification/benchmarking
void matrixMulCPU(float* A, float* B, float* C, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += A[i * width + k] * B[k * width + j];
            }
            C[i * width + j] = sum;
        }
    }
}

// Verification function
void verify_result(float* h_C_GPU, float* h_C_CPU, int width) {
    printf("Verifying results...\n");
    int correct = 1;
    int errors = 0;
    for (int i = 0; i < width * width; i++) {
        if (fabs(h_C_GPU[i] - h_C_CPU[i]) > 1e-3) {
            if (errors < 10) printf("Mismatch at index %d: GPU %.6f, CPU %.6f\n", i, h_C_GPU[i], h_C_CPU[i]);
            correct = 0;
            errors++;
        }
    }
    if (correct) printf("Verification PASSED!\n");
    else printf("Verification FAILED with %d mismatches!\n", errors);
}

void run_benchmark() {
    int width = N;
    size_t size = width * width * sizeof(float);
    
    printf("\n=== Matrix Multiplication (N=%d) ===\n", width);
    printf("Matrix Size: %d x %d\n", width, width);
    printf("Memory usage per matrix: %.2f MB\n", (float)size / (1024*1024));
    
    // Allocate Host Memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_CPU = (float*)malloc(size);
    float *h_C_GPU = (float*)malloc(size);
    
    // Initialize
    srand(2025);
    for(int i=0; i<width*width; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }
    
    // 1. CPU Run (only once for verification baseline if N is not too huge)
    // If N is very large, this will take forever.
    if (width <= 2048) {
        printf("Running Sequential CPU...\n");
        double start = getCurrentTime();
        matrixMulCPU(h_A, h_B, h_C_CPU, width);
        double end = getCurrentTime();
        printf("CPU Time: %.6f s\n", end - start);
    } else {
        printf("Skipping CPU run for large N\n");
    }
    
    // 2. OpenCL Setup
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int ret;
    cl_uint num_platforms, num_devices;
    
    clGetPlatformIDs(1, &platform, &num_platforms);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    // Fallback to CPU if no GPU
    if (num_devices == 0) clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
    
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &ret);
    queue = clCreateCommandQueue(context, device, 0, &ret);
    
    // Buffers using copy/host ptr for simplicity or create then write
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, h_A, &ret);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, h_B, &ret);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, &ret);
    
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &ret);
    ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *) malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("%s\n", log);
        return;
    }
    
    kernel = clCreateKernel(program, "matMul", &ret);
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &width);
    
    // Work Sizes
    // Global: 2D range [width, width]
    // Local: 2D range [TX, TY] e.g., 16x16 = 256 threads per block
    size_t globalSize[2] = { (size_t)width, (size_t)width };
    size_t localSize[2] = { 16, 16 }; 
    
    // Improvement area: Pad globalSize if width is not multiple of 16
    
    printf("Running OpenCL GPU (Naive)...\n");
    double gpu_start = getCurrentTime();
    
    ret = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    checkError(ret, "EnqueueNDRangeKernel");
    
    clFinish(queue);
    
    double gpu_end = getCurrentTime();
    printf("GPU Time: %.6f s\n", gpu_end - gpu_start);
    
    // Read Comparison
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, size, h_C_GPU, 0, NULL, NULL);
    
    if (width <= 2048) {
        verify_result(h_C_GPU, h_C_CPU, width);
    }
    
    // Speedup Calculation
    // Only if CPU was run
    if (width <= 2048) {
        // approximate comparison
        // double cpu_time = ... need to pass it or store it.
        // For this simple script, printed output is enough.
    }
    
    // Cleanup
    clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);
    free(h_A); free(h_B); free(h_C_CPU); free(h_C_GPU);
}

int main() {
    printOpenCLInfo();
    run_benchmark();
    return 0;
}
"""

run_code = "!g++ -O3 ocl_matmul.cpp -o ocl_matmul -lOpenCL && ./ocl_matmul"

# Load notebook
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Keep metadata, replace cells.
# We will create a fresh list of cells.
new_cells = []

# Cell 1: Header
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# OpenCL Matrix Multiplication\n",
        "Exercise 3 - Problem 2"
    ]
})

# Cell 1.5: Install Dependencies (Fix for Colab/Linux)
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

# Cell 2: System Info
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

# Cell 3: C++ Code
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": cpp_code.splitlines(keepends=True)
})

# Cell 4: Run
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [run_code]
})

nb['cells'] = new_cells

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Created ocl_matmul.ipynb successfully.")
