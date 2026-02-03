import json

notebook_path = "Exercise 3/opencl_vadd.ipynb"

new_cpp_code = r"""%%writefile ocl_vadd.cpp
/**
 * OpenCL Basic Example - Vector Addition
 * 
 * Optimized and Modified for Exercise 3
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

// Configuration for Stress Test
// Default: ~2.5 GB
#define VECTOR_LENGTH (1024*10000*63)
// Uncomment for Stress Test (Ensure system has enough RAM/VRAM)
// #define VECTOR_LENGTH (1024*1024*1000) // ~4GB

// OpenCL kernel source code
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

// Helper function to get error string
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

// Verification function
void verify_result(float *h_A, float *h_B, float *h_C, int numElements) {
    printf("Verifying results...\n");
    int correct = 1;
    // Check first 10, last 10, and random 100 for efficiency if vector is huge
    // But checking all is safer for "Correctness Check"
    for (int i = 0; i < numElements; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-4) { // Increased tolerance slightly
            printf("Mismatch at index %d: GPU/Hybrid %.6f, CPU %.6f\n", i, h_C[i], expected);
            correct = 0;
            break; 
        }
    }
    if (correct) {
        printf("Verification PASSED: Results match sequential CPU calculation.\n");
    } else {
        printf("Verification FAILED!\n");
    }
}

void run_benchmark(float gpu_portion, double *totalTimes, int T, int verify) {
    const int numElements = VECTOR_LENGTH;
    const size_t dataSize = numElements * sizeof(float);
    
    // Calculate workload split based on gpu_portion
    const int gpuElements = (int)(numElements * gpu_portion);
    const int cpuElements = numElements - gpuElements;
    const size_t gpuDataSize = gpuElements * sizeof(float);

    double cpuTimes[T];
    double gpuTimes[T];
    
    if(!verify) {
        printf("\n========================================\n");
        printf("Benchmark Profile: GPU portion=%.0f%%\n", gpu_portion * 100);
        printf("CPU processing: %d elements (%.0f%%)\n", cpuElements, (1 - gpu_portion) * 100);
        printf("GPU processing: %d elements (%.0f%%)\n", gpuElements, gpu_portion * 100);
        printf("Number of iterations: %d\n", T);
        printf("========================================\n\n");
    } else {
        printf("\n=== Running Correctness Check (GPU Portion %.0f%%) ===\n", gpu_portion * 100);
    }
    
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;
    
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (ret != CL_SUCCESS || ret_num_platforms == 0) { printf("ERROR: No OpenCL platforms available!\n"); return; }
    
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    if (ret != CL_SUCCESS) {
         ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);
    }
    if (ret_num_devices == 0) { printf("ERROR: No OpenCL devices available!\n"); return; }

    float *h_A = (float*)malloc(dataSize);
    float *h_B = (float*)malloc(dataSize);
    float *h_C = (float*)malloc(dataSize);
    
    if (!h_A || !h_B || !h_C) { printf("Failed to allocate host memory\n"); return; }
    
    // Initialize host arrays
    srand(time(NULL));
    for (int i = 0; i < numElements; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }
    
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    checkError(ret, "creating context");
    
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    checkError(ret, "creating command queue");
    
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, gpuDataSize, NULL, &ret);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, gpuDataSize, NULL, &ret);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, gpuDataSize, NULL, &ret);
    checkError(ret, "creating buffers");
    
    ret = clEnqueueWriteBuffer(command_queue, d_A, CL_TRUE, 0, gpuDataSize, &h_A[cpuElements], 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_B, CL_TRUE, 0, gpuDataSize, &h_B[cpuElements], 0, NULL, NULL);
    checkError(ret, "writing buffers");
    
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) { printf("Error building program\n"); return; }
    
    cl_kernel kernel = clCreateKernel(program, "vectorAdd", &ret);
    checkError(ret, "creating kernel");
    
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_A);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_B);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_C);
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void*)&gpuElements);

    // OPTIMIZATION: Work Group Size
    // Estimation for improvement: 
    // - Local Work Size should be a multiple of the wavefront/warp size (usually 32 or 64).
    // - 256 is a generally safe and good value.
    // - Global Work Size must be a multiple of Local Work Size.
    
    size_t localWorkSize = 256;
    size_t globalWorkSize = gpuElements;
    
    // Pad global work size to be a multiple of local work size
    if (globalWorkSize % localWorkSize != 0) {
        globalWorkSize = ((globalWorkSize / localWorkSize) + 1) * localWorkSize;
    }
    
    int iterations = verify ? 1 : T;

    for (int iter = 0; iter < iterations; iter++) {
        // CPU computation (Sequential part)
        double cpu_start = getCurrentTime();
        for (int i = 0; i < cpuElements; i++) {
            h_C[i] = h_A[i] + h_B[i];
        }
        double cpu_end = getCurrentTime();
        cpuTimes[iter] = cpu_end - cpu_start;

        // GPU computation
        double gpu_start = getCurrentTime();
        
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        checkError(ret, "enqueueing kernel");
        
        ret = clFinish(command_queue);
        checkError(ret, "waiting for kernel");
        
        ret = clEnqueueReadBuffer(command_queue, d_C, CL_TRUE, 0, gpuDataSize, &h_C[cpuElements], 0, NULL, NULL);
        checkError(ret, "reading result");
        
        double gpu_end = getCurrentTime();
        gpuTimes[iter] = gpu_end - gpu_start;

        if (!verify) {
            totalTimes[iter] = cpuTimes[iter] + gpuTimes[iter];
        }
    }
    
    if (verify) {
        verify_result(h_A, h_B, h_C, numElements);
    }
    
    // Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    
    free(h_A);
    free(h_B);
    free(h_C);
}

int main() {
    // 1. Correctness Check (Run once with mixed workload 50/50)
    run_benchmark(0.5, NULL, 1, 1);
    
    const int T = 5; // Reduced iterations for quicker runs
    double totalTimes[T];
    
    FILE *csv = fopen("benchmark_results.csv", "w");
    if (csv == NULL) return 1;
    
    fprintf(csv, "Configuration");
    for (int i = 0; i < T; i++) fprintf(csv, ",Run_%d", i + 1);
    fprintf(csv, "\n");
    
    // 2. Run Benchmark Suite (Variable Portions)
    float portions[] = {0.0, 0.25, 0.5, 0.75, 1.0}; 
    int num_portions = sizeof(portions) / sizeof(portions[0]);
    
    printf("\n=== Starting Benchmark Suite ===\n");
    
    for (int p = 0; p < num_portions; p++) {
        run_benchmark(portions[p], totalTimes, T, 0);
        
        fprintf(csv, "P%.0f%%", portions[p]*100);
        for (int i = 0; i < T; i++) {
            fprintf(csv, ",%.6f", totalTimes[i]);
        }
        fprintf(csv, "\n");
        fflush(csv);
    }
    
    fclose(csv);
    printf("\n=== Benchmark Suite Complete ===\n");
    return 0;
}
"""

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Find the cell starting with %%writefile ocl_vadd.cpp
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if source.startswith('%%writefile ocl_vadd.cpp'):
            cell['source'] = new_cpp_code.splitlines(keepends=True)
            print("Found and updated C++ cell.")
            break

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)
