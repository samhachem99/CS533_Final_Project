//Write some simple baseline for our 

#include <cmath>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_KERNELS_PER_PROC 10
#define BLOCK_SIZE 512
#define NUM_PROCS 4

/*******************************
 * CUDA KERNEL
 * ****************************/

__global__ void vector_add_kernel(const float *x, const float *y, float *z, int size) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(i < size)
        z[i] = x[i] + y[i];
}

/*******************************
 * MAIN CODE FOR BASELINE
 * ****************************/

int main()
{
	float* h_a[NUM_PROCS][NUM_KERNELS_PER_PROC];
	float* h_b[NUM_PROCS][NUM_KERNELS_PER_PROC];
	float* h_c[NUM_PROCS][NUM_KERNELS_PER_PROC];
	float* d_a[NUM_PROCS][NUM_KERNELS_PER_PROC];
	float* d_b[NUM_PROCS][NUM_KERNELS_PER_PROC];
	float* d_c[NUM_PROCS][NUM_KERNELS_PER_PROC];

	int size = 0;
	for (int my_id = 1; my_id < NUM_PROCS; my_id++)
	{
		for (int task = 0; task < NUM_KERNELS_PER_PROC; task++)
		{
			size = (my_id+1) << 20;
			dim3 gridDim(1+(1.0*size/BLOCK_SIZE));
    		dim3 blockDim(BLOCK_SIZE);
			
			//Allocate host
			h_a[my_id][task] = (float*) malloc(size * sizeof(float));
			h_b[my_id][task] = (float*) malloc(size * sizeof(float));
			h_c[my_id][task] = (float*) malloc(size * sizeof(float));

			//Init Host
			for (int i = 0; i < size; i++)
			{
		        h_a[my_id][task][i] = 20.0;
		        h_b[my_id][task][i] = 40.0;
		    }

		    //Allocate Device
		    cudaMallocAsync((float **)&d_a[my_id][task], size * sizeof(float), stream);
		    cudaMallocAsync((float **)&d_b[my_id][task], size * sizeof(float), stream);
		    cudaMallocAsync((float **)&d_c[my_id][task], size * sizeof(float), stream);

			//Create Stream
			cudaStream_t stream;
			cudaStreamCreate(&stream);

			//Do Work
		    cudaMemcpyAsync(d_a[my_id][task], h_a[my_id][task], size * sizeof(float), cudaMemcpyHostToDevice, stream);
		    cudaMemcpyAsync(d_b[my_id][task], h_b[my_id][task], size * sizeof(float), cudaMemcpyHostToDevice, stream);
		    vector_add_kernel<<<gridDim, blockDim, 0, stream>>>(d_a[my_id][task], d_b[my_id][task], d_c[my_id][task], size);
		    cudaMemcpyAsync(h_c[my_id][task], d_c[my_id][task], size * sizeof(float), cudaMemcpyDeviceToHost, stream);
		}
	}

	cudaDeviceSynchronize();

	for (int my_id = 1; my_id < NUM_PROCS; my_id++)
	{
		for (int task = 0; task < NUM_KERNELS_PER_PROC; task++)
		{
			free(h_a[my_id][task]);
			free(h_b[my_id][task]);
			free(h_c[my_id][task]);
			cudaFree(d_a[my_id][task]);
			cudaFree(d_b[my_id][task]);
			cudaFree(d_c[my_id][task]);
		}
	}

	return 0;
}
