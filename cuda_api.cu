#include <cmath>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

typedef struct Node {
	char *ptr;
	char free;
	int64_t size;
	struct Node *next;
} Node;

#define TOTAL_MEM_ALLOC_SIZE_MB ((int64_t)4096)
#define BLOCK_SIZE 512
#define NUM_PROCS 4

static Node *head = NULL;
static int num_nodes = 0;
static void* mem_ptr;
static cudaStream_t streams[NUM_PROCS-1];

static void free_all_recurse(Node *n)
{
	if (NULL == n)
		return;

	free_all_recurse(n->next);
	free(n); //It's not our responsibility to free n->ptr
}

static int free_all_pre_alloc_mem()
{
	free_all_recurse(head);
	head = NULL;
	num_nodes = 0;
	return 0; //success
}

static int init_pre_alloc_mem(char *ptr, int64_t num_bytes)
{
	if (num_bytes <= 0)
		return 1; //fail

	if (head != NULL)
		free_all_pre_alloc_mem();

	head = (Node*)malloc(sizeof(Node));
	head->ptr = ptr;
	head->free = 1;
	head->size = num_bytes;
	head->next = NULL;
	num_nodes = 1;

	return 0; //success
}

static int get_pre_alloc_mem(int64_t num_bytes, char **ptr)
{
	*ptr = NULL;

	if (num_bytes <= 0)
		return 1; //fail

	if (NULL == head)
		return 1; //fail

	//Walk the LL to find smallest big-enough free Node
	Node *best = NULL;
	Node *curr = head;
	while (curr != NULL)
	{
		if (1 == curr->free && curr->size >= num_bytes &&
			(NULL == best || curr->size < best->size))
			best = curr;

		curr = curr->next;
	}

	if (NULL == best)
		return 1; //fail

	if (best->size == num_bytes)
		best->free = 0;
	else
	{
		Node *new_node = (Node*)malloc(sizeof(Node));
		new_node->ptr = best->ptr + num_bytes;
		new_node->free = 1;
		new_node->size = best->size - num_bytes;
		new_node->next = best->next;
		num_nodes++;
		
		best->free = 0;
		best->size = num_bytes;
		best->next = new_node;

	}

	*ptr = best->ptr;
	return 0; //success
}

/*
static int free_pre_alloc_mem(char *ptr) //must pass in the ptr to the beginning of the region
{
	if (NULL == head)
		return 1; //fail

	//Walk the LL to find ptr
	Node *curr = head;
	Node *prev = NULL;
	while (curr != NULL)
	{
		if (curr->ptr == ptr)
			break;

		prev = curr;
		curr = curr->next;
	}

	if (curr == NULL) //not found
		return 1; //fail

	//Case 1: previous and next are both free (combine all 3)
	if (prev && curr->next && prev->free && curr->next->free)
	{
		prev->size += (curr->size + curr->next->size);
		prev->next = curr->next->next;
		free(curr);
		free(curr->next);
		num_nodes -= 2;
		return 0; //success
	}

	//Case 2: combine prev and curr
	if (prev && prev->free)
	{
		prev->size += curr->size;
		prev->next = curr->next;
		free(curr);
		num_nodes--;
		return 0; //success
	}

	//Case 3: combine curr and next
	if (curr->next && curr->next->free)
	{
		curr->next->ptr = curr->ptr;
		curr->next->size += curr->size;
		if (prev)
			prev->next = curr->next;
		else
			head = curr->next;
		free(curr);
		num_nodes--;
		return 0; //success
	}

	//Case 4: Just mark curr free
	curr->free = 1;
	return 0; //success
}
*/

__global__ void vector_add_kernel(const float *x, const float *y, float *z, int size) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(i < size)
        z[i] = x[i] + y[i];
}

void sch_init() {
    int64_t mem_size = TOTAL_MEM_ALLOC_SIZE_MB << 20;
    cudaError_t e1 = cudaMallocHost(&mem_ptr, mem_size);

    init_pre_alloc_mem((char*)mem_ptr, mem_size);

    // initalize NUM_PROCS - 1 streams
    for(int i = 0; i < NUM_PROCS-1; i++)
        cudaError_t err = cudaStreamCreate(&streams[i]);
}

void execute_task(int id, int size) {
    cudaStream_t stream = streams[id-1];
    int arr_bsize = sizeof(float)*size;
    float* h_a, *h_b, *h_c;
    int err = get_pre_alloc_mem(arr_bsize, (char **)&h_a);
    err = get_pre_alloc_mem(arr_bsize, (char **)&h_b);
    err = get_pre_alloc_mem(arr_bsize, (char **)&h_c);
    for (int i = 0; i < size; i++) {
        h_a[i] = 20.0;
        h_b[i] = 40.0;
    }
    // memset(h_c, 0, arr_bsize);

    float *d_a, *d_b, *d_c;
    cudaMallocAsync((float **)&d_a, arr_bsize, stream);
    cudaMallocAsync((float **)&d_b, arr_bsize, stream);
    cudaMallocAsync((float **)&d_c, arr_bsize, stream);

    cudaMemcpyAsync(d_a, h_a, arr_bsize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, arr_bsize, cudaMemcpyHostToDevice, stream);
    dim3 gridDim(1+(1.0*size/BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE);
    vector_add_kernel<<<gridDim, blockDim, 0, stream>>>(d_a, d_b, d_c, size);
    cudaMemcpyAsync(h_c, d_c, arr_bsize, cudaMemcpyDeviceToHost, stream);
}

void sync() {
    cudaDeviceSynchronize();
    free_all_pre_alloc_mem();
    cudaFreeHost(mem_ptr);
}