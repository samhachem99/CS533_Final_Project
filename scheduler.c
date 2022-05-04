#include <mpi.h>
#include <stdio.h>

#define TOTAL_MEM_ALLOC_SIZE_MB 16

typedef struct task {
    
} task_t;


int main(int argc, char** argv) {
    int ierr, num_procs, my_id;

    ierr = MPI_Init(&argc, &argv);
    // printf("MPI_Init err code: %d\n", ierr);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if(my_id == 0) {
        // things that should run only once.
        int mem_size = TOTAL_MEM_ALLOC_SIZE_MB << 20;
        void *mem_ptr;
        cudaError_t e1 = cudaMallocHost(&mem_ptr, mem_size);

        task_t my_tasks[num_procs-1];
        
        printf("this is process %d, the scheduling process\n", my_id);
    } else {
        printf("this is process %d, and we are handling task %d\n", my_id, my_id);
    }

    ierr = MPI_Finalize();
    // printf("MPI_Finalize err code: %d\n", ierr);

    return 0;
}