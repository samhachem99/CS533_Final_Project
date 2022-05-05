#include <mpi.h>
#include <stdio.h>

#define MAIN_PROCESS 0
#define BLOCK_SIZE 512
#define TOTAL_MEM_ALLOC_SIZE_MB 1024
#define HtoD 1
#define Kernel 2
#define DtoH 3
#define NUM_KERNELS_PER_PROC 10
#define END 4

void sch_init();
void execute_task(int id, int size);
void sync();

typedef struct task {
    int id;
    int state;
    int size;
} task_t;

int check_tasks_states(task_t tasks[], int num);

int main(int argc, char** argv) {
    int ierr, num_procs, my_id;

    ierr = MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_Datatype mpi_task;
    MPI_Type_contiguous(3, MPI_INT, &mpi_task);
    MPI_Type_commit(&mpi_task);
    
    if(my_id == MAIN_PROCESS) {
        printf("MAIN started\n");
        task_t tasks[num_procs-1];
        for(int i = 0; i < num_procs-1; i++) {
            tasks[i].id = i+1;
            tasks[i].state = HtoD;
            tasks[i].size = (i+1) << 20;
        }
        sch_init();
        int num_messages = 0;
        while(1) {
            task_t tmp;
            MPI_Status status;
            int count;
            // data_to_recieve, count, datatype, source_id, tag, comm, status(which can be ignored with MPI_STATUS_IGNORE)
            ierr = MPI_Recv(&tmp, 1, mpi_task, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status); // ignore the status struct with MPI_STATUS_IGNORE
            MPI_Get_count(&status, MPI_BYTE, &count);

            task_t* task = &tasks[tmp.id-1];
            execute_task(task->id, task->size);

            // if (task->state == HtoD) {
            //     task->size = tmp.size;
            //     printf("MAIN task %d is in state 1 with size %d\n", task->id, task->size);
            // } else if (task->state == Kernel) {
            //     printf("MAIN task %d is in state 2 with size %d\n", task->id, task->size);
            // } else if (task->state == DtoH) {
            //     printf("MAIN task %d is in state 3 with size %d\n", task->id, task->size);
            // }
            // if(check_tasks_states(tasks, num_procs-1)) break;

            // task->state++;
            // tmp.id = task->id;
            // tmp.state = task->state;
            // tmp.size = task->size;
            // MPI_Send(&tmp, 1, mpi_task, task->id, 0, MPI_COMM_WORLD);
            num_messages++;
            if(num_messages == (num_procs-1)*NUM_KERNELS_PER_PROC) {
                sync();
                break;
            }
        }
        printf("MAIN Finished\n");
    } else {
        printf("PROCESS %d started\n", my_id);
        task_t task;
        task.id = my_id;
        task.state = HtoD;
        task.size = (my_id+1) << 20;
    
        for(int i = 0; i < NUM_KERNELS_PER_PROC; i++)
            ierr = MPI_Send(&task, 1, mpi_task, MAIN_PROCESS, 0, MPI_COMM_WORLD);
        printf("PROCESS %d Finished\n", my_id);
        // while(1) {
        //     // printf("process %d is sending data to process 0\n", my_id);
        //     // data_to_send, count, datatype, dest_id, tag, comm
           
            
        //     printf("task %d progressed to state: %d\n", task.id, task.state);
        //     if(task.state == END) break;
        //     task_t tmp;
        //     MPI_Recv(&tmp, 1, mpi_task, MAIN_PROCESS, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // ignore the status struct with MPI_STATUS_IGNORE
        //     // printf("PROCESS %d: tmp recieved has id: %d, state: %d, size: %d\n", my_id, tmp.id, tmp.state, tmp.size);
        //     task.state = tmp.state;
        // }
        // printf("PROCESS %d Finished with state %d\n", my_id, task.state);
    }

    ierr = MPI_Finalize();
    // printf("MPI_Finalize err code: %d\n", ierr);

    return 0;
}

int check_tasks_states(task_t tasks[], int num) {
    for(int i = 0; i < num; i++) {
        if(tasks[i].state < END) return 0;
    }
    return 1;
}