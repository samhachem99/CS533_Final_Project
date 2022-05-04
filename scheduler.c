#include <mpi.h>
#include <stdio.h>


int main(int args, char** argv) {

    printf("the number of arguments are %d\n", args);
    for(int i = 0; i < args; i++) {
        printf("arg %d is %s\n", i, argv[i]);
    }

    int ierr, num_procs, my_id;

    ierr = MPI_Init(&argc, &argv);
    printf("MPI_Init err code: %d\n", ierr);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

    if(my_id == 0) {
        printf("this is the scheduling process");
    } else {
        printf("this is process %d, and we are handling task %d\n", my_id, my_id);
    }

    ierr = MPI_Finalize();
    printf("MPI_Finalize err code: %d\n", ierr);

    return 0;
}