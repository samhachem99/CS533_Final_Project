rm scheduler.o
rm cuda_code.o
rm cuda_mpi_exec

mpicxx -c -o scheduler.o ./scheduler.c 
nvcc -ccbin g++ -c -o cuda_code.o ./cuda_api.cu 
mpicxx -o cuda_mpi_exec scheduler.o cuda_code.o -L/usr/local/cuda-11.4/lib64 -lcudart