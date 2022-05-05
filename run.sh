rm scheduler.o
rm cuda_code.o
rm cuda_mpi_exec

rm cuda_api_exec.*
rm report*
rm nsys_output.txt
rm nvprof_api_output.txt
rm nvprof_gpu_output.txt
rm nvprof_output.txt

mpicxx -c -o scheduler.o ./scheduler.c 
nvcc -ccbin g++ -c -o cuda_code.o ./cuda_api.cu 
mpicxx -o cuda_mpi_exec scheduler.o cuda_code.o -L/usr/local/cuda-11.4/lib64 -lcudart
mpiexec -np 4 nvprof -f -o cuda_api_exec.%q{OMPI_COMM_WORLD_RANK}.nvprof ./cuda_mpi_exec
mpiexec -np 4 nsys profile --stats=true --force-overwrite true ./cuda_mpi_exec > nsys_output.txt
nvprof --print-api-trace --import-profile ./cuda_api_exec.0.nvprof --log-file nvprof_api_output.txt
nvprof --print-gpu-trace --import-profile ./cuda_api_exec.0.nvprof --log-file nvprof_gpu_output.txt
nvprof --import-profile ./cuda_api_exec.0.nvprof --log-file nvprof_output.txt