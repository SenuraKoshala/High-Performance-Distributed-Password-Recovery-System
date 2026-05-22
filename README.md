# High-Performance-Distributed-Password-Recovery-System

Serial_code_run
g++ -O2 -o serial_cracker serial_password_cracker.cpp -lssl -lcrypto

# Password recovery serial

sudo apt install libssl-dev
g++ -O2 -std=c++17 password_recovery_serial.cpp -lssl -lcrypto -o recover
./recover

# Password recovery OpenMP

sudo apt install libssl-dev # if not already done
g++ -O2 -std=c++17 -fopenmp password_recovery_openmp.cpp -lssl -lcrypto -o recover_omp
./recover_omp

To control thread count:

OMP_NUM_THREADS=8 ./recover_omp

# Password recovery MPI

sudo apt install libssl-dev mpich # once only
mpicxx -O2 -std=c++17 password_recovery_mpi.cpp -lssl -lcrypto -o recover_mpi

# 4 processes on one machine:

mpirun -np 4 ./recover_mpi

# More processes than cores (for testing):

mpirun -np 8 --oversubscribe ./recover_mpi

# Password Recovery Hybrid

mpicxx -O2 -std=c++17 -fopenmp \
    password_recovery_hybrid_mpi_openmp.cpp \
    -lssl -lcrypto \
    -o recover_hybrid

OMP_NUM_THREADS=4 mpirun -np 4 ./recover_hybrid
