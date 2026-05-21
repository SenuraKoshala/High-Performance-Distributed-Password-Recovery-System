# High-Performance-Distributed-Password-Recovery-System

Serial_code_run
g++ -O2 -o serial_cracker serial_password_cracker.cpp -lssl -lcrypto

Password recovery serial

sudo apt install libssl-dev
g++ -O2 -std=c++17 password_recovery_serial.cpp -lssl -lcrypto -o recover
./recover

OpenMP

sudo apt install libssl-dev # if not already done
g++ -O2 -std=c++17 -fopenmp password_recovery_openmp.cpp -lssl -lcrypto -o recover_omp
./recover_omp

# To control thread count:

OMP_NUM_THREADS=8 ./recover_omp
