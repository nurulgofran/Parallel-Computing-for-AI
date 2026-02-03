#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ITERS 1000

int main(int argc, char **argv) {
    int rank, nprocs;
    MPI_Status st;
    int sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144};
    int nsizes = sizeof(sizes)/sizeof(sizes[0]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (nprocs != 2) {
        if (rank == 0) printf("Need exactly 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) printf("MsgSize(bytes),Latency(us),Bandwidth(MB/s)\n");

    for (int i = 0; i < nsizes; i++) {
        int sz = sizes[i];
        char *buf = malloc(sz);
        memset(buf, 'x', sz);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        for (int j = 0; j < ITERS; j++) {
            if (rank == 0) {
                MPI_Send(buf, sz, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buf, sz, MPI_CHAR, 1, 0, MPI_COMM_WORLD, &st);
            } else {
                MPI_Recv(buf, sz, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &st);
                MPI_Send(buf, sz, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }

        double elapsed = MPI_Wtime() - t0;
        if (rank == 0) {
            double lat = (elapsed / (2 * ITERS)) * 1e6;
            double bw = (2.0 * sz * ITERS) / (elapsed * 1e6);
            printf("%d,%.3f,%.3f\n", sz, lat, bw);
        }
        free(buf);
    }

    MPI_Finalize();
    return 0;
}
