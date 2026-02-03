#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int cmp(const void *a, const void *b) { return *(char*)a - *(char*)b; }

void merge(char *a, int na, char *b, int nb, char *out) {
    int i=0, j=0, k=0;
    while (i < na && j < nb) out[k++] = (a[i] <= b[j]) ? a[i++] : b[j++];
    while (i < na) out[k++] = a[i++];
    while (j < nb) out[k++] = b[j++];
}

int main(int argc, char **argv) {
    int rank, nprocs;
    char *data = NULL;
    long fsize = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank == 0) {
        FILE *fp = fopen("database.txt", "rb");
        if (!fp) { MPI_Abort(MPI_COMM_WORLD, 1); }
        fseek(fp, 0, SEEK_END);
        fsize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        data = malloc(fsize);
        fread(data, 1, fsize, fp);
        fclose(fp);
    }

    MPI_Bcast(&fsize, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    int *scnt = malloc(nprocs * sizeof(int));
    int *disp = malloc(nprocs * sizeof(int));
    int base = fsize / nprocs, rem = fsize % nprocs, off = 0;
    for (int i = 0; i < nprocs; i++) {
        scnt[i] = base + (i < rem ? 1 : 0);
        disp[i] = off;
        off += scnt[i];
    }

    int mylen = scnt[rank];
    char *local = malloc(mylen);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    MPI_Scatterv(data, scnt, disp, MPI_CHAR, local, mylen, MPI_CHAR, 0, MPI_COMM_WORLD);
    qsort(local, mylen, 1, cmp);

    char *gathered = NULL;
    if (rank == 0) gathered = malloc(fsize);
    MPI_Gatherv(local, mylen, MPI_CHAR, gathered, scnt, disp, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        char *sorted = malloc(fsize);
        char *tmp = malloc(fsize);
        memcpy(sorted, gathered, scnt[0]);
        int len = scnt[0];
        for (int i = 1; i < nprocs; i++) {
            merge(sorted, len, gathered + disp[i], scnt[i], tmp);
            len += scnt[i];
            memcpy(sorted, tmp, len);
        }
        double elapsed = MPI_Wtime() - t0;
        printf("%d,%.4f\n", nprocs, elapsed);
        free(sorted); free(tmp); free(gathered); free(data);
    }

    free(local); free(scnt); free(disp);
    MPI_Finalize();
    return 0;
}
