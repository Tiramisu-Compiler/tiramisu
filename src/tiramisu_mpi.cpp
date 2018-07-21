#include <cstdlib>
#include <cstdio>
#include <cassert>
#include "tiramisu/mpi_comm.h"

#ifdef WITH_MPI

int tiramisu_MPI_init() {
    int provided = -1;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
    assert(provided == MPI_THREAD_FUNNELED && "Did not get the appropriate MPI thread requirement.");
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

void tiramisu_MPI_cleanup() {
    MPI_Finalize();
}

void tiramisu_MPI_global_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}

extern "C" {

#define make_Send(suffix, c_datatype, mpi_datatype) \
void tiramisu_MPI_Send_##suffix(int count, int dest, int tag, c_datatype *data) \
{ \
    check_MPI_error(MPI_Send(data, count, mpi_datatype, dest, tag, MPI_COMM_WORLD)); \
}

#define make_Ssend(suffix, c_datatype, mpi_datatype) \
void tiramisu_MPI_Ssend_##suffix(int count, int dest, int tag, c_datatype *data) \
{ \
    check_MPI_error(MPI_Ssend(data, count, mpi_datatype, dest, tag, MPI_COMM_WORLD)); \
}

#define make_Isend(suffix, c_datatype, mpi_datatype) \
void tiramisu_MPI_Isend_##suffix(int count, int dest, int tag, c_datatype *data, long *reqs) \
{ \
    ((MPI_Request**)reqs)[0] = (MPI_Request*)malloc(sizeof(MPI_Request)); \
    check_MPI_error(MPI_Isend(data, count, mpi_datatype, dest, tag, MPI_COMM_WORLD, ((MPI_Request**)reqs)[0])); \
}

#define make_Issend(suffix, c_datatype, mpi_datatype) \
void tiramisu_MPI_Issend_##suffix(int count, int dest, int tag, c_datatype *data, long *reqs) \
{ \
    ((MPI_Request**)reqs)[0] = (MPI_Request*)malloc(sizeof(MPI_Request)); \
    check_MPI_error(MPI_Issend(data, count, mpi_datatype, dest, tag, MPI_COMM_WORLD, ((MPI_Request**)reqs)[0])); \
}

#define make_Recv(suffix, c_datatype, mpi_datatype) \
void tiramisu_MPI_Recv_##suffix(int count, int source, int tag, \
                                c_datatype *store_in) \
{ \
    MPI_Status status; \
    check_MPI_error(MPI_Recv(store_in, count, mpi_datatype, source, tag, MPI_COMM_WORLD, &status)); \
}

#define make_Irecv(suffix, c_datatype, mpi_datatype) \
void tiramisu_MPI_Irecv_##suffix(int count, int source, int tag, \
                                 c_datatype *store_in, long *reqs) \
{ \
    ((MPI_Request**)reqs)[0] = (MPI_Request*)malloc(sizeof(MPI_Request)); \
    check_MPI_error(MPI_Irecv(store_in, count, mpi_datatype, source, tag, MPI_COMM_WORLD, \
                              ((MPI_Request**)reqs)[0])); \
}

inline void check_MPI_error(int ret_val) 
{
    if (ret_val != MPI_SUCCESS) {
        fprintf(stderr, "Non-zero exit value returned from MPI: %d", ret_val);
        exit(28);
    }
}

int tiramisu_MPI_Comm_rank(int offset) 
{
    int rank;
    check_MPI_error(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    return rank + offset;
}

void tiramisu_MPI_Wait(void *request) 
{
    MPI_Status status;
    check_MPI_error(MPI_Wait((MPI_Request*)request, &status));
}

void tiramisu_MPI_Send(int count, int dest, int tag, char *data, MPI_Datatype type) 
{
    check_MPI_error(MPI_Send(data, count, type, dest, tag, MPI_COMM_WORLD));
}

make_Send(int8, char, MPI_SIGNED_CHAR)
make_Send(uint8, unsigned char, MPI_UNSIGNED_CHAR)
make_Send(int16, short, MPI_SHORT)
make_Send(uint16, unsigned short, MPI_UNSIGNED_SHORT)
make_Send(int32, int, MPI_INT)
make_Send(uint32, unsigned int, MPI_UNSIGNED)
make_Send(int64, long, MPI_LONG)
make_Send(uint64, unsigned long, MPI_UNSIGNED_LONG)
make_Send(f32, float, MPI_FLOAT)
make_Send(f64, double, MPI_DOUBLE)

void tiramisu_MPI_Ssend(int count, int dest, int tag, char *data, MPI_Datatype type) 
{
    check_MPI_error(MPI_Ssend(data, count, type, dest, tag, MPI_COMM_WORLD));
}

make_Ssend(int8, char, MPI_SIGNED_CHAR)
make_Ssend(uint8, unsigned char, MPI_UNSIGNED_CHAR)
make_Ssend(int16, short, MPI_SHORT)
make_Ssend(uint16, unsigned short, MPI_UNSIGNED_SHORT)
make_Ssend(int32, int, MPI_INT)
make_Ssend(uint32, unsigned int, MPI_UNSIGNED)
make_Ssend(int64, long, MPI_LONG)
make_Ssend(uint64, unsigned long, MPI_UNSIGNED_LONG)
make_Ssend(f32, float, MPI_FLOAT)
make_Ssend(f64, double, MPI_DOUBLE)

void tiramisu_MPI_Isend(int count, int dest, int tag, char *data, MPI_Datatype type, long *reqs) 
{
    ((MPI_Request**)reqs)[0] = (MPI_Request*)malloc(sizeof(MPI_Request));
    check_MPI_error(MPI_Isend(data, count, type, dest, tag, MPI_COMM_WORLD, ((MPI_Request**)reqs)[0]));
}

make_Isend(int8, char, MPI_SIGNED_CHAR)
make_Isend(uint8, unsigned char, MPI_UNSIGNED_CHAR)
make_Isend(int16, short, MPI_SHORT)
make_Isend(uint16, unsigned short, MPI_UNSIGNED_SHORT)
make_Isend(int32, int, MPI_INT)
make_Isend(uint32, unsigned int, MPI_UNSIGNED)
make_Isend(int64, long, MPI_LONG)
make_Isend(uint64, unsigned long, MPI_UNSIGNED_LONG)
make_Isend(f32, float, MPI_FLOAT)
make_Isend(f64, double, MPI_DOUBLE)

void tiramisu_MPI_Issend(int count, int dest, int tag, char *data, MPI_Datatype type, long *reqs) 
{
    ((MPI_Request**)reqs)[0] = (MPI_Request*)malloc(sizeof(MPI_Request));
    check_MPI_error(MPI_Issend(data, count, type, dest, tag, MPI_COMM_WORLD, ((MPI_Request**)reqs)[0]));
}

make_Issend(int8, char, MPI_SIGNED_CHAR)
make_Issend(uint8, unsigned char, MPI_UNSIGNED_CHAR)
make_Issend(int16, short, MPI_SHORT)
make_Issend(uint16, unsigned short, MPI_UNSIGNED_SHORT)
make_Issend(int32, int, MPI_INT)
make_Issend(uint32, unsigned int, MPI_UNSIGNED)
make_Issend(int64, long, MPI_LONG)
make_Issend(uint64, unsigned long, MPI_UNSIGNED_LONG)
make_Issend(f32, float, MPI_FLOAT)
make_Issend(f64, double, MPI_DOUBLE)

void tiramisu_MPI_Recv(int count, int source, int tag,
                     char *store_in, MPI_Datatype type) 
{
    MPI_Status status;
    check_MPI_error(MPI_Recv(store_in, count, type, source, tag, MPI_COMM_WORLD, &status));
}

make_Recv(int8, char, MPI_SIGNED_CHAR)
make_Recv(uint8, unsigned char, MPI_UNSIGNED_CHAR)
make_Recv(int16, short, MPI_SHORT)
make_Recv(uint16, unsigned short, MPI_UNSIGNED_SHORT)
make_Recv(int32, int, MPI_INT)
make_Recv(uint32, unsigned int, MPI_UNSIGNED)
make_Recv(int64, long, MPI_LONG)
make_Recv(uint64, unsigned long, MPI_UNSIGNED_LONG)
make_Recv(f32, float, MPI_FLOAT)
make_Recv(f64, double, MPI_DOUBLE)

void tiramisu_MPI_Irecv(int count, int source, int tag,
                      char *store_in, MPI_Datatype type, long *reqs) 
{
    ((MPI_Request**)reqs)[0] = (MPI_Request*)malloc(sizeof(MPI_Request));
    check_MPI_error(MPI_Irecv(store_in, count, type, source, tag, MPI_COMM_WORLD,
                              ((MPI_Request**)reqs)[0]));
}

make_Irecv(int8, char, MPI_SIGNED_CHAR)
make_Irecv(uint8, unsigned char, MPI_UNSIGNED_CHAR)
make_Irecv(int16, short, MPI_SHORT)
make_Irecv(uint16, unsigned short, MPI_UNSIGNED_SHORT)
make_Irecv(int32, int, MPI_INT)
make_Irecv(uint32, unsigned int, MPI_UNSIGNED)
make_Irecv(int64, long, MPI_LONG)
make_Irecv(uint64, unsigned long, MPI_UNSIGNED_LONG)
make_Irecv(f32, float, MPI_FLOAT)
make_Irecv(f64, double, MPI_DOUBLE)

}

#endif
