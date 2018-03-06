#ifndef TIRAMISU_MPI_H
#define TIRAMISU_MPI_H

#include <mpi.h>

extern "C" {

inline void check_MPI_error(int ret_val);

int tiramisu_MPI_Comm_rank(int offset);

void tiramisu_MPI_Wait(void *request);

void tiramisu_MPI_Send(int count, int dest, int tag, char *data, MPI_Datatype type);
void tiramisu_MPI_Send_int8(int count, int dest, int tag, char *data);
void tiramisu_MPI_Send_int16(int count, int dest, int tag, short *data);
void tiramisu_MPI_Send_int32(int count, int dest, int tag, int *data);
void tiramisu_MPI_Send_int64(int count, int dest, int tag, long *data);
void tiramisu_MPI_Send_uint8(int count, int dest, int tag, unsigned char *data);
void tiramisu_MPI_Send_uint16(int count, int dest, int tag, unsigned short *data);
void tiramisu_MPI_Send_uint32(int count, int dest, int tag, unsigned int *data);
void tiramisu_MPI_Send_uint64(int count, int dest, int tag, unsigned long *data);
void tiramisu_MPI_Send_f32(int count, int dest, int tag, float *data);
void tiramisu_MPI_Send_f64(int count, int dest, int tag, double *data);

void tiramisu_MPI_Ssend(int count, int dest, int tag, char *data, MPI_Datatype type);
void tiramisu_MPI_Ssend_int8(int count, int dest, int tag, char *data);
void tiramisu_MPI_Ssend_int16(int count, int dest, int tag, short *data);
void tiramisu_MPI_Ssend_int32(int count, int dest, int tag, int *data);
void tiramisu_MPI_Ssend_int64(int count, int dest, int tag, long *data);
void tiramisu_MPI_Ssend_uint8(int count, int dest, int tag, unsigned char *data);
void tiramisu_MPI_Ssend_uint16(int count, int dest, int tag, unsigned short *data);
void tiramisu_MPI_Ssend_uint32(int count, int dest, int tag, unsigned int *data);
void tiramisu_MPI_Ssend_uint64(int count, int dest, int tag, unsigned long *data);
void tiramisu_MPI_Ssend_f32(int count, int dest, int tag, float *data);
void tiramisu_MPI_Ssend_f64(int count, int dest, int tag, double *data);

void tiramisu_MPI_Isend(int count, int dest, int tag, char *data, MPI_Datatype type, long *reqs);
void tiramisu_MPI_Isend_int8(int count, int dest, int tag, char *data, long *reqs);
void tiramisu_MPI_Isend_int16(int count, int dest, int tag, short *data, long *reqs);
void tiramisu_MPI_Isend_int32(int count, int dest, int tag, int *data, long *reqs);
void tiramisu_MPI_Isend_int64(int count, int dest, int tag, long *data, long *reqs);
void tiramisu_MPI_Isend_uint8(int count, int dest, int tag, unsigned char *data, long *reqs);
void tiramisu_MPI_Isend_uint16(int count, int dest, int tag, unsigned short *data, long *reqs);
void tiramisu_MPI_Isend_uint32(int count, int dest, int tag, unsigned int *data, long *reqs);
void tiramisu_MPI_Isend_uint64(int count, int dest, int tag, unsigned long *data, long *reqs);
void tiramisu_MPI_Isend_f32(int count, int dest, int tag, float *data, long *reqs);
void tiramisu_MPI_Isend_f64(int count, int dest, int tag, double *data, long *reqs);

void tiramisu_MPI_Issend(int count, int dest, int tag, char *data, MPI_Datatype type, long *reqs);
void tiramisu_MPI_Issend_int8(int count, int dest, int tag, char *data, long *reqs);
void tiramisu_MPI_Issend_int16(int count, int dest, int tag, short *data, long *reqs);
void tiramisu_MPI_Issend_int32(int count, int dest, int tag, int *data, long *reqs);
void tiramisu_MPI_Issend_int64(int count, int dest, int tag, long *data, long *reqs);
void tiramisu_MPI_Issend_uint8(int count, int dest, int tag, unsigned char *data, long *reqs);
void tiramisu_MPI_Issend_uint16(int count, int dest, int tag, unsigned short *data, long *reqs);
void tiramisu_MPI_Issend_uint32(int count, int dest, int tag, unsigned int *data, long *reqs);
void tiramisu_MPI_Issend_uint64(int count, int dest, int tag, unsigned long *data, long *reqs);
void tiramisu_MPI_Issend_f32(int count, int dest, int tag, float *data, long *reqs);
void tiramisu_MPI_Issend_f64(int count, int dest, int tag, double *data, long *reqs);

void tiramisu_MPI_Recv(int count, int source, int tag, char *store_in, MPI_Datatype type);
void tiramisu_MPI_Recv_int8(int count, int source, int tag, char *store_in);
void tiramisu_MPI_Recv_int16(int count, int source, int tag, short *store_in);
void tiramisu_MPI_Recv_int32(int count, int source, int tag, int *store_in);
void tiramisu_MPI_Recv_int64(int count, int source, int tag, long *store_in);
void tiramisu_MPI_Recv_uint8(int count, int source, int tag, unsigned char *store_in);
void tiramisu_MPI_Recv_uint16(int count, int source, int tag, unsigned short *store_in);
void tiramisu_MPI_Recv_uint32(int count, int source, int tag, unsigned int *store_in);
void tiramisu_MPI_Recv_uint64(int count, int source, int tag, unsigned long *store_in);
void tiramisu_MPI_Recv_f32(int count, int source, int tag, float *store_in);
void tiramisu_MPI_Recv_f64(int count, int source, int tag, double *store_in);

void tiramisu_MPI_Irecv(int count, int source, int tag, char *store_in, MPI_Datatype type, long *reqs);
void tiramisu_MPI_Irecv_int8(int count, int source, int tag, char *store_in, long *reqs);
void tiramisu_MPI_Irecv_int16(int count, int source, int tag, short *store_in, long *reqs);
void tiramisu_MPI_Irecv_int32(int count, int source, int tag, int *store_in, long *reqs);
void tiramisu_MPI_Irecv_int64(int count, int source, int tag, long *store_in, long *reqs);
void tiramisu_MPI_Irecv_uint8(int count, int source, int tag, unsigned char *store_in, long *reqs);
void tiramisu_MPI_Irecv_uint16(int count, int source, int tag, unsigned short *store_in, long *reqs);
void tiramisu_MPI_Irecv_uint32(int count, int source, int tag, unsigned int *store_in, long *reqs);
void tiramisu_MPI_Irecv_uint64(int count, int source, int tag, unsigned long *store_in, long *reqs);
void tiramisu_MPI_Irecv_f32(int count, int source, int tag, float *store_in, long *reqs);
void tiramisu_MPI_Irecv_f64(int count, int source, int tag, double *store_in, long *reqs);

}

#endif
