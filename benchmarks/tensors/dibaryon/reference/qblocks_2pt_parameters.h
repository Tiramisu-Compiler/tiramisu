#ifndef QBLOCKS_2PT_PARAMETERS_H
#define QBLOCKS_2PT_PARAMETERS_H

#define DATA_SET_1 1
#define DATA_SET_2 0

#if DATA_SET_1

#define P_Vsrc 5
#define P_Vsnk 1
#define P_Nsrc 4
#define P_Nsnk 4
#define P_Nt 1
#define P_NsrcHex 1
#define P_NsnkHex 1
#define P_Nc 3
#define P_Ns 2
#define P_Nq 3
#define P_Nw 9
#define P_Nperms 9
#define P_mq 1.0

#elif DATA_SET_2

#define P_Vsrc 123
#define P_Vsnk 512
#define P_Nsrc 4
#define P_Nsnk 4
#define P_Nt 1
#define P_NsrcHex 1
#define P_NsnkHex 1
#define P_Nc 3
#define P_Ns 2
#define P_Nq 3
#define P_Nw 9
#define P_Nperms 9
#define P_mq 1.0

#endif

#endif
