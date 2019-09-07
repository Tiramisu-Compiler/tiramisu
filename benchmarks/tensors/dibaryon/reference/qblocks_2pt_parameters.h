#ifndef QBLOCKS_2PT_PARAMETERS_H
#define QBLOCKS_2PT_PARAMETERS_H

#define DATA_SET_1 0
#define DATA_SET_2 1

#if DATA_SET_1

#define P_Vsrc 1
#define P_Vsnk 1
#define P_Nsrc 1
#define P_Nsnk 3
#define P_Nt 1
#define P_NsrcHex 2
#define P_NsnkHex 4
#define P_Nc 3
#define P_Ns 2
#define P_Nq 3
#define P_Nw 9
#define P_Nperms 9
#define P_mq 1.0

#elif DATA_SET_2

#define P_Vsrc 16
#define P_Vsnk 16
#define P_Nsrc 16
#define P_Nsnk 16
#define P_Nt 1
#define P_NsrcHex 2
#define P_NsnkHex 4
#define P_Nc 3
#define P_Ns 2
#define P_Nq 3
#define P_Nw 9
#define P_Nperms 9
#define P_mq 1.0

#endif

#endif
