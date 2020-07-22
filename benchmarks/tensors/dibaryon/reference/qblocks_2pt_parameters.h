#ifndef QBLOCKS_2PT_PARAMETERS_H
#define QBLOCKS_2PT_PARAMETERS_H

#define DATA_SET_1 1
#define DATA_SET_2 0

#if DATA_SET_1

#define P_Vsrc 512
#define P_Vsnk 512
#define P_Nsrc 44
#define P_Nsnk 44
#define P_NEntangled 3
#define P_NsrcHex 1
#define P_NsnkHex 1
#define P_Nperms 9
#define P_Nw 9
#define P_Nw2Hex 32
#define P_Nt 1
#define P_Nc 3
#define P_Ns 2
#define P_Nq 3
#define P_B2Nrows 6
#define P_B1Nrows 2
#define P_Nb 2
#define P_mq 1.0

#elif DATA_SET_2

#define P_Vsrc 8
#define P_Vsnk 8
#define P_Nsrc 9
#define P_Nsnk 9
#define P_NEntangled 3
#define P_NsrcHex 1
#define P_NsnkHex 1
#define P_Nperms 9
#define P_Nw 9
#define P_Nw2Hex 32
#define P_Nt 1
#define P_Nc 3
#define P_Ns 2
#define P_Nq 3
#define P_Nr 6
#define P_Nb 2
#define P_mq 1.0

#endif

#endif
