#ifndef QBLOCKS_2PT_PARAMETERS_H
#define QBLOCKS_2PT_PARAMETERS_H

// 8^3 sparse lattice deuteron (wrapper tests work)
#define DATA_SET_1 0
// 2^3 sparse lattice deuteron (wrapper tests work)
#define DATA_SET_2 1
// 6^3 sparse lattice deuteron (wrapper tests work)
#define DATA_SET_3 0
// 8^3 sparse lattice H-dibaryon (wrapper tests dont work)
#define DATA_SET_4 0
// 2^3 sparse lattice H-dibaryon (wrapper tests dont work)
#define DATA_SET_5 0
// 6^3 sparse lattice H-dibaryon (wrapper tests dont work)
#define DATA_SET_6 0

#if DATA_SET_1

#define P_Vsrc 512
#define P_Vsnk 512
#define P_Nsrc 44
#define P_Nsnk 44
#define P_NEntangled 3
#define P_NsrcHex 1
#define P_NsnkHex 1
#define P_Nperms 36
#define P_B1Nperms 2
#define P_Nw 12
#define P_Nw2 288
#define P_Nw2Hex 32
#define P_Nt 1
#define P_Nc 3
#define P_Ns 2
#define P_Nq 3
#define P_B2Nrows 4
#define P_B1Nrows 2
#define P_Nb 2
#define P_mq 1.0
#define P_Mq 2
#define P_B0Nrows 1
#define P_Mw 12
#define P_NsFull 4
#define P_sites_per_rank 4
#define P_src_sites_per_rank 4

#elif DATA_SET_2

#define P_Vsrc 8
#define P_Vsnk 8
#define P_Nsrc 44
#define P_Nsnk 44
#define P_NEntangled 3
#define P_NsrcHex 1
#define P_NsnkHex 1
#define P_Nperms 36
#define P_B1Nperms 2
#define P_B1Nweights 1
#define P_B1NsrcSC 1
#define P_B1NsnkSC 1
#define P_B2NsrcSC 1
#define P_B2NsnkSC 1
#define P_Nw 12
#define P_Nw2 288
#define P_Nw2Hex 32
#define P_Nt 1
#define P_Nc 3
#define P_Ns 2
#define P_Nq 3
#define P_B2Nrows 4
#define P_B1Nrows 2
#define P_Nb 2
#define P_mq 1.0
#define P_Mq 2
#define P_B0Nrows 1
#define P_Mw 12
#define P_NsFull 4
#define P_sites_per_rank 2
#define P_src_sites_per_rank 2
#define P_tiling_factor 2 // smaller means more dibaryon threads per block */

#elif DATA_SET_3

#define P_Vsrc 216
#define P_Vsnk 216
#define P_Nsrc 20
#define P_Nsnk 20
#define P_NEntangled 3
#define P_NsrcHex 1
#define P_NsnkHex 1
#define P_Nperms 36
#define P_B1Nperms 2
#define P_Nw 24
#define P_Nw2 1152
//#define P_Nw 12
//#define P_Nw2 288
#define P_Nw2Hex 600
#define P_B1NsrcSC 3
#define P_B1NsnkSC 3
#define P_B2NsrcSC 16
#define P_B2NsnkSC 16
#define P_Nt 1
#define P_Nc 3
#define P_Ns 4
#define P_Nq 3
#define P_B2Nrows 4
#define P_B1Nrows 2
#define P_Nb 2
#define P_mq 1.0
#define P_Mq 2
#define P_B0Nrows 1
#define P_Mw 12
#define P_NsFull 4
#define P_sites_per_rank 1
#define P_src_sites_per_rank 1

#elif DATA_SET_4

#define P_Vsrc 216
#define P_Vsnk 216
#define P_Nsrc 20
#define P_Nsnk 20
#define P_NEntangled 3
#define P_NsrcHex 1
#define P_NsnkHex 1
#define P_Nperms 8
#define P_B1Nperms 1
#define P_B1Nweights 3
#define P_Nw 18 
#define P_Nw2 648
#define P_Nw2Hex 174
#define P_B1NsrcSC 6
#define P_B1NsnkSC 6
#define P_B2NsrcSC 4
#define P_B2NsnkSC 4
#define P_Nt 1
#define P_Nc 3
#define P_Ns 2
#define P_Nq 3
#define P_B2Nrows 4
#define P_B1Nrows 2
#define P_Nb 2
#define P_mq 1.0
#define P_Mq 2
#define P_B0Nrows 1
#define P_Mw 12
#define P_NsFull 4
#define P_sites_per_rank 1
#define P_src_sites_per_rank 1

#elif DATA_SET_5

#define P_Vsrc 8
#define P_Vsnk 8
#define P_Nsrc 20
#define P_Nsnk 20
#define P_NEntangled 3
#define P_NsrcHex 1
#define P_NsnkHex 1
#define P_Nperms 8
#define P_B1Nperms 1
#define P_B1Nweights 3
#define P_Nw 18 
#define P_Nw2 648
#define P_Nw2Hex 174
#define P_B1NsrcSC 6
#define P_B1NsnkSC 6
#define P_B2NsrcSC 4
#define P_B2NsnkSC 4
#define P_Nt 1
#define P_Nc 3
#define P_Ns 2
#define P_Nq 3
#define P_B2Nrows 4
#define P_B1Nrows 2
#define P_Nb 2
#define P_mq 1.0
#define P_Mq 2
#define P_B0Nrows 1
#define P_Mw 12
#define P_NsFull 4
#define P_sites_per_rank 1
#define P_src_sites_per_rank 1

#elif DATA_SET_6

#define P_Vsrc 216
#define P_Vsnk 216
#define P_Nsrc 20
#define P_Nsnk 20
#define P_NEntangled 3
#define P_NsrcHex 1
#define P_NsnkHex 1
#define P_Nperms 8
#define P_B1Nperms 1
#define P_B1Nweights 3
#define P_Nw 18 
#define P_Nw2 648
#define P_Nw2Hex 174
#define P_B1NsrcSC 6
#define P_B1NsnkSC 6
#define P_B2NsrcSC 4
#define P_B2NsnkSC 4
#define P_Nt 1
#define P_Nc 3
#define P_Ns 2
#define P_Nq 3
#define P_B2Nrows 4
#define P_B1Nrows 2
#define P_Nb 2
#define P_mq 1.0
#define P_Mq 2
#define P_B0Nrows 1
#define P_Mw 12
#define P_NsFull 4
#define P_sites_per_rank 1
#define P_src_sites_per_rank 1

#endif
#endif
