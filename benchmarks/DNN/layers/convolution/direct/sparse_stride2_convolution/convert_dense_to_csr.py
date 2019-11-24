import numpy as np
# Blocking to set in order to generate a data where FIN is blocked (for blocked traversal)

############ PARAMETERS TO SET
dense_filename = "resnet_10.npy"
sparse_output_filename = "resnet_10.csr"
N = 224 # Input height and width
FIN_BL = 8
###############################
# The output file's format is :
'''
FOut, FIn, K, N, NNZ
/
all non zero values separated by line breaks
/
all rowptr values separated by line breaks
/
all colidx values separated by line breaks
'''

def denseToCSR(arr, N):
    nnz=0
    FOut = arr.shape[0] # get number of output features
    FIn = arr.shape[1] # get number of input features
    K = arr.shape[2] # get kernel size
    vals=[] # csr array of values (size NNZ)
    indexes=[] # csr array of col indexes (size NNZ)
    finptr=[] # csr array of row ptr (size FOut + 1)
    for fout in range(FOut):
        finptr.append(nnz)
        for fin_b in range(int(FIn/FIN_BL)):
            for ky in range(K):
                for(kx) in range(K):
                    for ffin in range(FIN_BL):
                        if v[fout, fin_b * FIN_BL + ffin, ky, kx] != 0:
                            vals.append(v[fout, (fin_b * FIN_BL + ffin), ky, kx])
                            indexes.append((fin_b * FIN_BL + ffin) * (N + 2) * (N + 2) + ky * (N + 2) + kx)
                            nnz+=1
    finptr.append(nnz)
    print("FOUT = ", FOut)
    print("FIn = ", FIn)
    print("K = ", K)
    print("Density = ", nnz / (FOut * FIn * K * K))
    return vals, finptr, indexes, nnz

v = np.load(dense_filename)
values, rowptr, colidx, nnz = denseToCSR(v, N)
with open(sparse_output_filename, 'w') as f:
    f.write("%d, %d, %d, %d, %d \n" % (v.shape[0], v.shape[1], v.shape[2], N, nnz))
    f.write("/\n")
    # Write values
    for item in values:
        f.write("%s\n" % item)
    f.write("/\n")
    # Write rowptr
    for item in rowptr:
        f.write("%s\n" % item)
    f.write("/\n")
    # Write colidx
    for item in colidx:
        f.write("%s\n" % item)
