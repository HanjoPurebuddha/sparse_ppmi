import numpy as np
import math
import scipy.sparse as sp

def convertPPMISparse(mat):
    """
     Converted from code from svdmi
     https://github.com/Bollegala/svdmi/blob/master/src/svdmi.py
     """
    (nrows, ncols) = mat.shape
    print("no. of rows =", nrows)
    print("no. of cols =", ncols)
    colTotals = mat.sum(axis=0)
    rowTotals = mat.sum(axis=1).T
    N = np.sum(rowTotals)
    rowMatSparse = np.zeros(nrows, dtype=np.float)
    for i in range(nrows):
        if rowTotals[0, i] != 0:
            rowMatSparse[i] = 1.0 / rowTotals[0, i]
    colMatSparse = np.zeros(ncols, dtype=np.float)
    for j in range(ncols):
        if colTotals[0, j] != 0:
            colMatSparse[j] = 1.0 / colTotals[0, j]
    P = N * mat
    P = P.astype(np.float64)
    for i in range(len(rowMatSparse)):
        P[i] *= rowMatSparse[i]
    for i in range(len(colMatSparse)):
        P[:,i] *= colMatSparse[i]
    cx = sp.coo_matrix(P)
    for i, j, v in zip(cx.row, cx.col, cx.data):
        P[i, j] = 0 if v <= 0 else P[i,j] = max(math.log(v), 0)
    return P


def convertPPMI(mat):
    """
    This is ripped directly from SVDMI for comparison
     https://github.com/Bollegala/svdmi/blob/master/src/svdmi.py

     Compute the PPMI values for the raw co-occurrence matrix.
     PPMI values will be written to mat and it will get overwritten.
     """
    (nrows, ncols) = mat.shape
    print("no. of rows =", nrows)
    print("no. of cols =", ncols)
    colTotals = mat.sum(axis=0)
    rowTotals = mat.sum(axis=1).T
    N = np.sum(rowTotals)
    rowMat = np.ones((nrows, ncols), dtype=np.float)
    for i in range(nrows):
        rowMat[i, :] = 0 \
            if rowTotals[0,i] == 0 \
            else rowMat[i, :] * (1.0 / rowTotals[0,i])
    colMat = np.ones((nrows, ncols), dtype=np.float)
    for j in range(ncols):
        colMat[:,j] = 0 if colTotals[0,j] == 0 else (1.0 / colTotals[0,j])
    mat = mat.toarray()
    P = N * mat * rowMat * colMat
    P = np.fmax(np.zeros((nrows,ncols), dtype=np.float64), np.log(P))

    return P


def test():
    # Create a random sparse array
    tf = np.random.randint(low=0, high=20, size=(10, 15), dtype=np.int)
    for i in range(len(tf)):
        for j in range(len(tf[i])):
            if np.random.randint(low=0, high=2, size=1) == 0:
                tf[i][j] = 0
    tf = sp.csr_matrix(tf)

    # Get our sparse matrix
    sparse_ppmi = convertPPMISparse(tf)

    # Get normal matrix
    ppmi = convertPPMI(tf)

    # Check values are equivalent
    for i in range(len(ppmi)):
        broke = False
        for j in range(len(ppmi[i])):
            if ppmi[i][j] != sparse_ppmi[i,j]:
                print(i, j, "sparse", sparse_ppmi[i,j], "non-sparse", ppmi[i][j])
                broke = True
        if broke is False:
            print("Clear")

test()