import numpy as np


def metric(a,b):
        # euclid
        return np.linalg.norm(a-b)

def first(x):
        return x[0]

def minVal(v1,v2,v3):
        if first(v1) <= min(first(v2), first(v3)):
            return v1, 0
        elif first(v2) <= first(v3):
            return v2, 1
        else:
            return v3, 2

def calculation_dtw(A,B):
        M = len(A)
        N = len(B)

        m = [[0 for j in range(N)] for i in range(M)]
        # m = np.zeros((M,N))
        m[0][0] = (metric(A[0],B[0]), (-1,-1))
        #print(m)

        for i in range(1,M):
            m[i][0] = (m[i-1][0][0] + metric(A[i], B[0]), (i-1,0))
        for j in range(1,N):
            m[0][j] = (m[0][j-1][0] + metric(A[0], B[j]), (0,j-1))

        for i in range(1,M):
            for j in range(1,N):
                minimum, index =  minVal(m[i-1][j], m[i][j-1], m[i-1][j-1])
                indexes = [(i-1,j), (i,j-1), (i-1,j-1)]
                m[i][j] = (first(minimum) + metric(A[i],B[j]), indexes[index])
        return m

def dtw(A, B):
    """DTWの計算

    Computes the DTW(Dynamic Time Warping) between two 2-D arrays.
    2次元のDTW(Dynamic Time Warping)の計算

    Args:
        A (numpy.ndarray): 2次元の系列データ
        B (numpy.ndarray): 比較する2次元の系列データ

    Returns:
        float: DTWの計算結果

    Examples:
        >>> import numpy as np
        >>> from trajectory_similarity import similarity_calculation as simi
        >>> A = np.array([[1, 1], [2, 2], [3, 3]])
        >>> B = np.array([[0, 1], [0, 2], [0, 3]])
        >>> simi.dtw(A,B)
        6.0

    """
    return calculation_dtw(A, B)[-1][-1][0]