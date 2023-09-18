""" benchmark python matmul """

import numpy as np
from timeit import timeit


class Matrix:
    def __init__(self, value, rows, cols):
        self.value = value
        self.rows = rows
        self.cols = cols
        
    def __getitem__(self, idxs):
        return self.value[idxs[0]][idxs[1]]
    
    def __setitem__(self, idxs, value):
        self.value[idxs[0]][idxs[1]] = value
    
    def __matmul__(self, other):
        assert self.cols == other.rows
        result = Matrix(list(np.zeros((self.rows, other.cols))), self.rows, other.cols)
        for m in range(self.rows):
            for k in range(self.cols):
                for n in range(other.cols):
                    result[m, n] += self[m, k] * other[k, n]
        return result


def benchmark_matmul_python(M, N, K):
    A = Matrix(list(np.random.rand(M, K)), M, K)
    B = Matrix(list(np.random.rand(K, N)), K, N)
    C = Matrix(list(np.zeros((M, N))), M, N)
    def testbench():
        C = A @ B
    secs = timeit(testbench, number=2)/2
    gflops = ((2*M*N*K)/secs) / 1e9
    print("Python baseline: ", gflops, "GFLOP/s")
    return gflops

def benchmark_matmul_numpy(M, N, K, baseline):
    A = np.random.rand(M, K)
    B = np.random.rand(K, N)
    C = np.zeros((M, N))
    def testbench():
        C = A @ B
    secs = timeit(testbench, number=2)/2
    gflops = ((2*M*N*K)/secs) / 1e9
    speedup = gflops / baseline
    print("Python numpy   : ", gflops, f"GFLOP/s, a {speedup} x speedup over Python")
    
    return gflops

if __name__ == "__main__":
    baseline = benchmark_matmul_python(128, 128, 128)
    benchmark_matmul_numpy(128, 128, 128, baseline)
