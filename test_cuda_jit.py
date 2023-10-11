from numba import jit, cuda
import numpy as np
# to measure exec time
from timeit import default_timer as timer


@jit
def func(a, n, nr):

    for i in range(n):
        for j in range(nr):
            a[i] += 1


@cuda.jit
def func2(a, i):

    tx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    for j in range(i):
        a[tx] += 1


if __name__ == "__main__":
    n = 10000000

    nr = 1
    for i in range(10):

        start = timer()
        a = np.random.rand(n)
        func(a, n, nr)
        print("without GPU:", nr, timer() - start, max(a))

        start = timer()
        a = np.random.rand(n)
        func2[n // 1000, 1000](a, nr)
        print("with GPU:", nr, timer() - start, max(a))
        nr *= 10
