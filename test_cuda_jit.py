from numba import jit, cuda
import numpy as np
# to measure exec time
from timeit import default_timer as timer


def func(a, n):

    for i in range(n):
        a[i] += 1


@cuda.jit
def func2(a):

    tx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    a[tx] += 1


if __name__ == "__main__":
    n = 100000000

    start = timer()
    a = np.random.rand(n)
    func(a, n)
    print("without GPU:", timer() - start, max(a))

    start = timer()
    a = np.random.rand(n)
    func2[n // 1000, 1000](a)
    print("with GPU:", timer() - start, max(a))