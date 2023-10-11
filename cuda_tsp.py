import numpy as np
from numba import cuda
import math
from timeit import default_timer as timer

from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

# Define cities and their coordinates
c_array = np.array([(0, 0), (1, 2), (3, 1), (2, 4), (1, 1)], dtype=np.float32)
cities = cuda.to_device(c_array)


# Define distance function
@cuda.jit('float32(int32,int32,float32[:,:])', device=True)
def calculate_distance(city1, city2, cities):
    return math.sqrt((cities[city1][0] - cities[city2][0])**2 + (cities[city1][1] - cities[city2][1])**2)


# Define cost function
@cuda.jit('float32(int32[:],float32[:,:])', device=True)
def calculate_total_distance(solution, cities):
    total_distance = 0.0
    for i in range(solution.size - 1):
        total_distance += calculate_distance(solution[i], solution[i + 1], cities)
    # Close the loop by returning to the starting city
    return total_distance + calculate_distance(solution[-1], solution[0], cities)


@cuda.jit('void(int32[:],int32[:],float32,float32)', device=True)
def generate_neighbor(source, target, r1, r2):
    i1 = np.int_(math.floor(r1 * len(source)))
    i2 = np.int_(math.floor(r2 * len(source)))
    for i, elem in enumerate(source):
        if i == i1:
            target[i] = source[i2]
        elif i == i2:
            target[i] = source[i1]
        else:
            target[i] = elem


@cuda.jit('void(int32[:],int32[:])', device=True)
def copy_solution(source, target):
    for i, elem in enumerate(source):
        target[i] = elem


# Main optimization loop (CUDA kernel)
@cuda.jit
def simulated_annealing_kernel(rng_states, cities, current_solution, new_solution, best_solution, best_cost, iter):
    tx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    #shared_data = cuda.shared.array((shared_memory_size,), dtype=numba.float32)

    current_cost = best_cost[tx]
    current_temperature = 100

    for i in range(iter):

        generate_neighbor(best_solution[tx], new_solution[tx], xoroshiro128p_uniform_float32(rng_states, tx), xoroshiro128p_uniform_float32(rng_states, tx))
        new_cost = calculate_total_distance(new_solution[tx], cities)
        delta_cost = new_cost - current_cost

        if delta_cost < 0 or xoroshiro128p_uniform_float32(rng_states, tx) < math.exp(-delta_cost / current_temperature):
            # Accept the new solution
            copy_solution(new_solution[tx], current_solution[tx])
            current_cost = new_cost

            if new_cost < best_cost[tx]:
                # Update the best solution
                copy_solution(current_solution[tx], best_solution[tx])
                best_cost[tx] = new_cost

        current_temperature *= .99


n = 1000

# Define initial solution (random permutation)
best_solution = cuda.to_device([np.random.permutation(list(range(len(c_array)))) for _ in range(n)])
current_solution = cuda.to_device([np.zeros(len(c_array), dtype=np.int32) for _ in range(n)])
new_solution = cuda.to_device([np.zeros(len(c_array), dtype=np.int32) for _ in range(n)])
best_cost = cuda.to_device([np.inf for _ in range(n)])  # Initialize with a high cost

rng_states = create_xoroshiro128p_states(n, seed=1)

# Run simulated annealing kernel
start = timer()
simulated_annealing_kernel[1, n](rng_states, cities, current_solution, new_solution, best_solution, best_cost, 1000)

# Transfer the best_solution back to the CPU
optimized_route = best_solution.copy_to_host()
optimized_cost = best_cost.copy_to_host()

argmin = np.argmin(optimized_cost)

print("with GPU:", timer() - start)
print("Best tour:", optimized_route[argmin], optimized_cost[argmin])