import numpy as np
from numba import jit, prange
import math
from timeit import default_timer as timer

from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

# Define cities and their coordinates
num_cities = 100
cities = np.array([(np.random.randint(0, 100), np.random.randint(0, 100)) for _ in range(num_cities)], dtype=np.float32)


# Define distance function
@jit('float32(int32,int32,float32[:,:])')
def calculate_distance(city1, city2, cities):
    return math.sqrt((cities[city1][0] - cities[city2][0])**2 + (cities[city1][1] - cities[city2][1])**2)


# Define cost function
@jit('float32(int32[:],float32[:,:])')
def calculate_total_distance(solution, cities):
    total_distance = 0.0
    for i in range(solution.size - 1):
        total_distance += calculate_distance(solution[i], solution[i + 1], cities)
    # Close the loop by returning to the starting city
    return total_distance + calculate_distance(solution[-1], solution[0], cities)


@jit('void(int32[:],int32[:],float32,float32)')
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


@jit('void(int32[:],int32[:])')
def copy_solution(source, target):
    for i, elem in enumerate(source):
        target[i] = elem


# Main optimization loop (CUDA kernel)
@jit
def simulated_annealing_kernel(cities, current_solution, new_solution, best_solution, best_cost, iter):

    current_cost = best_cost
    current_temperature = 100

    for i in range(iter):

        generate_neighbor(best_solution, new_solution, np.random.random(), np.random.random())
        new_cost = calculate_total_distance(new_solution, cities)
        delta_cost = new_cost - current_cost

        if delta_cost < 0 or np.random.random() < math.exp(-delta_cost / current_temperature):
            # Accept the new solution
            copy_solution(new_solution, current_solution)
            current_cost = new_cost

            if new_cost < best_cost:
                # Update the best solution
                copy_solution(current_solution, best_solution)
                best_cost = new_cost

        current_temperature *= .99

    return best_solution, best_cost


@jit(parallel=True)
def parallelize(n, cities, current_solution, new_solution, best_solution, best_cost, iter):
    for i in prange(n):
        best_solution[i], best_cost[i] = simulated_annealing_kernel(cities, current_solution[i], new_solution[i], best_solution[i], best_cost[i], iter)


n = 1000000

# Define initial solution (random permutation)
best_solution = [np.random.permutation(list(range(num_cities))) for _ in range(n)]
current_solution = [np.zeros(num_cities, dtype=np.int32) for _ in range(n)]
new_solution = [np.zeros(num_cities, dtype=np.int32) for _ in range(n)]
best_cost = [np.inf for _ in range(n)]  # Initialize with a high cost

rng_states = create_xoroshiro128p_states(n, seed=1)

# Run simulated annealing kernel
start = timer()
parallelize(n, cities, current_solution, new_solution, best_solution, best_cost, 1000)

argmin = np.argmin(best_cost)

print("without GPU, with JIT parallel:", timer() - start)
print("Best tour:", best_solution[argmin], best_cost[argmin])
