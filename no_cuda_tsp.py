import math
import numpy as np
from timeit import default_timer as timer

# Define cities and their coordinates
cities = np.array([(0, 0), (1, 2), (3, 1), (2, 4), (1, 1)], dtype=np.float32)

# Define distance function
def calculate_distance(city1, city2):
    return math.sqrt((cities[city1][0] - cities[city2][0])**2 + (cities[city1][1] - cities[city2][1])**2)

# Define cost function
def calculate_total_distance(solution):
    total_distance = 0.0
    for i in range(len(solution) - 1):
        total_distance += calculate_distance(solution[i], solution[i + 1])
    # Close the loop by returning to the starting city
    return total_distance + calculate_distance(solution[-1], solution[0])

def generate_neighbor(solution):
    new_solution = solution.copy()
    indices = np.random.randint(0, len(solution), 2)
    new_solution[indices[0]] = solution[indices[1]]
    new_solution[indices[1]] = solution[indices[0]]
    return new_solution

# Main optimization loop (CUDA kernel)
def simulated_annealing_kernel(best_solution, best_cost, iter, tx):
    #shared_data = cuda.shared.array((shared_memory_size,), dtype=numba.float32)

    current_solution = best_solution[tx]
    current_cost = calculate_total_distance(current_solution)
    current_temperature = 100

    for i in range(iter):

        new_solution = generate_neighbor(current_solution)
        new_cost = calculate_total_distance(new_solution)
        delta_cost = new_cost - current_cost

        if delta_cost < 0 or np.random.random() < math.exp(-delta_cost / current_temperature):
            # Accept the new solution
            current_solution = new_solution
            current_cost = new_cost

            if new_cost < best_cost[tx]:
                # Update the best solution
                best_solution[tx] = new_solution
                best_cost[tx] = new_cost

        current_temperature *= .99


n = 1000

# Define initial solution (random permutation)
best_solution = [np.random.permutation(list(range(len(cities)))) for _ in range(n)]
best_cost = [np.inf for _ in range(n)]  # Initialize with a high cost

start = timer()
# Run simulated annealing kernel
for i in range(1000):
    simulated_annealing_kernel(best_solution, best_cost, 1000, i)

argmin = np.argmin(best_cost)

print("without GPU:", timer() - start)
print("Best tour:", best_solution[argmin], best_cost[argmin])