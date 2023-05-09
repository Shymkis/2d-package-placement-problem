# Joe Shymanski
# Evolutionary Computation Project: 2D Package Placement Problem
# Simulated Annealing

from itertools import combinations
from math import exp
import numpy as np
import random
from time import time

class Simulated_Annealing:
    def __init__(self, dims,
                 weights = None, optimal_fitness = None,
                 t0 = 100, i0 = 100, alpha = 0.9, beta = 1.01, max_stalls = 50, max_time = 300,
                 h = "weighted_manhattan_distance",
                 perturb = "pairwise_exchange"):
        self.dims = dims
        self.num_genes = np.prod(dims)
        self.labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"[:self.num_genes]
        self.dist_matrix = np.zeros([self.num_genes, self.num_genes])
        self.weights = self.generate_weights() if weights is None else weights
        self.optimal_fitness = self.weighted_manhattan_distance(np.arange(self.num_genes)) if weights is None else optimal_fitness
        self.t0 = t0
        self.i0 = i0
        self.alpha = alpha
        self.beta = beta
        self.max_stalls = max_stalls
        self.max_time = max_time
        self.h = getattr(self, h)
        self.perturb = getattr(self, perturb)

    '''

    Generate Toy Weights

    '''

    def coords(self, i):
        r = i // self.dims[1]
        c = i % self.dims[1]
        return [r, c]

    def manhattan_distance(self, i, j):
        if self.dist_matrix[i, j] == 0:
            r_i, c_i = self.coords(i)
            r_j, c_j = self.coords(j)
            self.dist_matrix[i, j] = self.dist_matrix[j, i] = abs(r_i - r_j) + abs(c_i - c_j)
        return self.dist_matrix[i, j]

    def generate_weights(self):
        weights = np.zeros([self.num_genes, self.num_genes])
        max_dist = self.manhattan_distance(0, self.num_genes - 1)
        for i, j in combinations(range(self.num_genes), 2):
            weights[i, j] = weights [j, i] = max_dist - self.manhattan_distance(i, j) + 1
        return weights

    '''

    Heuristic

    '''

    def weighted_manhattan_distance(self, order):
        d_weighted = 0
        for index_i, index_j in combinations(range(self.num_genes), 2):
            label_i = order[index_i]
            label_j = order[index_j]
            w = self.weights[label_i, label_j]
            d = self.manhattan_distance(index_i, index_j)
            d_weighted += w * d
        return d_weighted

    '''

    Perturbation

    '''

    def single_move(self, s):
        r, i = random.sample(range(s.shape[0]), k=2) # "replace" and "insert" indices
        a, b = min(r, i), max(r, i) + 1 # array splice points
        s[a:b] = np.roll(s[a:b], -1) if r < i else np.roll(s[a:b], 1) # roll splice forward or backward
        return s

    def pairwise_exchange(self, s):
        g1, g2 = random.sample(range(s.shape[0]), k=2) # two random genes
        tmp = s[g2] # swap them
        s[g2] = s[g1]
        s[g1] = tmp
        return s

    def cycle_of_three(self, s):
        g1, g2, g3 = random.sample(range(s.shape[0]), k=3) # three random genes
        tmp = s[g3] # cycle them
        s[g3] = s[g2]
        s[g2] = s[g1]
        s[g1] = tmp
        return s

    '''

    Run Simulated Annealing

    '''

    def run(self):
        # Start timer
        start_time = time()
        deadline = start_time + self.max_time

        # Generate initial, random solution
        s = np.random.permutation(self.num_genes)
        
        # Output solution and its heuristic score
        print(s)
        print(self.h(s))

        # Initialize variables
        t = self.t0
        iterations = self.i0
        stalls = l = 0

        # Loop until stopping a criterion is met
        while time() <= deadline and stalls < self.max_stalls and self.h(s) != self.optimal_fitness:
            l += 1
            print("Loop " + str(l) + ":", t, iterations, stalls)
            old_s_fitness = self.h(s)

            # Perturb, choose, iterate
            for _ in range(iterations):
                new_s = self.perturb(s.copy())
                # if self.h(new_s) < self.h(s) or random.random() < exp((self.h(s) - self.h(new_s))/t):
                if self.h(new_s) < self.h(s):
                    s = new_s
            # Output solution and its heuristic score
            print(s)
            print(self.h(s))

            # Update variables
            t = max(self.alpha*t, 1e-6)
            iterations = round(self.beta*iterations)
            stalls = stalls + 1 if self.h(s) == old_s_fitness else 0
        # End timer and assign final values
        self.end_time = time() - start_time
        self.solution = s
        self.solution_fitness = self.h(s)
        self.end_loops = l
        self.optimal_found = self.solution_fitness == self.optimal_fitness

    '''

    Print Solution

    '''

    def print_solution(self):
        a = []
        for i in range(self.num_genes):
            r, _ = self.coords(i)
            b = self.solution[i]
            if len(a) <= r:
                a.append(self.labels[b])
            else:
                a[r] += " " + self.labels[b]
        print(np.array(a)[..., None])

'''

Main Method

'''

if __name__ == "__main__":
    total_time = total_loops = total_sol_fits = total_opts = 0
    dims = [4, 12]
    num_trials = 5
    for _ in range(num_trials):
        sa = Simulated_Annealing(dims, perturb="single_move")
        sa.run()

        print("Solution:")
        sa.print_solution()

        print("Solution Fitness:", sa.solution_fitness)
        print("Optimal Fitness:", sa.optimal_fitness)

        print(sa.end_loops, "loops in", sa.end_time, "seconds")

        total_time += sa.end_time
        total_loops += sa.end_loops
        total_sol_fits += sa.solution_fitness
        total_opts += int(sa.optimal_found)

    print()
    print("Avg time", total_time / num_trials)
    print("Avg loops", total_loops / num_trials)
    print("Avg solution fitness", total_sol_fits / num_trials)
    print("Optimal fitness", sa.optimal_fitness)
    print("Percentage optimal", total_opts / num_trials * 100)
