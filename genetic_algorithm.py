# Joe Shymanski
# Evolutionary Computation Project: 2D Package Placement Problem
# Genetic Algorithm

from itertools import combinations
import numpy as np
import random
from time import time

class Genetic_Algorithm:
    def __init__(self, dims,
                 weights = None, optimal_fitness = None,
                 max_time = 300, num_generations = 100, pop_size = 1000,
                 fitness_func = "weighted_manhattan_distance", minimize = True, top_n = 5,
                 selection_strat = "rank",
                 crossover_strat = "cut_and_crossfill", num_elites = 2,
                 mutation_strat = "pairwise_exchange", min_mp = .01, max_mp = .1):
        self.dims = dims
        self.num_genes = np.prod(dims)
        self.labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"[:self.num_genes]
        self.dist_matrix = np.zeros([self.num_genes, self.num_genes])
        self.weights = self.generate_weights() if weights is None else weights
        self.optimal_fitness = self.weighted_manhattan_distance(np.arange(self.num_genes)) if weights is None else optimal_fitness
        self.max_time = max_time
        self.num_generations = num_generations
        self.pop_size = pop_size
        self.fitness_func = getattr(self, fitness_func)
        self.minimize = minimize
        self.top_n = top_n
        self.selection_strat = getattr(self, selection_strat)
        self.crossover_strat = getattr(self, crossover_strat)
        self.num_elites = num_elites
        self.mutation_strat = getattr(self, mutation_strat)
        self.min_mp = min_mp
        self.max_mp = max_mp

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

    Fitness

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

    def fitness(self):
        fit_scores = []
        for ind in self.population:
            f = self.fitness_func(ind)
            fit_scores.append(f)
        self.fit_scores = np.array(fit_scores)

    def order_pop_by_fitness(self):
        self.fitness()
        p = self.fit_scores.argsort() if self.minimize else self.fit_scores.argsort()[::-1]
        self.fit_scores = self.fit_scores[p]
        self.population = self.population[p]
        if self.top_n:
            print(self.population[:self.top_n])
            print(self.fit_scores[:self.top_n])

    '''

    Selection

    '''

    def diff(self):
        return self.fit_scores.max() - self.fit_scores + 1 if self.minimize else self.fit_scores - self.fit_scores.min() + 1

    def rank(self):
        return np.arange(1, self.pop_size + 1)[::-1]

    def roulette(self):
        return self.fit_scores.sum() / self.fit_scores if self.minimize else self.fit_scores

    def selection(self):
        self.parent_pool = np.array(random.choices(self.population, self.selection_strat(), k=self.pop_size))

    '''

    Crossover

    '''

    def order_1(self, parents):
        parent1, parent2 = parents
        s1, s2 = random.sample(range(self.num_genes), k=2)
        while abs(s1 - s2) == self.num_genes - 1:
            s1, s2 = random.sample(range(self.num_genes), k=2)
        a, b = min(s1, s2), max(s1, s2) + 1
        cut1 = parent1[a:b]
        cut2 = parent2[a:b]
        remainder1 = [g for g in np.roll(parent2, -b) if g not in cut1]
        remainder2 = [g for g in np.roll(parent1, -b) if g not in cut2]
        child1 = np.roll(np.concatenate([cut1, remainder1]), a)
        child2 = np.roll(np.concatenate([cut2, remainder2]), a)
        return [child1, child2]

    def cut_and_crossfill(self, parents):
        parent1, parent2 = parents
        s = random.randint(1, self.num_genes - 1)
        r = random.random()
        cut1 = parent1[:s] if r < 0.5 else parent1[s:]
        cut2 = parent2[:s] if r < 0.5 else parent2[s:]
        crossfill1 = [g for g in parent2 if g not in cut1]
        crossfill2 = [g for g in parent1 if g not in cut2]
        child1 = np.concatenate([cut1, crossfill1]) if r < 0.5 else np.concatenate([crossfill1, cut1])
        child2 = np.concatenate([cut2, crossfill2]) if r < 0.5 else np.concatenate([crossfill2, cut2])
        return [child1, child2]

    def crossover(self):
        child_pool = []
        for _ in range((self.pop_size - self.num_elites) // 2):
            ids = random.sample(range(self.pop_size), k=2)
            parents = self.parent_pool[ids, :]
            children = self.crossover_strat(parents)
            child_pool.extend(children)
        self.child_pool = np.array(child_pool)

    '''

    Mutation

    '''

    def single_move(self, individuals):
        for ind in individuals:
            r, i = random.sample(range(ind.shape[0]), k=2) # replace and insert indices
            a, b = min(r, i), max(r, i) + 1 # array splice points
            ind[a:b] = np.roll(ind[a:b], -1) if r < i else np.roll(ind[a:b], 1) # roll splice forward or backward
        return individuals

    def pairwise_exchange(self, individuals):
        for ind in individuals:
            g1, g2 = random.sample(range(ind.shape[0]), k=2) # two random genes
            tmp = ind[g2] # swap them
            ind[g2] = ind[g1]
            ind[g1] = tmp
        return individuals

    def cycle_of_three(self, individuals):
        for ind in individuals:
            g1, g2, g3 = random.sample(range(ind.shape[0]), k=3) # three random genes
            tmp = ind[g3] # cycle them
            ind[g3] = ind[g2]
            ind[g2] = ind[g1]
            ind[g1] = tmp
        return individuals

    def mutation(self):
        p = np.random.random(self.child_pool.shape[0])
        self.child_pool[p < self.mut_prob] = self.mutation_strat(self.child_pool[p < self.mut_prob])

    '''

    Run Genetic Algorithm

    '''

    def run(self):
        # Start timer
        start_time = time()
        deadline = start_time + self.max_time

        # Generate initial, random population
        pop0 = []
        for _ in range(self.pop_size):
            pop0.append(np.random.permutation(self.num_genes))
        self.population = np.array(pop0)

        # Order initial population by fitness
        self.order_pop_by_fitness()

        # Loop through generations
        for g in range(1, self.num_generations + 1):
            # print("Generation", g)

            # Keep elites
            elites = self.population[:self.num_elites]

            # Adaptive mutation probability
            self.mut_prob = (g/self.num_generations)*(self.max_mp - self.min_mp) + self.min_mp

            # Selection, crossover, mutation
            self.selection()
            self.crossover()
            self.mutation()

            # Add elites to next generation
            self.population = np.concatenate([self.child_pool, elites])

            # Order population by fitness
            self.order_pop_by_fitness()

            # Stopping criteria
            self.solution = self.population[0]
            self.solution_fitness = self.fit_scores[0]
            if self.solution_fitness == self.optimal_fitness:
                self.end_time = time() - start_time
                self.end_gens = g
                self.optimal_found = True
                return
            if time() > deadline:
                self.end_time = time() - start_time
                self.end_gens = g
                self.optimal_found = False
                return
        self.end_time = time() - start_time
        self.end_gens = self.num_generations
        self.optimal_found = False

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
    for i in range(72):
        total_time = total_gens = total_sol_fits = total_opts = 0
        
        if i < 6:
            dims = [3, 4]
        elif i < 12:
            dims = [4, 3]
        elif i < 18:
            dims = [3, 8]
        elif i < 24:
            dims = [8, 3]
        elif i < 30:
            dims = [4, 7]
        elif i < 36:
            dims = [7, 4]
        elif i < 42:
            dims = [2, 12]
        elif i < 48:
            dims = [12, 2]
        elif i < 54:
            dims = [4, 8]
        elif i < 60:
            dims = [8, 4]
        elif i < 63:
            dims = [4, 5]
        elif i < 66:
            dims = [5, 4]
        elif i < 69:
            dims = [4, 6]
        else:
            dims = [6, 4]

        if i % 6 < 3 and i < 60:
            crossover_strat = "cut_and_crossfill"
        else:
            crossover_strat = "order_1"

        if i % 3 == 0:
            mutation_strat = "cycle_of_three"
        elif i % 3 == 1:
            mutation_strat = "pairwise_exchange"
        else:
            mutation_strat = "single_move"
        
        num_trials = 5
        for _ in range(num_trials):
            ga = Genetic_Algorithm(dims, top_n=0, selection_strat="roulette", crossover_strat=crossover_strat, mutation_strat=mutation_strat)
            ga.run()

            # print("Solution:")
            # ga.print_solution()

            # print("Solution Fitness:", ga.solution_fitness)
            # print("Optimal Fitness:", ga.optimal_fitness)

            # print(ga.end_gens, "generations of", ga.pop_size, "individuals in", ga.end_time, "seconds")

            total_time += ga.end_time
            total_gens += ga.end_gens
            total_sol_fits += ga.solution_fitness
            total_opts += int(ga.optimal_found)

        print(dims, "roulette", crossover_strat, mutation_strat)
        print("Avg time", total_time / num_trials)
        print("Avg generations", total_gens / num_trials)
        print("Avg solution fitness", total_sol_fits / num_trials)
        print("Optimal fitness", ga.optimal_fitness)
        print("Percentage optimal", total_opts / num_trials * 100)
