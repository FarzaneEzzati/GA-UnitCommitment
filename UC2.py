import copy
import pickle
import random
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gurobipy
from gurobipy import quicksum, GRB
import time
from tqdm import tqdm
env = gurobipy.Env()
env.setParam('OutputFlag', 0)

T = 24
DC = 10
RNGT = range(1, T + 1)
RNGDvc = range(1, DC + 1)


S = [4500, 550, 560, 170, 30,
     170, 260, 30, 30, 30]
S = S[:DC]
a = [1000, 700, 680, 370, 660,
     370, 480, 660, 665, 670]
a = a[:DC]
b = [0.00048, 0.002, 0.00211, 0.00712, 0.00413,
     0.00712, 0.00079, 0.00413, 0.00222, 0.00173]
b = b[:DC]
Load = [400, 450, 480, 500, 530, 550, 580, 600, 620, 650,
        680, 700, 650, 620, 600, 550, 500, 550, 600, 650,
        600, 550, 500, 450]
Pmax = [455, 130, 130, 80, 55,
        80, 85, 55, 55, 55]
Pmax = Pmax[:DC]
Pmin = [150, 20, 20, 20, 40,
        20, 25, 10, 10, 10]
Pmin = Pmin[:DC]
lmda0 = 10000
lmda = copy.copy(lmda0)

class MIP:
    def __init__(self, exact=False):
        # Build the model
        model = gurobipy.Model('MIP', env=env)
        Y_indices = [(d, t) for d in RNGDvc for t in RNGT]
        Y = model.addVars(Y_indices, vtype=GRB.CONTINUOUS)
        U = model.addVars(Y_indices, vtype=GRB.BINARY)

        self.U = U
        self.Y = Y
        self.model = model
        if exact:
            # Apply Exact Constraint for Demand
            self.ExactConst()
        else:
            # Apply Relaxed Constraint for Demand
            self.RelaxedConst()

        for d in RNGDvc:
            for t in RNGT:
                model.addConstr(Y[(d, t)] <= Pmax[d - 1] * U[(d, t)])
                model.addConstr(Pmin[d - 1] * U[(d, t)] <= Y[(d, t)])

        C = {(d, t): a[d - 1] * Y[(d, t)] + b[d - 1] * U[(d, t)] for d in RNGDvc for t in RNGT}
        StartUpCost = quicksum(U[(d, t)] * S[d - 1] for d in RNGDvc for t in RNGT)
        FuelCost = quicksum(C[(d, t)] for d in RNGDvc for t in RNGT)
        ViolationCost = lmda * quicksum(Load[t - 1] - quicksum(Y[(d, t)] for d in RNGDvc) for t in RNGT)
        model.setObjective(StartUpCost + FuelCost + ViolationCost, sense=GRB.MINIMIZE)
        model.update()

    def ExactConst(self):
        self.model.addConstrs(Load[t - 1] == quicksum(self.Y[(d, t)] for d in RNGDvc) for t in RNGT)
        self.model.update()
    def RelaxedConst(self):
        self.model.addConstrs(Load[t - 1] >= quicksum(self.Y[(d, t)] for d in RNGDvc) for t in RNGT)
        self.model.update()

    def Fitness(self, ind):
        ''' U must be a distionary with indices [d][t] '''
        UU = [ind[(d-1)*24:d*24] for d in RNGDvc]
        for d in RNGDvc:
            for t in RNGT:
                self.model.addConstr(self.U[(d, t)] == UU[d-1][t-1], name=f'U[{d},{t}]')

        self.model.update()
        self.model.optimize()
        obj = self.model.ObjVal

        for d in RNGDvc:
            for t in RNGT:
                self.model.remove(self.model.getConstrByName(f'U[{d},{t}]'))

        self.model.update()
        return obj

    def GetInitialInd(self):
        U = {}
        for d in RNGDvc:
            for t in RNGT:
                U[d, t] = random.choice([0, 1])
        return U

    def ToBinList(self, U):
        bin_list = [U[(d, t)] for d in RNGDvc for t in RNGT]
        return bin_list

    def Solve(self):
        self.model.optimize()
        return self.model.ObjVal

class GA:
    def __init__(self, gen_size, gen_count, selection_size, first_gen, mut_prob, cros_prob, cros_prob_det, cros_len, parents_save):
        # gen_size: size of the generation  |  gen_count:  number of iteration  |  selection_size: count of parent pairs
        # first_gen: first generation (population)  |  mut_prob: mutation probability  |  cros_prob: crossover probability
        # parents_save: parents to be saved for the next generation
        self.gen_size = gen_size
        self.generation = first_gen
        self.generation_fitness = []
        self.gen_count = gen_count
        self.mut_prob = mut_prob
        self.cros_prob = cros_prob
        self.cros_prob_det = cros_prob_det
        self.cros_len = cros_len
        self.selection_size = selection_size
        self.parents_save = parents_save

    def RunGA(self, MIP_model):
        # First find fitness of the first generation
        self.FirstGenFitness(MIP_model)

        # Initialize the best solution
        first_best = np.argmax(self.generation_fitness)
        best_fitness_found = self.generation_fitness[first_best]
        best_solution_found = self.generation[first_best]

        # Create a list to save the generation data which is: [itr, avg(fitness), max(fitness)
        solution = {'Iteration': [],
                    'Avg Fitness': [],
                    'Best Fitness': []}
        RunTime = {'Selection':0,
                   'CrossOver': 0,
                   'Mutation': 0,
                   'OffspringFitness': 0,
                   'NextGeneration': 0}

        # Start GA
        for itr in tqdm(range(1, self.gen_count+1)):
            ratio = itr/self.gen_count

            # Select parents list/dict. The input is generation and output must be a list of pair of parents.
            tt = time.time()
            parents_pair = self.SelectionT()
            ff = time.time()
            RunTime['Selection'] += ff - tt


            # Crossover to generate offspring. The input to the function must be a list of pair of parents
            tt = time.time()
            offspring = self.CrossOver(parents_pair)
            ff = time.time()
            RunTime['CrossOver'] += ff - tt


            # Mutate to improve offspring
            tt = time.time()
            offspring_mut = self.Mutation(offspring)
            ff = time.time()
            RunTime['Mutation'] += ff - tt

            # Calculate offspring fitness
            tt = time.time()
            offspring_fitness = self.OffspringFitness(offspring_mut, MIP_model)
            ff = time.time()
            RunTime['OffspringFitness'] += ff - tt

            # Create the next generation
            tt = time.time()
            self.NextGeneration(offspring_mut, offspring_fitness)
            ff = time.time()
            RunTime['NextGeneration'] += ff - tt

            # Save the results
            avg_fitness = np.mean(self.generation_fitness)
            index_best = np.argmin(self.generation_fitness)
            best_fitness = self.generation_fitness[index_best]

            if best_fitness <= best_fitness_found:
                best_fitness_found = best_fitness
                best_solution_found = self.generation[index_best]
                solution['Iteration'].append(itr+1)
                solution['Avg Fitness'].append(avg_fitness)
                solution['Best Fitness'].append(best_fitness_found)
            else:
                solution['Iteration'].append(itr + 1)
                solution['Avg Fitness'].append(avg_fitness)
                solution['Best Fitness'].append(best_fitness_found)

        print('Run Time: ', RunTime)
        return solution, best_solution_found

    def FirstGenFitness(self, MIP_model):
        for ind in self.generation:
            self.generation_fitness.append(MIP_model.Fitness(ind))

    def OffspringFitness(self, offspring_list, MIP_model):
        offspring_fitness = []
        for U in offspring_list:
            offspring_fitness.append(MIP_model.Fitness(U))
        return offspring_fitness

    def CrossOver(self, pairs):
        offspring_list = []

        for pair in pairs:
            off1, off2 = copy.copy(pair[0]), copy.copy(pair[1])

            if random.random() <= self.cros_prob:
                # Find positions for crossover
                positions = []
                index = 0
                while T - index > cross_lenght:
                    rand = random.choice(range(1, cross_lenght+1))
                    positions.append(index + rand)
                    index = index + rand
                positions.append(T)

                # Apply crossover
                start = 0
                # positions is a list of the lenght for crossover. For example [2, 6, 10] means crossover possibility
                # for bits 0-2, 3-6, 7-10
                for p in positions:
                    if random.random() <= self.cros_prob_det:
                        temp = copy.copy(off1[start:p + 1])
                        off1[start:p + 1] = copy.copy(off2[start:p+1])
                        off2[start:p + 1] = temp
                    start = p+1

            offspring_list.append(off1)
            offspring_list.append(off2)

        return offspring_list

    def Mutation(self, offspring_list):
        for off in offspring_list:
            for index in range(len(off)):
                if random.random() < self.mut_prob:
                    off[index] = 1 - off[index]
        return offspring_list

    def SelectionT(self):
        parents = []
        while len(parents) < self.selection_size:
            pair = []
            while len(pair) < 2:
                selected = random.sample(range(len(self.generation)), 2)
                p1 = selected[0]
                p2 = selected[1]
                if self.generation_fitness[p1] < self.generation_fitness[p2]:
                    pair.append(self.generation[p1])
                else:
                    pair.append(self.generation[p2])
            parents.append(pair)
        return parents

    def NextGeneration(self, offspring_list, offspring_fitness):
        next_generation = []
        next_generation_fitness = []
        # First add the best parents
        while len(next_generation) < self.parents_save:
            best_parent_index = np.argmin(self.generation_fitness)
            next_generation.append(self.generation[best_parent_index])
            next_generation_fitness.append(self.generation_fitness[best_parent_index])

            self.generation.remove(self.generation[best_parent_index])
            self.generation_fitness.remove(self.generation_fitness[best_parent_index])

        while len(next_generation) < self.gen_size:
            best_child_index = np.argmin(offspring_fitness)
            next_generation.append(offspring_list[best_child_index])
            next_generation_fitness.append(offspring_fitness[best_child_index])

            offspring_list.remove(offspring_list[best_child_index])
            offspring_fitness.remove(offspring_fitness[best_child_index])

        # Replace the next generation
        self.generation = next_generation
        self.generation_fitness = next_generation_fitness

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    M = MIP(exact=False)
    parents_save = 4
    generation_size = 15
    selection_size = generation_size
    generation_count = 10
    mutation_probability = 1/generation_size
    crossover_probability = 0.8
    crossover_probability_detail = 1/3
    cross_lenght = 5
    trials = 1
    summary = {'Avg Avg Fitness': [0 for _ in range(generation_count)],
               'Avg Best Fitness': [0 for _ in range(generation_count)]}

    for trial in range(1, trials+1):

        lmda = lmda0 * trial/generation_count
        print(f'{100 * trial/trials:0.0f}%|{trial * ">"}{(trials-trial) * " "}| Trial {trial}/{trials}')
        U = [M.ToBinList(M.GetInitialInd()) for _ in range(generation_size)]
        ga = GA(gen_size=generation_size, gen_count=generation_count, selection_size=selection_size, first_gen=U,
                mut_prob=mutation_probability, cros_prob=crossover_probability,
                cros_prob_det=crossover_probability_detail, cros_len=cross_lenght, parents_save=parents_save)
        solution, optimal = ga.RunGA(M)


        # Recall: solution is a dictionary with keys: Iteration, Avg Fitness, Best Fitness
        summary['Avg Avg Fitness'] = np.add(summary['Avg Avg Fitness'], solution['Avg Fitness'])
        summary['Avg Best Fitness'] = np.add(summary['Avg Best Fitness'], solution['Best Fitness'])

    # Obtain the exact solution
    MM = MIP(exact=True)
    Optimal = MM.Solve()

    # Compare with exact solution
    name = f'GC({generation_count})'
    summary['Avg Avg Fitness'] = np.divide(summary['Avg Avg Fitness'], trials)
    summary['Avg Best Fitness'] = np.divide(summary['Avg Best Fitness'], trials)
    DG = 100 * abs(summary["Avg Best Fitness"][-1] - Optimal)/Optimal
    summary['Avg Duality Gap'] = [DG for _ in range(generation_count)]
    pd.DataFrame(summary).to_csv(f'Result/Summary-{name}.csv')

    fig1 = plt.figure(dpi=300)
    plt.boxplot(summary['Avg Avg Fitness'])
    plt.title(f'Avg Duality Gap: {DG:0.4f}%')
    plt.savefig(f'IMG/2BP-{name}.jpg', bbox_inches='tight')

    fig2 = plt.figure(dpi=300)
    df = pd.DataFrame(pd.DataFrame(solution))
    plt.plot(range(1, generation_count + 1), df['Best Fitness'], label='Best Fitness')
    plt.plot(range(1, generation_count + 1), df['Avg Fitness'], label='Avg Fitness')
    plt.plot([1, generation_count + 1], [Optimal, Optimal], label='Optimal Value')
    plt.legend()
    plt.title(f'Avg Duality Gap: {DG:0.4f}%')
    plt.savefig(f'IMG/2FR-{name}.jpg')






