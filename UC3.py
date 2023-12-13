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
DC = 5
RNGT = range(1, T + 1)
RNGDvc = range(1, DC + 1)


S = [4500, 550, 560, 170, 30,
     170, 260, 30, 30, 30]
S = S[:DC]
a = [1000, 970, 700, 680, 450,
     370, 480, 660, 665, 670]
a = a[:DC]
b = [16.19, 17.26, 16.60, 16.60, 19.7,
     22.26, 27.74, 25.92, 27.27, 27.79]
b = b[:DC]
c = [0.00048, 0.00031, 0.002, 0.00211, 0.00398,
     0.00712, 0.00079, 0.00413, 0.00222, 0.00173]
c = c[:DC]
Load = [700, 750, 850, 950, 1000, 1100, 1150, 1200, 1300, 1400, 1450, 1500,
        1400, 1300, 1200, 1050, 1000, 1100, 1200, 1400, 1300, 1100, 900, 800]
Pmax = [455, 455, 130, 130, 162,
        80, 85, 55, 55, 55]
Pmax = Pmax[:DC]
Pmin = [150, 150, 20, 20, 25,
        20, 25, 10, 10, 10]
Pmin = Pmin[:DC]
HotStart = [4500, 5000, 550, 560, 900,
            170, 260, 30, 30, 30]
HotStart = HotStart[:DC]
ColdStart = [9000, 10000, 1100, 1120, 1800,
             340, 520, 60, 60, 60]
ColdStart = ColdStart[:DC]
ColdStartHour = [5, 5, 4, 4, 4,
                2, 2, 0, 0, 0]
ColdStartHour = ColdStartHour[:DC]
InitialState = [0, 0, 5, 5, 6,
                3, 3, 1, 1, 1]

MinUpDown = [8, 8, 5, 5, 6,
             3, 3, 1, 1, 1]
MinUpDown = MinUpDown[:DC]
lmda0 = 10000
lmda1 = 10000


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
        for d in RNGDvc:
            for t in RNGT:
                model.addConstr(Y[(d, t)] <= Pmax[d - 1] * U[(d, t)], name='UB')
                model.addConstr(Pmin[d - 1] * U[(d, t)] <= Y[(d, t)], name='LB')

        if exact:
            self.ExactConst()
        else:
            self.RelaxedConst()

        self.C = {(d, t): a[d - 1] * Y[(d, t)] + b[d - 1] * U[(d, t)] for d in RNGDvc for t in RNGT}
        fuel_cost = quicksum(self.C[(d, t)] for d in RNGDvc for t in RNGT)
        violation_cost = lmda0 * quicksum(Load[t - 1]-quicksum(Y[(d, t)] for d in RNGDvc) for t in RNGT)
        model.setObjective(fuel_cost + violation_cost, sense=GRB.MINIMIZE)
        model.update()

    def ExactConst(self):
        self.model.addConstrs(Load[t - 1] == quicksum(self.Y[(d, t)] for d in RNGDvc) for t in RNGT)
        self.model.update()
    def RelaxedConst(self):
        self.model.addConstrs(Load[t - 1] >= quicksum(self.Y[(d, t)] for d in RNGDvc) for t in RNGT)
        self.model.update()

    @staticmethod
    def StartUpCost(ind):
        StartUpCost = 0
        for d in RNGDvc:
            counter = InitialState[d - 1]
            for t in RNGT:
                if ind[d - 1][t - 1] == 0:
                    counter += 1
                else:
                    if counter != 0:
                        if counter <= ColdStartHour[d - 1]:
                            StartUpCost += HotStart[d - 1]
                        else:
                            StartUpCost += ColdStart[d - 1]
                    counter = 0
        return StartUpCost

    @staticmethod
    def MinUpDownCost(ind):
        MinUpDownCost = 0
        UUWhere1 = []
        UUSubtracted = [[0] for _ in RNGDvc]
        for d in RNGDvc:
            for t in range(1, T):
                UUSubtracted[d - 1].append(abs(ind[d - 1][t] - ind[d - 1][t - 1]))
            UUWhere1.append(np.where(np.array(UUSubtracted[d - 1]) == 1)[0])  # output is the index where status changes
        for d in RNGDvc:
            pre = 0
            for dur in UUWhere1[d - 1]:
                if dur - pre > MinUpDown[d - 1]:
                    MinUpDownCost += lmda1
                pre = dur
        return MinUpDownCost


    def FuelCost(self, ind):
        for d in RNGDvc:
            for t in RNGT:
                self.model.addConstr(self.U[(d, t)] == ind[d - 1][t - 1], name=f'U[{d},{t}]')

        self.model.update()
        self.model.optimize()
        FuelCost = np.sum([a[d - 1] * self.Y[(d, t)].x + b[d - 1] * ind[d-1][t-1] for d in RNGDvc for t in RNGT])

        for d in RNGDvc:
            for t in RNGT:
                self.model.remove(self.model.getConstrByName(f'U[{d},{t}]'))
        self.model.update()
        return FuelCost

    def Fitness(self, ind):
        UU = [ind[(d-1)*24:d*24] for d in RNGDvc]

        # Calculate startup costs
        StartUpCost = self.StartUpCost(UU)

        # Violation of min up/down hours
        MinUpDownCost = self.MinUpDownCost(UU)

        # Add fixed constraint
        for d in RNGDvc:
            for t in RNGT:
                self.model.addConstr(self.U[(d, t)] == UU[d-1][t-1], name=f'U[{d},{t}]')

        self.model.update()
        self.model.optimize()
        obj = StartUpCost + MinUpDownCost + self.model.ObjVal

        for d in RNGDvc:
            for t in RNGT:
                self.model.remove(self.model.getConstrByName(f'U[{d},{t}]'))

        self.model.update()
        return obj

    @staticmethod
    def GetInitialInd():
        Init = {}
        for d in RNGDvc:
            for t in RNGT:
                Init[d, t] = random.choice([0, 1])
        return Init

    @staticmethod
    def ToBinList(string):
        bin_list = [string[(d, t)] for d in RNGDvc for t in RNGT]
        return bin_list

    def Solve(self):
        self.model.optimize()
        return self.model.ObjVal


class GA:
    def __init__(self, gen_size, gen_count, select_size, first_gen, mut_prob, cros_prob, cros_prob_det, cros_len,
                 parent_save, selection):
        self.gen_size = gen_size
        self.generation = first_gen
        self.generation_fitness = []
        self.gen_count = gen_count
        self.mut_prob = mut_prob
        self.cros_prob = cros_prob
        self.cros_prob_det = cros_prob_det
        self.cros_len = cros_len
        self.selection_size = select_size
        self.parent_save = parent_save
        self.selection = selection

    def RunGA(self, MIP_model):
        # First find fitness of the first generation
        self.FirstGenFitness(MIP_model)

        # Initialize the best solution
        first_best = np.argmin(self.generation_fitness)
        best_fitness_found = self.generation_fitness[first_best]
        best_solution_found = self.generation[first_best]

        # Create a list to save the generation data which is: [itr, avg(fitness), max(fitness)
        SOL = {'Iteration': [],
                'Avg Fitness': [],
                'Best Fitness': []}

        # Start GA
        for itr in tqdm(range(1, self.gen_count+1)):
            # Select parents list/dict. The input is generation and output must be a list of pair of parents.

            if self.selection == 'T':
                parents_pair = self.SelectionT()
            elif self.selection == 'R':
                parents_pair = self.SelectionR()
            else:
                raise ValueError('Selection must be one of "T" or "R"')
            # Crossover to generate offspring. The input to the function must be a list of pair of parents
            offspring = self.CrossOver(parents_pair)

            # Mutate to improve offspring
            offspring_mut = self.Mutation(offspring)

            # Calculate offspring fitness
            offspring_fitness = self.OffspringFitness(offspring_mut, MIP_model)

            # Create the next generation
            self.NextGeneration(offspring_mut, offspring_fitness)

            # Save the results
            avg_fitness = np.mean(self.generation_fitness)
            index_best = np.argmin(self.generation_fitness)
            best_fitness = self.generation_fitness[index_best]
            best_solution = self.generation[index_best]

            if best_fitness <= best_fitness_found:
                best_fitness_found = best_fitness
                best_solution_found = best_solution
                SOL['Iteration'].append(itr+1)
                SOL['Avg Fitness'].append(avg_fitness)
                SOL['Best Fitness'].append(best_fitness_found)
            else:
                SOL['Iteration'].append(itr + 1)
                SOL['Avg Fitness'].append(avg_fitness)
                SOL['Best Fitness'].append(best_fitness_found)

        return SOL, best_solution_found, best_fitness_found

    def FirstGenFitness(self, MIP_model):
        for ind in self.generation:
            self.generation_fitness.append(MIP_model.Fitness(ind))

    @staticmethod
    def OffspringFitness(offspring_list, MIP_model):
        offspring_fitness = []
        for ind in offspring_list:
            offspring_fitness.append(MIP_model.Fitness(ind))
        return offspring_fitness

    def CrossOver(self, pairs):
        offspring_list = []

        for pair in pairs:
            off1, off2 = copy.copy(pair[0]), copy.copy(pair[1])

            if random.random() <= self.cros_prob:
                # Find positions for crossover
                positions = []
                index = 0
                while T*DC - index > cross_lenght:
                    rand = random.choice(range(1, cross_lenght+1))
                    positions.append(index + rand)
                    index = index + rand
                positions.append(T*DC)

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

    def SelectionR(self):
        generation_fitness = [0]
        for fit in self.generation_fitness:
            generation_fitness.append(fit)
        fitness_sum = sum(generation_fitness)

        # cum_probability is used to select pairs
        probability = [i / fitness_sum for i in generation_fitness][::-1]
        cum_probability = np.cumsum(probability)

        # Create list to save pairs of parents
        parents = []

        # To avoid reference before assignment alert, I define these two variables.
        parent1, parent2 = self.generation[0], self.generation[1]
        p1 = 0
        while len(parents) < self.selection_size:
            # First, find the fist parent for the parent pair
            rand1 = random.random()
            for j in range(len(self.generation)):
                if cum_probability[j] <= rand1:
                    if rand1 <= cum_probability[j + 1]:
                        parent1 = self.generation[j]
                        p1 = j  # save the fist parent
                        break

            # Then find the second parent for the pair
            rand2 = random.random()
            for j in range(len(self.generation)):
                if cum_probability[j] <= rand2:
                    if rand2 <= cum_probability[j + 1]:
                        # if the second parent is not the same as the first, select it and break.
                        if j != p1:
                            parent2 = self.generation[j]
                            break
            parents.append([parent1, parent2])
        return parents

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
        while len(next_generation) < self.parent_save:
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
    parents_save = 4
    generation_size = 15
    selection_size = 15
    generation_count = 500
    mutation_probability = 0.05
    crossover_probability = 0.9
    crossover_probability_detail = 0.3
    pair_select = 'T'  # 'T' or 'R'
    cross_lenght = 5
    trials = 20
    avg_fitness = {'Avg Avg Fitness': [0 for _ in range(generation_count)],
               'Avg Best Fitness': [0 for _ in range(generation_count)]}
    trials_result = {'Final Cost': [],
                     'Final UC': [],
                     'Run Time': [],
                     'StartUp': [],
                     'MinUpDown': [],
                     'Fuel': [],
                     'Demand Viol': []}

    for trial in range(1, trials+1):
        M = MIP(exact=False)
        # Save start time
        s = time.time()

        print(f'{100 * trial/trials:0.0f}%|{trial * ">"}{(trials-trial) * " "}| Trial {trial}/{trials}')
        U = [M.ToBinList(M.GetInitialInd()) for _ in range(generation_size)]
        ga = GA(gen_size=generation_size, gen_count=generation_count,
                select_size=selection_size, first_gen=U,
                mut_prob=mutation_probability, cros_prob=crossover_probability,
                cros_prob_det=crossover_probability_detail, cros_len=cross_lenght,
                parent_save=parents_save, selection=pair_select)
        solution, best_uc, best_cost = ga.RunGA(M)

        # Calculate Run time
        TRT = time.time() - s

        # Recall: solution is a dictionary with keys: Iteration, Avg Fitness, Best Fitness
        avg_fitness['Avg Avg Fitness'] = np.add(avg_fitness['Avg Avg Fitness'], solution['Avg Fitness'])
        avg_fitness['Avg Best Fitness'] = np.add(avg_fitness['Avg Best Fitness'], solution['Best Fitness'])
        trials_result['Final Cost'].append(best_cost)
        trials_result['Final UC'].append(best_uc)
        trials_result['Run Time'].append(TRT)

        uc_list = [best_uc[(d-1)*24:d*24] for d in RNGDvc]
        SUC = M.StartUpCost(uc_list)
        trials_result['StartUp'].append(SUC)

        MUDC = M.MinUpDownCost(uc_list)
        trials_result['MinUpDown'].append(MUDC)

        FC = M.FuelCost(uc_list)
        trials_result['Fuel'].append(FC)

        trials_result['Demand Viol'].append((best_cost - (SUC+MUDC+FC))/lmda0)

    name = f'GC({generation_count}){pair_select}'
    avg_fitness['Avg Avg Fitness'] = np.divide(avg_fitness['Avg Avg Fitness'], trials)
    avg_fitness['Avg Best Fitness'] = np.divide(avg_fitness['Avg Best Fitness'], trials)

    with open(f'Result/{name}.pkl', 'wb') as handle:
        pickle.dump([trials_result, avg_fitness], handle)
    handle.close()
    pd.DataFrame(trials_result).to_csv(f'Result/TR-{name}.csv')








