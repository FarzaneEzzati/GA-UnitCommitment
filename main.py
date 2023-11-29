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


T = 20
RNGT = range(1, T+1)

def OpenData():
    # Load profile of households
    load_profile_1 = pd.read_csv('Data/Load_profile_1.csv')
    load_profile_2 = pd.read_csv('Data/Load_profile_2.csv')

    # PV output for one unit (4kW)
    pv_profile = pd.read_csv('Data/PV_profiles.csv')
    return load_profile_1, load_profile_2, pv_profile


class MIP:
    def __init__(self):
        # write the model
        load1, load2, pv = OpenData()
        # Initialization of parameters
        Budget = 100000
        Horizon = 20
        interest_rate = 0.08
        operational_rate = 0.01
        PA_factor = ((1 + interest_rate) ** Horizon - 1) / (interest_rate * (1 + interest_rate) ** Horizon)
        C = {1: 600, 2: 2780 / 4, 3: 150}
        O = {i: operational_rate * PA_factor * C[i] for i in (1, 2, 3)}
        CO = {i: (C[i] + O[i]) / (365 * 24) for i in (1, 2, 3)}
        UB = [166, 80, 40]
        LB = [20, 10, 2]
        FuelPrice = 3.7
        alpha, beta = 0.5, 0.2
        GridPlus = 0.1497
        GridMinus = alpha * GridPlus
        LoadPrice = GridPlus
        GenerPrice = beta * GridPlus
        VoLL = np.array([2.1, 1.8, 1.4]) * GridPlus
        PVSellPrice = (alpha + beta) * GridPlus
        DGSellPrice = PVSellPrice
        PVCurPrice = (alpha + beta) * GridPlus
        DGCurPrice = (alpha + beta) * GridPlus
        SOC_UB, SOC_LB = 0.9, 0.1
        ES_gamma = 0.85
        DG_gamma = 0.4
        Eta_c = 0.8
        Eta_i = 0.9

        # Ranges need to be used
        DVCCount = 3
        HCount = 2
        OutageStart = 4
        RNGDvc = range(1, DVCCount + 1)
        RNGTime = range(1, T + 1)
        RNGTimeMinus = range(1, T)
        RNGHouse = range(1, HCount + 1)

        self.RNGTime = RNGTime

        # Define the load profiles and PV profiles
        load = [load1, load2]
        Load = {(h, t): load[h-1][f'Month {1}'].iloc[t - 1] for h in RNGHouse for t in RNGTime}
        PV_unit = {t: pv[f'Month {1}'].iloc[t - 1] for t in RNGTime}
        Out_Time = [OutageStart+i for i in range(3)]

        # Build the model
        model = gurobipy.Model('MIP', env=env)
        X_indices = [j for j in RNGDvc]  # 1: ES, 2: PV, 3: DG
        X = model.addVars(X_indices, vtype=GRB.INTEGER, name='X')
        self.X = X

        # Bounds on X decisions
        model.addConstrs(X[d] <= UB[d - 1] for d in RNGDvc)
        model.addConstrs(X[d] >= LB[d - 1] for d in RNGDvc)

        model.addConstr(quicksum([X[j] * C[j] for j in RNGDvc]) <= Budget, name='budget constraint')

        # Second Stage Variables
        Y_indices = [t for t in RNGTime]
        Yh_indices = [(h, t) for h in RNGHouse for t in RNGTime]
        Ytg_indices = [t for t in RNGTime]

        Y_PVES = model.addVars(Y_indices, name='Y_PVES')
        Y_DGES = model.addVars(Y_indices, name='Y_DGES')
        Y_GridES = model.addVars(Y_indices, name='Y_GridES')

        Y_PVL = model.addVars(Y_indices, name='Y_PVL')
        Y_DGL = model.addVars(Y_indices, name='Y_DGL')
        Y_ESL = model.addVars(Y_indices, name='Y_ESL')
        Y_GridL = model.addVars(Y_indices, name='Y_GridL')

        Y_LH = model.addVars(Yh_indices, name='Y_LH')
        Y_LL = model.addVars(Yh_indices, name='Y_LL')

        Y_PVCur = model.addVars(Y_indices, name='Y_PVCur')
        Y_DGCur = model.addVars(Y_indices, name='Y_DGCur')

        Y_PVGrid = model.addVars(Y_indices, name='Y_DGGrid')
        Y_DGGrid = model.addVars(Y_indices, name='Y_DGGrid')
        Y_ESGrid = model.addVars(Y_indices, name='Y_ESGrid')

        Y_GridPlus = model.addVars(Y_indices, name='Y_GridPlus')
        Y_GridMinus = model.addVars(Y_indices, name='Y_GridMinus')

        E = model.addVars(Y_indices, name='E')

        u = model.addVars(Y_indices, vtype=GRB.BINARY, name='u')
        self.u = u

        # Energy storage level
        model.addConstr(E[1] == SOC_UB * X[1])
        model.addConstr(E[1] == E[T])
        model.addConstrs(SOC_LB * X[1] <= E[t] for t in RNGTime)
        model.addConstrs(E[t] <= SOC_UB * X[1] for t in RNGTime)

        # Balance of power flow
        model.addConstrs(E[t + 1] == E[t] +
                         ES_gamma * (Y_PVES[t] + Y_DGES[t] + Eta_c * Y_GridES[t]) -
                         Eta_i * (Y_ESL[t] + Y_ESGrid[t]) / ES_gamma
                         for t in RNGTimeMinus)

        # The share of Load
        model.addConstrs(quicksum(Load[(h, t)] for h in RNGHouse) >=
                         Eta_i * (Y_ESL[t] + Y_DGL[t] + Y_PVL[t]) + Y_GridL[t]
                         for t in RNGTime)

        model.addConstrs(Y_LH[(h, t)] <= Load[(h, t)]
                         for h in RNGHouse for t in RNGTime)

        model.addConstrs(quicksum(Y_LH[(h, t)] for h in RNGHouse) ==
                         Eta_i * (Y_ESL[t] + Y_DGL[t] + Y_PVL[t]) + Y_GridL[t]
                         for t in RNGTime)

        model.addConstrs(Y_PVL[t] + Y_PVES[t] + Y_PVCur[t] + Y_PVGrid[t] == PV_unit[t] * X[2]
                         for t in RNGTime)

        model.addConstrs(Y_GridPlus[t] == Y_GridES[t] + Y_GridL[t]
                         for t in RNGTime)

        model.addConstrs(Y_GridMinus[t] == Eta_i * (Y_ESGrid[t] + Y_PVGrid[t] + Y_DGGrid[t])
                         for t in RNGTime)

        model.addConstrs(Y_DGL[t] + Y_DGES[t] + Y_DGGrid[t] + Y_DGCur[t] == X[3]
                         for t in RNGTime)

        model.addConstrs(Y_ESL[t] + Y_ESGrid[t] <= UB[0] * u[t]
                         for t in RNGTime)

        model.addConstrs(Y_PVES[t] + Y_GridES[t] + Y_DGES[t] <= UB[0] * (1 - u[t])
                         for t in RNGTime)

        model.addConstrs(Y_GridPlus[t] == 0 for t in Out_Time)
        model.addConstrs(Y_GridMinus[t] == 0 for t in Out_Time)
        model.update()

        # Objective
        Cost1 = quicksum(X[d] * CO[d] for d in RNGDvc)
        Cost2 = PVCurPrice * quicksum((Y_PVCur[t] + Y_DGCur[t])
                         for t in RNGTime)

        Cost3 = quicksum(VoLL[h - 1] * (Load[(h, t)] - Y_LH[(h, t)])
                          for h in RNGHouse for t in RNGTime)

        Cost4 = FuelPrice * DG_gamma * quicksum(Y_DGL[t] + Y_DGGrid[t] + Y_DGCur[t] + Y_DGES[t]
                                                for t in RNGTime)

        Cost5 = quicksum(GridPlus * Y_GridPlus[t] - GridMinus * Y_GridMinus[t] -
                                              GenerPrice * X[2] * PV_unit[t] - quicksum(LoadPrice * Y_LH[(h, t)]
                                                                                             for h in RNGHouse)
                        for t in RNGTime)
        model.setObjective(Cost1 + 12 * (31*24/T) * (Cost2 + Cost3 + Cost4 + Cost5), sense=GRB.MINIMIZE)
        model.update()
        self.model = model

    def UpdateCommitment(self, U):
        model = self.model
        for t in self.RNGTime:
            model.addConstr(self.u[t] == U[t], name=f'U[{t}]')

        self.model.update()
        self.model.optimize()
        obj = self.model.ObjVal

        for t in self.RNGTime:
            self.model.remove(self.model.getConstrByName(f'U[{t}]'))
        self.model.update()
        return obj

    def GetInitialInd(self):
        U = {}
        for t in self.RNGTime:
            U[t] = random.choice([0, 1])
        return U

    def ToBinList(self, U):
        bin_list = [U[i] for i in RNGT]
        return bin_list

    def TDict(self, U):
        bin_dict = {t: int(U[t-1]) for t in RNGT}
        return bin_dict


class GA:
    def __init__(self, gen_size, gen_count, selection_size, first_gen, mut_prob, cros_prob, parents_save):
        # gen_size: size of the generation  |  gen_count:  number of iteration  |  selection_size: count of parent pairs
        # first_gen: first generation (population)  |  mut_prob: mutation probability  |  cros_prob: crossover probability
        # parents_save: parents to be saved for the next generation
        self.gen_size = gen_size
        self.generation = first_gen
        self.generation_fitness = []
        self.gen_count = gen_count
        self.mut_prob = mut_prob
        self.cros_prob = cros_prob
        self.selection_size = selection_size
        self.parents_save = parents_save

    def RunGA(self, MIP_model):
        # First find fitness of the first generation
        self.FirstGenFitness(MIP_model)

        # Initialize the best solution
        first_best = np.argmin(self.generation_fitness)
        best_fitness_found = self.generation_fitness[first_best]
        best_solution_found = self.generation[first_best]

        # Create a list to save the generation data which is: [itr, avg(fitness), min(fitness), argmin(fitness)
        solution = []
        start_time = time.time()

        # Start GA
        for itr in tqdm(range(self.gen_count)):
            # Select parents list/dict. The input is generation and output must be a list of pair of parents.
            parents_pair = self.Selection(MIP_model)

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
            if best_fitness <= best_fitness_found:
                best_fitness_found = best_fitness
                best_solution_found = self.generation[index_best]
                solution.append([itr+1, avg_fitness, best_fitness_found])
            else:
                solution.append([itr + 1, avg_fitness, best_fitness_found])

        finish_time = time.time()-start_time
        return solution, best_solution_found, finish_time

    def FirstGenFitness(self, MIP_model):
        for ind in self.generation:
            U = {t: ind[t - 1] for t in RNGT}
            self.generation_fitness.append(MIP_model.UpdateCommitment(U))

    def OffspringFitness(self, offspring_list, MIP_model):
        offspring_fitness = []
        for ind in offspring_list:
            U = {t: ind[t - 1] for t in RNGT}
            offspring_fitness.append(MIP_model.UpdateCommitment(U))
        return offspring_fitness

    def CrossOver(self, pairs):
        offspring_list = []

        for pair in pairs:
            off1, off2 = copy.copy(pair[0]), copy.copy(pair[1])

            if random.random() < self.cros_prob:
                # Find positions for crossover
                position_counts = math.floor(T/4)
                positions = []
                while len(positions) < position_counts:
                    r = random.choice(range(T))
                    if r not in positions:
                        positions.append(r)  # Note that it states the position for crossover and the position is between 0-T
                positions.sort()

                if T-1 not in positions:
                    positions.append(T-1)

                # Apply crossover
                start = 0
                crossing = random.choice([0, 1])
                for p in positions:
                    if crossing == 1:
                        temp = copy.copy(off1[start:p+1])
                        off1[start:p+1] = copy.copy(off2[start:p+1])
                        off2[start:p + 1] = temp
                        crossing = 0
                    else:
                        crossing = 1
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

    def Selection(self, MIP_model):
        # Calculating the fitness for all individuals in the generation
        # fitness list would be: [0, f1, f2, ..., fN]
        generation_fitness = [0]
        for fit in self.generation_fitness:
            generation_fitness.append(fit)
        fitness_sum = sum(generation_fitness)

        # cum_probability is used to select pairs
        probability = [i / fitness_sum for i in generation_fitness]
        cum_probability = np.cumsum(probability)

        # Create list to save pairs of parents
        parents = []

        # To avoid reference before assignment alert, I define these two variables.
        parent1, parent2 = self.generation[0], self.generation[1]
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
    M = MIP()
    parents_save = 2
    generation_size = 10
    selection_size = generation_size - parents_save
    generation_count = 200
    mutation_probability = 1/generation_size
    crossover_probability = 0.9

    U = [M.ToBinList(M.GetInitialInd()) for _ in range(generation_size)]

    ga = GA(gen_size=generation_size, gen_count=generation_count, selection_size=selection_size, first_gen=U,
            mut_prob=mutation_probability, cros_prob=crossover_probability, parents_save=parents_save)
    solution, optimal, runtime = ga.RunGA(M)
    pd = pd.DataFrame(solution, columns=['itr', 'avg-fitness', 'best-fitness'])
    plt.plot(pd['best-fitness'], label='Best Fitness')
    plt.plot(pd['avg-fitness'], label='Avg Fitness')
    plt.legend()
    plt.savefig('Fitness-Reduction.jpg')



