import copy
import pickle
import random
import math
import numpy as np
import pandas as pd
import gurobipy
from gurobipy import quicksum, GRB
import time
env = gurobipy.Env()
env.setParam('OutputFlag', 0)

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
        T = 15
        DVCCount = 3
        MCount = 1
        HCount = 2
        OutageStart = 4
        RNGDvc = range(1, DVCCount + 1)
        RNGTime = range(1, T + 1)
        RNGTimeMinus = range(1, T)
        RNGMonth = range(1, MCount + 1)
        RNGHouse = range(1, HCount + 1)

        self.RNGTime, self.RNGMonth = RNGTime, RNGMonth

        # Define the load profiles and PV profiles
        load = [load1, load2]
        Load = {(h, t, g): load[h-1][f'Month {g}'].iloc[t - 1] for h in RNGHouse for t in RNGTime for g in RNGMonth}
        PV_unit = {(t, g): pv[f'Month {g}'].iloc[t - 1] for t in RNGTime for g in RNGMonth}
        Out_Time = {g: [OutageStart+i for i in range(3)] for g in RNGMonth}

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
        Y_indices = [(t, g) for t in RNGTime for g in RNGMonth]
        Yh_indices = [(h, t, g) for h in RNGHouse for t in RNGTime for g in RNGMonth ]
        Ytg_indices = [(t, g) for t in RNGTime for g in RNGMonth]

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
        model.addConstrs(E[(1, g)] == SOC_UB * X[1] for g in RNGMonth)
        model.addConstrs(E[(1, g)] == E[(T, g)] for g in RNGMonth)
        model.addConstrs(SOC_LB * X[1] <= E[(t, g)] for t in RNGTime for g in RNGMonth)
        model.addConstrs(E[(t, g)] <= SOC_UB * X[1] for t in RNGTime for g in RNGMonth)

        # Balance of power flow
        model.addConstrs(E[(t + 1, g)] == E[(t, g)] +
                         ES_gamma * (Y_PVES[(t, g)] + Y_DGES[(t, g)] + Eta_c * Y_GridES[(t, g)]) -
                         Eta_i * (Y_ESL[(t, g)] + Y_ESGrid[(t, g)]) / ES_gamma
                         for t in RNGTimeMinus for g in RNGMonth)

        # The share of Load
        model.addConstrs(quicksum(Load[(h, t, g)] for h in RNGHouse) >=
                         Eta_i * (Y_ESL[(t, g)] + Y_DGL[(t, g)] + Y_PVL[(t, g)]) + Y_GridL[(t, g)]
                         for t in RNGTime  for g in RNGMonth)

        model.addConstrs(Y_LH[(h, t, g)] <= Load[(h, t, g)]
                         for h in RNGHouse for t in RNGTime  for g in RNGMonth)

        model.addConstrs(quicksum(Y_LH[(h, t, g)] for h in RNGHouse) ==
                         Eta_i * (Y_ESL[(t, g)] + Y_DGL[(t, g)] + Y_PVL[(t, g)]) + Y_GridL[(t, g)]
                         for t in RNGTime  for g in RNGMonth)

        model.addConstrs(Y_PVL[(t, g)] + Y_PVES[(t, g)] + Y_PVCur[(t, g)] + Y_PVGrid[(t, g)] == PV_unit[(t, g)] * X[2]
                         for t in RNGTime  for g in RNGMonth)

        model.addConstrs(Y_GridPlus[(t, g)] == Y_GridES[(t, g)] + Y_GridL[(t, g)]
                         for t in RNGTime  for g in RNGMonth)

        model.addConstrs(Y_GridMinus[(t, g)] == Eta_i * (Y_ESGrid[(t, g)] + Y_PVGrid[(t, g)] + Y_DGGrid[(t, g)])
                         for t in RNGTime  for g in RNGMonth)

        model.addConstrs(Y_DGL[(t, g)] + Y_DGES[(t, g)] + Y_DGGrid[(t, g)] + Y_DGCur[(t, g)] == X[3]
                         for t in RNGTime  for g in RNGMonth)

        model.addConstrs(Y_ESL[(t, g)] + Y_ESGrid[(t, g)] <= UB[0] * u[(t, g)]
                         for t in RNGTime  for g in RNGMonth)

        model.addConstrs(Y_PVES[(t, g)] + Y_GridES[(t, g)] + Y_DGES[(t, g)] <= UB[0] * (1 - u[(t, g)])
                         for t in RNGTime  for g in RNGMonth)

        for g in RNGMonth:
            model.addConstrs(Y_GridPlus[(t, g)] == 0 for t in Out_Time[g])
            model.addConstrs(Y_GridMinus[(t, g)] == 0 for t in Out_Time[g])
        model.update()

        # Objective
        Cost1 = quicksum(X[d] * CO[d] for d in RNGDvc)
        Cost2 = PVCurPrice * quicksum((Y_PVCur[(t, g)] + Y_DGCur[(t, g)])
                         for t in RNGTime for g in RNGMonth)

        Cost3 = quicksum(VoLL[h - 1] * (Load[(h, t, g)] - Y_LH[(h, t, g)])
                          for h in RNGHouse for t in RNGTime for g in RNGMonth)

        Cost4 = FuelPrice * DG_gamma * quicksum(Y_DGL[(t, g)] + Y_DGGrid[(t, g)] + Y_DGCur[(t, g)] + Y_DGES[(t, g)]
                                                for t in RNGTime for g in RNGMonth)

        Cost5 = quicksum(GridPlus * Y_GridPlus[(t, g)] - GridMinus * Y_GridMinus[(t, g)] -
                                              GenerPrice * X[2] * PV_unit[(t, g)] - quicksum(LoadPrice * Y_LH[(h, t, g)]
                                                                                             for h in RNGHouse)
                        for t in RNGTime for g in RNGMonth)
        model.setObjective(Cost1 + (12/MCount) * (31*24/T) * (Cost2 + Cost3 + Cost4 + Cost5), sense=GRB.MINIMIZE)
        model.update()
        self.model = model

    def UpdateCommitment(self, U):
        model = self.model
        for g in self.RNGMonth:
            for t in self.RNGTime:
                model.addConstr(self.u[(t, g)] == U[(t, g)], name=f'U[({t}, {g})]')

        self.model.update()
        self.model.optimize()
        obj = self.model.ObjVal
        print(self.X)

        for g in self.RNGMonth:
            for t in self.RNGTime:
                self.model.remove(self.model.getConstrByName(f'U[({t}, {g})]'))
        self.model.update()
        return obj

    def GetInitialInd(self):
        U = {}
        for g in self.RNGMonth:
            for t in self.RNGTime:
                U[(t, g)] = random.choice([0, 1])
        return U


def GA(maxItr, gen_count, first_pop):

    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    M = MIP()
    U = M.GetInitialInd()
    print(M.UpdateCommitment(U))


