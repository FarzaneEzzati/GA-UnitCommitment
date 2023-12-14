import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


def BoxPlot(fileR, fileT, GC, type, labels):
    fig = plt.figure(figsize=(4, 4), dpi=300)
    plt.boxplot([fileR['Final Cost'], fileT['Final Cost']], labels=[labels[0], labels[1]],
                notch=False,  # notch shape
                vert=True,  # vertical box alignment
                patch_artist=True  # fill with color
                )

    plt.ylabel('Generation Best Fitness ($)')
    plt.title(f'{GC} Generations')
    plt.savefig(f'IMG/BP-{GC}-{type}.jpg', bbox_inches='tight')


with open('Result/GC(200)T-MP(0.05).pkl', 'rb') as handle:
    [TR, AF] = pickle.load(handle)
handle.close()
with open('Result/GC(200)T-MP(0.15).pkl', 'rb') as handle:
    [TR1, AF1] = pickle.load(handle)
handle.close()
BoxPlot(TR, TR1, 200, 'MP', ['Mutation Prob 0.05', 'Mutation Prob 0.15'])
# ======================================================================================================================
with open('Result/GC(200)T-CP(0.3).pkl', 'rb') as handle:
    [TR, AF] = pickle.load(handle)
handle.close()
with open('Result/GC(200)T-CP(0.5).pkl', 'rb') as handle:
    [TR1, AF1] = pickle.load(handle)
handle.close()
BoxPlot(TR, TR1, 200, 'CP', ['Crossover Prob 0.3', 'Crossover Prob 0.5'])
# ======================================================================================================================
with open('Result/GC(200)T-CP(0.3).pkl', 'rb') as handle:
    [TR, AF] = pickle.load(handle)
handle.close()
with open('Result/GC(200)R-CP(0.3).pkl', 'rb') as handle:
    [TR1, AF1] = pickle.load(handle)
handle.close()
BoxPlot(TR, TR1, 200, 'SM', ['Tournament', 'Roulette'])
# ======================================================================================================================

