import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


with open('Result/GC(50)R.pkl', 'rb') as handle:
    [TR50R, AF50R] = pickle.load(handle)
handle.close()
with open('Result/GC(50)T.pkl', 'rb') as handle:
    [TR50T, AF50T] = pickle.load(handle)
handle.close()

fig1 = plt.figure(dpi=300)
plt.plot(AF50R['Avg Avg Fitness'], label='Roulette-Mean Fitness', color='orange', linestyle=':')
plt.plot(AF50R['Avg Best Fitness'], label='Roulette-Best Fitness', color='orange')
plt.plot(AF50T['Avg Avg Fitness'], label='Tornament-Mean Fitness', color='green', linestyle='--')
plt.plot(AF50T['Avg Best Fitness'], label='Tornament-Best Fitness', color='green')
plt.grid()
plt.legend()
plt.xlabel('Generation')
plt.ylabel('Objective Function Cost ($)')
plt.savefig(f'IMG/50.jpg', bbox_inches='tight')
# ======================================================================================================================
with open('Result/GC(100)R.pkl', 'rb') as handle:
    [TR100R, AF100R] = pickle.load(handle)
handle.close()
with open('Result/GC(100)T.pkl', 'rb') as handle:
    [TR100T, AF100T] = pickle.load(handle)
handle.close()

fig1 = plt.figure(dpi=300)
plt.plot(AF100R['Avg Avg Fitness'], label='Roulette-Mean Fitness', color='orange', linestyle=':')
plt.plot(AF100R['Avg Best Fitness'], label='Roulette-Best Fitness', color='orange')
plt.plot(AF100T['Avg Avg Fitness'], label='Tornament-Mean Fitness', color='green', linestyle='--')
plt.plot(AF100T['Avg Best Fitness'], label='Tornament-Best Fitness', color='green')
plt.grid()
plt.legend()
plt.xlabel('Generation')
plt.ylabel('Objective Function Cost ($)')
plt.savefig(f'IMG/100.jpg', bbox_inches='tight')

# ======================================================================================================================
with open('Result/GC(500)R.pkl', 'rb') as handle:
    [TR500R, AF500R] = pickle.load(handle)
handle.close()
with open('Result/GC(500)T.pkl', 'rb') as handle:
    [TR500T, AF500T] = pickle.load(handle)
handle.close()

fig2 = plt.figure(dpi=300)
plt.plot(AF500R['Avg Avg Fitness'], label='Roulette-Mean Fitness', color='orange', linestyle=':')
plt.plot(AF500R['Avg Best Fitness'], label='Roulette-Best Fitness', color='orange')
plt.plot(AF500T['Avg Avg Fitness'], label='Tornament-Mean Fitness', color='green', linestyle='--')
plt.plot(AF500T['Avg Best Fitness'], label='Tornament-Best Fitness', color='green')
plt.grid()
plt.legend()
plt.xlabel('Generation')
plt.ylabel('Objective Function Cost ($)')
plt.savefig(f'IMG/500.jpg', bbox_inches='tight')

# ======================================================================================================================

