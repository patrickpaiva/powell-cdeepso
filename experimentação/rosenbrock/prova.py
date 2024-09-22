import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scipy.optimize import rosen
import numpy as np
import pandas as pd
import ast

df = pd.read_excel('experimento_rosen_100_dimensoes_pcdeepso.xlsx')

global_best = df['global_best'].iloc[0]

global_best = ast.literal_eval(global_best)

valor = rosen(global_best)

print(f"Prova do Fitness: {valor}")