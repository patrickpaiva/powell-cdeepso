import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scipy.optimize import rosen
import numpy as np
import pandas as pd
import ast

current_dir = os.path.dirname(__file__)

file_path = os.path.join(current_dir, 'Testes_max_fun_calls' , 'experimento_rosen_100_dimensoes_pcdeepso.xlsx')

df = pd.read_excel(file_path)

global_best = df['global_best'].iloc[0]

global_best = ast.literal_eval(global_best)

valor = rosen(global_best)

print(f"Prova do Fitness: {valor}")

