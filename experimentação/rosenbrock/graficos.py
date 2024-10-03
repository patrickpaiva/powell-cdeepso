import os
import pandas as pd
import matplotlib.pyplot as plt

def ler_dados_excel(file_path):
    df = pd.read_excel(file_path, sheet_name='Dados')
    return df['best_fitness']

dimensoes = [30, 50, 100]
algoritmos = ['cdeepso', 'pcdeepso']
function = 'rosen'

current_dir = os.path.dirname(__file__)
        

base_path = os.path.join(current_dir, 'Testes_max_fun_calls')

def generate_graphics():
    for dim in dimensoes:
        resultados = {algoritmo: [] for algoritmo in algoritmos}
        
        for algoritmo in algoritmos:
            file_name = f'experimento_{function}_{dim}_dimensoes_{algoritmo}.xlsx'
            file_path = os.path.join(base_path, file_name)
            
            best_fitness = ler_dados_excel(file_path)
            resultados[algoritmo] = best_fitness
        
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.boxplot([resultados['cdeepso'], resultados['pcdeepso']], labels=['cdeepso', 'powell_cdeepso'])

        ax.set_title(f'Rosenbrock {dim} dimensões')
        ax.set_ylabel('Best Fitness')

        output_file = os.path.join(base_path, f'boxplot_comparacao_{dim}.png')
        plt.savefig(output_file)

        plt.show()

if __name__ == "__main__":
    generate_graphics()