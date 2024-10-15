import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

def ler_dados_excel(file_path, sheet_name='Dados'):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    if sheet_name == 'Dados':
        return df['best_fitness']
    elif sheet_name == 'Convergencia_Media':
        return df['Convergencia_Media']

dimensoes = [30, 50, 100]
algoritmos = ['cdeepso', 'pcdeepso']
function = 'rosen'

current_dir = os.path.dirname(__file__)

base_path = os.path.join(current_dir, 'Testes_max_fun_calls')

def generate_graphics():
    for dim in dimensoes:
        resultados = {algoritmo: [] for algoritmo in algoritmos}
        convergencias = {algoritmo: [] for algoritmo in algoritmos}

        for algoritmo in algoritmos:
            file_name = f'experimento_{function}_{dim}_dimensoes_{algoritmo}.xlsx'
            file_path = os.path.join(base_path, file_name)
            
            best_fitness = ler_dados_excel(file_path, sheet_name='Dados')
            convergencia_media = ler_dados_excel(file_path, sheet_name='Convergencia_Media')
            
            resultados[algoritmo] = best_fitness
            convergencias[algoritmo] = convergencia_media

        # Gerar Boxplot para best_fitness
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot([resultados['cdeepso'], resultados['pcdeepso']], labels=['cdeepso', 'powell_cdeepso'])
        ax.set_title(f'Rosenbrock {dim} dimensões')
        ax.set_ylabel('Best Fitness')
        output_file = os.path.join(base_path, f'boxplot_comparacao_{dim}.png')
        plt.savefig(output_file)
        plt.show()

        # Gerar gráfico de convergência média
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(convergencias['cdeepso'], label='cdeepso')
        ax.plot(convergencias['pcdeepso'], label='powell_cdeepso')
        ax.set_title(f'Curva de Convergência Média - Rosenbrock {dim} dimensões')
        ax.set_xlabel('Avaliações de Função (em $10^5$)')
        ax.set_ylabel('Convergência Média')
        
        real_ticks = [0, 0.25 * 10**5, 0.50 * 10**5, 0.75 * 10**5, 1 * 10**5]  # Mapeando 0 a 10^5
        display_ticks = [0, 0.25, 0.50, 0.75, 1]
        ax.set_xticks(real_ticks)  # Definindo os valores reais no eixo x
        ax.set_xticklabels([str(x) for x in display_ticks])

        ax.set_yscale('log')

        ax.legend()
        output_file = os.path.join(base_path, f'curva_convergencia_media_{dim}.png')
        plt.savefig(output_file)
        plt.show()

if __name__ == "__main__":
    generate_graphics()

# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# def ler_dados_excel(file_path):
#     df = pd.read_excel(file_path, sheet_name='Dados')
#     return df['best_fitness']

# dimensoes = [30, 50, 100]
# algoritmos = ['cdeepso', 'pcdeepso']
# function = 'rosen'

# current_dir = os.path.dirname(__file__)
        

# base_path = os.path.join(current_dir, 'Testes_max_fun_calls')

# def generate_graphics():
#     for dim in dimensoes:
#         resultados = {algoritmo: [] for algoritmo in algoritmos}
        
#         for algoritmo in algoritmos:
#             file_name = f'experimento_{function}_{dim}_dimensoes_{algoritmo}.xlsx'
#             file_path = os.path.join(base_path, file_name)
            
#             best_fitness = ler_dados_excel(file_path)
#             resultados[algoritmo] = best_fitness
        
#         fig, ax = plt.subplots(figsize=(10, 6))

#         ax.boxplot([resultados['cdeepso'], resultados['pcdeepso']], labels=['cdeepso', 'powell_cdeepso'])

#         ax.set_title(f'Rosenbrock {dim} dimensões')
#         ax.set_ylabel('Best Fitness')

#         output_file = os.path.join(base_path, f'boxplot_comparacao_{dim}.png')
#         plt.savefig(output_file)

#         plt.show()

# if __name__ == "__main__":
#     generate_graphics()