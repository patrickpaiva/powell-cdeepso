import sys
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scipy.optimize import rosen
import numpy as np
from tqdm import tqdm
import pandas as pd
from utils import calculate_statistics
from powell_cdeepso import c_deepso_powell_global_best
from cec2013lsgo.cec2013 import Benchmark

bench = Benchmark()

def rosenbrock_shifted(sol):
    fun_fitness = bench.get_function(12)
    return fun_fitness(sol)

dimension = bench.get_info(12)['dimension']

# Função individual para executar c_deepso_powell_global_best
def executar_experimento(function, dimension, swarm_size, lower_bound, upper_bound, percent_powell_start_moment, percent_powell_func_evals, wi, wa, wc, tcom, tmut, max_v, max_fun_evals, max_iter):
    try:
        best_fitness, g_best, g_best_list, _, _, function_evals, g_best_fitness_120k_evals, g_best_fitness_600k_evals = c_deepso_powell_global_best(
            function, dimension, swarm_size, lower_bound, upper_bound,
            percent_powell_start_moment=percent_powell_start_moment,
            percent_powell_func_evals=percent_powell_func_evals,
            max_iter=max_iter,
            max_fun_evals=max_fun_evals,
            type='pb', W_i=wi, W_a=wa, W_c=wc,
            T_mut=tmut, T_com=tcom, max_v=max_v
        )
        return {
            'best_fitness': best_fitness,
            'global_best': g_best,
            'function_evals': function_evals,
            'global_best_list': g_best_list,
            'g_best_fitness_120k_evals': g_best_fitness_120k_evals,
            'g_best_fitness_600k_evals': g_best_fitness_600k_evals
        }
    except Exception as e:
        print(f"Erro na execução do experimento: {e}")
        raise e


def experimentacao_powell(function, dimension, swarm_size, lower_bound, upper_bound, percent_powell_start_moment, percent_powell_func_evals, wi, wa, wc, tcom, tmut, max_v, max_fun_evals, max_iter):
    results = []
    global_best_data = []

    # Use ProcessPoolExecutor para paralelismo em múltiplos processos
    max_processes = 3  # Ajuste para o número de núcleos que você deseja usar

    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        futures = [
            executor.submit(executar_experimento, function, dimension, swarm_size, lower_bound, upper_bound,
                            percent_powell_start_moment, percent_powell_func_evals, wi, wa, wc, tcom, tmut, max_v, max_fun_evals, max_iter)
            for _ in range(25)
        ]

        with tqdm(total=25, desc="Executando em paralelo...", unit="iter") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append({
                    'best_fitness': result['best_fitness'],
                    'global_best': result['global_best'],
                    'function_evals': result['function_evals'],
                    'g_best_fitness_120k_evals': result['g_best_fitness_120k_evals'],
                    'g_best_fitness_600k_evals': result['g_best_fitness_600k_evals']
                })
                global_best_data.append(result['global_best_list'])
                pbar.update(1)

    # Calcula as estatísticas e salva na planilha
    best_fitnesses = [res['best_fitness'] for res in results]
    best_fitnesses_120k_evals = [res['g_best_fitness_120k_evals'] for res in results if res['g_best_fitness_120k_evals'] is not None]
    best_fitnesses_600k_evals = [res['g_best_fitness_600k_evals'] for res in results if res['g_best_fitness_600k_evals'] is not None]
    fun_evals = [res['function_evals'] for res in results]

    for result in results:
        result['global_best'] = ', '.join(map(str, result['global_best']))

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='best_fitness', ascending=True)

    function_evals_mean = np.mean(fun_evals)
    minimum, maximum, mean, std_dev, median = calculate_statistics(best_fitnesses)
    minimum_120k, maximum_120k, mean_120k, std_dev_120k, median_120k = calculate_statistics(best_fitnesses_120k_evals)
    minimum_600k, maximum_600k, mean_600k, std_dev_600k, median_600k = calculate_statistics(best_fitnesses_600k_evals)

    statistics = [{
        'Minimo': minimum,
        'Maximo': maximum,
        'Media': mean,
        'Mediana': median,
        'Desvio_Padrao': std_dev,
        'Aval_Func_Media': function_evals_mean
    }]
    df_stats = pd.DataFrame(statistics)

    statistics_120k = [{
        'Minimo': minimum_120k,
        'Maximo': maximum_120k,
        'Media': mean_120k,
        'Mediana': median_120k,
        'Desvio_Padrao': std_dev_120k
    }]
    df_stats_120k = pd.DataFrame(statistics_120k)

    statistics_600k = [{
        'Minimo': minimum_600k,
        'Maximo': maximum_600k,
        'Media': mean_600k,
        'Mediana': median_600k,
        'Desvio_Padrao': std_dev_600k
    }]
    df_stats_600k = pd.DataFrame(statistics_600k)

    global_best_array = np.array(global_best_data)
    global_best_mean = np.mean(global_best_array, axis=0)
    df_global_best_mean = pd.DataFrame(global_best_mean, columns=['Convergencia_Media'])

    nome_arquivo = f"experimento_{function.__name__}_{dimension}_dimensoes_pcdeepso.xlsx"

    with pd.ExcelWriter(nome_arquivo, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='Dados', index=False)
        df_stats.to_excel(writer, sheet_name='Estatisticas', index=False)
        df_stats_120k.to_excel(writer, sheet_name='Estatisticas_120k', index=False)
        df_stats_600k.to_excel(writer, sheet_name='Estatisticas_600k', index=False)
        df_global_best_mean.to_excel(writer, sheet_name='Convergencia_Media', index=False)

if __name__ == "__main__":
    experimentacao_powell(
        function=rosenbrock_shifted,
        dimension=1000,
        swarm_size=500,
        lower_bound=-100,
        upper_bound=100,
        percent_powell_start_moment=0.5,
        percent_powell_func_evals=0.05,
        wi=0.4019092098808389,
        wa=0.3791940368874607,
        wc=0.7539312405916303,
        tcom=0.5819630448962767,
        tmut=0.3,
        max_v=1.01,
        max_fun_evals=3_000_000,
        max_iter=None
    )
