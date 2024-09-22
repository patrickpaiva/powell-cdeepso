from powell_cdeepso import c_deepso_powell_global_best_com_kmeans, c_deepso, c_deepso_powell_global_best_com_kmeans_v2
from utils import calculate_statistics, print_statistics, perform_t_test
from scipy.optimize import rosen
import numpy as np
import time
from tqdm import tqdm

def schwefel(particle):
    dimension = len(particle)
    return 418.9829 * dimension - np.sum(particle * np.sin(np.sqrt(np.abs(particle))))

def schwefel_variation(x):
    # Dimensão D
    D = len(x)
    
    # Aplicar a fórmula da função de Schwefel
    sum_total = 0
    for i in range(1, D + 1):
        sum_inner = 0
        for j in range(1, i + 1):
            sum_inner += x[j - 1]
        sum_total += sum_inner ** 2
    
    return sum_total

def experimentacao(function, dimension, swarm_size, lower_bound, upper_bound, wi, wa, wc, tcom, tmut, max_v, max_fun_evals, max_iter, dispersion_threshold):
    results_PCDEEPSO = []
    results_CDEEPSO = []

    for _ in tqdm(range(30), desc="Executando...", unit="iter"):
        best_fitness_PCDEEPSO, g_best_PCDEEPSO, _, _, _, function_evals_PCDEEPSO = c_deepso_powell_global_best_com_kmeans(function, dimension, swarm_size, lower_bound, upper_bound, dispersion_threshold, max_iter=max_iter, max_fun_evals=max_fun_evals, type='pb', W_i=wi, W_a=wa, W_c=wc, T_mut=tmut, T_com=tcom, max_v=max_v)
        results_PCDEEPSO.append({
            'best_fitness': best_fitness_PCDEEPSO,
            'global_best': g_best_PCDEEPSO,
            'function_evals': function_evals_PCDEEPSO
        })
        best_fitnesses_PCDEEPSO = [res['best_fitness'] for res in results_PCDEEPSO]
        fun_evals_PCDEEPSO = [res['function_evals'] for res in results_PCDEEPSO]

        best_fitness_CDEEPSO, g_best_CDEEPSO, _, _, _, function_evals_CDEEPSO = c_deepso_powell_global_best_com_kmeans_v2(function, dimension, swarm_size, lower_bound, upper_bound, max_iter=max_iter, max_fun_evals=max_fun_evals, type='pb', W_i=wi, W_a=wa, W_c=wc, T_mut=tmut, T_com=tcom, max_v=max_v)
        results_CDEEPSO.append({
            'best_fitness': best_fitness_CDEEPSO,
            'global_best': g_best_CDEEPSO,
            'function_evals': function_evals_CDEEPSO
        })
        best_fitnesses_CDEEPSO = [res['best_fitness'] for res in results_CDEEPSO]
        fun_evals_CDEEPSO = [res['function_evals'] for res in results_CDEEPSO]

    function_evals_mean_PCDEEPSO = np.mean(fun_evals_PCDEEPSO)
    function_evals_mean_CDEEPSO = np.mean(fun_evals_CDEEPSO)
    PCDEEPSO_stats = calculate_statistics(best_fitnesses_PCDEEPSO)
    CDEEPSO_stats = calculate_statistics(best_fitnesses_CDEEPSO)
    print_statistics(function, dimension, PCDEEPSO_stats, CDEEPSO_stats, function_evals_mean_PCDEEPSO, function_evals_mean_CDEEPSO)
    perform_t_test(best_fitnesses_PCDEEPSO, best_fitnesses_CDEEPSO)
    
    # prova do fitness mínimo
    indice_minimo = best_fitnesses_PCDEEPSO.index(PCDEEPSO_stats[0])
    melhor_execucao = results_PCDEEPSO[indice_minimo]
    melhor_gb = melhor_execucao['global_best']
    teste = function(melhor_gb)
    print(f"\nProva do fitness: {teste}")

def main():
    inicio = time.time()
    print("Iniciando...")

    experimentacao(
            function=schwefel, 
            dimension=30, 
            swarm_size=30, 
            lower_bound=-500, 
            upper_bound=500,
            dispersion_threshold = 1200, 
            wi = 0.7115679640086889, 
            wa = 0.33431498083767774, 
            wc = 0.6519128879895824, 
            tcom= 0.5272319474395429, 
            tmut= 0.7980253597223532, 
            max_v=1.01,
            max_fun_evals=None,
            max_iter=100)

    fim = time.time()
    tempo_total = fim - inicio
    minutos = int(tempo_total // 60)
    segundos = tempo_total % 60
    print(f"\nTempo de execução: {minutos} minutos e {segundos:.2f} segundos")

if __name__ == "__main__":
    main()



# experimentacao(
#         function=schwefel, 
#         dimension=30, 
#         swarm_size=30, 
#         lower_bound=-500, 
#         upper_bound=500,
#         dispersion_threshold = 1200, 
#         wi = 0.7115679640086889, 
#         wa = 0.33431498083767774, 
#         wc = 0.6519128879895824, 
#         tcom= 0.5272319474395429, 
#         tmut= 0.7980253597223532, 
#         max_v=1.01,
#         max_fun_evals=None,
#         max_iter=100)

# experimentacao(
#         function=rosen, 
#         dimension=30, 
#         swarm_size=30, 
#         lower_bound=-2.048, 
#         upper_bound=2.048,
#         dispersion_threshold = 1e-1,
#         wi = 0.4019092098808389, 
#         wa = 0.3791940368874607, 
#         wc = 0.7539312405916303, 
#         tcom= 0.5819630448962767, 
#         tmut= 0.3, 
#         max_v=1.01,
#         max_fun_evals=None,
#         max_iter=100)