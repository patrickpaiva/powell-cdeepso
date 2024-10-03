import numpy as np
import bisect
from random import choice
from sklearn.cluster import KMeans
from utils import generatePopulation, generateMultiplicationMatrix
from powell_method import powell

def c_deepso_powell_global_best_com_limite(function, dimension, swarmSize, lowerBound, upperBound,
             max_iter=100,
             W_i=0.4019092098808389,
             W_a=0.3791940368874607,
             W_c=0.7539312405916303,
             max_v=1.01,
             T_com=0.5819630448962767,
             T_mut=0.1,
             type='sgpb',
             localSearch=False,
             searchRadius=0.5962189101390463,
             localSearchStartIter=99,
             localSearchEndIter=99,
             F=0.5):
    k = 0
    swarm = generatePopulation(dimension, swarmSize, lowerBound, upperBound)
    velocity = np.zeros((swarmSize, dimension))
    p_best = swarm.copy()
    p_best_fitness = np.array([function(p) for p in p_best])
    g_best = p_best[np.argmin(p_best_fitness)].copy()
    g_best_fitness_list = []
    velocities = []
    positions = []
    function_evals = 0
    improvement_tol = 1e-3
    no_improvement_counter = 0

    def evaluate_function(particle):
        nonlocal function_evals
        function_evals += 1
        return function(particle)

    g_best_fitness = evaluate_function(g_best)
    prev_g_best_fitness = g_best_fitness

    # Inicializa a lista ordenada para as 10% melhores partículas
    num_top_particles = max(1, swarmSize // 10)
    sorted_smb_particles = sorted((p_best_fitness[i], p_best[i].copy()) for i in range(num_top_particles))

    while k < max_iter:
        for i in range(swarmSize):
            # Mutação dos pesos
            W_i = np.clip(W_i + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_a = np.clip(W_a + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_c = np.clip(W_c + (T_mut * np.random.normal(0, 1)), 0, 1)

            # Atualiza Particle Best e Global Best
            particle = swarm[i]
            fitness = evaluate_function(particle)

            if fitness < p_best_fitness[i]:
                p_best[i] = particle.copy()
                p_best_fitness[i] = fitness
                if fitness < g_best_fitness:
                    g_best = particle.copy()
                    g_best_fitness = fitness

            # Atualiza lista com as 10% melhores partículas utilizando busca binária
            if fitness < sorted_smb_particles[-1][0]:
                sorted_smb_particles.pop()  # Remove o pior (último) elemento
                indice = bisect.bisect_left(sorted_smb_particles, fitness, key=lambda i: i[0])
                sorted_smb_particles.insert(indice, (fitness, particle.copy()))

            # Gera Matriz de Comunicação
            C = generateMultiplicationMatrix(dimension, T_com)

            # Extrai Xr conforme o tipo selecionado
            Xr = np.zeros((dimension))

            r = np.random.randint(0, swarmSize)
            while r == i:
                r = np.random.randint(0, swarmSize)

            if type == 'sg':  # Extraído aleatoriamente da população atual
                Xr = swarm[r].copy()
            elif type == 'pb':  # Extraído aleatoriamente dentre os 10% melhores salvos
                Xr = choice([p for _, p in sorted_smb_particles])
            elif type == 'sgpb':  # Média entre Sg e Pb
                pb = choice([p for _, p in sorted_smb_particles])
                Xr = ((swarm[r] + pb) / 2).copy()
            else:
                return "Tipo inválido. Aceitos: sg, pb e sgpb."

            if evaluate_function(Xr) > fitness:  # Fica com Xr apenas se o fitness dele for menor que o da minha partícula corrente
                Xr = particle.copy()

            # Realiza possível mutação do global best
            selected_global_best = g_best
            mutated_global_best = np.clip(g_best * (1 + T_mut * np.random.normal(0, 1)), lowerBound, upperBound)
            if evaluate_function(mutated_global_best) < g_best_fitness:
                selected_global_best = mutated_global_best

            # Implementa a estratégia current-to-best para o Xst
            r1, r2 = np.random.randint(0, swarmSize), np.random.randint(0, swarmSize)
            while r1 == i:
                r1 = np.random.randint(0, swarmSize)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, swarmSize)

            X_best = g_best.copy()
            X_r1 = swarm[r1].copy()
            X_r2 = swarm[r2].copy()

            X_st = Xr + F * (X_best - Xr)

            # Calcula vetor de velocidade da partícula
            inertia = W_i * velocity[i]
            cognitive = W_a * (X_st - particle)
            social = W_c * (C @ (selected_global_best - particle))

            velocity[i] = np.clip(inertia + cognitive + social, -max_v, max_v)

            # Atualiza posição somando a velocidade calculada
            swarm[i] = np.clip(particle + velocity[i], lowerBound, upperBound)

        # Avalia se está na hora de mudar para o powell
        improvement = abs(g_best_fitness - prev_g_best_fitness) / abs(prev_g_best_fitness)
        if improvement < improvement_tol:
            no_improvement_counter += 1
        else:
            no_improvement_counter = 0

        if no_improvement_counter >= 10 or k == max_iter - 1:
          result = powell(evaluate_function, g_best, (lowerBound, upperBound))
          candidate = result.copy()
          candidate_fitness = evaluate_function(candidate)
          # result = fmin_powell(evaluate_function, g_best, full_output=1, maxiter=100)
          # candidate = result[0].copy()
          # candidate_fitness = result[1]
          if candidate_fitness < g_best_fitness:
              g_best = candidate.copy()
              g_best_fitness = candidate_fitness
              break

        g_best_fitness_list.append(g_best_fitness)
        positions.append(swarm.copy())
        velocities.append(velocity.copy())
        prev_g_best_fitness = g_best_fitness
        k += 1

    return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals

def c_deepso_powell_global_best_com_kmeans(function, dimension, swarmSize, lowerBound, upperBound,
             dispersion_threshold,
             max_iter=None,
             max_fun_evals=None,
             W_i=0.4019092098808389,
             W_a=0.3791940368874607,
             W_c=0.7539312405916303,
             max_v=1.01,
             T_com=0.5819630448962767,
             T_mut=0.1,
             type='sgpb',
             F=0.5):
    k = 0
    swarm = generatePopulation(dimension, swarmSize, lowerBound, upperBound)
    velocity = np.zeros((swarmSize, dimension))
    p_best = swarm.copy()
    p_best_fitness = np.array([function(p) for p in p_best])
    g_best = p_best[np.argmin(p_best_fitness)].copy()
    g_best_fitness_list = []
    velocities = []
    positions = []
    function_evals = 0

    if max_iter is None and max_fun_evals is None:
        max_iter = 100
    
    def get_function_evals():
        nonlocal function_evals
        return function_evals
    
    def evaluate_function(particle):
        nonlocal function_evals
        function_evals += 1
        return function(particle)

    g_best_fitness = evaluate_function(g_best)

    # Inicializa a lista ordenada para as 10% melhores partículas
    num_top_particles = max(1, swarmSize // 10)
    sorted_smb_particles = sorted((p_best_fitness[i], p_best[i].copy()) for i in range(num_top_particles))

    while True:
        for i in range(swarmSize):
            # Mutação dos pesos
            W_i = np.clip(W_i + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_a = np.clip(W_a + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_c = np.clip(W_c + (T_mut * np.random.normal(0, 1)), 0, 1)

            # Atualiza Particle Best e Global Best
            particle = swarm[i]
            fitness = evaluate_function(particle)

            if fitness < p_best_fitness[i]:
                p_best[i] = particle.copy()
                p_best_fitness[i] = fitness
                if fitness < g_best_fitness:
                    g_best = particle.copy()
                    g_best_fitness = fitness

            # Atualiza lista com as 10% melhores partículas utilizando busca binária
            if fitness < sorted_smb_particles[-1][0]:
                sorted_smb_particles.pop()  # Remove o pior (último) elemento
                indice = bisect.bisect_left(sorted_smb_particles, fitness, key=lambda i: i[0])
                sorted_smb_particles.insert(indice, (fitness, particle.copy()))

            # Gera Matriz de Comunicação
            C = generateMultiplicationMatrix(dimension, T_com)

            # Extrai Xr conforme o tipo selecionado
            Xr = np.zeros((dimension))

            r = np.random.randint(0, swarmSize)
            while r == i:
                r = np.random.randint(0, swarmSize)

            if type == 'sg':  # Extraído aleatoriamente da população atual
                Xr = swarm[r].copy()
            elif type == 'pb':  # Extraído aleatoriamente dentre os 10% melhores salvos
                Xr = choice([p for _, p in sorted_smb_particles])
            elif type == 'sgpb':  # Média entre Sg e Pb
                pb = choice([p for _, p in sorted_smb_particles])
                Xr = ((swarm[r] + pb) / 2).copy()
            else:
                return "Tipo inválido. Aceitos: sg, pb e sgpb."

            if evaluate_function(Xr) > fitness:  # Fica com Xr apenas se o fitness dele for menor que o da minha partícula corrente
                Xr = particle.copy()

            # Realiza possível mutação do global best
            selected_global_best = g_best
            mutated_global_best = np.clip(g_best * (1 + T_mut * np.random.normal(0, 1)), lowerBound, upperBound)
            if evaluate_function(mutated_global_best) < g_best_fitness:
                selected_global_best = mutated_global_best

            # Implementa a estratégia current-to-best para o Xst
            r1, r2 = np.random.randint(0, swarmSize), np.random.randint(0, swarmSize)
            while r1 == i:
                r1 = np.random.randint(0, swarmSize)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, swarmSize)

            X_best = g_best.copy()
            X_r1 = swarm[r1].copy()
            X_r2 = swarm[r2].copy()

            X_st = Xr + F * (X_best - Xr)

            # Calcula vetor de velocidade da partícula
            inertia = W_i * velocity[i]
            cognitive = W_a * (X_st - particle)
            social = W_c * (C @ (selected_global_best - particle))

            velocity[i] = np.clip(inertia + cognitive + social, -max_v, max_v)

            # Atualiza posição somando a velocidade calculada
            swarm[i] = np.clip(particle + velocity[i], lowerBound, upperBound)

        # Avalia se está na hora de mudar para o powell
        if k > 10 and k % 5 == 0:  # Pode ajustar o intervalo de verificação
          kmeans = KMeans(n_clusters=2, n_init=10)  # Número de clusters pode ser ajustado
          swarm_positions = np.array(swarm)
          kmeans.fit(swarm_positions)
          labels = kmeans.labels_
          cluster_centers = kmeans.cluster_centers_

          # Calcula a dispersão entre os clusters
          intra_cluster_distance = np.mean([np.linalg.norm(swarm_positions[i] - cluster_centers[labels[i]])
                                            for i in range(swarmSize)])

          # Define um limiar de dispersão que, quando atingido, entrega para o Powell
          print(f"V1 - Geração: {k} Dispersão: {intra_cluster_distance} fitness: {g_best_fitness} funcalls: {function_evals} maxiter: {max_iter}")
          if intra_cluster_distance < dispersion_threshold \
            or (max_iter is not None and k == (max_iter - 5)) \
            or (max_fun_evals is not None and (function_evals > max_fun_evals/2)):
              print(f"V1 - Aconteceu na iteracao {k}")
              result = powell(evaluate_function, g_best, (lowerBound, upperBound), max_fun_evals, get_function_evals)
              candidate = result.copy()
              candidate_fitness = evaluate_function(candidate)
              if candidate_fitness < g_best_fitness:
                  g_best = candidate.copy()
                  g_best_fitness = candidate_fitness
                  g_best_fitness_list.append(g_best_fitness)
              break

        g_best_fitness_list.append(g_best_fitness)
        positions.append(swarm.copy())
        velocities.append(velocity.copy())
        k += 1

        if max_iter is not None and max_iter == k:
            break
        
        if max_fun_evals is not None and max_fun_evals <= function_evals:
            break

    return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals

def c_deepso_powell_global_best_com_kmeans_v3(function, dimension, swarmSize, lowerBound, upperBound,
             dispersion_threshold,
             max_iter=None,
             max_fun_evals=None,
             W_i=0.4019092098808389,
             W_a=0.3791940368874607,
             W_c=0.7539312405916303,
             max_v=1.01,
             T_com=0.5819630448962767,
             T_mut=0.1,
             type='sgpb',
             F=0.5,
             diff_window=3):
    k = 0
    swarm = generatePopulation(dimension, swarmSize, lowerBound, upperBound)
    velocity = np.zeros((swarmSize, dimension))
    p_best = swarm.copy()
    p_best_fitness = np.array([function(p) for p in p_best])
    g_best = p_best[np.argmin(p_best_fitness)].copy()
    g_best_fitness_list = []
    velocities = []
    positions = []
    function_evals = 0
    dispersions = []
    dispersion_diffs = []

    if max_iter is None and max_fun_evals is None:
        max_iter = 100
    
    def get_function_evals():
        nonlocal function_evals
        return function_evals
    
    def evaluate_function(particle):
        nonlocal function_evals
        function_evals += 1
        return function(particle)

    g_best_fitness = evaluate_function(g_best)

    # Inicializa a lista ordenada para as 10% melhores partículas
    num_top_particles = max(1, swarmSize // 10)
    sorted_smb_particles = sorted((p_best_fitness[i], p_best[i].copy()) for i in range(num_top_particles))

    while True:
        for i in range(swarmSize):
            # Mutação dos pesos
            W_i = np.clip(W_i + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_a = np.clip(W_a + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_c = np.clip(W_c + (T_mut * np.random.normal(0, 1)), 0, 1)

            # Atualiza Particle Best e Global Best
            particle = swarm[i]
            fitness = evaluate_function(particle)

            if fitness < p_best_fitness[i]:
                p_best[i] = particle.copy()
                p_best_fitness[i] = fitness
                if fitness < g_best_fitness:
                    g_best = particle.copy()
                    g_best_fitness = fitness

            # Atualiza lista com as 10% melhores partículas utilizando busca binária
            if fitness < sorted_smb_particles[-1][0]:
                sorted_smb_particles.pop()  # Remove o pior (último) elemento
                indice = bisect.bisect_left(sorted_smb_particles, fitness, key=lambda i: i[0])
                sorted_smb_particles.insert(indice, (fitness, particle.copy()))

            # Gera Matriz de Comunicação
            C = generateMultiplicationMatrix(dimension, T_com)

            # Extrai Xr conforme o tipo selecionado
            Xr = np.zeros((dimension))

            r = np.random.randint(0, swarmSize)
            while r == i:
                r = np.random.randint(0, swarmSize)

            if type == 'sg':  # Extraído aleatoriamente da população atual
                Xr = swarm[r].copy()
            elif type == 'pb':  # Extraído aleatoriamente dentre os 10% melhores salvos
                Xr = choice([p for _, p in sorted_smb_particles])
            elif type == 'sgpb':  # Média entre Sg e Pb
                pb = choice([p for _, p in sorted_smb_particles])
                Xr = ((swarm[r] + pb) / 2).copy()
            else:
                return "Tipo inválido. Aceitos: sg, pb e sgpb."

            if evaluate_function(Xr) > fitness:  # Fica com Xr apenas se o fitness dele for menor que o da minha partícula corrente
                Xr = particle.copy()

            # Realiza possível mutação do global best
            selected_global_best = g_best
            mutated_global_best = np.clip(g_best * (1 + T_mut * np.random.normal(0, 1)), lowerBound, upperBound)
            if evaluate_function(mutated_global_best) < g_best_fitness:
                selected_global_best = mutated_global_best

            # Implementa a estratégia current-to-best para o Xst
            r1, r2 = np.random.randint(0, swarmSize), np.random.randint(0, swarmSize)
            while r1 == i:
                r1 = np.random.randint(0, swarmSize)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, swarmSize)

            X_best = g_best.copy()
            X_r1 = swarm[r1].copy()
            X_r2 = swarm[r2].copy()

            X_st = Xr + F * (X_best - Xr)

            # Calcula vetor de velocidade da partícula
            inertia = W_i * velocity[i]
            cognitive = W_a * (X_st - particle)
            social = W_c * (C @ (selected_global_best - particle))

            velocity[i] = np.clip(inertia + cognitive + social, -max_v, max_v)

            # Atualiza posição somando a velocidade calculada
            swarm[i] = np.clip(particle + velocity[i], lowerBound, upperBound)

        # Avalia se está na hora de mudar para o powell
        if k > 10 and k % 5 == 0:  # Pode ajustar o intervalo de verificação
          kmeans = KMeans(n_clusters=2, n_init=10)  # Número de clusters pode ser ajustado
          swarm_positions = np.array(swarm)
          kmeans.fit(swarm_positions)
          labels = kmeans.labels_
          cluster_centers = kmeans.cluster_centers_

          # Calcula a dispersão entre os clusters
          intra_cluster_distance = np.mean([np.linalg.norm(swarm_positions[i] - cluster_centers[labels[i]])
                                            for i in range(swarmSize)])

          dispersions.append(intra_cluster_distance)

          if len(dispersions) > 1:
                dispersion_diffs.append(abs(dispersions[-1] - dispersions[-2]))
                
          # Verifica a estabilidade das últimas diferenças
          if len(dispersion_diffs) >= diff_window:
              recent_diffs = dispersion_diffs[-diff_window:]
              max_diff = max(recent_diffs)
              min_diff = min(recent_diffs)
              
              # Se as últimas diferenças estiverem abaixo do limiar e forem próximas
              diff = max_diff - min_diff
              print(f'geração: {k} diff: {diff} dispersion: {intra_cluster_distance} fitness: {g_best_fitness} funcalls: {function_evals}')
              if diff <= dispersion_threshold\
                or (max_iter is not None and k == (max_iter - 5)) \
                or (max_fun_evals is not None and (function_evals >= max_fun_evals*0.7)):
                  print(f'Aconteceu na geração: {k}')
                  result = powell(evaluate_function, g_best, (lowerBound, upperBound), max_fun_evals, get_function_evals)
                  candidate = result.copy()
                  candidate_fitness = evaluate_function(candidate)
                  if candidate_fitness < g_best_fitness:
                      g_best = candidate.copy()
                      g_best_fitness = candidate_fitness
                      g_best_fitness_list.append(g_best_fitness)
                  break

        g_best_fitness_list.append(g_best_fitness)
        positions.append(swarm.copy())
        velocities.append(velocity.copy())
        k += 1

        if max_iter is not None and max_iter == k:
            break
        
        if max_fun_evals is not None and max_fun_evals <= function_evals:
            break

    return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals

def c_deepso_powell_global_best_com_kmeans_v2(function, dimension, swarmSize, lowerBound, upperBound,
             max_iter=None,
             max_fun_evals=None,
             W_i=0.4019092098808389,
             W_a=0.3791940368874607,
             W_c=0.7539312405916303,
             max_v=1.01,
             T_com=0.5819630448962767,
             T_mut=0.1,
             type='sgpb',
             F=0.5):
    k = 0
    swarm = generatePopulation(dimension, swarmSize, lowerBound, upperBound)
    velocity = np.zeros((swarmSize, dimension))
    p_best = swarm.copy()
    p_best_fitness = np.array([function(p) for p in p_best])
    g_best = p_best[np.argmin(p_best_fitness)].copy()
    g_best_fitness_list = []
    velocities = []
    positions = []
    function_evals = 0
    D_max = np.sqrt(dimension)

    if max_iter is None and max_fun_evals is None:
        max_iter = 100
    
    def get_function_evals():
        nonlocal function_evals
        return function_evals
    
    def evaluate_function(particle):
        nonlocal function_evals
        function_evals += 1
        return function(particle)

    g_best_fitness = evaluate_function(g_best)

    # Inicializa a lista ordenada para as 10% melhores partículas
    num_top_particles = max(1, swarmSize // 10)
    sorted_smb_particles = sorted((p_best_fitness[i], p_best[i].copy()) for i in range(num_top_particles))

    while True:
        for i in range(swarmSize):
            # Mutação dos pesos
            W_i = np.clip(W_i + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_a = np.clip(W_a + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_c = np.clip(W_c + (T_mut * np.random.normal(0, 1)), 0, 1)

            # Atualiza Particle Best e Global Best
            particle = swarm[i]
            fitness = evaluate_function(particle)

            if fitness < p_best_fitness[i]:
                p_best[i] = particle.copy()
                p_best_fitness[i] = fitness
                if fitness < g_best_fitness:
                    g_best = particle.copy()
                    g_best_fitness = fitness

            # Atualiza lista com as 10% melhores partículas utilizando busca binária
            if fitness < sorted_smb_particles[-1][0]:
                sorted_smb_particles.pop()  # Remove o pior (último) elemento
                indice = bisect.bisect_left(sorted_smb_particles, fitness, key=lambda i: i[0])
                sorted_smb_particles.insert(indice, (fitness, particle.copy()))

            # Gera Matriz de Comunicação
            C = generateMultiplicationMatrix(dimension, T_com)

            # Extrai Xr conforme o tipo selecionado
            Xr = np.zeros((dimension))

            r = np.random.randint(0, swarmSize)
            while r == i:
                r = np.random.randint(0, swarmSize)

            if type == 'sg':  # Extraído aleatoriamente da população atual
                Xr = swarm[r].copy()
            elif type == 'pb':  # Extraído aleatoriamente dentre os 10% melhores salvos
                Xr = choice([p for _, p in sorted_smb_particles])
            elif type == 'sgpb':  # Média entre Sg e Pb
                pb = choice([p for _, p in sorted_smb_particles])
                Xr = ((swarm[r] + pb) / 2).copy()
            else:
                return "Tipo inválido. Aceitos: sg, pb e sgpb."

            if evaluate_function(Xr) > fitness:  # Fica com Xr apenas se o fitness dele for menor que o da minha partícula corrente
                Xr = particle.copy()

            # Realiza possível mutação do global best
            selected_global_best = g_best
            mutated_global_best = np.clip(g_best * (1 + T_mut * np.random.normal(0, 1)), lowerBound, upperBound)
            if evaluate_function(mutated_global_best) < g_best_fitness:
                selected_global_best = mutated_global_best

            # Implementa a estratégia current-to-best para o Xst
            r1, r2 = np.random.randint(0, swarmSize), np.random.randint(0, swarmSize)
            while r1 == i:
                r1 = np.random.randint(0, swarmSize)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, swarmSize)

            X_best = g_best.copy()
            X_r1 = swarm[r1].copy()
            X_r2 = swarm[r2].copy()

            X_st = Xr + F * (X_best - Xr)

            # Calcula vetor de velocidade da partícula
            inertia = W_i * velocity[i]
            cognitive = W_a * (X_st - particle)
            social = W_c * (C @ (selected_global_best - particle))

            velocity[i] = np.clip(inertia + cognitive + social, -max_v, max_v)

            # Atualiza posição somando a velocidade calculada
            swarm[i] = np.clip(particle + velocity[i], lowerBound, upperBound)

        # Avalia se está na hora de mudar para o powell
        if k > 10 and k % 5 == 0:  # Pode ajustar o intervalo de verificação
            swarm_positions = swarm.copy()
            current_lower_bound = np.min(swarm_positions, axis=0)
            current_upper_bound = np.max(swarm_positions, axis=0)

            normalized_swarm_positions = (swarm_positions - current_lower_bound) / (current_upper_bound - current_lower_bound)

            kmeans = KMeans(n_clusters=2, n_init=10)
            kmeans.fit(normalized_swarm_positions)
            labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_

            intra_cluster_distance = np.mean([np.linalg.norm(normalized_swarm_positions[i] - cluster_centers[labels[i]]) 
                                  for i in range(swarmSize)])
            
            proximity_threshold = 0.15 * D_max

            # print(f"V2 - Geração: {k} Dispersão: {intra_cluster_distance} LIMITE: {proximity_threshold}")
            if intra_cluster_distance < proximity_threshold:
                # print(f"V2 - Aconteceu na iteracao {k}")
                result = powell(evaluate_function, g_best, (lowerBound, upperBound), max_fun_evals, get_function_evals)
                candidate = result.copy()
                candidate_fitness = evaluate_function(candidate)
                if candidate_fitness < g_best_fitness:
                    g_best = candidate.copy()
                    g_best_fitness = candidate_fitness
                    g_best_fitness_list.append(g_best_fitness)
                break

        g_best_fitness_list.append(g_best_fitness)
        positions.append(swarm.copy())
        velocities.append(velocity.copy())
        k += 1

        if max_iter is not None and max_iter == k:
            break
        
        if max_fun_evals is not None and max_fun_evals <= function_evals:
            break

    return g_best_fitness, g_best, g_best_fitness_list, positions, velocities, function_evals

def c_deepso(function, dimension, swarmSize, lowerBound, upperBound,
             max_iter=None,
             max_fun_evals=None,
             W_i=0.4019092098808389,
             W_a=0.3791940368874607,
             W_c=0.7539312405916303,
             max_v=1.01,
             T_com=0.5819630448962767,
             T_mut=0.1,
             type='sgpb',
             F=0.5):
    k = 0
    swarm = generatePopulation(dimension, swarmSize, lowerBound, upperBound)
    velocity = np.zeros((swarmSize, dimension))
    p_best = swarm.copy()
    p_best_fitness = np.array([function(p) for p in p_best])
    g_best = p_best[np.argmin(p_best_fitness)].copy()
    g_best_fitness_list = []
    velocities = []
    positions = []
    function_evals = 0

    if max_iter is None and max_fun_evals is None:
        max_iter = 100

    def evaluate_function(particle):
        nonlocal function_evals
        function_evals += 1
        return function(particle)

    # Inicializa a lista ordenada para as 10% melhores partículas
    num_top_particles = max(1, swarmSize // 10)
    sorted_smb_particles = sorted((p_best_fitness[i], p_best[i].copy()) for i in range(num_top_particles))

    while True:
        for i in range(swarmSize):
            # Mutação dos pesos
            W_i = np.clip(W_i + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_a = np.clip(W_a + (T_mut * np.random.normal(0, 1)), 0, 1)
            W_c = np.clip(W_c + (T_mut * np.random.normal(0, 1)), 0, 1)

            # Atualiza Particle Best e Global Best
            particle = swarm[i]
            fitness = evaluate_function(particle)

            if fitness < p_best_fitness[i]:
                p_best[i] = particle.copy()
                p_best_fitness[i] = fitness
                if fitness < evaluate_function(g_best):
                    g_best = particle.copy()

            # Atualiza lista com as 10% melhores partículas utilizando busca binária
            if fitness < sorted_smb_particles[-1][0]:
                sorted_smb_particles.pop()  # Remove o pior (último) elemento
                indice = bisect.bisect_left(sorted_smb_particles, fitness, key=lambda i: i[0])
                sorted_smb_particles.insert(indice, (fitness, particle.copy()))

            # Gera Matriz de Comunicação
            C = generateMultiplicationMatrix(dimension, T_com)

            # Extrai Xr conforme o tipo selecionado
            Xr = np.zeros((dimension))

            r = np.random.randint(0, swarmSize)
            while r == i:
                r = np.random.randint(0, swarmSize)

            if type == 'sg':  # Extraído aleatoriamente da população atual
                Xr = swarm[r].copy()
            elif type == 'pb':  # Extraído aleatoriamente dentre os 10% melhores salvos
                Xr = choice([p for _, p in sorted_smb_particles])
            elif type == 'sgpb':  # Média entre Sg e Pb
                pb = choice([p for _, p in sorted_smb_particles])
                Xr = ((swarm[r] + pb) / 2).copy()
            else:
                return "Tipo inválido. Aceitos: sg, pb e sgpb."

            if evaluate_function(Xr) > fitness:  # Fica com Xr apenas se o fitness dele for menor que o da minha partícula corrente
                Xr = particle.copy()

            # Realiza possível mutação do global best
            selected_global_best = g_best
            mutated_global_best = np.clip(g_best * (1 + T_mut * np.random.normal(0, 1)), lowerBound, upperBound)
            if evaluate_function(mutated_global_best) < evaluate_function(g_best):
                selected_global_best = mutated_global_best

            # Implementa a estratégia current-to-best para o Xst
            r1, r2 = np.random.randint(0, swarmSize), np.random.randint(0, swarmSize)
            while r1 == i:
                r1 = np.random.randint(0, swarmSize)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, swarmSize)

            X_best = g_best.copy()
            X_r1 = swarm[r1].copy()
            X_r2 = swarm[r2].copy()

            X_st = Xr + F * (X_best - Xr)

            # Calcula vetor de velocidade da partícula
            inertia = W_i * velocity[i]
            cognitive = W_a * (X_st - particle)
            social = W_c * (C @ (selected_global_best - particle))

            velocity[i] = np.clip(inertia + cognitive + social, -max_v, max_v)

            # Atualiza posição somando a velocidade calculada
            swarm[i] = np.clip(particle + velocity[i], lowerBound, upperBound)

        g_best_fitness_list.append(evaluate_function(g_best))
        positions.append(swarm.copy())
        velocities.append(velocity.copy())
        k += 1

        if max_iter is not None and max_iter == k:
            break
        
        if max_fun_evals is not None and max_fun_evals <= function_evals:
            break

    return evaluate_function(g_best), g_best, g_best_fitness_list, positions, velocities, function_evals
