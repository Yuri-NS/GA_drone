############################################## GA #################################################

import json
import os
import random
import time
import numpy as np
import solution_priority
import task_priority
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.visualization.scatter import Scatter
from pymoo.util.plotting import plot
from pymoo.indicators.hv import HV
from scipy.spatial.distance import pdist, squareform
import csv
import os

from robot import Robot
import vnd_priority

DISTANCE_METRIC = 'euclidean'

# INICIALIZANDO ROBÔS
battery_times = 900
velocity = 2
initial_positions = (29, 136) # Exemplo de posições iniciais
num_robots = 10
pop_size = 50
time_limit = 100
ref_point = [15000, 2000, 2000]

robots = [Robot(i, battery_times, velocity, initial_position=initial_positions) for i in range(num_robots)]

tasks = task_priority.tasks



def save_results_to_file(evolutions, filename):
    filename = f'experiment_results_bnd/{filename}'
    """
    Salva as evoluções das métricas em um arquivo JSON.

    Args:
        evolutions (dict): Dicionário contendo as evoluções das métricas.
        filename (str): Nome do arquivo para salvar os dados.
    """
    with open(filename, "w") as f:
        json.dump(evolutions, f)
    print(f"Evoluções salvas em {filename}")

def load_evolutions_from_file(filename):
    filename = f'experiment_results_bnd/{filename}'
    """
    Carrega as evoluções das métricas de um arquivo JSON.

    Args:
        filename (str): Nome do arquivo de onde carregar os dados.

    Returns:
        dict: Dicionário contendo as evoluções das métricas.
    """
    with open(filename, "r") as f:
        evolutions = json.load(f)
    print(f"Evoluções carregadas de {filename}")
    return evolutions




# Função auxiliar para calcular distâncias
def calculate_distance(point1, point2, metric='euclidean'):
    if metric == 'manhattan':
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
    elif metric == 'euclidean':
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    else:
        raise ValueError("Invalid metric. Use 'euclidean' or 'manhattan'.")

def precompute_distances(tasks):
    n = len(tasks)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = calculate_distance(tasks[i].coordinates, tasks[j].coordinates, DISTANCE_METRIC)
            distances[j, i] = distances[i, j]
    return distances

distance_matrix = precompute_distances(tasks)

def select_random_parents(population):
    """
    Seleciona dois pais aleatoriamente da população.
    
    Args:
        population (list): Lista de soluções.
    
    Returns:
        tuple: Dois pais selecionados.
    """
    parent1 = random.choice(population)
    parent2 = random.choice(population)
    return parent1, parent2

def assign_fronts_to_population(population, fronts):
    """
    Atribui o número da frente a cada solução na população e inicializa o ranking.
    
    Args:
        population (list): Lista de soluções.
        fronts (list): Lista de frentes, onde cada frente contém índices das soluções.
    """
    for front_idx, front in enumerate(fronts):
        for idx in front:
            population[idx].front = front_idx  # Atribui o número da frente
            population[idx].rank = front_idx  # Salva o rank como atributo


############################# CALCULO DE METRICAS #################################################

def calculate_fronts(metrics):
    """
    Divide a população em frentes de Pareto.

    Args:
        metrics (np.ndarray): Matriz de métricas (n_individuals x n_objectives).

    Returns:
        list: Lista de frentes, onde cada frente é uma lista de índices.
    """
    non_dominated_sorting = NonDominatedSorting()
    fronts = non_dominated_sorting.do(metrics)
    return fronts

def calculate_crowding_distance(metrics, front):
    """
    Calcula a distância de crowding para um conjunto de indivíduos.

    Args:
        metrics (np.ndarray): Matriz de métricas (n_individuals x n_objectives).
        front (list): Índices dos indivíduos na frente de Pareto.

    Returns:
        dict: Dicionário com a distância de crowding de cada indivíduo no front.
    """
    n_objectives = metrics.shape[1]
    distances = np.zeros(len(front))

    for i in range(n_objectives):
        obj_values = metrics[front, i]
        sorted_indices = np.argsort(obj_values)
        sorted_front = np.array(front)[sorted_indices]

        # Distância infinita para os extremos
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')

        # Normalizar os valores do objetivo
        norm = obj_values[sorted_indices[-1]] - obj_values[sorted_indices[0]]
        if norm == 0:
            continue

        for j in range(1, len(sorted_front) - 1):
            distances[sorted_indices[j]] += (
                obj_values[sorted_indices[j + 1]] - obj_values[sorted_indices[j - 1]]
            ) / norm

    return {front[i]: distances[i] for i in range(len(front))}

def perform_survival(metrics, pop, n_survive):
    """
    Seleciona os indivíduos sobreviventes usando NSGA-II e atribui ranks e crowding distances.

    Args:
        metrics (np.ndarray): Matriz de métricas (n_individuals x n_objectives).
        pop (list): Lista de soluções correspondentes às métricas.
        n_survive (int): Número de indivíduos que devem sobreviver.

    Returns:
        list: Lista de indivíduos sobreviventes.
    """
    # Passo 1: Calcular as frentes de Pareto
    fronts = calculate_fronts(metrics)

    # Atribuir o número da frente (rank) às soluções
    assign_fronts_to_population(pop, fronts)

    survivors = []
    for front in fronts:
        if len(survivors) + len(front) <= n_survive:
            # Atribuir crowding distance infinita para estas soluções
            for idx in front:
                pop[idx].crowding_distance = float('inf')
            survivors.extend(front)
        else:
            # Passo 2: Calcular a distância de crowding para o último front
            crowding_distances = calculate_crowding_distance(metrics, front)

            # Atribuir a distância de crowding às soluções
            for idx in front:
                pop[idx].crowding_distance = crowding_distances[idx]

            # Ordenar pela distância de crowding em ordem decrescente
            sorted_by_crowding = sorted(front, key=lambda x: crowding_distances[x], reverse=True)

            # Adicionar até completar o número de sobreviventes
            survivors.extend(sorted_by_crowding[:n_survive - len(survivors)])
            break

    # Retorna a lista de sobreviventes como objetos da população
    return [pop[i] for i in survivors]

def visualize_pareto(population, generation):
    """
    Visualiza a frente de Pareto da população atual.
    """
    metrics = np.array([sol.metrics for sol in population])
    scatter = Scatter()
    scatter.add(metrics, label=f"Generation {generation}")
    scatter.show()

def calculate_hypervolume(population, ref_point, metrics_log=None):
    """
    Calcula o hipervolume usando o Pymoo e captura as métricas de cada geração.

    Args:
        population (list): População atual contendo soluções.
        ref_point (list): Ponto de referência para cálculo do hipervolume.
        metrics_log (list, optional): Lista para armazenar métricas de cada geração.

    Returns:
        float: Valor do hipervolume.
    """
    metrics = np.array([sol.metrics for sol in population])
    
    # Logar métricas, se necessário
    if metrics_log is not None:
        metrics_log.append(metrics)
    
    hv = HV(ref_point)
    return hv.do(metrics)

def calculate_spacing(pareto_front):
    """
    Calcula o spacing da frente de Pareto.

    Args:
        pareto_front (np.ndarray): Matriz de soluções na frente de Pareto (N x M),
                                   onde N é o número de soluções e M é o número de objetivos.

    Returns:
        float: Valor do spacing ou None se não for possível calcular.
    """
    # Verificar se há soluções suficientes na frente de Pareto
    if len(pareto_front) <= 1:
        # Não é possível calcular spacing com 1 ou 0 soluções
        return None

    # Distâncias euclidianas entre todas as soluções
    distances = squareform(pdist(pareto_front, metric='euclidean'))

    # Adicionar np.inf na diagonal para ignorar distâncias de si mesmo
    np.fill_diagonal(distances, np.inf)

    # Para cada solução, encontre a menor distância para outra solução
    min_distances = np.min(distances, axis=1)

    # Média das distâncias
    mean_distance = np.mean(min_distances)

    # Spacing: Desvio médio absoluto das distâncias mínimas em relação à média
    spacing = np.mean(np.abs(min_distances - mean_distance))
    
    return spacing

def analyze_metrics_evolution(metrics_log, normalize=False):
    """
    Analisa a evolução das métricas ao longo das gerações, com opção de normalização.

    Args:
        metrics_log (list): Lista de arrays contendo métricas de cada geração.
        normalize (bool): Se True, normaliza as métricas entre 0 e 1.

    Returns:
        dict: Contém as evoluções das melhores, piores, médias e diversidade das métricas.
    """
    evolutions = {
        "best_distance": [],
        "worst_distance": [],
        "mean_distance": [],
        "best_time": [],
        "worst_time": [],
        "mean_time": [],
        "best_balance_load": [],
        "worst_balance_load": [],
        "mean_balance_load": [],
        "diversity": []
    }

    # Min e Max de cada métrica para normalização, se necessário
    if normalize:
        all_metrics = np.vstack(metrics_log)
        min_values = all_metrics.min(axis=0)
        max_values = all_metrics.max(axis=0)
    
    for metrics in metrics_log:
        if normalize:
            metrics = (metrics - min_values) / (max_values - min_values)

        # Melhor, pior e média valores para cada métrica
        evolutions["best_distance"].append(metrics[:, 0].min())
        evolutions["worst_distance"].append(metrics[:, 0].max())
        evolutions["mean_distance"].append(metrics[:, 0].mean())
        evolutions["best_time"].append(metrics[:, 1].min())
        evolutions["worst_time"].append(metrics[:, 1].max())
        evolutions["mean_time"].append(metrics[:, 1].mean())
        evolutions["best_balance_load"].append(metrics[:, 2].min())
        evolutions["worst_balance_load"].append(metrics[:, 2].max())
        evolutions["mean_balance_load"].append(metrics[:, 2].mean())

        # Diversidade (média das distâncias pareadas)
        evolutions["diversity"].append(np.mean(pdist(metrics)))

    return evolutions

def plot_metrics_evolution(evolutions, normalize=False):
    """
    Plota a evolução das métricas ao longo das gerações, com opção de normalização.

    Args:
        evolutions (dict): Dicionário contendo as evoluções das métricas.
        normalize (bool): Se True, indica que os valores são normalizados.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 12))

    title_suffix = " (Normalizado)" if normalize else ""

    # Distância
    plt.subplot(3, 1, 1)
    plt.plot(evolutions["best_distance"], label="Melhor Distância", linestyle="--")
    plt.plot(evolutions["worst_distance"], label="Pior Distância", linestyle="--")
    plt.plot(evolutions["mean_distance"], label="Média Distância", linestyle="-")
    plt.xlabel("Geração")
    plt.ylabel("Distância")
    plt.title(f"Evolução da Distância{title_suffix}")
    plt.legend()
    plt.grid()

    # Tempo Total
    plt.subplot(3, 1, 2)
    plt.plot(evolutions["best_time"], label="Melhor Tempo", linestyle="--")
    plt.plot(evolutions["worst_time"], label="Pior Tempo", linestyle="--")
    plt.plot(evolutions["mean_time"], label="Média Tempo", linestyle="-")
    plt.xlabel("Geração")
    plt.ylabel("Tempo Total")
    plt.title(f"Evolução do Tempo Total{title_suffix}")
    plt.legend()
    plt.grid()

    # Tempo de Conclusão das Tarefas Prioritárias
    plt.subplot(3, 1, 3)
    plt.plot(evolutions["best_balance_load"], label="Melhor Consumo de Bateria", linestyle="--")
    plt.plot(evolutions["worst_balance_load"], label="Pior Consumo de Bateria", linestyle="--")
    plt.plot(evolutions["mean_balance_load"], label="Média Consumo de Bateria", linestyle="-")
    plt.xlabel("Geração")
    plt.ylabel("Consumo de Bateria")
    plt.title(f"Evolução de Consumo de Bateria{title_suffix}")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()




class Pareto_Front:
    def __init__(self) -> None:
        self.capacity = 50
        self.solutions = []
        self.iterations = []
        self.distances = []
        self.times = []
        self.max_balance_loads = []
    
    def add_solution(self, new_solution):
        self.solutions.append(new_solution)

def non_dominated_sorting(population):
    fronts = []
    current_front = []
    for p in population:
        p.dominated_count = 0
        p.dominated_solutions = []
        for q in population:
            if dominates_solution(p, q):
                p.dominated_solutions.append(q)
            elif dominates_solution(q, p):
                p.dominated_count += 1
        if p.dominated_count == 0:
            current_front.append(p)
    fronts.append(current_front)
    while current_front:
        next_front = []
        for p in current_front:
            for q in p.dominated_solutions:
                q.dominated_count -= 1
                if q.dominated_count == 0:
                    next_front.append(q)
        current_front = next_front
        fronts.append(current_front)
    return fronts


def crossover_solution(parent1, parent2):
    """
    Realiza o crossover entre dois pais, reorganizando as tarefas do Pai 2 com base no Pai 1.

    Args:
        parent1 (Solution): Objeto Solution representando o primeiro pai.
        parent2 (Solution): Objeto Solution representando o segundo pai.

    Returns:
        Solution: Descendente gerado a partir dos dois pais.
    """
    # Unificar todas as tarefas alocadas
    tasks_parent1 = [task for robot_tasks in parent1.allocations for task in robot_tasks]
    tasks_parent2 = [task for robot_tasks in parent2.allocations for task in robot_tasks]

    # Selecionar subconjunto de tarefas do Pai 1
    num_tasks = random.randint(3, max(3, len(tasks_parent1) // 3))
    selected_tasks = random.sample(tasks_parent1, num_tasks)

    # Remover as tarefas selecionadas do Pai 2
    tasks_removed_from_p2 = [task for task in tasks_parent2 if task not in selected_tasks]

    # Reorganizar as tarefas removidas na ordem do Pai 1
    new_order = tasks_removed_from_p2[:]
    for task in selected_tasks:
        idx = tasks_parent2.index(task)  # Posição original no Pai 2
        new_order.insert(idx, task)

    # Repartir as tarefas de volta nos robôs com base nas alocações originais do Pai 2
    robot_task_counts = [len(robot_tasks) for robot_tasks in parent2.allocations]
    child_allocations = []
    pointer = 0
    for count in robot_task_counts:
        child_allocations.append(new_order[pointer:pointer + count])
        pointer += count

    # Criar um novo objeto Solution com as alocações do filho
    child = solution_priority.Solution(parent1.robots, parent1.tasks, allocations=child_allocations)
    return child



# Adiciona as métricas da solução atual na lista
def log_metrics_func(solution, iteration, pareto_front):
    pareto_front.iterations.append(iteration)
    pareto_front.distances.append(solution.distance)
    pareto_front.times.append(solution.time)
    pareto_front.max_balance_loads.append(solution.balance_load)


def dominates_solution(sol1, sol2, improvement = 1.01):
        """Verifica se sol1 domina sol2 com base nos atributos de interesse"""
        # Compara distância, tempo e balanceamento de carga das soluções
        return (sol1.distance <= sol2.distance and 
                sol1.time <= sol2.time and
                sol1.max_balance_load <= sol2.max_balance_load and
                (sol1.distance < improvement*sol2.distance or sol1.time < improvement*sol2.time or sol1.max_balance_load < improvement*sol2.max_balance_load))





def genetic_algorithm_vnd(robots, pop_size, time_limit, population, local, pop_kind, run):
    start_time = time.time()  # Início da medição do tempo

    #population = solution.generate_hybrid_population(robots, tasks, pop_size)
    #population = solution.generate_random_population(robots, tasks, pop_size)
    
    generation = 0
    hv_values = []  # Armazena valores de hipervolume
    metrics_log = []  # Lista para armazenar métricas de cada geração

    while time.time() - start_time < time_limit:
        # print(f"Geração: {generation}")


        for _ in range (pop_size):
            parent1, parent2 = select_random_parents(population)
            child = crossover_solution(parent1, parent2)

            child.calculate_metrics(distance_matrix)
            #child.print_solution_metrics()
            child = vnd_priority.apply_vnd(child, len(robots), distance_matrix)
            # child = apply_movnd(child)
            child.calculate_metrics(distance_matrix)
            #child.print_solution_metrics()
            #print("#####################################")
            population.append(child)

        for pop in population:
            pop.calculate_metrics(distance_matrix)

        metricas_calculadas = []
        for pop in population:
            metricas_calculadas.append(pop.metrics)
        # Converte para NumPy
        F = np.array(metricas_calculadas, dtype=float)

        population = perform_survival(F, population, n_survive=pop_size)

        # Visualização e Análise
        # visualize_pareto(population, generation)
        hv = calculate_hypervolume(population, ref_point, metrics_log=metrics_log)
        hv_values.append(hv)
        # print(f"Geração {generation}: Hipervolume: {hv}")

        # Melhor solução
        best_solution = population[0]
        # print(f"Geração {generation}: Melhor distância: {best_solution.distance}, Melhor tempo: {best_solution.time}, Melhor max_priority_time: {best_solution.max_priority_time}")

        generation += 1
    """ for sol in population:
        plot_routes_with_priority(sol) """
    # visualize_pareto(population, generation)
    # Analisar a evolução das métricas
    evolutions = analyze_metrics_evolution(metrics_log)

    # Plotar os resultados
    plot_metrics_evolution(evolutions)
    save_results_to_file(evolutions, filename=f"evolutions_{local}_{pop_kind}{run}.json")
    save_results_to_file(hv_values, filename=f"hv_{local}_{pop_kind}){run}.json")

    
    """ # Plot da evolução do hipervolume
    plt.plot(hv_values)
    plt.title("Evolução do Hipervolume")
    plt.xlabel("Gerações")
    plt.ylabel("Hipervolume")
    plt.show()
 """
    # plot_graphs.plot_pareto_solutions(population.solutions)
    
    # Retorna a melhor solução encontrada
    return population, hv, metrics_log, generation

def genetic_algorithm_standard(robots, pop_size, time_limit, population, local, pop_kind, run):
    start_time = time.time()  # Início da medição do tempo

    #population = solution.generate_hybrid_population(robots, tasks, pop_size)
    #population = solution.generate_random_population(robots, tasks, pop_size)
    
    generation = 0
    hv_values = []  # Armazena valores de hipervolume
    metrics_log = []  # Lista para armazenar métricas de cada geração

    while time.time() - start_time < time_limit:
        # print(f"Geração: {generation}")


        for _ in range (pop_size):
            parent1, parent2 = select_random_parents(population)
            child = crossover_solution(parent1, parent2)

            child.calculate_metrics(distance_matrix)
            #child.print_solution_metrics()
            # child = vnd_priority.apply_vnd(child, len(robots), distance_matrix)
            # child = apply_movnd(child)
            child.calculate_metrics(distance_matrix)
            #child.print_solution_metrics()
            #print("#####################################")
            population.append(child)

        for pop in population:
            pop.calculate_metrics(distance_matrix)
        metricas_calculadas = []
        for pop in population:
            metricas_calculadas.append(pop.metrics)
        # Converte para NumPy
        F = np.array(metricas_calculadas, dtype=float)

        population = perform_survival(F, population, n_survive=pop_size)

        # Visualização e Análise
        # visualize_pareto(population, generation)
        hv = calculate_hypervolume(population, ref_point, metrics_log=metrics_log)
        hv_values.append(hv)
        # print(f"Geração {generation}: Hipervolume: {hv}")

        # Melhor solução
        best_solution = population[0]
        # print(f"Geração {generation}: Melhor distância: {best_solution.distance}, Melhor tempo: {best_solution.time}, Melhor max_priority_time: {best_solution.max_priority_time}")

        generation += 1

    # visualize_pareto(population, generation)
    # Analisar a evolução das métricas
    evolutions = analyze_metrics_evolution(metrics_log)

    save_results_to_file(evolutions, filename=f"evolutions_{local}_{pop_kind}){run}.json")
    save_results_to_file(hv_values, filename=f"hv_{local}_{pop_kind}){run}.json")

    # Plotar os resultados
    plot_metrics_evolution(evolutions)
    
    """ # Plot da evolução do hipervolume
    plt.plot(hv_values)
    plt.title("Evolução do Hipervolume")
    plt.xlabel("Gerações")
    plt.ylabel("Hipervolume")
    plt.show()
 """
    # plot_graphs.plot_pareto_solutions(population.solutions)

    
    return population, hv, metrics_log, generation

    

# genetic_algorithm(robots, tasks, pop_size, time_limit)




def run_experiments_nsga(robots, tasks, configurations, n_runs, output_dir):
    """
    Roda o algoritmo genético várias vezes com diferentes configurações e salva os resultados em CSV.

    Args:
        robots (list): Lista de robôs.
        tasks (list): Lista de tarefas.
        configurations (list): Lista de configurações a serem testadas.
        n_runs (int): Número de rodadas por configuração.
        output_dir (str): Diretório onde salvar os resultados.

    """
    os.makedirs(output_dir, exist_ok=True)  # Garantir que o diretório de saída existe
    
    csv_file = os.path.join(output_dir, "results.csv")
    fieldnames = [
        "config_id", "run", "pop_size", "time_limit", "method", "hybrid_population", 
        "total_generations", "hypervolume", "pareto_front", "spacing", "metrics_log"
    ]
    
    # Criar o arquivo CSV e escrever o cabeçalho, caso ainda não exista
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    for config_id, config in enumerate(configurations):
        print(f"Executando Configuração {config_id + 1}/{len(configurations)}")
        pop_size = config['pop_size']
        method = config['method']

        # Inicializar população
        if config['hybrid_population']:
            population = solution_priority.generate_hybrid_population(robots, tasks, pop_size)
            
            pop_kind = 'hybrid'
        else:
            population = solution_priority.generate_random_population(robots, tasks, pop_size)
            for sol in population:
                sol.calculate_metrics(distance_matrix)
                # sol.print_solution_metrics()
            pop_kind = 'random'

        for run in range(n_runs):
            print(f"Rodada {run + 1}/{n_runs} para Configuração {config_id + 1}")

            # Extrair parâmetros da configuração
            time_limit = config['time_limit']
            ref_point = config['ref_point']

            # Executar algoritmo genético
            if method == 'standard':
                final_population, hv, metrics_log, total_generations = genetic_algorithm_standard(
                    robots=robots,
                    pop_size=int(pop_size),
                    time_limit=time_limit,
                    population=population,
                    local= method,
                    pop_kind= pop_kind,
                    run= run
                )
            elif method == 'vnd':
                final_population, hv, metrics_log, total_generations = genetic_algorithm_vnd(
                    robots=robots,
                    pop_size=int(pop_size),
                    time_limit=time_limit,
                    population=population,
                    local= method,
                    pop_kind= pop_kind,
                    run= run
                )
            else:
                raise ValueError(f"Método desconhecido: {method}")

            # Obter dados de interesse
            pareto_front = [sol.metrics for sol in final_population]

            spacing = calculate_spacing(pareto_front)
            # print(f"spacing: {spacing}")

            # Salvar resultados no CSV
            with open(csv_file, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow({
                    "config_id": config_id,
                    "run": run,
                    "pop_size": pop_size,
                    "time_limit": time_limit,
                    "method": method,
                    "hybrid_population": config['hybrid_population'],
                    "total_generations": total_generations,
                    "hypervolume": hv,
                    "pareto_front": pareto_front,
                    "spacing": spacing
                })

    print("Experimentos concluídos e resultados salvos.")


configurations = [
    {'pop_size': 50, 'time_limit': time_limit, 'method': 'vnd', 'hybrid_population': False, 'ref_point': [30000, 8000, 8000]},
    # {'pop_size': 50, 'time_limit': time_limit, 'method': 'vnd', 'hybrid_population': True, 'ref_point': [30000, 8000, 8000]},
]
""" configurations = [
    {'pop_size': 50, 'time_limit': time_limit, 'method': 'vnd', 'hybrid_population': False, 'ref_point': [12000, 8000, 8000]},

] """

n_runs = 1  # Número de rodadas por configuração
output_dir = "experiment_results_bnd"  # Diretório onde salvar os resultados

run_experiments_nsga(robots, tasks, configurations, n_runs, output_dir)