import random
import numpy as np
import solution_priority


# Função auxiliar para calcular distâncias
def calculate_distance(point1, point2, metric='euclidean'):
    if metric == 'manhattan':
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
    elif metric == 'euclidean':
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    else:
        raise ValueError("Invalid metric. Use 'euclidean' or 'manhattan'.")
    
def get_distance_with_borders(robot, distance_matrix, before_task, current_task, after_task):
    """
    Calcula a distância considerando bordas (distância para o robô se for a primeira ou última tarefa).
    """
    distance = 0
    if before_task:
        distance += distance_matrix[before_task.id][current_task.id]
    else:
        distance += calculate_distance(robot.initial_position, current_task.coordinates)

    if after_task:
        distance += distance_matrix[current_task.id][after_task.id]
    else:
        distance += calculate_distance(current_task.coordinates, robot.initial_position)

    return distance

def dominates_pareto_with_tolerance(solution_metrics, other_metrics, tolerances=1.02, improvement_factor=1.02):
    """
    Verifica se uma solução domina outra no sentido de Pareto, considerando tolerâncias para piora leve em objetivos.

    Args:
        solution_metrics (list or np.ndarray): Métricas da solução candidata.
        other_metrics (list or np.ndarray): Métricas da solução a ser comparada.
        tolerances (list or np.ndarray, optional): Valores de tolerância para piora em cada objetivo. 
                                                   Se None, considera tolerância zero.
        improvement_factor (float, optional): Fator pelo qual um objetivo deve ser significativamente melhor.

    Returns:
        bool: True se `solution_metrics` domina `other_metrics` com tolerância, False caso contrário.
    """
    solution_metrics = np.array(solution_metrics)
    other_metrics = np.array(other_metrics)
    if tolerances is None:
        tolerances = np.zeros_like(solution_metrics)

    # Verifica se é "não pior" considerando tolerâncias
    is_not_worse = np.all(solution_metrics <= other_metrics*tolerances)

    # Verifica se é significativamente melhor em pelo menos um objetivo
    is_significantly_better = np.any(solution_metrics < other_metrics / improvement_factor)

    # Se for não pior considerando tolerâncias e significativamente melhor em pelo menos um, então domina
    return is_not_worse and is_significantly_better



def swap_intra_robot_random(solution, num_robots, distance_matrix, initial_position, tolerance=0.98):
    # print("entrei")
    """
    Realiza um swap aleatório em duas tarefas alocadas ao mesmo robô.
    Verifica se a solução após o swap melhora em relação à atual.

    Args:
        allocations (list): Vetores de alocações (vetor de vetores).
        robots (list): Lista de robôs.
        distance_matrix (np.ndarray): Matriz de distâncias entre tarefas.
        tolerance (float): Tolerância para aceitar uma piora.

    Returns:
        allocations, bool: Vetores de alocações atualizados e flag indicando se houve melhoria.
    """
    # Selecionar um robô aleatoriamente
    robot_idx = random.randint(0, num_robots - 1)
    # robot_tasks = allocations[robot_idx]
    robot_tasks = solution.allocations[robot_idx]  # Trabalhar diretamente com as alocações do robô

    # Se o robô não tiver ao menos duas tarefas, não há o que trocar
    if len(robot_tasks) < 2:
        return solution, False

    # Selecionar duas tarefas aleatórias para troca
    i, j = random.sample(range(len(robot_tasks)), 2)

    # Função auxiliar para obter a distância com bordas
    def get_distance(before_task, current_task, after_task):
        distance = 0
        if before_task is not None:
            distance += distance_matrix[before_task.id][current_task.id]
        else:
            distance += calculate_distance(current_task.coordinates, initial_position)
        if after_task is not None:
            distance += distance_matrix[current_task.id][after_task.id]
        else:
            distance += calculate_distance(current_task.coordinates, initial_position)
        return distance

    # Tarefas antes e depois
    task_before_i = robot_tasks[i - 1] if i > 0 else None
    task_after_i = robot_tasks[i + 1] if i < len(robot_tasks) - 1 else None
    task_before_j = robot_tasks[j - 1] if j > 0 else None
    task_after_j = robot_tasks[j + 1] if j < len(robot_tasks) - 1 else None

    # Distância antes do swap
    current_distance = (
        get_distance(task_before_i, robot_tasks[i], task_after_i) +
        get_distance(task_before_j, robot_tasks[j], task_after_j)
    )

    # Fazer o swap
    robot_tasks[i], robot_tasks[j] = robot_tasks[j], robot_tasks[i]

    # Tarefas antes e depois para o estado pós-swap
    new_task_before_i = robot_tasks[i - 1] if i > 0 else None
    new_task_after_i = robot_tasks[i + 1] if i < len(robot_tasks) - 1 else None
    new_task_before_j = robot_tasks[j - 1] if j > 0 else None
    new_task_after_j = robot_tasks[j + 1] if j < len(robot_tasks) - 1 else None

    # Distância após o swap
    new_distance = (
        get_distance(new_task_before_i, robot_tasks[i], new_task_after_i) +
        get_distance(new_task_before_j, robot_tasks[j], new_task_after_j)
    )

    # Avaliar impacto
    distance_gain = current_distance - new_distance

    if distance_gain > 0:
        # Atualizar alocação na solução se a troca for benéfica
        solution.allocations[robot_idx] = robot_tasks[:]
        solution.calculate_metrics(distance_matrix)
        return solution, True
    

    # Atualizar as alocações
    robot_tasks[i], robot_tasks[j] = robot_tasks[j], robot_tasks[i]
    # print("Swap aceito.")
    return solution, False


def swap_with_closest_task_single_robot(solution, num_robots, distance_matrix, initial_position, tolerance=0.98):
    # solution.print_solution_metrics()
    """
    Percorre todas as tarefas de um único robô e realiza swaps com as tarefas mais próximas.

    Args:
        solution (Solution): Objeto principal da solução contendo alocações.
        robots (list): Lista de robôs.
        distance_matrix (np.ndarray): Matriz de distâncias entre tarefas.
        tolerance (float): Critério de aceitação para uma piora.

    Returns:
        Solution, bool: Solução modificada e flag indicando se houve melhoria.
    """
    # print(f"Tarefas de todos os robôs ANTES: {[task.id for robot_tasks in solution.allocations for task in robot_tasks]}")

    # Selecionar um robô aleatoriamente
    robot_idx = random.randint(0, num_robots - 1)
    # robot = robots[robot_idx]
    robot_tasks = solution.allocations[robot_idx]  # Trabalhar diretamente com as alocações do robô

    # Se o robô não tiver ao menos duas tarefas, não há o que trocar
    if len(robot_tasks) < 2:
        return solution, False

    # Função auxiliar para calcular a distância com bordas
    def get_distance(before_task, current_task, after_task):
        distance = 0
        if before_task:
            distance += distance_matrix[before_task.id][current_task.id]
        else:
            distance += calculate_distance(initial_position, current_task.coordinates)

        if after_task:
            distance += distance_matrix[current_task.id][after_task.id]
        else:
            distance += calculate_distance(current_task.coordinates, initial_position)

        return distance

    # Iterar sobre todas as tarefas do robô
    for task_idx, selected_task in enumerate(robot_tasks):
        closest_task_idx = None
        min_distance = float('inf')

        for i, task in enumerate(robot_tasks):
            if i != task_idx:
                distance = distance_matrix[selected_task.id][task.id]
                if distance < min_distance:
                    min_distance = distance
                    closest_task_idx = i

        if closest_task_idx is None:
            continue

        # Tarefas antes e depois do índice atual e do mais próximo
        task_before_current = robot_tasks[task_idx - 1] if task_idx > 0 else None
        task_after_current = robot_tasks[task_idx + 1] if task_idx < len(robot_tasks) - 1 else None
        task_before_closest = robot_tasks[closest_task_idx - 1] if closest_task_idx > 0 else None
        task_after_closest = robot_tasks[closest_task_idx + 1] if closest_task_idx < len(robot_tasks) - 1 else None

        # Distância antes do swap
        current_distance = (
            get_distance(task_before_current, robot_tasks[task_idx], task_after_current) +
            get_distance(task_before_closest, robot_tasks[closest_task_idx], task_after_closest)
        )

        # Fazer o swap
        robot_tasks[task_idx], robot_tasks[closest_task_idx] = robot_tasks[closest_task_idx], robot_tasks[task_idx]

        # Tarefas antes e depois do índice atual e do mais próximo
        new_task_before_current = robot_tasks[task_idx - 1] if task_idx > 0 else None
        new_task_after_current = robot_tasks[task_idx + 1] if task_idx < len(robot_tasks) - 1 else None
        new_task_before_closest = robot_tasks[closest_task_idx - 1] if closest_task_idx > 0 else None
        new_task_after_closest = robot_tasks[closest_task_idx + 1] if closest_task_idx < len(robot_tasks) - 1 else None

        # Distância após o swap
        new_distance = (
            get_distance(new_task_before_current, robot_tasks[task_idx], new_task_after_current) +
            get_distance(new_task_before_closest, robot_tasks[closest_task_idx], new_task_after_closest)
        )


        

        # Avaliar impacto
        distance_gain = current_distance - new_distance


        if distance_gain > 0:
            # Atualizar alocação na solução se a troca for benéfica
            solution.allocations[robot_idx] = robot_tasks[:]
            solution.calculate_metrics(distance_matrix)
            return solution, True

        # Reverter o swap caso não melhore
        robot_tasks[task_idx], robot_tasks[closest_task_idx] = robot_tasks[closest_task_idx], robot_tasks[task_idx]
        #print(f"Swap revertido. current_distance: {current_distance}, new_distance: {new_distance}")

    return solution, False

def swap_with_farthest_task_single_robot(solution, num_robots, distance_matrix, initial_position, tolerance=0.98):
    """
    Percorre todas as tarefas de um único robô e realiza swaps com as tarefas mais distantes.

    Args:
        solution (Solution): Objeto principal da solução contendo alocações.
        num_robots (int): Número total de robôs.
        distance_matrix (np.ndarray): Matriz de distâncias entre tarefas.
        tolerance (float): Critério de aceitação para uma piora.

    Returns:
        Solution, bool: Solução modificada e flag indicando se houve melhoria.
    """
    # Selecionar um robô aleatoriamente
    robot_idx = random.randint(0, num_robots - 1)
    robot_tasks = solution.allocations[robot_idx]  # Trabalhar diretamente com as alocações do robô

    # Se o robô não tiver ao menos duas tarefas, não há o que trocar
    if len(robot_tasks) < 2:
        return solution, False

    # Função auxiliar para calcular a distância com bordas
    def get_distance(before_task, current_task, after_task):
        distance = 0
        if before_task:
            distance += distance_matrix[before_task.id][current_task.id]
        else:
            distance += calculate_distance(initial_position, current_task.coordinates)

        if after_task:
            distance += distance_matrix[current_task.id][after_task.id]
        else:
            distance += calculate_distance(current_task.coordinates, initial_position)

        return distance

    # Iterar sobre todas as tarefas do robô
    for task_idx, selected_task in enumerate(robot_tasks):
        farthest_task_idx = None
        max_distance = float('-inf')

        # Encontrar a tarefa mais distante da tarefa selecionada
        for i, task in enumerate(robot_tasks):
            if i != task_idx:
                distance = distance_matrix[selected_task.id][task.id]
                if distance > max_distance:
                    max_distance = distance
                    farthest_task_idx = i

        if farthest_task_idx is None:
            continue

        # Tarefas antes e depois do índice atual e do mais distante
        task_before_current = robot_tasks[task_idx - 1] if task_idx > 0 else None
        task_after_current = robot_tasks[task_idx + 1] if task_idx < len(robot_tasks) - 1 else None
        task_before_farthest = robot_tasks[farthest_task_idx - 1] if farthest_task_idx > 0 else None
        task_after_farthest = robot_tasks[farthest_task_idx + 1] if farthest_task_idx < len(robot_tasks) - 1 else None

        # Distância antes do swap
        current_distance = (
            get_distance(task_before_current, robot_tasks[task_idx], task_after_current) +
            get_distance(task_before_farthest, robot_tasks[farthest_task_idx], task_after_farthest)
        )

        # Fazer o swap
        robot_tasks[task_idx], robot_tasks[farthest_task_idx] = robot_tasks[farthest_task_idx], robot_tasks[task_idx]

        # Tarefas antes e depois do índice atual e do mais distante após o swap
        new_task_before_current = robot_tasks[task_idx - 1] if task_idx > 0 else None
        new_task_after_current = robot_tasks[task_idx + 1] if task_idx < len(robot_tasks) - 1 else None
        new_task_before_farthest = robot_tasks[farthest_task_idx - 1] if farthest_task_idx > 0 else None
        new_task_after_farthest = robot_tasks[farthest_task_idx + 1] if farthest_task_idx < len(robot_tasks) - 1 else None

        # Distância após o swap
        new_distance = (
            get_distance(new_task_before_current, robot_tasks[task_idx], new_task_after_current) +
            get_distance(new_task_before_farthest, robot_tasks[farthest_task_idx], new_task_after_farthest)
        )

        # Avaliar impacto
        distance_gain = current_distance - new_distance


        if distance_gain > 0:
            # Atualizar alocação na solução se a troca for benéfica
            solution.allocations[robot_idx] = robot_tasks[:]
            solution.calculate_metrics(distance_matrix)
            return solution, True

        # Reverter o swap caso não melhore
        robot_tasks[task_idx], robot_tasks[farthest_task_idx] = robot_tasks[farthest_task_idx], robot_tasks[task_idx]

    return solution, False

def swap_inter_robot(solution, num_robots, distance_matrix, initial_position, tolerance=1.02, improvement_threshold=0.02):
    """
    Realiza um swap inter-robôs entre tarefas baseando-se na proximidade dos clusters.

    Args:
        solution (Solution): Objeto principal da solução contendo alocações.
        num_robots (int): Número total de robôs.
        distance_matrix (np.ndarray): Matriz de distâncias entre tarefas.
        tolerance (float): Tolerância para aceitar uma piora.
        improvement_threshold (float): Percentual mínimo de melhora em pelo menos um objetivo para aceitar piora em outro.

    Returns:
        Solution, bool: Solução modificada e flag indicando se houve melhoria.
    """
    # Selecionar dois robôs aleatoriamente
    robot_1_idx, robot_2_idx = random.sample(range(num_robots), 2)
    robot_1_tasks = solution.allocations[robot_1_idx]
    robot_2_tasks = solution.allocations[robot_2_idx]

    # Se algum dos robôs não tiver tarefas, não há o que trocar
    if not robot_1_tasks or not robot_2_tasks:
        return solution, False

    # Selecionar a tarefa de robot_1 mais próxima de qualquer tarefa de robot_2
    task_1 = min(robot_1_tasks, key=lambda t1: min(
        distance_matrix[t1.id][t2.id] for t2 in robot_2_tasks
    ))

    # Selecionar a tarefa de robot_2 mais próxima de qualquer tarefa de robot_1
    task_2 = min(robot_2_tasks, key=lambda t2: min(
        distance_matrix[t2.id][t1.id] for t1 in robot_1_tasks
    ))

    # Índices das tarefas para facilitar a troca
    index_1 = robot_1_tasks.index(task_1)
    index_2 = robot_2_tasks.index(task_2)

    # Armazenar as métricas atuais antes do swap
    previous_metrics = np.array(solution.metrics)  # Métricas antes do swap

    # Fazer o swap
    robot_1_tasks[index_1], robot_2_tasks[index_2] = robot_2_tasks[index_2], robot_1_tasks[index_1]

    # Recalcular as métricas após o swap
    solution.calculate_metrics(distance_matrix)

    # Avaliar a nova solução
    current_metrics = np.array(solution.metrics)  # Métricas após o swap
    # previous_metrics = np.array(solution.previous_metrics)  # Métricas antes do swap

    # Critério de dominação Pareto
    if dominates_pareto_with_tolerance(current_metrics, previous_metrics):
        # print("Swap inter-robôs aceito por dominação Pareto.")
        # solution.previous_metrics = current_metrics  # Atualizar as métricas anteriores
        return solution, True

    """ # Critério de tolerância: melhorar significativamente um objetivo
    improvements = (previous_metrics - current_metrics) / previous_metrics
    if np.any(improvements > improvement_threshold) and np.all(current_metrics <= previous_metrics * tolerance):
        print("Swap inter-robôs aceito com tolerância.")
        # solution.previous_metrics = current_metrics  # Atualizar as métricas anteriores
        return solution, True """

    # Reverter o swap caso não melhore
    robot_1_tasks[index_1], robot_2_tasks[index_2] = robot_2_tasks[index_2], robot_1_tasks[index_1]
    solution.calculate_metrics(distance_matrix)  # Recalcular métricas após reverter
    # print("Swap inter-robôs revertido.")
    return solution, False

def move_task_between_robots(solution, num_robots, distance_matrix, initial_position, tolerance=0.99):
    """
    Move uma tarefa de um robô para outro, considerando proximidade de clusters.

    Args:
        solution (Solution): Objeto principal da solução contendo alocações.
        num_robots (int): Número total de robôs.
        distance_matrix (np.ndarray): Matriz de distâncias entre tarefas.
        tolerance (float): Critério de aceitação para uma piora.

    Returns:
        Solution, bool: Solução modificada e flag indicando se houve melhoria.
    """
    # Selecionar dois robôs aleatoriamente
    robot_from_idx, robot_to_idx = random.sample(range(num_robots), 2)
    robot_from_tasks = solution.allocations[robot_from_idx]
    robot_to_tasks = solution.allocations[robot_to_idx]

    # Se o robô de origem não tiver tarefas, não há o que mover
    if not robot_from_tasks:
        return solution, False

    # Selecionar a tarefa mais próxima de qualquer tarefa no robô de destino
    task_to_move = min(robot_from_tasks, key=lambda task_from: 
                       min(distance_matrix[task_from.id][task_to.id] for task_to in robot_to_tasks)) \
        if robot_to_tasks else robot_from_tasks[0]

    # Armazenar a posição original da tarefa no robô de origem
    original_index_from = robot_from_tasks.index(task_to_move)

    # Se o robô de destino tiver tarefas, encontrar a tarefa mais próxima de `task_to_move`
    if robot_to_tasks:
        closest_task_to = min(robot_to_tasks, key=lambda task_to: 
                              distance_matrix[task_to.id][task_to_move.id])
        insert_index = robot_to_tasks.index(closest_task_to) + 1
    else:
        # Caso contrário, insere no início
        insert_index = 0


    # Realizar o movimento
    robot_from_tasks.remove(task_to_move)
    robot_to_tasks.insert(insert_index, task_to_move)


    # Atualizar métricas
    previous_metrics = np.array(solution.metrics)
    solution.calculate_metrics(distance_matrix)
    current_metrics = np.array(solution.metrics)

    # Critério de aceitação
    if np.all(current_metrics <= previous_metrics * tolerance):
        # Movimento aceito
        solution.allocations[robot_from_idx] = robot_from_tasks
        solution.allocations[robot_to_idx] = robot_to_tasks
        # print("Movimento entre robôs aceito.")
        return solution, True

    # Reverter o movimento
    robot_to_tasks.pop(insert_index)  # Remove a tarefa do robô de destino
    robot_from_tasks.insert(original_index_from, task_to_move)  # Insere de volta na posição original no robô de origem
    solution.allocations[robot_from_idx] = robot_from_tasks
    solution.allocations[robot_to_idx] = robot_to_tasks
    # print("Movimento entre robôs revertido.")
    return solution, False


def apply_vnd(solution, robots, distance_matrix):
    """
    Aplica o VND diretamente no objeto Solution.

    Args:
        solution (Solution): Objeto principal da solução.
        robots (list): Lista de robôs.
        distance_matrix (np.ndarray): Matriz de distâncias entre tarefas.

    Returns:
        Solution: Solução atualizada após aplicar o VND.
    """
    improved = True
    tolerance = 0.98
    initial_position = (13, 124)

    while improved:
        improved = False
        for operation in ['swap_with_closest_task_single_robot', 'swap_intra_robot_random', 'swap_with_farthest_task_single_robot', 'swap_inter_robot', 'move_task_between_robots']:
        # for operation in ['swap_with_closest_task_single_robot', 'swap_with_farthest_task_single_robot', 'swap_intra_robot_random']:
            if operation == 'swap_with_closest_task_single_robot':
                solution, improvement = swap_with_closest_task_single_robot(solution, robots, distance_matrix, initial_position, tolerance)
                if improvement:
                    improved = True
                    break
            elif operation == 'swap_with_farthest_task_single_robot':
                solution, improvement = swap_with_farthest_task_single_robot(solution, robots, distance_matrix, initial_position, tolerance)
                if improvement:
                    improved = True
                    break
            elif operation == 'swap_intra_robot_random':
                solution, improvement = swap_intra_robot_random(solution, robots, distance_matrix, initial_position, tolerance)
                if improvement:
                    improved = True
                    break
            elif operation == 'swap_inter_robot':
                solution, improvement = swap_inter_robot(solution, robots, distance_matrix, initial_position, tolerance)
                if improvement:
                    improved = True
                    break
            elif operation == 'move_task_between_robots':
                solution, improvement = move_task_between_robots(solution, robots, distance_matrix, initial_position, tolerance)
                if improvement:
                    improved = True
                    break

    return solution

def apply_vnd_movns(solution, robots, distance_matrix):
    """
    Aplica o VND diretamente no objeto Solution.

    Args:
        solution (Solution): Objeto principal da solução.
        robots (list): Lista de robôs.
        distance_matrix (np.ndarray): Matriz de distâncias entre tarefas.

    Returns:
        Solution: Solução atualizada após aplicar o VND.
    """
    improved = True
    improvements = 0
    tolerance = 0.99
    initial_position = (150, 150)

    while improved:
        # print(f"Antes: Melhor distância: {solution.distance}, Melhor tempo: {solution.time}, Melhor max_priority_time: {solution.max_priority_time}")
        improved = False
        for operation in ['swap_with_closest_task_single_robot', 'swap_intra_robot_random', 'swap_with_farthest_task_single_robot', 'swap_inter_robot', 'move_task_between_robots']:
        # for operation in ['swap_with_closest_task_single_robot', 'swap_with_farthest_task_single_robot', 'swap_intra_robot_random']:
            if operation == 'swap_with_closest_task_single_robot':
                solution, improvement = swap_with_closest_task_single_robot(solution, robots, distance_matrix, initial_position, tolerance)
                if improvement:
                    improved = True
                    improvements += 1
                    break
            elif operation == 'swap_with_farthest_task_single_robot':
                solution, improvement = swap_with_farthest_task_single_robot(solution, robots, distance_matrix, initial_position, tolerance)
                if improvement:
                    improved = True
                    improvements += 1
                    break
            elif operation == 'swap_intra_robot_random':
                solution, improvement = swap_intra_robot_random(solution, robots, distance_matrix, initial_position, tolerance)
                if improvement:
                    improved = True
                    improvements += 1
                    break
            elif operation == 'swap_inter_robot':
                solution, improvement = swap_inter_robot(solution, robots, distance_matrix, initial_position, tolerance)
                if improvement:
                    improved = True
                    improvements += 1
                    break
            elif operation == 'move_task_between_robots':
                solution, improvement = move_task_between_robots(solution, robots, distance_matrix, initial_position, tolerance)
                if improvement:
                    improved = True
                    improvements += 1
                    break

    # print(f"Depois: Melhor distância: {solution.distance}, Melhor tempo: {solution.time}, Melhor max_priority_time: {solution.max_priority_time}")
    return solution, improvements

import random

def apply_vnd_movns_ains(solution, robots, distance_matrix):
    """
    Aplica vizinhanças de forma aleatória e adaptativa. Escolhe uma vizinhança aleatoriamente
    e aplica até que não haja mais melhorias.

    Args:
        solution (Solution): Solução inicial.
        robots (list): Lista de robôs.
        distance_matrix (np.ndarray): Matriz de distâncias entre tarefas.

    Returns:
        Solution: Solução atualizada após o processo.
        int: Número de melhorias realizadas.
    """
    improvements = 0
    tolerance = 0.99
    initial_position = (150, 150)

    # Lista de operações de vizinhança disponíveis
    neighborhoods = [
        'swap_with_closest_task_single_robot',
        'swap_with_farthest_task_single_robot',
        'swap_intra_robot_random',
        'swap_inter_robot',
        'move_task_between_robots'
    ]

    while True:
        # Escolha uma vizinhança aleatoriamente
        operation = random.choice(neighborhoods)
        improvement = False

        # Aplica a operação escolhida
        if operation == 'swap_with_closest_task_single_robot':
            solution, improvement = swap_with_closest_task_single_robot(solution, robots, distance_matrix, initial_position, tolerance)
        elif operation == 'swap_with_farthest_task_single_robot':
            solution, improvement = swap_with_farthest_task_single_robot(solution, robots, distance_matrix, initial_position, tolerance)
        elif operation == 'swap_intra_robot_random':
            solution, improvement = swap_intra_robot_random(solution, robots, distance_matrix, initial_position, tolerance)
        elif operation == 'swap_inter_robot':
            solution, improvement = swap_inter_robot(solution, robots, distance_matrix, initial_position, tolerance)
        elif operation == 'move_task_between_robots':
            solution, improvement = move_task_between_robots(solution, robots, distance_matrix, initial_position, tolerance)

        # Se houver melhoria, contabilize e escolha outra vizinhança
        if improvement:
            improvements += 1
        else:
            # Se não houver melhora, saia do loop
            break

    return solution, improvements





def apply_best_neighborhood(solution, robots, distance_matrix):
    # print("rodei best_neighborhood")
    """
    Aplica a vizinhança que causa a maior melhoria em cada iteração.

    Args:
        solution (Solution): Solução inicial.
        robots (list): Lista de robôs.
        distance_matrix (np.ndarray): Matriz de distâncias entre tarefas.

    Returns:
        Solution: Solução final após aplicar a estratégia de Best Neighborhood Descent.
        int: Número de melhorias realizadas.
    """
    improvements = 0
    tolerance = 0.99
    initial_position = (150, 150)

    # Lista de operações de vizinhança disponíveis
    neighborhoods = [
        'swap_with_closest_task_single_robot',
        'swap_with_farthest_task_single_robot',
        'swap_intra_robot_random',
        'swap_inter_robot',
        'move_task_between_robots'
    ]

    improved = True
    while improved:
        improved = False
        best_solution = None
        best_improvement = float('inf')
        best_operation = None

        # Avaliar todas as vizinhanças
        for operation in neighborhoods:
            temp_solution = solution.copy()  # Cria uma cópia da solução atual
            improvement = False

            # Aplica a vizinhança
            if operation == 'swap_with_closest_task_single_robot':
                temp_solution, improvement = swap_with_closest_task_single_robot(temp_solution, robots, distance_matrix, initial_position, tolerance)

            elif operation == 'swap_with_farthest_task_single_robot':
                temp_solution, improvement = swap_with_farthest_task_single_robot(temp_solution, robots, distance_matrix, initial_position, tolerance)

            elif operation == 'swap_intra_robot_random':
                temp_solution, improvement = swap_intra_robot_random(temp_solution, robots, distance_matrix, initial_position, tolerance)

            elif operation == 'swap_inter_robot':
                temp_solution, improvement = swap_inter_robot(temp_solution, robots, distance_matrix, initial_position, tolerance)

            elif operation == 'move_task_between_robots':
                temp_solution, improvement = move_task_between_robots(temp_solution, robots, distance_matrix, initial_position, tolerance)


            # Verifica se houve melhoria e se é a melhor até agora
            if improvement and temp_solution.get_improvement_metric() < best_improvement:
                best_solution = solution_priority.Solution(temp_solution.robots, temp_solution.tasks, temp_solution.allocations)
                # best_solution = temp_solution
                best_improvement = temp_solution.get_improvement_metric()
                best_operation = operation

        # Se encontrou uma melhoria, aplica a melhor solução e continua
        if best_solution:
            # print("achei best solution")
            solution = solution_priority.Solution(best_solution.robots, best_solution.tasks, best_solution.allocations)
            solution.calculate_metrics(distance_matrix)
            # solution = best_solution
            improved = True
            improvements += 1
            # print(f"Melhoria encontrada com {best_operation}: {best_improvement}")

    return solution, improvements 

