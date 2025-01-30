import random
import numpy as np

DISTANCE_METRIC = 'euclidean'

# Função auxiliar para calcular distâncias
def calculate_distance_and_time(point1, point2, velocity, metric=DISTANCE_METRIC):
    if metric == 'manhattan':
        distance = abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
        time = distance / velocity
        return distance, time
    elif metric == 'euclidean':
        distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
        time = distance / velocity
        return distance, time
    else:
        raise ValueError("Invalid metric. Use 'euclidean' or 'manhattan'.")

class Solution:
    def __init__(self, robots, tasks, allocations=None, id=None):
        self.robots = robots
        self.tasks = tasks
        self.allocations = allocations
        self.iteration = 0
        self.id = None
        self.distance = None
        self.time = None
        self.robots_number = None
        self.balance_load = None
        self.metrics = [0, 0, 0]
        self.dominated_count = 0
        self.dominated_solutions = []
        self.crowding_distance = 0  # Este será usado no próximo passo
        self.crowding_distance = 0  # Este será usado no próximo passo
        self.rank = None  # Adicione aqui para tornar explícito
        self.strength = 0
        self.fitness = 0

    def objectives(self):
        """
        Retorna os objetivos como uma lista, no formato esperado para cálculo de hipervolume.
        """
        return [self.distance, self.time, self.robots_number]

    # LÓGICA: A FUNÇÃO VAI ALOCANDO, PARA CADA ROBÕ ALTERNADAMENTE, A TASK MAIS PRÓXIMA A ELE RESPEITANDO SUA CAPACIDADE    
    def greedy_allocate_tasks(self, robots, tasks):
        remaining_tasks = tasks[:]

        while remaining_tasks:
            for robot in robots:
                if not remaining_tasks:
                    break

                # Se o robô já tiver tarefas alocadas, considerar o ponto de saída da última tarefa como posição atual
                if robot.allocations:
                    last_task = robot.allocations[-1]
                    closest_task = min(remaining_tasks, key=lambda task: 
                        calculate_distance_and_time(last_task.coordinates, task.coordinates, self.robots[robot].velocity))
                else:
                    # Se o robô não tiver tarefas, usa a posição inicial do robô para calcular a tarefa mais próxima
                    closest_task = min(remaining_tasks, key=lambda task: calculate_distance_and_time((robot.x, robot.y), task.coordinates, self.robots[robot].velocity))


                # Tentar alocar a tarefa ao robô
                if robot.can_allocate(closest_task):
                    robot.allocate(closest_task)
                    remaining_tasks.remove(closest_task)

                else:
                    # Se não puder alocar, marque o robô como inativo ou pule para o próximo robô
                    continue

            # Se nenhum robô puder alocar uma tarefa, encerrar o loop
            if not any(robot.can_allocate(task) for task in remaining_tasks for robot in robots):
                break
        return robots, remaining_tasks
    
    
    
    
    def calculate_robot_execution_distance_and_time(self, initial_position, battery_time, task_list, distance_matrix, robot_idx):
        """
        Calcula a distância total de execução para um conjunto de tarefas alocadas a um robô.

        Args:
            initial_position (tuple): Coordenadas iniciais do robô (x, y).
            battery_time (float): Tempo de bateria disponível para o robô.
            task_list (list): Lista de tarefas alocadas ao robô.
            distance_matrix (np.ndarray): Matriz de distâncias entre tarefas.

        Returns:
            float: Distância total de execução do robô ou float('inf') se exceder a capacidade da bateria.
        """
        inspection_time = sum(task.inspection_time for task in task_list)
        travel_distance = 0
        travel_time = 0

        if task_list:
            task_indices = [task.id for task in task_list]

            # Distância do ponto inicial até a primeira tarefa
            first_task_coordinates = task_list[0].coordinates 
            distance, time = calculate_distance_and_time(initial_position, first_task_coordinates, self.robots[robot_idx].velocity)
            travel_distance += distance
            travel_time += time

            # Distâncias entre tarefas consecutivas
            for i in range(len(task_indices) - 1):
                travel_distance += distance_matrix[task_indices[i]][task_indices[i + 1]]
                travel_time += travel_distance / self.robots[robot_idx].velocity

            # Distância da última tarefa de volta ao ponto inicial
            last_task_coordinates = task_list[-1].coordinates
            distance, time = calculate_distance_and_time(last_task_coordinates, initial_position, self.robots[robot_idx].velocity)
            travel_distance += distance
            travel_time += time

        # Penalidade por exceder a capacidade da bateria
        if travel_time > battery_time:
            #return float('inf'), float('inf')  # Penalidade se a bateria for excedida
            return battery_time * 2, battery_time * 2  # Penalidade se a bateria for excedida

        return travel_distance, inspection_time + travel_time

        
    # Calcula a distância total de execução
    def calculate_execution_distance_and_time(self, distance_matrix):
        """
        Calcula a distância total de execução para a solução atual.

        Args:
            distance_matrix (np.ndarray): Matriz de distâncias entre tarefas.

        Returns:
            float: A distância total de execução para a solução.
        """
        total_distance = 0
        last_time = 0
        max_time = 0
        num_robots = 0

        # Itera sobre os robôs e suas respectivas alocações
        for robot_idx, task_list in enumerate(self.allocations):
            initial_position = self.robots[robot_idx].initial_position
            battery_time = self.robots[robot_idx].initial_battery_time

            # Calcula a distância e tempo de execução para o robô atual
            robot_distance, robot_time = self.calculate_robot_execution_distance_and_time(
                initial_position, battery_time, task_list, distance_matrix, robot_idx
            )

            total_distance += robot_distance
            last_time = robot_time
            self.robots[robot_idx].battery_time = self.robots[robot_idx].initial_battery_time - robot_time

            if last_time > max_time:
                max_time = last_time

            if robot_idx + 1 > num_robots:
                num_robots = robot_idx + 1

            #print(f"battery {self.robots[robot_idx].battery_time}")

        # Atualiza as métricas da solução
        self.distance = total_distance
        self.metrics[0] = total_distance
        self.time = max_time
        self.metrics[1] = max_time
        #self.robots_number = num_robots
        #self.metrics[2] = num_robots

        return total_distance, max_time
    

    # SEGUNDA VERSÃO DO BALANCE LOAD: baseado na energia restante de cada robô
    def calculate_balance_load(self):
        """
        Calcula o balanceamento de carga baseado na energia remanescente dos robôs.

        Args:
            distance_matrix (np.ndarray): Matriz de distâncias entre tarefas.

        Returns:
            float: O desvio padrão das energias restantes entre os robôs.
        """
        if self.robots:
            self.balance_load = np.std([robot.battery_time for robot in self.robots]) if all(robot.battery_time for robot in self.robots) else 0
        else:
            self.balance_load = 0
        self.metrics[2] = self.balance_load
        return self.balance_load

    # Calcula todas as métricas principais
    def calculate_metrics(self, distance_matrix):
        self.calculate_execution_distance_and_time(distance_matrix)
        self.calculate_balance_load()
        # self.calculate_execution_time(distance_matrix)
    
    # Exibe métricas da solução
    def print_solution_metrics(self, label="Solution"):
        print(f"{label} distance: {self.distance}")
        print(f"{label} time: {self.time}")
        print(f"{label} balance_load: {self.balance_load}")
        print("---------------------------------------------------------")

    def to_dict(self):
        return {
            "distance": self.distance,
            "time": self.time,
            "robots_number": self.balance_load,
            "robots": [robot.to_dict() for robot in self.robots],
            "tasks": [task.to_dict() for task in self.tasks] if self.tasks else None,
        }
    
    def shallow_copy(self):
        new_robots = [robot.shallow_copy() for robot in self.robots]
        return Solution(new_robots, self.tasks)
    
    def copy(self):
        """
        Cria uma cópia da solução atual.

        Returns:
            Solution: Uma nova instância da solução com os mesmos dados.
        """
        # Cria cópias profundas dos atributos que precisam ser independentes
        copied_allocations = [list(robot_tasks) for robot_tasks in self.allocations]
        return Solution(self.robots, self.tasks, copied_allocations)
    
    def get_improvement_metric(self):
        """
        Calcula a métrica de melhoria para a solução atual.

        Returns:
            float: Valor da métrica de melhoria.
        """
        # Exemplo: Combinação ponderada de distância, tempo e balanceamento
        weight_distance = 0.3
        weight_time = 0.5
        weight_balance_load = 0.2
        weight_robots_number = 0.2

        # Inverter métricas para minimizar
        improvement_metric = (
            self.distance * weight_distance +
            self.time * weight_time +
            self.balance_load * weight_balance_load
        )

        return improvement_metric
        

    ###########################################################################################
    
    def generate_greedy_randomized_solution(self, tasks, robots, alpha=0.3):
        """
        Gera uma solução inicial usando uma estratégia gulosa randomizada.

        Args:
            tasks (list): Lista de objetos Task.
            robots (list): Lista de objetos Robot.
            alpha (float): Parâmetro de aleatoriedade para escolha gulosa.

        Returns:
            list: Lista de listas, onde cada sublista contém objetos Task alocados a cada robô.
        """
        remaining_tasks = tasks[:]
        allocations = [[] for _ in robots]  # Lista de listas para alocações

        while remaining_tasks:
            for robot_idx, robot in enumerate(robots):
                if not remaining_tasks:
                    break

                # Escolhe tarefas baseadas em distância
                distances = [
                    (task, calculate_distance_and_time(robot.initial_position, task.coordinates, self.robots[robot_idx].velocity))
                    for task in remaining_tasks
                ]
                distances.sort(key=lambda x: x[1])
                limit = max(1, int(len(distances) * alpha))
                closest_task, _ = random.choice(distances[:limit])

                # Se o robô puder alocar a tarefa (verificação externa)
                if robot.can_allocate(allocations[robot_idx], closest_task):
                    allocations[robot_idx].append(closest_task)
                    remaining_tasks.remove(closest_task)

        return allocations


    def generate_random_solution(self, tasks, num_robots):
        """
        Gera uma solução aleatória, distribuindo tarefas entre os robôs.

        Args:
            tasks (list): Lista de objetos Task.
            num_robots (int): Número de robôs.

        Returns:
            list: Lista de listas, onde cada sublista contém objetos Task alocados a cada robô.
        """
        allocations = [[] for _ in range(num_robots)]  # Lista de listas para alocar as tarefas
        for task in tasks:
            chosen_robot = random.choice(range(num_robots))  # Escolhe um robô aleatoriamente
            allocations[chosen_robot].append(task)  # Adiciona a tarefa ao robô selecionado
        return allocations


    
def generate_hybrid_population(robots, tasks, pop_size):
    print("gerei populacao")
    """
    Gera uma população inicial com soluções híbridas (aleatórias + baseadas em heurísticas).
    """
    population = []

    for _ in range(pop_size // 2):
        sol_1 = Solution(robots, tasks)
        sol_1.allocations = sol_1.generate_random_solution(tasks, len(robots))
        population.append(sol_1)

    for _ in range(pop_size // 2):
        sol_2 = Solution(robots, tasks)
        sol_2.allocations = sol_2.generate_greedy_randomized_solution(tasks, robots)
        population.append(sol_2)

    return population

def generate_random_population(robots, tasks, pop_size):
    """
    Gera uma população inicial com soluções híbridas (aleatórias + baseadas em heurísticas).
    """
    population = []

    for _ in range(pop_size):
        sol_1 = Solution(robots, tasks)
        sol_1.allocations = sol_1.generate_random_solution(tasks, len(robots))
        population.append(sol_1)

    return population