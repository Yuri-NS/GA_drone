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


class Robot:
    def __init__(self, id, battery_time, velocity, initial_position = (0,0)):
        self.id = id
        self.battery_time = battery_time
        self.velocity = velocity
        self.initial_battery_time = battery_time
        self.allocations = []
        self.initial_position = initial_position

    def can_allocate(self, task):
        return self.battery_time >= task.inspection_time

    def allocate(self, task):
        if not self.allocations:
            last_task_id = None
        else:
            last_task_id = self.allocations[-1]

        if last_task_id is not None:
            distance_to_new_task, time_to_new_task = calculate_distance_and_time(last_task_id.coordinates, task.coordinates, self.velocity, DISTANCE_METRIC)

        else:
            # Se não há tarefas alocadas, considere a distância e tempo inicial como 0
            distance_to_new_task = 0
            time_to_new_task = 0

        if self.can_allocate(task) and self.battery_time >= (task.inspection_time + time_to_new_task):
            self.allocations.append(task)
            self.battery_time -= (task.inspection_time + time_to_new_task)
            return True
        
        return False
    
    def can_allocate_all(self):
        total_distance = 0
        total_time = 0
        total_inspection_time = sum(task.inspection_time for task in self.allocations)

        # Se o robô tem tarefas, calculamos as distâncias entre os pontos de entrada e saída
        if self.allocations:
            # Distância do robô até a primeira tarefa
            first_task = self.allocations[0]
            distance, time = calculate_distance_and_time(first_task.coordinates, self.initial_position, self.velocity, DISTANCE_METRIC)
            total_distance += distance
            total_time += time

            # Distância entre os pontos de saída e entrada das tarefas
            for i in range(len(self.allocations) - 1):
                current_task = self.allocations[i]
                next_task = self.allocations[i + 1]
                distance, time = calculate_distance_and_time(current_task.coordinates, next_task.coordinates, self.velocity, DISTANCE_METRIC)
                total_distance += distance
                total_time += time

            # Distância de volta para a posição inicial do robô
            last_task = self.allocations[-1]
            distance, time = calculate_distance_and_time(last_task.coordinates, self.initial_position, self.velocity, DISTANCE_METRIC)
            total_distance += distance
            total_time += time

        # Verifica se a soma do deslocamento e da inspeção está dentro da capacidade de bateria
        total_required_energy = total_inspection_time + total_time

        return total_required_energy <= self.battery_time
    
    def shallow_copy(self):
        return Robot(self.id, self.battery_time, self.velocity, self.initial_position)
    

    def can_allocate(self, allocated_tasks, task):
        """
        Verifica se uma tarefa pode ser alocada ao robô com base em suas restrições.

        Args:
            robot (Robot): Objeto representando o robô.
            allocated_tasks (list): Lista de tarefas já alocadas ao robô.
            task (Task): Tarefa a ser alocada.

        Returns:
            bool: True se a tarefa puder ser alocada, False caso contrário.
        """
        # Exemplo simples: verifica se há capacidade de bateria ou carga
        total_workload = sum(t.inspection_time for t in allocated_tasks) + task.inspection_time
        return total_workload <= self.battery_time
