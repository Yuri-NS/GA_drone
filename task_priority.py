class Task:
    def __init__(self, id, inspection_time, coordinates):
        self.id = id
        self.inspection_time = inspection_time
        self.coordinates = coordinates

    def to_dict(self):
        return {
            "id": self.id,
            "inspection_time": self.inspection_time,
            "coordinates": self.coordinates
        }

### TASKS ALEATÓRIAS ###
tasks = [
Task(id=0, inspection_time=1, coordinates=(0, 0)),
Task(id=1, inspection_time=1, coordinates=(2, 0)),
Task(id=2, inspection_time=1, coordinates=(4, 0)),
Task(id=3, inspection_time=1, coordinates=(6, 0)),
Task(id=4, inspection_time=1, coordinates=(8, 0)),
Task(id=5, inspection_time=1, coordinates=(10, 0)),
Task(id=6, inspection_time=1, coordinates=(12, 0)),
Task(id=7, inspection_time=1, coordinates=(14, 0)),
Task(id=8, inspection_time=1, coordinates=(16, 0)),
Task(id=9, inspection_time=1, coordinates=(18, 0)),
Task(id=10, inspection_time=1, coordinates=(0, 4)),
Task(id=11, inspection_time=1, coordinates=(2, 4)),
Task(id=12, inspection_time=1, coordinates=(4, 4)),
Task(id=13, inspection_time=1, coordinates=(6, 4)),
Task(id=14, inspection_time=1, coordinates=(8, 4)),
Task(id=15, inspection_time=1, coordinates=(10, 4)),
Task(id=16, inspection_time=1, coordinates=(12, 4)),
Task(id=17, inspection_time=1, coordinates=(14, 4)),
Task(id=18, inspection_time=1, coordinates=(16, 4)),
Task(id=19, inspection_time=1, coordinates=(18, 4)),
Task(id=20, inspection_time=1, coordinates=(0, 8)),
Task(id=21, inspection_time=1, coordinates=(2, 8)),
Task(id=22, inspection_time=1, coordinates=(4, 8)),
Task(id=23, inspection_time=1, coordinates=(6, 8)),
Task(id=24, inspection_time=1, coordinates=(8, 8)),
Task(id=25, inspection_time=1, coordinates=(10, 8)),
Task(id=26, inspection_time=1, coordinates=(12, 8)),
Task(id=27, inspection_time=1, coordinates=(14, 8)),
Task(id=28, inspection_time=1, coordinates=(16, 8)),
Task(id=29, inspection_time=1, coordinates=(18, 8)),
Task(id=30, inspection_time=1, coordinates=(0, 12)),
Task(id=31, inspection_time=1, coordinates=(2, 12)),
Task(id=32, inspection_time=1, coordinates=(4, 12)),
Task(id=33, inspection_time=1, coordinates=(6, 12)),
Task(id=34, inspection_time=1, coordinates=(8, 12)),
Task(id=35, inspection_time=1, coordinates=(10, 12)),
Task(id=36, inspection_time=1, coordinates=(12, 12)),
Task(id=37, inspection_time=1, coordinates=(14, 12)),
Task(id=38, inspection_time=1, coordinates=(16, 12)),
Task(id=39, inspection_time=1, coordinates=(18, 12)),
Task(id=40, inspection_time=1, coordinates=(0, 16)),
Task(id=41, inspection_time=1, coordinates=(2, 16)),
Task(id=42, inspection_time=1, coordinates=(4, 16)),
Task(id=43, inspection_time=1, coordinates=(6, 16)),
Task(id=44, inspection_time=1, coordinates=(8, 16)),
Task(id=45, inspection_time=1, coordinates=(10, 16)),
Task(id=46, inspection_time=1, coordinates=(12, 16)),
Task(id=47, inspection_time=1, coordinates=(14, 16)),
Task(id=48, inspection_time=1, coordinates=(16, 16)),
Task(id=49, inspection_time=1, coordinates=(18, 16)),
]