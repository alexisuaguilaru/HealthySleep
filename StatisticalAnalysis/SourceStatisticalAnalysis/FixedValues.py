from pathlib import Path

RANDOM_STATE = 8013
PATH = './Datasets/' if any(file.name == 'Datasets' for file in Path('.').iterdir()) else '../Datasets/'