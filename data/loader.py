from typing import Tuple, List
import csv
import numpy as np

class DataLoader:
    def __init__(self, file_name:str, batch_size:int):
        self.batch_size = batch_size
        self.file_name = file_name
        self.data_entries = self.get_data()
        self.current_index = 0

    def get_data(self):
        raise NotImplementedError
    
    def next_batch(self) -> Tuple[List[List[float]], List[List[float]]]:
        raise NotImplementedError

class CSVLoader(DataLoader):
    def __init__(self, file_name:str, batch_size:int = 32):
        super().__init__(file_name, batch_size)
    
    def get_data(self) -> List[List[str]]:
        result = []
        with open(self.file_name, 'r') as csv_file:
            next(csv_file, None)
            for line in csv_file:
                row_data = []
                for item in line.strip().split(","):
                    row_data.append(float(item.strip()))
                result.append(row_data)
        return result
    
    def next_batch(self) -> Tuple[List[List[float]], List[List[float]]]:
        np.random.shuffle(self.data_entries)
        if self.current_index + self.batch_size >= len(self.data_entries):
            np.random.shuffle(self.data_entries)        
            self.current_index = 0
        next_index = self.current_index + self.batch_size
        data = self.data_entries[self.current_index: next_index]
        self.current_index = next_index
        inputs = []
        targets = []

        for row in data:
            row_input_data = []
            row_target_data = []

            for items in row[:-1]:
                row_input_data.append(items)
            row_target_data.append(row[-1])
            
            inputs.append(row_input_data)
            targets.append(row_target_data)
        return (inputs, targets)



