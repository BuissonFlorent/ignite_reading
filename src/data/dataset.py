from torch.utils.data import Dataset
import pandas as pd
import torch

class ReadingScoreDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        
        # Create student index mapping
        self.student_ids = self.data['student_id'].unique()
        
        # Preprocessing
        self._preprocess_data()
    
    def _preprocess_data(self):
        # Keep protocol as raw integers
        self.data['protocol'] = self.data['protocol'].astype(int)
    
    def get_student_data(self, student_id):
        """Get all data for a specific student"""
        student_data = self.data[self.data['student_id'] == student_id].sort_values('test_time')
        return student_data
    
    def __len__(self):
        """Return the number of students in the dataset"""
        return len(self.student_ids)
    
    def __getitem__(self, idx):
        """Returns data for one student"""
        student_id = self.student_ids[idx]
        student_data = self.get_student_data(student_id)
        
        # Create tensors - X is just the protocol numbers
        X = torch.FloatTensor(student_data['protocol'].values)
        y = torch.FloatTensor(student_data['accuracy'].values)
        
        return X, y 