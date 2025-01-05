import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class ReadingScoreDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        
        # Create student index mapping
        self.student_ids = self.data['student_id'].unique()
        
        # Preprocessing
        self._preprocess_data()
    
    def _preprocess_data(self):
        # Convert test_time to datetime and calculate relative days
        self.data['test_time'] = pd.to_datetime(self.data['test_time'])
        start_time = self.data['test_time'].min()
        self.data['days'] = (self.data['test_time'] - start_time).dt.total_seconds() / (24 * 3600)
        
        # Keep protocol as raw integers
        self.data['protocol'] = self.data['protocol'].astype(int)
        
        # Store feature names
        self.feature_names = ['days']
    
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
        
        # Create tensors
        X = torch.FloatTensor(student_data[self.feature_names].values)
        protocols = torch.LongTensor(student_data['protocol'].values)
        y = torch.FloatTensor(student_data['accuracy'].values)
        
        return X, protocols, y
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader"""
        # Unzip the batch
        sequences, protocols, targets = zip(*batch)
        
        # Get lengths for packing
        lengths = [len(seq) for seq in sequences]
        
        # Sort by length for pack_padded_sequence
        sorted_idx = np.argsort(lengths)[::-1]
        sequences = [sequences[i] for i in sorted_idx]
        protocols = [protocols[i] for i in sorted_idx]
        targets = [targets[i] for i in sorted_idx]
        lengths = [lengths[i] for i in sorted_idx]
        
        # Pack sequences
        X_packed = pack_padded_sequence(
            pad_sequence(sequences, batch_first=True),
            lengths,
            batch_first=True,
            enforce_sorted=True
        )
        
        # Pack protocols
        protocols_packed = pack_padded_sequence(
            pad_sequence(protocols, batch_first=True),
            lengths,
            batch_first=True,
            enforce_sorted=True
        )
        
        # Pack targets
        y_packed = pack_padded_sequence(
            pad_sequence(targets, batch_first=True),
            lengths,
            batch_first=True,
            enforce_sorted=True
        )
        
        return X_packed, protocols_packed, y_packed 