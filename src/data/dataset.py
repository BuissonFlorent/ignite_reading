from torch.utils.data import Dataset
import pandas as pd
import torch

class ReadingScoreDataset(Dataset):
    def __init__(self, test_data_path, student_data_path):
        """
        Initialize dataset with both test data and student data paths.
        
        Args:
            test_data_path: Path to raw_test_data.csv with test scores
            student_data_path: Path to raw_student_data.csv with student info
        """
        # Load both datasets
        self.test_data = pd.read_csv(test_data_path)
        self.student_data = pd.read_csv(student_data_path)
        
        # Convert dates to datetime
        self.student_data['program_start_date'] = pd.to_datetime(self.student_data['program_start_date'])
        self.test_data['test_time'] = pd.to_datetime(self.test_data['test_time'])
        
        # Merge datasets
        self.data = self.test_data.merge(
            self.student_data[['student_id', 'program_start_date']],
            on='student_id',
            how='left'
        )
        
        # Create student index mapping
        self.student_ids = self.data['student_id'].unique()
        
        # Preprocessing
        self._preprocess_data()
    
    def _preprocess_data(self):
        # Keep protocol as raw integers
        self.data['protocol'] = self.data['protocol'].astype(int)
        
        # Calculate days since program start for each test
        self.data['days_since_start'] = (
            self.data['test_time'] - self.data['program_start_date']
        ).dt.total_seconds() / (24 * 3600)  # Convert to days
    
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
        
        # Create tensors including days since start
        X = torch.tensor(student_data[['protocol', 'days_since_start']].values, dtype=torch.float)
        y = torch.FloatTensor(student_data['accuracy'].values)
        
        return X, y 