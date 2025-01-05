import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ReadingScoreDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        """
        Args:
            csv_path: Path to the CSV file
            transform: Optional transform to be applied on a sample
        """
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        
        # Create student index mapping
        self.student_ids = self.data['student_id'].unique()
        self.student_idx = {id_: idx for idx, id_ in enumerate(self.student_ids)}
        
        # Create title and protocol mappings
        self.title_mapping = {title: idx for idx, title in enumerate(self.data['title'].unique())}
        self.protocol_mapping = {protocol: idx for idx, protocol in enumerate(self.data['protocol'].unique())}
        
        # Preprocessing
        self._preprocess_data()
    
    def _preprocess_data(self):
        # Convert accuracy to float
        self.data['accuracy'] = self.data['accuracy'].astype(float)
        
        # Convert protocol and lesson to integers
        self.data['protocol'] = self.data['protocol'].astype(int)
        self.data['lesson'] = self.data['lesson'].astype(int)
        
        # Convert test_time to datetime with flexible parsing
        self.data['test_time'] = pd.to_datetime(
            self.data['test_time'],
            format='mixed',  # Allow mixed formats
            errors='coerce'  # Handle any parsing errors by setting them to NaT
        )
        
        # Check for and handle any NaT values
        if self.data['test_time'].isna().any():
            print(f"Warning: {self.data['test_time'].isna().sum()} datetime values could not be parsed")
        
        # Calculate days since first observation
        start_time = self.data['test_time'].min()
        self.data['days'] = (self.data['test_time'] - start_time).dt.total_seconds() / (24 * 3600)
        
        # Normalize numeric features
        for col in ['days', 'protocol', 'lesson']:
            self.data[f'{col}_normalized'] = (self.data[col] - self.data[col].mean()) / self.data[col].std()
        
        # Create day of week one-hot encoding
        dow_dummies = pd.get_dummies(self.data['test_time'].dt.dayofweek, prefix='dow')
        
        # Create one-hot encodings for titles only
        title_dummies = pd.get_dummies(self.data['title'], prefix='title')
        
        # Combine all features
        self.data = pd.concat([
            self.data,
            dow_dummies,
            title_dummies
        ], axis=1)
        
        # Store feature names for later use
        self.feature_names = (
            ['days_normalized', 'protocol_normalized', 'lesson_normalized'] +
            [col for col in dow_dummies.columns] +
            [col for col in title_dummies.columns]
        )
    
    def get_student_data(self, student_id):
        """Get all data for a specific student"""
        return self.data[self.data['student_id'] == student_id]
    
    def get_student_tensor(self, student_id):
        """Get student data as tensor"""
        student_data = self.get_student_data(student_id)
        return self._convert_to_tensor(student_data)
    
    def _convert_to_tensor(self, df):
        """Convert relevant columns to tensor"""
        X = torch.FloatTensor(df[self.feature_names].values)
        y = torch.FloatTensor(df['accuracy'].values)
        return X, y
    
    def __len__(self):
        return len(self.student_ids)
    
    def __getitem__(self, idx):
        """
        Returns data for one student
        """
        student_id = self.student_ids[idx]
        return self.get_student_tensor(student_id)
    
    @property
    def input_size(self):
        """Return the size of the input feature vector"""
        return len(self.feature_names) 