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
        
        Raises:
            ValueError: If required columns are missing or data types are incorrect
        """
        # Load both datasets
        self.test_data = pd.read_csv(test_data_path)
        self.student_data = pd.read_csv(student_data_path)
        
        # Validate required columns
        required_test_columns = {'student_id', 'test_time', 'protocol', 'accuracy'}
        required_student_columns = {'student_id', 'program_start_date', 'grade_level', 'school_name'}
        
        missing_test = required_test_columns - set(self.test_data.columns)
        missing_student = required_student_columns - set(self.student_data.columns)
        
        if missing_test:
            raise ValueError(f"Missing required columns in test data: {missing_test}")
        if missing_student:
            raise ValueError(f"Missing required columns in student data: {missing_student}")
        
        # Validate data types
        if not pd.api.types.is_numeric_dtype(self.test_data['student_id']):
            raise ValueError("student_id in test data must be numeric")
        if not pd.api.types.is_numeric_dtype(self.test_data['protocol']):
            raise ValueError("protocol must be numeric")
        if not pd.api.types.is_numeric_dtype(self.test_data['accuracy']):
            raise ValueError("accuracy must be numeric")
        
        # Convert dates to datetime with flexible parsing
        try:
            self.student_data['program_start_date'] = pd.to_datetime(
                self.student_data['program_start_date'],
                format='mixed'
            )
            self.test_data['test_time'] = pd.to_datetime(
                self.test_data['test_time'],
                format='mixed'
            )
        except Exception as e:
            raise ValueError(f"Error converting dates: {e}")
        
        # Validate value ranges
        if (self.test_data['accuracy'] < 0).any() or (self.test_data['accuracy'] > 1).any():
            raise ValueError("Accuracy values must be between 0 and 1")
        if (self.test_data['protocol'] < 0).any():
            raise ValueError("Protocol values must be non-negative")
        
        # Merge datasets to keep all student information
        self.data = self.test_data.merge(
            self.student_data,
            on='student_id',
            how='inner'
        )
        
        if len(self.data) == 0:
            raise ValueError("No data after merging - check student_id matches")
        
        # Sort by student_id and test_time
        self.data = self.data.sort_values(['student_id', 'test_time'])
        
        # Print merge info
        print(f"Original test records: {len(self.test_data)}")
        print(f"Students with data: {len(self.student_data)}")
        print(f"Records after merge: {len(self.data)}")
        
        # Create student index mapping
        self.student_ids = self.data['student_id'].unique()
        print(f"Number of students in final dataset: {len(self.student_ids)}")
        
        # Preprocessing
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess the raw data."""
        # Merge test data with student data
        self.data = pd.merge(self.test_data, self.student_data, on='student_id', how='inner')
        
        # Convert dates to datetime
        self.data['test_time'] = pd.to_datetime(self.data['test_time'])
        self.data['program_start_date'] = pd.to_datetime(self.data['program_start_date'])
        
        # Calculate days since start
        self.data['days_since_start'] = (self.data['test_time'] - self.data['program_start_date']).dt.days
        
        # Filter out invalid sequences (lessons before start date)
        invalid_sequences = self.data['days_since_start'] < 0
        if invalid_sequences.any():
            print(f"Removing {invalid_sequences.sum()} sequences with lessons before start date")
            self.data = self.data[~invalid_sequences].copy()
        
        # Validate remaining data
        if len(self.data) == 0:
            raise ValueError("No valid sequences remain after filtering")
        
        # Sort by student and time
        self.data = self.data.sort_values(['student_id', 'test_time'])
    
    def __getitem__(self, idx):
        """Get single sequence"""
        row = self.data.iloc[idx]
        X = torch.tensor([[row['protocol'], row['days_since_start']]], dtype=torch.float32)
        y = torch.tensor([row['accuracy']], dtype=torch.float32)
        return X, y
    
    def get_student_tensor(self, student_id):
        """Get all sequences for a student"""
        student_data = self.data[self.data['student_id'] == student_id]
        X = torch.tensor([
            [row['protocol'], row['days_since_start']] 
            for _, row in student_data.iterrows()
        ], dtype=torch.float32)
        y = torch.tensor(student_data['accuracy'].values, dtype=torch.float32)
        return X, y
    
    def get_student_data(self, student_id: int) -> pd.DataFrame:
        """Get all data for a specific student.
        
        Args:
            student_id: The student's ID number
            
        Returns:
            DataFrame containing all sequences for the student
            
        Raises:
            ValueError: If student_id is not found in the dataset
        """
        student_data = self.data[self.data['student_id'] == student_id].copy()
        if len(student_data) == 0:
            raise ValueError(f"No data found for student_id {student_id}")
        return student_data.sort_values('test_time').reset_index(drop=True)
    
    def get_sequence(self, idx: int) -> pd.DataFrame:
        """Get data for a specific sequence by index.
        
        Args:
            idx: The sequence index
            
        Returns:
            DataFrame containing the single sequence
            
        Raises:
            IndexError: If idx is out of bounds
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} is out of bounds")
        return self.data.iloc[[idx]]
    
    def __len__(self):
        return len(self.data)    
  
