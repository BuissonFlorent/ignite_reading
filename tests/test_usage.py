import unittest
from pathlib import Path
from src.data.dataset import ReadingScoreDataset
from src.data.student import Student
from torch.utils.data import DataLoader

class TestUsage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = Path(__file__).parent / 'data' / 'dummy_data.csv'
        cls.dataset = ReadingScoreDataset(cls.data_path)

    def test_basic_usage(self):
        # Test dataset creation
        self.assertIsNotNone(self.dataset)
        
        # Test feature dimensionality
        self.assertTrue(self.dataset.input_size > 0)
        
        # Test student access
        first_student_id = self.dataset.student_ids[0]
        student = Student(first_student_id, self.dataset)
        
        # Test student properties
        self.assertEqual(student.n_observations, 3)
        self.assertTrue(0 <= student.mean_accuracy <= 1)
        self.assertTrue(len(student.unique_titles) > 0)

    def test_dataloader(self):
        loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        batch = next(iter(loader))
        
        # Check if batch contains features and targets
        self.assertEqual(len(batch), 2)
        X, y = batch
        
        # Check batch dimensions
        # X shape should be [batch_size, sequence_length, n_features]
        self.assertEqual(len(X.shape), 3)  # Changed from 2 to 3
        # y shape should be [batch_size, sequence_length]
        self.assertEqual(len(y.shape), 2)
        
        # Additional shape checks
        self.assertEqual(X.shape[0], 1)  # batch_size
        self.assertTrue(X.shape[2] == self.dataset.input_size)  # number of features 