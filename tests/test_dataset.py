import unittest
import pandas as pd
import torch
from pathlib import Path
from src.data.dataset import ReadingScoreDataset

class TestReadingScoreDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get path to dummy data
        cls.data_path = Path(__file__).parent / 'data' / 'dummy_data.csv'
        cls.dataset = ReadingScoreDataset(cls.data_path)

    def test_dataset_loading(self):
        self.assertIsNotNone(self.dataset)
        self.assertEqual(len(self.dataset.student_ids), 2)  # Two unique students

    def test_feature_creation(self):
        # Check if all expected features exist
        expected_features = ['days_normalized', 'protocol_normalized', 'lesson_normalized']
        for feature in expected_features:
            self.assertIn(feature, self.dataset.feature_names)
        
        # Check if day-of-week features exist (only test for presence, not exact number)
        dow_features = [f for f in self.dataset.feature_names if f.startswith('dow_')]
        self.assertTrue(len(dow_features) > 0, "No day-of-week features found")
        # Our dummy data only has a few days, so we can't expect all 7 days

    def test_student_tensor(self):
        # Test for first student
        student_id = self.dataset.student_ids[0]
        X, y = self.dataset.get_student_tensor(student_id)
        
        # Check tensor shapes
        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(X.dim(), 2)
        self.assertEqual(y.dim(), 1)
        
        # First student should have 3 observations
        self.assertEqual(len(y), 3)

    def test_data_normalization(self):
        # Test if normalized columns have mean close to 0 and std close to 1
        # With small datasets, we'll use a wider tolerance
        for col in ['days_normalized', 'protocol_normalized', 'lesson_normalized']:
            values = self.dataset.data[col].values
            self.assertAlmostEqual(values.mean(), 0, places=1)
            # For small datasets, std might not be exactly 1
            std = values.std()
            self.assertTrue(0.5 <= std <= 1.5, f"Standard deviation {std} is too far from 1")

    def test_accuracy_range(self):
        # Test if accuracy values are between 0 and 1
        all_accuracies = self.dataset.data['accuracy'].values
        self.assertTrue(all(0 <= acc <= 1 for acc in all_accuracies)) 