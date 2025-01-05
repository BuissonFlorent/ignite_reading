import pandas as pd
import tempfile
import os

class TestData:
    """Shared test data for all test files"""
    
    @staticmethod
    def create_base_test_data(num_students=2, sequences_per_student=3):
        """Create configurable test datasets"""
        print(f"\nCreating test data:")
        print(f"num_students: {num_students}")
        print(f"sequences_per_student: {sequences_per_student}")
        
        student_ids = []
        test_times = []
        protocols = []
        accuracies = []
        
        for sid in range(1, num_students + 1):
            for seq in range(sequences_per_student):
                student_ids.append(sid)
                test_times.append(f'2023-01-{seq+1:02d}')
                protocols.append(seq + 1)
                accuracies.append(0.4 + 0.1 * seq)
        
        test_data = {
            'student_id': student_ids,
            'test_time': pd.to_datetime(test_times),
            'protocol': protocols,
            'accuracy': accuracies
        }
        
        student_data = {
            'student_id': list(range(1, num_students + 1)),
            'program_start_date': pd.to_datetime(['2023-01-01'] * num_students),
            'grade_level': [f'{g}th' for g in range(3, 3 + num_students)],
            'school_name': ['Test School'] * num_students
        }
        
        test_df = pd.DataFrame(test_data)
        student_df = pd.DataFrame(student_data)
        
        print(f"Generated test data shape: {test_df.shape}")
        print(f"Student IDs in test data: {test_df['student_id'].unique()}")
        print(f"Test data head:\n{test_df.head()}")
        print(f"Student data:\n{student_df}")
        
        return test_df, student_df
    
    @staticmethod
    def create_temp_file(filename, cleanup_files):
        """Create a temporary file and track it for cleanup"""
        fd, path = tempfile.mkstemp(prefix=filename)
        os.close(fd)
        cleanup_files.append(path)
        return path
    
    @staticmethod
    def create_test_csvs(cleanup_files, num_students=2, sequences_per_student=3):
        """Create temporary CSV files with configurable test data"""
        test_data, student_data = TestData.create_base_test_data(
            num_students=num_students,
            sequences_per_student=sequences_per_student
        )
        
        test_csv = TestData.create_temp_file('test_data.csv', cleanup_files)
        student_csv = TestData.create_temp_file('student_data.csv', cleanup_files)
        
        test_data.to_csv(test_csv, index=False)
        student_data.to_csv(student_csv, index=False)
        
        return test_csv, student_csv 