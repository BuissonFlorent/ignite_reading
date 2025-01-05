# Test Suite Documentation

## Test Data Management

### Overview
The test suite uses a centralized test data management system through the `TestData` class in `test_data.py`. This ensures consistency across all tests and makes data dependencies explicit.

### Core Design Decisions
1. **Single Source of Truth**: All test data comes from TestData class
2. **Consistent Structure**: Fixed number of sequences per student, predictable values
3. **File Management**: Centralized handling of temporary test files

### Implementation Details

The TestData class provides two main methods:

def create_base_test_data(num_students=2, sequences_per_student=3):
    """Creates consistent test data structure
    Returns: test_df, student_df with:
    - student_ids: [1,1,1, 2,2,2] (for 2 students, 3 sequences each)
    - test_times: datetime objects
    - protocols: sequential (1,2,3 per student)
    - accuracies: increasing values
    """

def create_test_csvs(cleanup_files, num_students=2, sequences_per_student=3):
    """Creates and tracks temporary CSV files
    Returns: test_csv_path, student_csv_path
    """

### Usage Pattern

Example test case:
    def setUp(self):
        self.temp_files = []
        self.test_csv, self.student_csv = TestData.create_test_csvs(self.temp_files)
        self.dataset = ReadingScoreDataset(self.test_csv, self.student_csv)

### Important Guidelines
1. Do not modify dataset.py to fix test issues
2. Do not create separate test data in individual test files
3. Do not bypass TestData utilities for file creation

### Troubleshooting
When encountering test failures:
1. First check if test is using TestData correctly
2. Verify test expectations match TestData structure
3. Only consider TestData updates if multiple tests have the same need

### Data Structure
Default test data includes:
- 2 students
- 3 sequences per student
- Increasing accuracy values
- Sequential protocol numbers
- Properly formatted dates