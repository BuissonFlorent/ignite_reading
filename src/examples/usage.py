from src.data.dataset import ReadingScoreDataset

# Create dataset with both data sources
dataset = ReadingScoreDataset(
    test_data_path='raw_test_data.csv',
    student_data_path='raw_student_data.csv'
)

# Example: Get data for first student
X, y = dataset[0]
print(f"First student has {len(X)} reading tests")
print(f"Protocol numbers: {X}")
print(f"Accuracy scores: {y}")

# Example: Iterate through all students
for i in range(len(dataset)):
    protocols, accuracies = dataset[i]
    print(f"Student {i} has {len(protocols)} tests") 