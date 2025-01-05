class Student:
    def __init__(self, student_id, dataset):
        """
        Args:
            student_id: Student identifier
            dataset: ReadingScoreDataset instance
        """
        self.student_id = student_id
        self.dataset = dataset
        self.data = dataset.get_student_data(student_id)
    
    @property
    def n_observations(self):
        return len(self.data)
    
    @property
    def time_range(self):
        return self.data['test_time'].min(), self.data['test_time'].max()
    
    @property
    def mean_accuracy(self):
        return self.data['accuracy'].mean()
    
    @property
    def unique_titles(self):
        return self.data['title'].unique()
    
    @property
    def protocols_used(self):
        return self.data['protocol'].unique()
    
    def get_tensor_data(self):
        return self.dataset.get_student_tensor(self.student_id) 