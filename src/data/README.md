# Data Management

## Core Classes and Responsibilities

### ReadingScoreDataset
Primary class for data access and management. Handles:
- Data loading and preprocessing
- Sequence access
- Student data management
- Data validation

## Access Patterns

### Current Implementation
1. **By Index (get_sequence)**:
   - Returns single sequence
   - Used by training loop via __getitem__
   - Used by visualization
   ```python
   sequence = dataset.get_sequence(idx)  # Returns DataFrame with single row
   ```

2. **By Student ID (get_student_data)**:
   - Returns all sequences for a student
   - Used for student analysis
   - Ensures time-ordered sequences
   ```python
   student_data = dataset.get_student_data(student_id)  # Returns DataFrame with all student sequences
   ```

### Issues We've Encountered
1. **Unclear Relationships**:
   - No clear mapping between indices and student_ids
   - Confusion between sequence access and student data access
   - Mixed usage patterns in visualization code

2. **Data Validation Gaps**:
   - Invalid sequences (lessons before start date)
   - Missing data handling
   - Type consistency issues

3. **Common Mistakes**:
   - Using get_sequence when full student data needed
   - Assuming index corresponds to student_id
   - Not checking data validity before use

## Best Practices

### Data Access
1. Use appropriate method for your needs:
   - get_sequence() for single sequences in training
   - get_student_data() for complete student history
   - Never access self.data directly from outside

### Data Validation
1. Validate at data loading:
   - Check date consistency
   - Verify required columns
   - Ensure proper data types

2. Handle edge cases:
   - Missing data
   - Invalid sequences
   - Out-of-order dates

### Error Handling
1. Provide clear error messages
2. Validate inputs
3. Document assumptions

## Future Improvements
1. Add helper methods:
   ```python
   get_student_id(idx: int) -> int
   get_student_indices(student_id: int) -> List[int]
   ```

2. Strengthen data validation:
   - Add sequence validity checks
   - Improve type checking
   - Add data quality metrics

3. Improve documentation:
   - Add method-level examples
   - Document return types
   - Clear error conditions

## Testing Guidelines
1. Use TestData class for consistent test data
2. Test both valid and invalid cases
3. Verify data integrity after operations
4. Test all access patterns 