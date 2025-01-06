# Visualization Requirements and Patterns

## Student Learning Trajectories

### Data Display Requirements
1. Protocol Display
   - X-axis should show protocol numbers
   - Protocol numbers must be displayed as integers
   - Protocol ticks should align with actual protocol values

2. Accuracy Display
   - Y-axis shows accuracy values (0-1)
   - Format to show 2 decimal places
   - Grid lines for easier reading

3. Test Results
   - Show actual test results as scatter points
   - Connect points with dotted line in chronological order
   - Use consistent color scheme (blue for actual data)

4. Model Predictions
   - Show model predictions as solid line
   - Use contrasting color (red) for predictions
   - Predictions should align with actual protocol numbers

### Implementation Patterns
1. Data Access
   - Always use dataset interface methods
   - Get complete student data using get_student_data()
   - Validate student eligibility using get_students_with_min_tests()

2. Plot Configuration
   - Set integer ticks for protocols
   - Format y-axis to show 2 decimal places
   - Connect points chronologically with dotted line
   - Use consistent colors (blue for actual, red for predictions)
   - Include grid for readability

3. File Output
   - Save to specified directory
   - Include timestamp in filename
   - Include student ID in filename
   - Use consistent figure size (12, 6)

### Best Practices
1. Always close figures after saving
2. Create directories if they don't exist
3. Use consistent color schemes
4. Include legends and labels
5. Separate plotting logic from data preparation
6. Document any deviations from standard patterns

### Error Handling
1. Validate input parameters
2. Check for eligible students
3. Verify data availability
4. Handle missing directories
5. Clean up resources properly