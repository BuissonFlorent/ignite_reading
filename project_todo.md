# Asymptotic Regression Model for Reading Scores - Project TODO

## Model Structure Setup
- [x] Set up basic project structure
  - [x] Create src directory with data and examples subdirectories
  - [x] Add necessary __init__.py files
  - [x] Set up imports and PYTHONPATH
- [x] Choose appropriate asymptotic function
  - [x] Test exponential approach
  - [ ] Test Michaelis-Menten
  - [ ] Test logistic with upper bound
- [x] Design latent variable structure
  - [x] Define growth rate function f(t)
  - [x] Define score transformation function g(θ)
  - [ ] Implement random effects
  - [ ] Add measurement error component

## Implementation
- [x] Create data handling classes
  - [x] Implement ReadingScoreDataset class
  - [x] Add basic data preprocessing
  - [x] Add student metadata integration
- [x] Set up gradient descent framework
  - [x] Choose automatic differentiation library (PyTorch)
  - [x] Create basic model class
  - [ ] Add batch processing capability
- [x] Build covariate handling
  - [x] Create numeric features (protocol)
  - [ ] Add day-of-week effects
  - [ ] Process categorical variables (title)

## Training Implementation
- [x] Basic training loop
  - [x] Single sequence training
  - [x] Basic visualization
  - [ ] Batch processing with packed sequences
  - [ ] Training on full dataset

## Data Preprocessing
- [x] Handle scale issues
  - [x] Standardize time variables
  - [x] Handle accuracy values
  - [x] Normalize protocol and lesson
- [x] Process temporal data
  - [x] Convert test_time to datetime
  - [x] Calculate days since first observation
  - [x] Create day of week features

## Completed ✅
- [x] Implement data loading and preprocessing
- [x] Create basic asymptotic function (exponential)
- [x] Add initial unit tests

## Next Steps (Training Implementation) 🚀
- [ ] Create basic training loop
  - [ ] Define loss function (MSE)
  - [ ] Set up optimizer (Adam)
  - [ ] Add training metrics
  - [ ] Implement basic validation
- [ ] Add model class
  - [ ] Wrap asymptotic function in nn.Module
  - [ ] Add forward method
  - [ ] Handle parameter initialization
- [ ] Create training utilities
  - [ ] Add train/validation split
  - [ ] Implement early stopping
  - [ ] Add basic logging

## Future Enhancements
- [ ] Add more asymptotic functions
  - [ ] Michaelis-Menten
  - [ ] Logistic
- [ ] Improve model architecture
  - [ ] Add student-specific parameters
  - [ ] Handle covariate effects
- [ ] Add visualization tools
  - [ ] Learning curves
  - [ ] Individual student trajectories
  - [ ] Parameter distributions

## Documentation and Testing
- [ ] Document training procedure
- [ ] Add model validation tests
- [ ] Create example notebooks