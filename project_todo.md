# Asymptotic Regression Model for Reading Scores - Project TODO

## Model Structure Setup
- [x] Set up basic project structure
  - [x] Create src directory with data and examples subdirectories
  - [x] Add necessary __init__.py files
  - [x] Set up imports and PYTHONPATH
- [ ] Choose appropriate asymptotic function
  - [ ] Test exponential approach
  - [ ] Test Michaelis-Menten
  - [ ] Test logistic with upper bound
- [ ] Design latent variable structure
  - [ ] Define growth rate function f(t)
  - [ ] Define score transformation function g(θ)
  - [ ] Implement random effects
  - [ ] Add measurement error component

## Implementation
- [x] Create data handling classes
  - [x] Implement ReadingScoreDataset class
  - [x] Implement Student class
  - [x] Add basic data preprocessing
- [ ] Set up gradient descent framework
  - [ ] Choose automatic differentiation library (PyTorch)
  - [ ] Implement Adam optimizer
  - [ ] Add batch processing capability
- [ ] Build covariate handling
  - [x] Create numeric features (protocol, lesson)
  - [x] Add day-of-week effects
  - [x] Process categorical variables (title)

## Data Preprocessing
- [x] Handle scale issues
  - [x] Standardize time variables
  - [x] Handle accuracy values
  - [x] Normalize protocol and lesson
- [x] Process temporal data
  - [x] Convert test_time to datetime
  - [x] Calculate days since first observation
  - [x] Create day of week features

## Model Development
- [ ] Create base model class
  - [ ] Implement loss function
  - [ ] Add prediction method
  - [ ] Include gradient computation
- [ ] Add model diagnostics
  - [ ] Residual analysis
  - [ ] Growth curve validation
  - [ ] Comparison with simpler models

## Validation Framework
- [ ] Implement cross-validation
  - [ ] Set up time-series CV
  - [ ] Create student-level holdout method
- [ ] Add sensitivity analysis
  - [ ] Function comparison tests
  - [ ] Initialization tests
  - [ ] Covariate combination tests

## Advanced Features
- [ ] Add student-level random effects
  - [ ] Implement hierarchical structure
  - [ ] Handle repeated measures
- [ ] Include time-varying covariates
  - [ ] Learning environment changes
  - [ ] Growth rate interactions

## Documentation
- [x] Initial code documentation
- [x] Create basic usage examples
- [ ] Document model assumptions
- [ ] Add validation results

## Testing
- [x] Basic functionality testing
- [ ] Unit tests for core functions
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Edge case handling

## Optimization
- [ ] Profile code performance
- [ ] Optimize critical paths
- [ ] Add parallel processing where beneficial

## Deployment
- [ ] Package code
- [ ] Create requirements.txt
- [ ] Add setup instructions
- [ ] Include example notebooks

## Next Steps (Immediate Priorities)
1. Create the asymptotic regression model class
2. Implement the basic training loop
3. Add model validation functionality
4. Create visualization utilities for growth curves 