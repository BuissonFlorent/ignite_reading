# Asymptotic Regression Model for Reading Scores - Project TODO

## Data Processing ✅
- [x] Set up basic project structure
  - [x] Create src directory with data and examples subdirectories
  - [x] Add necessary __init__.py files
  - [x] Set up imports and PYTHONPATH
- [x] Create data handling classes
  - [x] Implement ReadingScoreDataset class
  - [x] Implement Student class
  - [x] Add basic data preprocessing
- [x] Handle data preprocessing
  - [x] Standardize time variables
  - [x] Handle accuracy values
  - [x] Normalize protocol and lesson
  - [x] Process categorical variables
- [x] Implement unit tests
  - [x] Create dummy dataset
  - [x] Test dataset functionality
  - [x] Test data loading and preprocessing
  - [x] Test PyTorch integration

## Model Development (Next Priority) 🚀
- [ ] Implement basic asymptotic functions
  - [ ] Create model/asymptotic.py
    - [ ] Implement exponential approach: y = A * (1 - e^(-k*x)) + b
    - [ ] Implement Michaelis-Menten: y = (V_max * x) / (K_m + x)
    - [ ] Implement logistic: y = L / (1 + e^(-k*(x-x0)))
  - [ ] Create tests/test_asymptotic.py
    - [ ] Test function shapes
    - [ ] Test derivatives
    - [ ] Test parameter effects
- [ ] Create base model class
  - [ ] Implement loss function
  - [ ] Add prediction method
  - [ ] Include gradient computation
- [ ] Design latent variable structure
  - [ ] Define growth rate function f(t)
  - [ ] Define score transformation function g(θ)
  - [ ] Implement random effects
  - [ ] Add measurement error component

## Training Framework
- [ ] Set up gradient descent framework
  - [ ] Implement Adam optimizer
  - [ ] Add batch processing capability
  - [ ] Create training loop
- [ ] Build covariate handling
  - [ ] Integrate protocol and lesson effects
  - [ ] Add day-of-week effects
  - [ ] Handle title effects

## Validation Framework
- [ ] Implement cross-validation
  - [ ] Set up time-series CV
  - [ ] Create student-level holdout method
- [ ] Add sensitivity analysis
  - [ ] Function comparison tests
  - [ ] Initialization tests
  - [ ] Covariate combination tests
- [ ] Add model diagnostics
  - [ ] Residual analysis
  - [ ] Growth curve validation
  - [ ] Comparison with simpler models

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
- [ ] Create example notebooks

## Next Steps (Immediate Tasks)
1. Create tests/test_asymptotic.py with test cases for:
   - Basic function evaluation
   - Parameter bounds and constraints
   - Gradient computation
   - Edge cases
2. Implement model/asymptotic.py with the simplest function first (exponential)
3. Validate with synthetic data before moving to real data
4. Add remaining asymptotic functions with tests