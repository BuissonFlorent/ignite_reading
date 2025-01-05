import unittest
import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import all test modules
from test_asymptotic import TestAsymptoticFunctions
from test_asymptotic_model import TestAsymptoticModel
from test_dataset import TestReadingScoreDataset
from test_parameter_handling import TestParameterHandling
from test_training import TestTraining
from test_usage import TestUsage

def run_test_case(test_case):
    """Run a single test case and return results"""
    suite = unittest.TestLoader().loadTestsFromTestCase(test_case)
    result = unittest.TestResult()
    start_time = time.time()
    suite.run(result)
    duration = time.time() - start_time
    
    return {
        'name': test_case.__name__,
        'success': result.wasSuccessful(),
        'failures': len(result.failures),
        'errors': len(result.errors),
        'duration': duration,
        'details': {
            'failures': result.failures,
            'errors': result.errors
        }
    }

def run_tests(max_workers=2):
    """
    Run all tests in parallel with limited workers to avoid file conflicts.
    Args:
        max_workers: Maximum number of parallel processes (default: 2)
    """
    test_cases = [
        # 1. Core functionality
        TestAsymptoticFunctions,
        TestAsymptoticModel,
        
        # 2. Data handling
        TestReadingScoreDataset,
        TestUsage,
        
        # 3. Training and parameter handling
        TestParameterHandling,
        TestTraining
    ]
    
    print("\nRunning tests in parallel:")
    print("1. Core functionality (asymptotic functions and model)")
    print("2. Data handling (dataset and usage)")
    print("3. Training and parameter handling\n")
    
    start_time = time.time()
    results = []
    
    # Run tests in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_test = {
            executor.submit(run_test_case, test_case): test_case.__name__
            for test_case in test_cases
        }
        
        # Process results as they complete
        for future in as_completed(future_to_test):
            test_name = future_to_test[future]
            try:
                result = future.result()
                results.append(result)
                
                # Print result immediately
                status = "✓" if result['success'] else "✗"
                print(f"{status} {test_name:<25} "
                      f"({result['duration']:.2f}s) - "
                      f"Failures: {result['failures']}, "
                      f"Errors: {result['errors']}")
                
                # Print failure/error details if any
                if not result['success']:
                    print("\nFailures:")
                    for failure in result['details']['failures']:
                        print(f"  - {failure[0]}")
                        print(f"    {failure[1]}")
                    print("\nErrors:")
                    for error in result['details']['errors']:
                        print(f"  - {error[0]}")
                        print(f"    {error[1]}")
                
            except Exception as e:
                print(f"✗ {test_name} - Exception: {str(e)}")
                results.append({
                    'name': test_name,
                    'success': False,
                    'failures': 0,
                    'errors': 1,
                    'duration': 0,
                    'details': {'error': str(e)}
                })
    
    # Print summary
    total_time = time.time() - start_time
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r['success'])
    total_failures = sum(r['failures'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    
    print(f"\nTest Summary:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Tests: {successful_tests}/{total_tests} passed")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}\n")
    
    return all(r['success'] for r in results)

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 