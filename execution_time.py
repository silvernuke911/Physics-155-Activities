import time

def execution_time(func):
    """
    Decorator to measure and print the time taken by a function to execute.
    Parameters:
        func (callable): The function to be decorated.
    Returns:
        callable: Decorated function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate execution time
        print(f"Execution time of '{func.__name__}': {execution_time:.6f} seconds")
        return result
    return wrapper