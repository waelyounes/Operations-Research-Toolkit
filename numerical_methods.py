import math

def newton_raphson(func, deriv, x0, tolerance=1e-7, max_iterations=100):
    """
    Finds the root of a function using the Newton-Raphson iterative method.
    
    Parameters:
    func: The target function for which we want to find the root.
    deriv: The derivative of the function.
    x0: Initial guess for the root.
    tolerance: The stopping criterion for accuracy.
    max_iterations: Maximum number of iterations to prevent infinite loops.
    """
    
    x = x0
    for i in range(max_iterations):
        f_value = func(x)
        d_value = deriv(x)
        
        # Edge Case: Division by zero protection (Critical for AI Safety evaluations)
        if abs(d_value) < 1e-12:
            return "Derivative is too small; the method cannot converge."
        
        # Iterative update rule
        x_new = x - f_value / d_value
        
        # Check for convergence
        if abs(x_new - x) < tolerance:
            return x_new
        
        x = x_new
        
    return "The method did not converge within the maximum number of iterations."

# Example: Finding the square root of 2 (solving x^2 - 2 = 0)
if __name__ == "__main__":
    f = lambda x: x**2 - 2
    df = lambda x: 2*x
    
    initial_guess = 1.0
    result = newton_raphson(f, df, initial_guess)
    
    print(f"Approximated Root: {result}")
