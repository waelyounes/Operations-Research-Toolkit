import numpy as np

def simplex_solver(c, A, b):
    """
    Solves a Linear Programming Maximization problem using the Simplex Method.
    
    Parameters:
    c: Coefficients of the Objective Function.
    A: Constraint Matrix.
    b: Right-hand side (RHS) values.
    """
    
    # Convert inputs to NumPy arrays for high-performance numerical computation
    num_constraints = len(b)
    num_vars = len(c)
    
    # Initialize the Simplex Tableau
    # Including slack variables to convert inequalities into equalities
    tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))
    
    tableau[:num_constraints, :num_vars] = A
    tableau[:num_constraints, num_vars:num_vars + num_constraints] = np.eye(num_constraints)
    tableau[:num_constraints, -1] = b
    tableau[-1, :num_vars] = -c
    
    # Iterate while there are negative indicators in the bottom row
    while np.any(tableau[-1, :-1] < 0):
        # Pivot Column Selection: Choosing the most negative indicator
        pivot_col = np.argmin(tableau[-1, :-1])
        
        # Pivot Row Selection: Applying the Minimum Ratio Test
        # Includes edge-case handling for division by zero or negative values
        ratios = []
        for i in range(num_constraints):
            val = tableau[i, pivot_col]
            if val > 0:
                ratios.append(tableau[i, -1] / val)
            else:
                ratios.append(np.inf)
        
        pivot_row = np.argmin(ratios)
        
        # Check for unbounded solution case
        if ratios[pivot_row] == np.inf:
            return "Unbounded solution detected."

        # Perform the Pivot Operation to update the Tableau
        pivot_val = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_val
        
        for i in range(num_constraints + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
                
    # Return the optimized value of the Objective Function (bottom-right cell)
    return tableau[-1, -1]

# Example Case: Maximize Z = 3x + 2y
c = np.array([3, 2])
A = np.array([[2, 1], [1, 2]])
b = np.array([18, 16])

result = simplex_solver(c, A, b)
print(f"The optimal value is: {result}")
