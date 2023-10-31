import numpy as np
import math

def add(A: list, B: list) -> list:
    """
    Add two polynomial vectors
    """

    m, n = len(A), len(B)
    result = [0] * max(m, n)

    for i in range(m):
        result[i] += A[i]
    for i in range(n):
        result[i] += B[i]

    return result

def multiply(A: list, B: list) -> list:
    """
    Multiply two polynomials
    """

    m, n = len(A), len(B)
    result = [0] * (m + n - 1)

    for i in range(m):
        for j in range(n):
            result[i + j] += A[i] * B[j]

    return result

def trim(A: np.array) -> np.array:
    """
    Remove zeroes at the bounds of the matrix
    """
    
    if A.shape == (1, 1):
        return A

    non_zero_rows, non_zero_cols = np.nonzero(A)
    min_row, max_row = min(non_zero_rows), max(non_zero_rows)
    min_col, max_col = min(non_zero_cols), max(non_zero_cols)

    result = A[min_row:max_row+1, min_col:max_col+1]

    return result

def xor(A: np.array, B: np.array) -> np.array:
    """
    Compute the XOR of two matrices
    """

    result = np.array(A)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            result[i, j] = A[i, j] ^ B[i, j]

    return result

def find_active(A: np.array, i: int = 0, j: int = 0) -> tuple:
    """
    Find the index of the active element in A from (i, j)
    """
    
    m, n = A.shape

    for i in range(i, m):
        for j in range(j, n):
            if A[i][j] == 1:
                return (i, j)
            
    return (-1, -1)

def spreading(A: np.array, i: int = 0, j: int = 0) -> np.array:
    """
    Spreading algorithm to find disjoint submatrices
    """

    result = np.zeros(A.shape, dtype=int)
    m, n = A.shape

    def recursion(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or A[i][j] == 0 or result[i][j] == 1:
            return

        result[i][j] = 1

        # Spread in all four directions
        recursion(i + 1, j)
        recursion(i - 1, j)
        recursion(i, j + 1)
        recursion(i, j - 1)

    if A[i][j] == 1:
        recursion(i, j)

    return result


def first_disjointed(A: np.array, i: int, j: int) -> np.array:
    """
    Find the first disjoint submatrix of A from (i, j)
    """

    def recursion(i, j):
        result = spreading(A.copy(), i, j)
        m, n = trim(result).shape

        is_disjoint = True

        if np.array_equal(A, result):
            return None

        for i in range(m):
            sum_i_result = np.sum(result[i, :])
            sum_i_A = np.sum(A[i, :])
            if sum_i_result != sum_i_A:
                is_disjoint = False
                break
        
        if is_disjoint:
            for j in range(n):
                sum_j_result = np.sum(result[:, j])
                sum_j_A = np.sum(A[:, j])
                if sum_j_result != sum_j_A:
                    is_disjoint = False
                    break

        if not is_disjoint:
            next_i, next_j = find_active(A, i, j)
            if next_i != -1 and next_j != -1:
                return
            recursion(A, next_i, next_j)
        else:
            return result
        
    result = recursion(i, j)
    return result

def separate(A: np.array) -> list:
    """
    Separate a matrix into disjoint submatrices

    > Two parts A1 and A2 of matrix A are said to be disjoint
    > if there is no active element in A1 in the same row or column 
    > with any active element in A2.

    E.g:
      O    
    O O    
        O O
        O O

    Result:
      O
    O O
    and
    O O
    O O
    """

    result = []
    m, n = A.shape

    def recursion(A: np.array) -> list:
        if m == 1 or n == 1:
            result.append(A)
            return result

        i, j = find_active(A)
        matrix_1 = first_disjointed(A, i, j)

        if matrix_1 is None:
            result.append(A)
            return result

        matrix_2 = xor(A, matrix_1)
        matrix_1, matrix_2 = trim(matrix_1), trim(matrix_2)

        result.append(matrix_1)
        return recursion(matrix_2)

    recursion(A)
    return result

def place_rook(A: np.array) -> tuple:
    """
    Put a rook on the matrix

    E.g:
    O O
    O O
      O

    Result:
      O
      O
    and
      O
    O O
      O
    """

    i, j = find_active(A)

    # Delete all row and column where the rook is
    left = np.delete(A, i, axis=0)
    left = np.delete(left, j, axis=1)
    
    # Delete the element where the rook is
    right = A.copy()
    right[i, j] = 0

    left, right = trim(left), trim(right)

    return (left, right)
    
def rook(A: np.array) -> list:
    """
    Compute the number of ways to place rooks on a matrix
    """

    def recursion(A, polynomial = []):
        m, n = A.shape

        if m == 1 or n == 1:
            # polynomial of a vector is [1, length]
            if np.sum(A) == 0:
                return [1]
            return [1, max(A.shape)]

        matrix_1, matrix_2 = place_rook(A)
        # x * left_polynomial + right_polynomial
        left_result = [0] + recursion(matrix_1, polynomial)
        right_result = recursion(matrix_2, polynomial)
        result = add(left_result, right_result)

        polynomial = add(polynomial, result)
        return polynomial

    result = recursion(A)
    return result

    
def polynomial(A: np.array) -> np.array:
    """
    Compute the rook polynomial of a matrix
    """

    m, n = A.shape

    if m == 1 or n == 1:
        return 1
    
    submatrices = separate(A)
    result = [1]
 
    for i in range(len(submatrices)):
        matrix = submatrices[i]
        polynomial = rook(matrix)
        result = multiply(result, polynomial)

    return result

def combinations(polynomial: list) -> int:
    """
    Compute the number of combinations
    """
    
    result = 0
    n = len(polynomial)

    for i in range(n):
        result = result + (-1)**i * math.factorial(n - i) * polynomial[i]

    return result
    
            
if __name__ == "__main__":
    # A = np.array([
    #     [1, 1, 0, 0, 0],
    #     [1, 0, 0, 0, 0],
    #     [0, 0, 1, 1, 1],
    #     [0, 0, 1, 1, 0],
    #     [0, 0, 0, 0, 0]
    # ])

    # A = np.array([
    #     [1, 0, 1],
    #     [1, 1, 1],
    #     [0, 1, 0]
    # ])

    # A = np.array([
    #     [0, 1, 0, 0, 1],
    #     [0, 0, 0, 0, 1],
    #     [0, 0, 1, 0, 0],
    #     [1, 0, 1, 1, 0],
    #     [0, 0, 0, 0, 0]
    # ])

    A = np.array([
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1]
    ])

    poly = polynomial(A)
    print(poly)
    comb = combinations(poly)
    print(comb)