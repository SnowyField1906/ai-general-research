import numpy as np
import matplotlib.pyplot as plt


def least_squares(A: np.array, b: np.array, method: str = "") -> np.array:
    """
    Solve Ax = b using least squares method
    """
    A = A.astype(float)
    b = b.astype(float)

    if method == "qr":
        """
        We have: A * x = b
             <=> QRx   = b
             <=> Rx    = Q^T * b
             <=> x     = R^-1 * Q^T * b
        """
        Q, R = qr_decomposition(A)

        # Calculate R^-1
        inverse_R = inverse(R)

        # Calculate x
        x = inverse_R @ Q.T @ b
        return x
    
    elif method == "svd":
        """
        We have: A * x           = b
             <=> U * Σ * V^T * x = b
             <=> Σ * V^T * x     = U^T * b
             <=> x               = V * Σ^-1 * U^T * b
             <=> x               = V * (Σ^T * Σ)^-1 * Σ^T * U^T * b
        """
        U, S, V = singular_value_decomposition(A)

        # Calculate (Σ^T * Σ)^-1 * Σ^T
        pseudo_inverse_S = pseudo_inverse(S)

        # Calculate x
        x = V @ pseudo_inverse_S @ U.T @ b

        return x
    
    else:
        """
        We have: A * x = b
             <=> x     = A^-1 * b
             <=> x     = (A^T * A)^-1 * A^T * b
        """
        # Calculate (A^T * A)^-1 * A^T
        inverse_project_A = pseudo_inverse(A)

        # Calculate x
        x = inverse_project_A @ A.T @ b
        return x

def singular_value_decomposition(A: np.array) -> (np.array, np.array, np.array):
    """
    Compute singular value decomposition of a matrix
    """
    A = A.astype(float)
    m, n = A.shape

    # Setup V
    projection_A = A.T @ A
    eigen = np.linalg.eig(projection_A)
    values, vectors = sort_eigen_by_values(eigen)
    V = vectors

    # Setup Σ
    S = np.zeros((m, n), float)
    for i in range(min(m, n)):
        S[i][i] = values[i] ** 0.5

    # Setup U
    """
    We have: A            = U * Σ * V^T
         <=> A * V        = U * Σ
         <=> A * V * Σ^-1 = U
         <=> U            = A * V * Σ^-1
         <=> U            = A * V * (Σ^T * Σ)^-1 * Σ^T
    """
    # Calculate (Σ^T * Σ)^-1 * Σ^T
    pseudo_inverse_S = pseudo_inverse(S)

    # Calculate U
    U = A @ V @ pseudo_inverse_S

    return U, S, V

def qr_decomposition(A: np.array) -> (np.array, np.array):
    """
    Compute QR decomposition of a matrix
    """
    A = A.astype(float)
    m, n = A.shape

    # Orthogonalization
    V = np.empty((m, 0), float)
    for i in range(n):
        u_i = get(A, i)

        for j in range(i):
            v_j = get(V, j)
            u_i -= orthogonalize(u_i, v_j)
        
        v_i = u_i
        V = append(V, v_i)

    # Orthonormalization
    Q = np.empty((m, 0), float)
    for i in range(n):
        v_i = get(V, i)
        q_i = v_i / normalize(v_i)
        Q = append(Q, q_i)

    # Compute R
    """
    We have: R = I * R
         <=> R = (Q^-1 * Q) * R
         <=> R = Q^-1 * (Q * R)
         <=> R = Q^-1 * A
    Q is orthogonal, so Q^-1 = Q^T
         <=> R = Q^T * A
    """
    R = Q.T @ A

    return Q, R


# Matrix manipulation functions ----------------

def pseudo_inverse(A: np.array) -> np.array:
    """
    Compute pseudo inverse of a matrix

    A^-1 = (A^T * A)^-1 * A^T
    """
    m, n = A.shape
    if m == n:
        inverse_A = inverse(A)
        return inverse_A
    elif m < n:
        projection = A @ A.T
        inverse_projection = inverse(projection)
        pseudo_inverse = A.T @ inverse_projection

        return pseudo_inverse
    else:
        projection = A.T @ A
        inverse_projection = inverse(projection)
        pseudo_inverse = inverse_projection @ A.T

        return pseudo_inverse
    
def inverse(A: np.array) -> np.array:
    """
    Compute inverse of a matrix
    """
    return np.linalg.inv(A)

def orthogonalize(u: np.array, v: np.array) -> np.array:
    """
    Orthogonalize two vectors
    """
    return (inner_product(u, v) * v) / (normalize(v) ** 2)

def normalize(v: np.array) -> float:
    """
    Compute norm of a vector
    """
    norm = 0
    for elem in v:
        norm += (elem ** 2)

    return norm ** 0.5

def inner_product(v1: np.array, v2: np.array) -> float:
    """
    Compute inner product of two vectors
    """
    if len(v1) != len(v2):
        return None
    
    product = 0
    for i in range(len(v1)):
        product += v1[i] * v2[i]
    
    return product

def get(A: np.array, i: int) -> np.array:
    """
    Get a column of a matrix
    """
    a_i = A.copy()[:, i]
    return a_i

def append(A: np.array, v: np.array) -> np.array:
    """
    Append a column to a matrix
    """
    return np.insert(A, len(A[0]), v, axis=1)


# Helper functions ----------------------------

def sort_eigen_by_values(eigen):
    eigenvalues, eigenvectors = eigen

    sorted_indices = np.argsort(eigenvalues)[::-1]

    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    sorted_eigen = [sorted_eigenvalues, sorted_eigenvectors]

    return sorted_eigen

def points_to_matrix(points: list, degree: int) -> (np.array, np.array):
    """
    Convert a list of points to a matrix
    """
    A = []
    B = []

    for x, y in points:
        row = [x ** i for i in range(degree + 1)]
        A.append(row)
        B.append(y)

    return np.array(A), np.array(B)

def get_poly_label(coefficients: np.array) -> str:
    """
    Get polynomial label from coefficients
    """
    res = ""
    for i in range(len(coefficients)):
        coeff = coefficients[len(coefficients) - i - 1]

        if coeff == 0:
            continue
        
        if i == 0:
            res += f"{coeff:.2f}"
        elif i == 1:
            res += f" {' + ' if coeff > 0 else ' - '} {abs(coeff):.2f}x"
        else:
            res += f" {' + ' if coeff > 0 else ' - '} {abs(coeff):.2f}x^{i}"
    
    return res

def print_matrix(A: np.array, PRINT_PRECISION: int = 5):
    """
    Print formatted matrix
    """
    for row in A:
        for elem in row:
            if isinstance(elem, float):
                formatted_elem = f"{elem:.{PRINT_PRECISION}f}"
                if float(formatted_elem).is_integer():
                    formatted_elem = f"{int(float(formatted_elem))}"
                print(f"{formatted_elem:>10s}", end="")
            else:
                print(f"{elem:>10}", end="")
        print()
    print()

def print_vector(v: np.array, PRINT_PRECISION: int = 5):
    """
    Print formatted vector
    """
    for elem in v:
        if isinstance(elem, float):
            formatted_elem = f"{elem:.{PRINT_PRECISION}f}"
            if float(formatted_elem).is_integer():
                formatted_elem = f"{int(float(formatted_elem))}"
            print(f"{formatted_elem:>10s}", end="")
        else:
            print(f"{elem:>10}", end="")
    print()

def plot_solution(x: np.array, A: np.array, B: np.array):
    """
    Plot least squares solution of Ax = b
    """
    degree = len(x) - 1

    poly_coefficients = np.polyfit(A[:, 1], B, degree)
    x_curve = np.linspace(min(A[:, 1]), max(A[:, 1]), 100)
    y_curve = np.polyval(poly_coefficients, x_curve)
    plt.scatter(A[:, 1], B, label="points")

    poly_label = get_poly_label(poly_coefficients)
    plt.plot(x_curve, np.polyval(poly_coefficients, x_curve), color='red', label=f"y = {poly_label}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Least Squares Polynomial Solution")
    plt.grid(True)
    plt.show()


# Main -----------------------------------------

if __name__ == "__main__":
    # Data input -------------------------------
    data_points = [(1, 1), (2, 3), (3, 2), (4, 5), (5, 7), (6, 8), (7, 8), (8, 9), (9, 10), (10, 12)]
    degree = 9

    # Least squares solution --------------------
    A, B = points_to_matrix(data_points, degree)
    print("A:"), print_matrix(A)
    print("B:"), print_vector(B)
    x = least_squares(A, B, method="svd")
    plot_solution(x, A, B)

    # QR decomposition -------------------------
    Q, R = qr_decomposition(A)
    print("Q:"), print_matrix(Q)
    print("R:"), print_matrix(R)
    print("Q @ R:"), print_matrix(Q @ R)

    # Singular value decomposition -------------
    U, S, V = singular_value_decomposition(A)
    print("U:"), print_matrix(U)
    print("S:"), print_matrix(S)
    print("V:"), print_matrix(V)
    print("U @ S @ V.T:"), print_matrix(U @ S @ V.T)