import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
import base64
def least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ax = b using least squares method
    """
    A = A.astype(float)
    b = b.astype(float)

    Q, R = qr_decomposition(A)
    d = Q.T @ b

    # Calculate Rx = d
    inverse_R = np.linalg.inv(R)
    x = inverse_R @ d

    return x

def qr_decomposition(A: np.ndarray) -> (np.ndarray, np.ndarray):
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

    # Normalization
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

        # R = np.zeros((n, m), float)
        # for i in range(n):
        #     q_i = Q[i]
        #
        #     for j in range(i, n):
        #         u_j = A[j]
        #         R[i][j] = inner_product(u_j, q_i)

    return Q, R

def gaussian_elimination(A: np.ndarray) -> np.ndarray:
    """
    Convert a matrix to row echelon form
    """
    m, n = len(A), len(A[0])
    for i in range(m):
        if A[i][i] == 0:
            for j in range(i + 1, m):
                if A[j][i] != 0:
                    A[i], A[j] = A[j], A[i]
                    break

        for j in range(i + 1, m):
            c = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= c * A[i][k]

    return A

def orthogonalize(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Orthogonalize two vectors
    """
    return (inner_product(u, v) * v) / (normalize(v) ** 2)

def is_quadratic(A: np.ndarray) -> bool:
    """
    Check if a matrix is quadratic
    """
    m, n = len(A), len(A[0])
    return m == n

def normalize(v: np.ndarray) -> float:
    """
    Compute norm of a vector
    """
    norm = 0
    for elem in v:
        norm += (elem ** 2)

    return norm ** 0.5

def inner_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute inner product of two vectors
    """
    if len(v1) != len(v2):
        return None
    
    dot_product = 0
    for i in range(len(v1)):
        dot_product += v1[i] * v2[i]
    
    return dot_product

def get(A: np.ndarray, i: int) -> np.ndarray:
    """
    Get a column of a matrix
    """
    a_i = A.copy()[:, i]
    return a_i

def append(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Append a column to a matrix
    """
    return np.insert(A, len(A[0]), v, axis=1)



### Helper functions ###

def get_poly_label(coefficients: np.ndarray) -> str:
    """
    Get polynomial label from coefficients
    """
    res = ""
    for i in range(len(coefficients)):
        coeff = coefficients[i]

        if coeff == 0:
            continue
        
        if i == 0:
            res += f"{coeff:.2f}"
        elif i == 1:
            res += f" {' + ' if coeff > 0 else ' - '} {abs(coeff):.2f}x"
        else:
            res += f" {' + ' if coeff > 0 else ' - '} {abs(coeff):.2f}x^{i}"
    
    return res

def points_to_matrix(points: list, degree: int) -> (np.ndarray, np.ndarray):
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

def print_matrix(A: np.ndarray, PRINT_PRECISION: int = 5):
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

def print_vector(v: np.ndarray, PRINT_PRECISION: int = 5):
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

def plot_solution(A: np.ndarray, B: np.ndarray):
    """
    Plot least squares solution of Ax = b
    """
    x = least_squares(A, B)
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
    plt.savefig(f'./img/{x}.png')
    
    # convert plot to image
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png')
    plt.close()
    img_stream.seek(0)
    img = Image.open(img_stream)

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64


if __name__ == "__main__":
    data_points = [(-2, 4), (-1, 1), (1, 2), (2, 1), (3, 5), (4, 6)]
    degree = 2


    A, B = points_to_matrix(data_points, degree)

    print("A:"), print_matrix(A)
    print("B:"), print_vector(B)

    Q, R = qr_decomposition(A)

    print("Q:"), print_matrix(Q)
    print("R:"), print_matrix(R)
    print("Q @ R:"), print_matrix(Q @ R)

    plot_solution(A, B)