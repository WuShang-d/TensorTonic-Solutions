import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    A = np.array(A)
    B = np.zeros((A.shape[1], A.shape[0]), dtype=A.dtype)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[j, i] = A[i, j]

    return B
