from catnip.config import *

#              /$$     /$$ /$$
#             | $$    |__/| $$
#  /$$   /$$ /$$$$$$   /$$| $$
# | $$  | $$|_  $$_/  | $$| $$
# | $$  | $$  | $$    | $$| $$
# | $$  | $$  | $$ /$$| $$| $$
# |  $$$$$$/  |  $$$$/| $$| $$
#  \______/    \___/  |__/|__/
                            

def svd(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Computes the singular value decomposition (SVD) of a matrix.

    Args:
        matrix: A 2D numpy array representing the input matrix.

    Returns:
        A tuple containing the SVD results (U, S, V), and the rank.
    """
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    rank = np.sum(s>num_threshold)
    U = u[:,:rank]
    S = s[:rank]
    V = vh[:rank,:]
    return U, S, V, rank

def neumann(s: np.ndarray) -> float:
    """Computes the von Neumann entropy from singular values.

    Args:
        s: Array of singular values/Schmidt coefficients.

    Returns:
        The von Neumann entropy.
    """
    p = s / np.sqrt(np.sum(s**2))
    return np.abs(-np.sum((p**2) * np.log(p**2) / np.log(hsd)))
