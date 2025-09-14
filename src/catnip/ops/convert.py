from catnip.config import *
from catnip.util import *

#                                                                /$$    
#                                                               | $$    
#   /$$$$$$$  /$$$$$$  /$$$$$$$  /$$    /$$ /$$$$$$   /$$$$$$  /$$$$$$  
#  /$$_____/ /$$__  $$| $$__  $$|  $$  /$$//$$__  $$ /$$__  $$|_  $$_/  
# | $$      | $$  \ $$| $$  \ $$ \  $$/$$/| $$$$$$$$| $$  \__/  | $$    
# | $$      | $$  | $$| $$  | $$  \  $$$/ | $$_____/| $$        | $$ /$$
# |  $$$$$$$|  $$$$$$/| $$  | $$   \  $/  |  $$$$$$$| $$        |  $$$$/
#  \_______/ \______/ |__/  |__/    \_/    \_______/|__/         \___/  
                                                                      
                                                                      
def vector_to_mps(vector: np.ndarray) -> list[np.ndarray]:
    """Converts a state vector to a Matrix Product State (MPS) representation.

    Args:
        vector: A 1D numpy array representing the state vector.

    Returns:
        A list of numpy arrays representing the MPS tensors.
    """
    sites = int(round(math.log(len(vector), hsd)))
    tensor = vector.reshape([hsd]*sites, order='F')

    mps = []
    bonds = []

    U, S, V, rank = svd(tensor.reshape(hsd, -1, order = 'F'))
    bonds.append(rank)
    mps.append(U.reshape(hsd, bonds[-1], order = 'F'))
    tensor = np.diag(S) @ V
    tensor = tensor.reshape(bonds[-1]*hsd, -1, order = 'F')

    for site in range(1, sites-1):
        U, S, V, rank = svd(tensor)
        bonds.append(rank)
        mps.append(U.reshape(bonds[-2], hsd, bonds[-1], order = 'F').transpose(1,0,2))
        tensor = np.diag(S) @ V
        tensor = tensor.reshape(bonds[-1]*hsd, -1, order = 'F')
    mps.append(tensor.reshape(bonds[-1], hsd, order = 'F').transpose(1,0))
    return mps

def mps_to_vector(mps: list[np.ndarray]) -> np.ndarray:
    """Converts a Matrix Product State (MPS) representation back to a state vector.

    Args:
        mps: A list of numpy arrays representing the MPS tensors.

    Returns:
        A 1D numpy array representing the state vector.
    """
    tensor = mps[0]
    for site in range(1, len(mps)):
        tensor = np.tensordot(tensor, mps[site], axes=(-1,1))
    vector = tensor.reshape(-1, order='F')
    return vector


def mpo_to_matrix(mpo: list[np.ndarray]) -> np.ndarray:
    """Converts a Matrix Product Operator (MPO) representation to a matrix.

    Args:
        mpo: A list of numpy arrays representing the MPO tensors.

    Returns:
        A 2D numpy array representing the matrix.
    """
    sites = len(mpo)
    tensor = mpo[0]
    for i in range(1, len(mpo)):
        tensor = np.tensordot(tensor, mpo[i], axes=(-1,2))


    tp = tuple(range(0,2*sites,2)) + tuple(range(1, 2*sites+1, 2)) 
    rs = (hsd**sites, (hsd**sites))
    matrix = tensor.transpose(tp).reshape(rs, order='F')
    return matrix


def matrix_to_mpo(matrix: np.ndarray) -> list[np.ndarray]:
    """Converts a matrix to a Matrix Product Operator (MPO) representation.

    Args:
        matrix: A 2D numpy array representing the matrix.

    Returns:
        A list of numpy arrays representing the MPO tensors.
    """
    sites = int(round(math.log(len(matrix[0]), hsd)))
    rs = tuple(2*sites*[hsd])
    tp = tuple(j for i in range(sites) for j in (i, i + sites))
    tensor = matrix.reshape(rs, order = 'F').transpose(tp)

    mpo = []
    bonds = []

    tensor = tensor.reshape(hsd*hsd, -1, order = 'F')
    U, S, V, rank = svd(tensor)
    bonds.append(rank)
    tp = (0,1,2)
    mpo.append(U.reshape(hsd, hsd, bonds[-1], order ='F').transpose(tp))
    tensor = np.diag(S) @ V
    tensor = tensor.reshape(bonds[-1]*hsd*hsd, -1, order ='F')

    for site in range(1, sites-1):
        U, S, V, rank = svd(tensor)
        bonds.append(rank)
        tp = (1,2,0,3)
        mpo.append(U.reshape(bonds[-2], hsd, hsd, bonds[-1], order ='F').transpose(tp))
        tensor = np.diag(S) @ V
        tensor = tensor.reshape(bonds[-1]*hsd*hsd, -1, order ='F')
    tp = (1,2,0)
    mpo.append(tensor.reshape(bonds[-1], hsd, hsd, order ='F').transpose(tp))
    return mpo
