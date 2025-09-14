from catnip.config import *


#                                  /$$                                    /$$    
#                                 | $$                                   | $$    
#   /$$$$$$$  /$$$$$$  /$$$$$$$  /$$$$$$    /$$$$$$  /$$$$$$   /$$$$$$$ /$$$$$$  
#  /$$_____/ /$$__  $$| $$__  $$|_  $$_/   /$$__  $$|____  $$ /$$_____/|_  $$_/  
# | $$      | $$  \ $$| $$  \ $$  | $$    | $$  \__/ /$$$$$$$| $$        | $$    
# | $$      | $$  | $$| $$  | $$  | $$ /$$| $$      /$$__  $$| $$        | $$ /$$
# |  $$$$$$$|  $$$$$$/| $$  | $$  |  $$$$/| $$     |  $$$$$$$|  $$$$$$$  |  $$$$/
#  \_______/ \______/ |__/  |__/   \___/  |__/      \_______/ \_______/   \___/  
                                                                               
                                                                               
def mpo_mps_contraction(mpo: List[np.ndarray], mps: List[np.ndarray]) -> List[np.ndarray]:
    """Contracts MPS with MPO to form a new MPS.

    This function performs the contraction of a Matrix Product State (MPS)
    with a Matrix Product Operator (MPO) to produce a new MPS. We assume a
    contraction of the form MPO @ MPS, where the zero-th index of the MPS is
    contracted with the first index of the MPO. This operation is analogous to
    applying an operator to a state vector as matrix @ vector. The naming
    mpo_mps is prefered to mps_mpo to keep the analogy to matrices and vectors
    clear, where the matrix is on the _left_ side of the vector.

    Schematic of the state of the mps after the first tensordot:
      o--0   0--o--1   0--o--1   0--o
      |         |         |         |
      #--2   3--#--4   3--#--4   2--#
      |         |         |         |
      1         2         2         1

    Args:
        mps: List of MPS tensors.
        mpo: List of MPO tensors.

    Returns:
        List of contracted MPS tensors.
    """
    n = len(mpo)
    mps2: List[np.ndarray] = []

    tensorA, tensorB = mps[0], mpo[0]
    td = (0, 1)
    tp = (1,0,2) 
    rs = (hsd, tensorA.shape[1]*tensorB.shape[2])
    tensorC = np.tensordot(tensorA, tensorB, axes=td).transpose(tp).reshape(rs, order='F')
    mps2.append(tensorC)

    for i in range(1, n-1):
        tensorA, tensorB = mps[i], mpo[i]
        td = (0, 1)
        tp = (2, 0, 3, 1, 4) 
        rs = (hsd, tensorA.shape[1]*tensorB.shape[2],tensorA.shape[2]*tensorB.shape[3])
        tensorC = np.tensordot(tensorA, tensorB, axes=td).transpose(tp).reshape(rs, order='F')
        mps2.append(tensorC)

    tensorA, tensorB = mps[-1], mpo[-1]
    td = (0, 1)
    tp = (1,0,2) 
    rs = (hsd, tensorA.shape[1]*tensorB.shape[2])
    tensorC = np.tensordot(tensorA, tensorB, axes=td).transpose(tp).reshape(rs, order='F')
    mps2.append(tensorC)

    return mps2

def mpo_mpo_contraction(mpoA: List[np.ndarray], mpoB: List[np.ndarray]) -> List[np.ndarray]:
    """Contracts MPS with MPO to form a new MPS.

    This function performs the contraction of a Matrix Product State (MPO)
    with a Matrix Product Operator (MPO) to produce a new MPO. The contraction
    follows the schematic:

    Args:
        mpoA: List of MPO tensors.
        mpoB: List of MPO tensors.

    Returns:
        List of MPO tensors.
    """
    sites = len(mpoA)
    mpoC = []

    tensorA, tensorB = mpoA[0], mpoB[0]
    td = (1, 0)
    tp = (0, 2, 1, 3)
    rs = (hsd, hsd, tensorA.shape[2] * tensorB.shape[2])
    tensorC = np.tensordot(tensorA, tensorB, axes=td).transpose(tp).reshape(rs, order='F')
    mpoC.append(tensorC)

    for i in range(1, sites-1):
        tensorA, tensorB = mpoA[i], mpoB[i]
        td = (1, 0)
        tp = (0, 3, 1, 4, 2, 5)
        rs = (hsd, hsd, tensorA.shape[2] * tensorB.shape[2], tensorA.shape[3] * tensorB.shape[3])
        tensorC = np.tensordot(tensorA, tensorB, axes=td).transpose(tp).reshape(rs, order='F')
        mpoC.append(tensorC)

    tensorA, tensorB = mpoA[-1], mpoB[-1]
    td = (1, 0)
    tp = (0, 2, 1, 3)
    rs = (hsd, hsd, tensorA.shape[2] * tensorB.shape[2])
    tensorC = np.tensordot(tensorA, tensorB, axes=td).transpose(tp).reshape(rs, order='F')
    mpoC.append(tensorC)
    
    return mpoC