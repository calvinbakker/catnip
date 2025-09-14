from catnip.config import *
from catnip.util import * 

#                        /$$          /$$      /$$ /$$$$$$$   /$$$$$$ 
#                       | $$         | $$$    /$$$| $$__  $$ /$$__  $$
#   /$$$$$$   /$$$$$$  /$$$$$$       | $$$$  /$$$$| $$  \ $$| $$  \__/
#  /$$__  $$ /$$__  $$|_  $$_//$$$$$$| $$ $$/$$ $$| $$$$$$$/|  $$$$$$ 
# | $$  \ $$| $$  \ $$  | $$ |______/| $$  $$$| $$| $$____/  \____  $$
# | $$  | $$| $$  | $$  | $$ /$$     | $$\  $ | $$| $$       /$$  \ $$
# |  $$$$$$/| $$$$$$$/  |  $$$$/     | $$ \/  | $$| $$      |  $$$$$$/
#  \______/ | $$____/    \___/       |__/     |__/|__/       \______/ 
#           | $$                                                      
#           | $$                                                      
#           |__/                                                      

def optimize_mps_R(mps: List[np.ndarray]) -> List[np.ndarray]:
    """Right-canonicalize an MPS using successive SVDs.

    Args:
        mps: List of MPS tensors.

    Returns:
        List of right-canonicalized MPS tensors.
    """
    mps2 = []
    sites = len(mps)

    # First pair of tensors
    tensorA, tensorB = mps[0], mps[1]
    td = (1, 1)
    rs = (hsd, hsd * tensorB.shape[2])
    tensorC = np.tensordot(tensorA, tensorB , axes=td).reshape(rs, order='F')
    U, S, V, rank = svd(tensorC)
    rs = (hsd, rank)
    mps2.append(U.reshape(rs, order='F'))

    # Loop over middle tensors
    td = (1, 0)
    rs = (rank, hsd, tensorB.shape[2])
    tp = (1, 0, 2)
    tensorA = np.tensordot(np.diag(S), V, axes=td).reshape(rs, order='F').transpose(tp)
    for i in range(2,sites-1):
        tensorB = mps[i]
        td = (2, 1)
        rs = (hsd * tensorA.shape[1], hsd * tensorB.shape[2])
        tensorC = np.tensordot(tensorA, tensorB, axes=td).reshape(rs, order='F')
        U, S, V, rank = svd(tensorC)
        rs = (hsd, tensorA.shape[1], rank)
        mps2.append(U.reshape(rs, order='F'))

        td = (1, 0)
        rs = (rank, hsd, tensorB.shape[2])
        tp = (1, 0, 2)
        tensorA = np.tensordot(np.diag(S), V, axes=td).reshape(rs, order='F').transpose(tp)

    # Last pair of tensors
    td = (2, 1)
    rs = (hsd * tensorA.shape[1], hsd)
    tensorC = np.tensordot(tensorA, mps[-1], axes=td).reshape(rs, order='F')
    U, S, V, rank = svd(tensorC)
    rs = (hsd, tensorA.shape[1], rank)
    mps2.append(U.reshape(rs, order='F'))

    td = (1, 0)
    rs = (rank, hsd)
    tp = (1, 0)
    mps2.append(np.tensordot(np.diag(S), V, axes=td).reshape(rs, order='F').transpose(tp))
    return mps2

def optimize_mps_L(mps: List[np.ndarray]) -> List[np.ndarray]:
    """Left-canonicalize an MPS using successive SVDs.

    Args:
        mps: List of MPS tensors.

    Returns:
        List of left-canonicalized MPS tensors.
    """
    mps2 = []
    sites = len(mps)

    # First pair of tensors
    tensorA, tensorB = mps[-2], mps[-1]
    td = (2, 1)
    rs = (hsd*tensorA.shape[1], hsd)
    tensorC = np.tensordot(tensorA, tensorB , axes=td).reshape(rs, order='F')
    U, S, V, rank = svd(tensorC)
    rs = (rank,hsd)
    tp = (1, 0)
    mps2.append(V.reshape(rs, order='F').transpose(tp))

    # Loop over middle tensors
    td = (1, 0)
    rs = (hsd, tensorA.shape[1], rank)
    tensorB = np.tensordot(U, np.diag(S), axes=td).reshape(rs, order='F')
    for i in range(2,sites-1):
        tensorA = mps[-i-1]
        td = (2, 1)
        rs = (hsd * tensorA.shape[1], hsd * tensorB.shape[2])
        tensorC = np.tensordot(tensorA, tensorB, axes=td).reshape(rs, order='F')
        U, S, V, rank = svd(tensorC)
        rs = (rank, hsd, tensorB.shape[2])
        tp = (1,0,2)
        mps2.append(V.reshape(rs, order='F').transpose(tp))

        td = (1, 0)
        rs = (hsd, tensorA.shape[1], rank)
        tensorB = np.tensordot(U,np.diag(S), axes=td).reshape(rs, order='F')

    # Last pair of tensors
    tensorA = mps[0]
    td = (1, 1)
    rs = (hsd, hsd * tensorB.shape[2])
    tensorC = np.tensordot(tensorA, tensorB, axes=td).reshape(rs, order='F')
    U, S, V, rank = svd(tensorC)
    rs = (rank, hsd, tensorB.shape[2])
    tp = (1, 0, 2)
    mps2.append(V.reshape(rs, order='F').transpose(tp))

    td = (1, 0)
    rs = (hsd, rank)
    mps2.append(np.tensordot(U,np.diag(S), axes=td).reshape(rs, order='F'))

    return mps2[::-1] # reverse the order

def mps_entanglement_entropy(mps: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
    """Middle-canonicalize an MPS using successive SVDs, compute entanglement entropy.

    The entanglement entropy is calculated at the middle bond of the MPS.

    Args:
        mps: List of MPS tensors.
    
    Returns:
        Tuple containing the middle-canonicalized MPSs, and the entanglement entropy.
    """
    mpsL = []
    mpsR = []
    mpsM = []
    sites = len(mps)
    middle = sites//2 # rounded down to the left-side when sites is odd
    plusone = 0 if sites % 2 == 0 else 1 # extra +1 for in the for-loop when sites is odd

    
    tensorA, tensorB = mps[0], mps[1]
    td = (1, 1)
    rs = (hsd, hsd * tensorB.shape[2])
    tensorC = np.tensordot(tensorA, tensorB , axes=td).reshape(rs, order='F')
    U, S, V, rank = svd(tensorC)
    rs = (hsd, rank)
    mpsL.append(U.reshape(rs, order='F'))

    td = (1, 0)
    rs = (rank, hsd, tensorB.shape[2])
    tp = (1, 0, 2)
    tensorA = np.tensordot(np.diag(S), V, axes=td).reshape(rs, order='F').transpose(tp)
    for i in range(2,middle):
        tensorB = mps[i]
        td = (2, 1)
        rs = (hsd * tensorA.shape[1], hsd * tensorB.shape[2])
        tensorC = np.tensordot(tensorA, tensorB, axes=td).reshape(rs, order='F')
        U, S, V, rank = svd(tensorC)
        rs = (hsd, tensorA.shape[1], rank)
        mpsL.append(U.reshape(rs, order='F'))

        td = (1, 0)
        rs = (rank, hsd, tensorB.shape[2])
        tp = (1, 0, 2)
        tensorA = np.tensordot(np.diag(S), V, axes=td).reshape(rs, order='F').transpose(tp)
    tensorL = tensorA

    tensorA, tensorB = mps[-2], mps[-1]
    td = (2, 1)
    rs = (hsd*tensorA.shape[1], hsd)
    tensorC = np.tensordot(tensorA, tensorB , axes=td).reshape(rs, order='F')
    U, S, V, rank = svd(tensorC)
    rs = (rank,hsd)
    tp = (1, 0)
    mpsR.append(V.reshape(rs, order='F').transpose(tp))

    td = (1, 0)
    rs = (hsd, tensorA.shape[1], rank)
    tensorB = np.tensordot(U, np.diag(S), axes=td).reshape(rs, order='F')
    for i in range(2,middle+plusone):
        tensorA = mps[-i-1]
        td = (2, 1)
        rs = (hsd * tensorA.shape[1], hsd * tensorB.shape[2])
        tensorC = np.tensordot(tensorA, tensorB, axes=td).reshape(rs, order='F')
        U, S, V, rank = svd(tensorC)
        rs = (rank, hsd, tensorB.shape[2])
        tp = (1,0,2)
        mpsR.append(V.reshape(rs, order='F').transpose(tp))

        td = (1, 0)
        rs = (hsd, tensorA.shape[1], rank)
        tensorB = np.tensordot(U,np.diag(S), axes=td).reshape(rs, order='F')
    tensorR = tensorB


    td = (2, 1)
    rs = (hsd * tensorL.shape[1], hsd * tensorR.shape[2])
    tensorC = np.tensordot(tensorL, tensorR, axes=td).reshape(rs, order='F')
    U, S, V, rank = svd(tensorC)

    td = (1, 0)
    rs = (hsd, tensorL.shape[1], rank)
    mpsM.append(np.tensordot(U,np.diag(S), axes=td).reshape(rs, order='F'))

    rs = (rank, hsd, tensorR.shape[2])
    tp = (1,0,2)
    mpsM.append(V.reshape(rs, order='F').transpose(tp))

    mps2 = mpsL + mpsM + mpsR[::-1]
    entanglement_entropy = neumann(S)

    return mps2, entanglement_entropy
