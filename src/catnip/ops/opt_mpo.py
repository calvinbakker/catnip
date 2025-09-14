from catnip.config import *
from catnip.util import *

#                        /$$          /$$      /$$ /$$$$$$$   /$$$$$$ 
#                       | $$         | $$$    /$$$| $$__  $$ /$$__  $$
#   /$$$$$$   /$$$$$$  /$$$$$$       | $$$$  /$$$$| $$  \ $$| $$  \ $$
#  /$$__  $$ /$$__  $$|_  $$_//$$$$$$| $$ $$/$$ $$| $$$$$$$/| $$  | $$
# | $$  \ $$| $$  \ $$  | $$ |______/| $$  $$$| $$| $$____/ | $$  | $$
# | $$  | $$| $$  | $$  | $$ /$$     | $$\  $ | $$| $$      | $$  | $$
# |  $$$$$$/| $$$$$$$/  |  $$$$/     | $$ \/  | $$| $$      |  $$$$$$/
#  \______/ | $$____/    \___/       |__/     |__/|__/       \______/ 
#           | $$                                                      
#           | $$                                                      
#           |__/                                                      

def optimize_mpo_R(mpo: List[np.ndarray]) -> List[np.ndarray]:
    """Right-canonicalize an MPO using successive SVDs.

    Args:
        mpo: List of MPO tensors.

    Returns:
        List of right-canonicalized MPO tensors.
    """
    mpo2 = []
    sites = len(mpo)

    tensorA, tensorB = mpo[0], mpo[1]
    td = (2, 2)
    rs = (hsd * hsd, hsd * hsd * tensorB.shape[3])
    tensorC = np.tensordot(tensorA, tensorB , axes=td).reshape(rs, order='F')
    U, S, V, rank = svd(tensorC)
    rs = (hsd, hsd, rank)
    mpo2.append(U.reshape(rs, order='F'))

    td = (1, 0)
    rs = (rank, hsd, hsd, tensorB.shape[3])
    tp = (1, 2, 0, 3)
    tensorA = np.tensordot(np.diag(S), V, axes=td).reshape(rs, order='F').transpose(tp)
    for i in range(2,sites-1):
        tensorB = mpo[i]
        td = (3, 2)
        rs = (hsd * hsd * tensorA.shape[2], hsd * hsd * tensorB.shape[3])
        tensorC = np.tensordot(tensorA, tensorB, axes=td).reshape(rs, order='F')
        U, S, V, rank = svd(tensorC)
        rs = (hsd, hsd, tensorA.shape[2], rank)
        mpo2.append(U.reshape(rs, order='F'))

        td = (1, 0)
        rs = (rank, hsd, hsd, tensorB.shape[3])
        tp = (1, 2, 0, 3)
        tensorA = np.tensordot(np.diag(S), V, axes=td).reshape(rs, order='F').transpose(tp)

    td = (3, 2)
    rs = (hsd * hsd* tensorA.shape[2], hsd * hsd)
    tensorC = np.tensordot(tensorA, mpo[-1], axes=td).reshape(rs, order='F')
    U, S, V, rank = svd(tensorC)
    rs = (hsd, hsd, tensorA.shape[2], rank)
    mpo2.append(U.reshape(rs, order='F'))

    td = (1, 0)
    rs = (rank, hsd, hsd)
    tp = (1, 2, 0)
    mpo2.append(np.tensordot(np.diag(S), V, axes=td).reshape(rs, order='F').transpose(tp))
    return mpo2


def optimize_mpo_L(mpo: List[np.ndarray]) -> List[np.ndarray]:
    """Left-canonicalize an MPO using successive SVDs.

    Args:
        mpo: List of MPO tensors.

    Returns:
        List of left-canonicalized MPO tensors.
    """
    mpo2 = []
    sites = len(mpo)

    tensorA, tensorB = mpo[-2], mpo[-1]
    td = (3, 2)
    rs = (hsd * hsd * tensorA.shape[2], hsd * hsd)
    tensorC = np.tensordot(tensorA, tensorB , axes=td).reshape(rs, order='F')
    U, S, V, rank = svd(tensorC)
    rs = (rank, hsd, hsd)
    tp = (1, 2, 0)
    mpo2.append(V.reshape(rs, order='F').transpose(tp))

    td = (1, 0)
    rs = (hsd, hsd, tensorA.shape[2], rank)
    tensorB = np.tensordot(U, np.diag(S), axes=td).reshape(rs, order='F')
    for i in range(2,sites-1):
        tensorA = mpo[-i-1]
        td = (3, 2)
        rs = (hsd * hsd * tensorA.shape[2], hsd * hsd * tensorB.shape[3])
        tensorC = np.tensordot(tensorA, tensorB, axes=td).reshape(rs, order='F')
        U, S, V, rank = svd(tensorC)
        rs = (rank, hsd, hsd, tensorB.shape[3])
        tp = (1, 2, 0, 3)
        mpo2.append(V.reshape(rs, order='F').transpose(tp))

        td = (1, 0)
        rs = (hsd, hsd, tensorA.shape[2], rank)
        tensorB = np.tensordot(U,np.diag(S), axes=td).reshape(rs, order='F')

    tensorA = mpo[0]
    td = (2, 2)
    rs = (hsd * hsd, hsd * hsd * tensorB.shape[3])
    tensorC = np.tensordot(tensorA, tensorB, axes=td).reshape(rs, order='F')
    U, S, V, rank = svd(tensorC)
    rs = (rank, hsd, hsd, tensorB.shape[3])
    tp = (1, 2, 0, 3)
    mpo2.append(V.reshape(rs, order='F').transpose(tp))

    td = (1, 0)
    rs = (hsd, hsd, rank)
    mpo2.append(np.tensordot(U,np.diag(S), axes=td).reshape(rs, order='F'))

    return mpo2[::-1] # reverse the order


def mpo_entanglement_entropy(mpo: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
    """Middle-canonicalize an MPO using successive SVDs, compute entanglement entropy.

    The entanglement entropy is calculated at the middle bond of the MPO.
    Args:
        mpo: List of MPO tensors.
        
    Returns:
        Tuple containing the middle-canonicalized MPOs, and the entanglement entropy.
    """
    mpoL = []
    mpoR = []
    mpoM = []
    sites = len(mpo)
    middle = sites//2 # rounded down to the left-side when sites is odd
    plusone = 0 if sites % 2 == 0 else 1 # extra +1 for in the for-loop when sites is odd

    
    tensorA, tensorB = mpo[0], mpo[1]
    td = (2, 2)
    rs = (hsd * hsd, hsd * hsd * tensorB.shape[3])
    tensorC = np.tensordot(tensorA, tensorB , axes=td).reshape(rs, order='F')
    U, S, V, rank = svd(tensorC)
    rs = (hsd, hsd, rank)
    mpoL.append(U.reshape(rs, order='F'))

    td = (1, 0)
    rs = (rank, hsd, hsd, tensorB.shape[3])
    tp = (1, 2, 0, 3)
    tensorA = np.tensordot(np.diag(S), V, axes=td).reshape(rs, order='F').transpose(tp)
    for i in range(2,middle):
        tensorB = mpo[i]
        td = (3, 2)
        rs = (hsd * hsd * tensorA.shape[2], hsd * hsd * tensorB.shape[3])
        tensorC = np.tensordot(tensorA, tensorB, axes=td).reshape(rs, order='F')
        U, S, V, rank = svd(tensorC)
        rs = (hsd, hsd, tensorA.shape[2], rank)
        mpoL.append(U.reshape(rs, order='F'))

        td = (1, 0)
        rs = (rank, hsd, hsd, tensorB.shape[3])
        tp = (1, 2, 0, 3)
        tensorA = np.tensordot(np.diag(S), V, axes=td).reshape(rs, order='F').transpose(tp)
    tensorL = tensorA

    tensorA, tensorB = mpo[-2], mpo[-1]
    td = (3, 2)
    rs = (hsd * hsd * tensorA.shape[2], hsd * hsd)
    tensorC = np.tensordot(tensorA, tensorB , axes=td).reshape(rs, order='F')
    U, S, V, rank = svd(tensorC)
    rs = (rank, hsd, hsd)
    tp = (1, 2, 0)
    mpoR.append(V.reshape(rs, order='F').transpose(tp))

    td = (1, 0)
    rs = (hsd, hsd, tensorA.shape[2], rank)
    tensorB = np.tensordot(U, np.diag(S), axes=td).reshape(rs, order='F')
    for i in range(2,middle+plusone):
        tensorA = mpo[-i-1]
        td = (3, 2)
        rs = (hsd * hsd * tensorA.shape[2], hsd * hsd * tensorB.shape[3])
        tensorC = np.tensordot(tensorA, tensorB, axes=td).reshape(rs, order='F')
        U, S, V, rank = svd(tensorC)
        rs = (rank, hsd, hsd, tensorB.shape[3])
        tp = (1, 2, 0, 3)
        mpoR.append(V.reshape(rs, order='F').transpose(tp))

        td = (1, 0)
        rs = (hsd, hsd, tensorA.shape[2], rank)
        tensorB = np.tensordot(U,np.diag(S), axes=td).reshape(rs, order='F')
    tensorR = tensorB

    td = (3, 2)
    rs = (hsd * hsd * tensorL.shape[2], hsd * hsd * tensorR.shape[3])
    tensorC = np.tensordot(tensorL, tensorR, axes=td).reshape(rs, order='F')
    U, S, V, rank = svd(tensorC)

    td = (1, 0)
    rs = (hsd, hsd, tensorL.shape[2], rank)
    mpoM.append(np.tensordot(U,np.diag(S), axes=td).reshape(rs, order='F'))

    rs = (rank, hsd, hsd, tensorR.shape[3])
    tp = (1,2,0,3)
    mpoM.append(V.reshape(rs, order='F').transpose(tp))

    mpo2 = mpoL + mpoM + mpoR[::-1]
    entanglement_entropy = neumann(S)

    return mpo2, entanglement_entropy