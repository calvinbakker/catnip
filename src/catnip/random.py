from catnip.config import *


#                                     /$$                        
#                                    | $$                        
#   /$$$$$$  /$$$$$$  /$$$$$$$   /$$$$$$$  /$$$$$$  /$$$$$$/$$$$ 
#  /$$__  $$|____  $$| $$__  $$ /$$__  $$ /$$__  $$| $$_  $$_  $$
# | $$  \__/ /$$$$$$$| $$  \ $$| $$  | $$| $$  \ $$| $$ \ $$ \ $$
# | $$      /$$__  $$| $$  | $$| $$  | $$| $$  | $$| $$ | $$ | $$
# | $$     |  $$$$$$$| $$  | $$|  $$$$$$$|  $$$$$$/| $$ | $$ | $$
# |__/      \_______/|__/  |__/ \_______/ \______/ |__/ |__/ |__/
                                                               

def random_vector(sites: int) -> np.ndarray:
    """Generates a random state vector.

    Args:
        sites: The number of sites.

    Returns:
        A 1D numpy array representing the random state vector.
    """
    vector = np.random.randn(hsd**sites)
    vector /= np.sqrt(np.sum(vector**2))
    return vector

def random_mps(sites: int, bond_limits: list) -> list[np.ndarray]:
    """Generates a random Matrix Product State (MPS).

    Args:
        sites: The number of sites.
        bond_limits: A list with minimum and maximum bond dimensions.

    Returns:
        A list of numpy arrays representing the MPS tensors.
    """
    mps = []
    bonds = np.random.randint(bond_limits[0], bond_limits[1]+1, size=sites-1)
    mps.append(np.random.randn(hsd, bonds[0]))
    for i in range(sites-2):
        mps.append(np.random.randn(hsd, bonds[i], bonds[i+1]))
    mps.append(np.random.randn(hsd, bonds[-1]))
    # normalize
    return mps

def random_matrix(sites: int) -> np.ndarray:
    """Generates a random matrix.

    Args:
        sites: The number of sites.

    Returns:
        A 2D numpy array representing the random matrix.
    """
    matrix = np.random.randn((hsd**sites), (hsd**sites))
    # matrix /= np.sqrt(np.sum(matrix**2))
    return matrix

def random_mpo(sites: int, bond_limits: list) -> list[np.ndarray]:
    """Generates a random Matrix Product Operator (MPO) representation.

    Args:
        sites: The number of sites.
        bond_limits: A list with minimum and maximum bond dimensions.

    Returns:
        A list of numpy arrays representing the MPO tensors.
    """
    mpo = []
    bonds = np.random.randint(bond_limits[0], bond_limits[1]+1, size=sites-1)
    mpo.append(np.random.randn(hsd, hsd, bonds[0]))
    for i in range(sites-2):
        mpo.append(np.random.randn(hsd, hsd, bonds[i], bonds[i+1]))
    mpo.append(np.random.randn(hsd, hsd, bonds[-1]))
    # normalize 
    return mpo                                                               
                             