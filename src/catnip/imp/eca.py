
from catnip.config import *
from catnip.util import *
from catnip.ops import *


import cellpylib as cpl
from functools import reduce

# the 88 unique eca rules
unique_eca_rules = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                    11, 12, 13, 14, 15, 18, 19, 22, 23, 
                    24, 25, 26, 27, 28, 29, 30, 32, 33, 
                    34, 35, 36, 37, 38, 40, 41, 42, 43, 
                    44, 45, 46, 50, 51, 54, 56, 57, 58, 
                    60, 62, 72, 73, 74, 76, 77, 78, 90, 
                    94, 104, 105, 106, 108, 110, 122, 
                    126, 128, 130, 132, 134, 136, 138, 
                    140, 142, 146, 150, 152, 154, 156, 
                    160, 162, 164, 168, 170, 172, 178, 
                    184, 200, 204, 232] 

def get_rule_number_string(rule_number: int) -> str:
    """Takes a rule number as input and returns the bit-string for a specific hsd.

    Args:
        rule_number: A rule number in decimal.

    Returns:
        A string of the rule number in the number-system of the hsd,
        where the lowest integer is on the RIGHT side.
    """
    n = rule_number
    if n == 0:
        return '0'
    nums = []
    while n:
        n, r = divmod(n, hsd)
        nums.append(str(r))
    y = ''.join(reversed(nums))
    return y


def number_to_bit_array(number: int, sites: int) -> np.ndarray:
    """Maps a number to a bit-array.

    Args:
        number: A number.
        sites: An amount of sites.

    Returns:
        A bit-array representation of the number.
    """
    bit_string = get_rule_number_string(number)[::-1]
    bit_string += (sites - len(bit_string)) * "0"
    bit_string = bit_string[::-1]
    bit_array = np.array([int(bit) for bit in bit_string])
    return bit_array[::-1]


def bit_array_to_vector(bit_array: np.ndarray) -> np.ndarray:
    """Takes a bit-array and maps this to a vector in Hilbert space.

    Args:
        bit_array: A bit-array.

    Returns:
        A vector.
    """
    vector = [np.eye(hsd, dtype = int)[x] for x in bit_array]
    return reduce(np.kron, vector[::-1])


def vector_to_bit_array(vector: np.ndarray, sites: int) -> np.ndarray:
    """Takes a vector and maps it to a bit-array.

    Args:
        vector: A vector.
        sites: The amount of sites.

    Returns:
        A bit-array.
    """
    decomposition = []
    state = np.rint(vector)
    for _ in range(sites):
        indices = np.where(state == 1)[0]
        if len(indices) == 0:
            nn = 0
        else:
            nn = int(np.floor((indices[0] / len(state)) * hsd))
        state = state[int(nn * len(state) / hsd):int((nn + 1) * len(state) / hsd)]
        decomposition.append(nn)
    return np.array(decomposition)[::-1]


def bit_array_to_number(bit_array: np.ndarray) -> int:
    """Takes a bit-array and maps it to a decimal number.

    Args:
        bit_array: A bit-array.

    Returns:
        A number.
    """
    bit_string = ''.join(str(int(z)) for z in bit_array[::-1])
    return int(bit_string, hsd)


def create_mps_from_bit_array(array: np.ndarray) -> List[np.ndarray]:
    """Creates a Matrix Product State (MPS) from a bit array.

    This function constructs an MPS representation where each tensor corresponds
    to a site in the bit array. The MPS is built using identity matrices reshaped
    to match the physical dimension.

    Args:
        array: A numpy array representing the bit array for each site.

    Returns:
        A list of numpy arrays representing the MPS tensors.
    """
    I = np.eye(hsd)
    L = []
    n_sites = len(array)

    # First tensor (left boundary)
    L.append(I[array[0]].reshape((hsd, 1)))
    
    # Middle tensors
    for i in range(1, n_sites - 1):
        L.append(I[array[i]].reshape((hsd, 1, 1)))
    
    # Last tensor (right boundary)
    if n_sites > 1:
        L.append(I[array[-1]].reshape((hsd, 1)))
    
    return L



def rule_number_to_array(rule_number: int) -> np.ndarray:
    """Converts a rule number to its corresponding array representation.

    For a given rule, a mapping is returned (extended bit-string).

    Args:
        rule_number: A rule number in decimal.

    Returns:
        A numpy array of the rule number in hsd, where the lowest integer is on the
        left side of the array. This is the mapping between states as will be used in the
        time-evolution of the cellular automata.
    """
    n = rule_number
    if n == 0:
        return np.array([0])
    nums = []
    while n:
        n, r = divmod(n, hsd)
        nums.append(str(r))
    rule_string = ''.join(nums)

    p = np.array([int(i) for i in rule_string])
    # Pad with zeros to reach length hsd**3
    padding = np.zeros(hsd**3 - len(p), dtype=int)
    return np.concatenate([p, padding])


def get_rule_tensor(rule_number: int) -> np.ndarray:
    """Creates the Wolfram rule-tensor for a given rule number.

    For a given rule, the Wolfram rule-tensor is returned that performs the mapping,
    and has an extra index for performing the algorithm.

    Shape:          Indices:
    0 1 2           i j k
    | | |
    #####--6        . . . o
    | | |
    3 4 5           l m n

    [i,j,k] are the input indices.
    [l,m,n] are the output indices.
    [o] is the extra index used in the algorithm.

    Args:
        rule_number: A rule number in decimal.

    Returns:
        A numpy array representing the tensor that reproduces the Wolfram rule-mapping.
    """
    rule_array = rule_number_to_array(rule_number)
    tensor = np.zeros((hsd,) * 7, order='F')
    for x in range(hsd**3):
        i, j, k = number_to_bit_array(x, 3)
        l = rule_array[x]
        tensor[i, j, k, i, j, k, l] = 1.0
    return tensor


def rule_tensor_to_mpo(rule_tensor: np.ndarray) -> List[np.ndarray]:
    """Converts a rule-tensor to its matrix-product operator (MPO) representation.

    For a given rule-tensor, return the matrix-product representation of this tensor.

    Args:
        rule_tensor: A Wolfram rule-tensor.

    Returns:
        A list of numpy arrays representing the matrix-product operator.
    """
    d = len(rule_tensor.shape)
    mpo = []
    bonds = []

    # Original tensor structure:
    #   0 1 2        i j k
    #   | | |
    #   #####--6     . . . o
    #   | | |
    #   3 4 5        l m n

    tensor = rule_tensor

    # First transformation:
    #   i--#--j        0    2
    #      #--k             3
    #      #--m             4
    #      #--n             5
    #   l--#--o        1    6
    tp = (0, 3, 1, 2, 4, 5, 6)
    rs = (hsd**2, hsd**(d-2))
    tensor = tensor.transpose(tp).reshape(rs, order='F')

    # Second transformation:
    #      #--j             1
    #      #--k             2
    #   d--#--m        0    3
    #      #--n             4
    #      #--o             5

    U, S, V, rank = svd(tensor)
    bonds.append(rank)
    rs = (hsd, hsd, bonds[-1])
    mpo.append(U.reshape(rs, order='F'))
    rs = [bonds[-1]] + (d-2) * [hsd]
    tensor = np.tensordot(np.diag(S), V, axes=(1, 0)).reshape(rs, order='F')

    # Third transformation:
    #   d--#--k      0    2
    #   j--#         1
    #   m--#         3
    #   o--#--n      5    4
    tp = (0, 1, 3, 5, 2, 4)
    rs = (bonds[-1] * hsd * hsd * hsd, -1)
    tensor = tensor.transpose(tp).reshape(rs, order='F')
    U, S, V, rank = svd(tensor)
    bonds.append(rank)

    # Fourth transformation:
    #   d--#         0
    #   j--#--d2     1     4
    #   m--#         2
    #   o--#         3
    tp = (1, 2, 0, 4, 3)
    rs = (bonds[-2], hsd, hsd, hsd, bonds[-1])
    mpo.append(U.reshape(rs, order='F').transpose(tp))

    # Fifth transformation:
    #       #--k           1
    #   d2--#         0
    #       #--n           2
    td = (1, 0)
    rs = [bonds[-1]] + [hsd] * 2
    tensor = np.tensordot(np.diag(S), V, axes=td).reshape(rs, order='F')
    tp = [1, 2, 0]
    rs = (hsd, hsd, bonds[-1])
    mpo.append(tensor.transpose(tp).reshape(rs, order='F'))
    return mpo

def mpo_to_tensor(mpo: list[np.ndarray]) -> np.ndarray:
    """Reconstructs the original tensor from its matrix-product operator (MPO) representation.

    This function takes an MPO (a list of tensors) and contracts them back to form
    the original high-dimensional tensor that the MPO represents.

    Args:
        mpo: A list of numpy arrays representing the matrix-product operator.

    Returns:
        A numpy array representing the reconstructed tensor.
    """

    tensorA, tensorB, tensorC = mpo
    td1 = (2, 2)
    td2 = (4, 2)
    tp = (0, 2, 5, 1, 3, 6, 4)

    tensor = np.rint(np.abs(
        np.transpose(
            np.tensordot(
                np.tensordot(
                    tensorA, tensorB, axes=td1
                ), tensorC, axes=td2
            ), tp
        )
    ))
    return tensor

def get_bound_tensor() -> np.ndarray:
    """Produces a copying tensor needed for the construction of the Wolframmatrix-product operator.

    This function creates a tensor required for implementing the algorithm
    to obtain the matrix-product operator for Wolfram cellular automata systems.

    Returns:
        A numpy array representing the copying tensor.
    """
    tensor = np.zeros((hsd, hsd, hsd), dtype=float, order='F')
    I = np.eye(hsd)
    for x in range(hsd):
        tensor[x, x] = I[x]
    return tensor


def construct_eca_mpo(rule_number: int, amount_of_sites: int) -> List[np.ndarray]:
    """Creates the matrix-product operator of a Wolfram elementary cellular automata (ECA) rule.

    For a given Wolfram cellular automata rule and an amount of sites, this function
    returns the matrix-product operator that produces the rule-mapping.

    Args:
        rule_number: A rule number.
        amount_of_sites: The number of sites.

    Returns:
        A list of numpy arrays representing the Wolfram cellular automata
        matrix-product operator.
    """
    if amount_of_sites < 3:
        raise ValueError("Wolfram cellular automata tensor network requires at least three tensors to be constructed. "
                        "The amount of sites given is too small.")

    rule_tensor = get_rule_tensor(rule_number)
    a, b, c = rule_tensor_to_mpo(rule_tensor)
    if amount_of_sites == 3:
        b = np.tensordot(b, [1, 1], axes=(1, 0)).transpose(0, 3, 1, 2)
        return [a.transpose(1, 0, 2), b.transpose(1, 0, 2, 3), c.transpose(1, 0, 2)]
    else:
        bound = get_bound_tensor()

        A = np.tensordot(bound, a, axes=(1, 0)).transpose((0, 2, 3, 1))
        B = np.tensordot(b, a, axes=(1, 0)).transpose(0, 1, 2, 5, 4, 3).reshape((hsd, b.shape[2], b.shape[3] * a.shape[2], hsd, hsd), order='F').transpose(0, 3, 1, 2, 4)
        C = np.tensordot(c, B, axes=(1, 0)).transpose(0, 1, 3, 4, 2, 5).reshape((hsd, c.shape[2] * B.shape[2], B.shape[3], hsd, hsd), order='F').transpose(0, 3, 1, 2, 4)
        D = np.tensordot(c, b, axes=(1, 0)).transpose(0, 1, 3, 4, 2, 5).reshape((hsd, c.shape[2] * b.shape[2], b.shape[3], hsd, hsd), order='F').transpose(0, 3, 1, 2, 4)
        E = np.tensordot(c, bound, axes=(1, 0)).transpose(0, 2, 1, 3)

        alpha = np.tensordot(A, [1, 1], axes=(1, 0)).transpose((0, 2, 1))
        beta = np.tensordot(B, [1, 1], axes=(1, 0)).transpose((0, 3, 1, 2))
        gamma = np.tensordot(C, [1, 1], axes=(1, 0)).transpose((0, 3, 1, 2))
        delta = np.tensordot(D, [1, 1], axes=(1, 0)).transpose((0, 3, 1, 2))
        epsilon = np.tensordot(E, [1, 1], axes=(1, 0)).transpose((0, 2, 1))

        alpha = alpha.transpose((1, 0, 2))
        beta = beta.transpose((1, 0, 2, 3))
        gamma = gamma.transpose((1, 0, 2, 3))
        delta = delta.transpose((1, 0, 2, 3))
        epsilon = epsilon.transpose((1, 0, 2))

        network = [alpha, beta] + (amount_of_sites - 4) * [gamma] + [delta, epsilon]
        return network


def ca_step(array: np.ndarray, rule_number: int, Lbound: int, Rbound: int) -> np.ndarray:
    """Performs one step of cellular automaton evolution.

    Uses the cellpylib library to evolve the array according to the
    specified Wolfram rule, then applies boundary conditions to fit
    the evolution obtained from the tensor-network based evolution.

    Due to the design of the tensor-network operator, the boundary
    cells are not updated, and stay constant. Since the cellpylib
    evolution uses periodic boundary conditions, there will be 
    a mismatch. To overcome this we use the Lbound and Rbound values
    to replace the boundary values after each evolution step.

    Args:
        array: State array.
        rule_number: Wolfram rule number.
        Lbound: Value for the left boundary.
        Rbound: Value for the right boundary.

    Returns:
        The evolved array with boundary conditions applied.

    Notes:
        - cpl.evolve() has the confusing convention to use a value
          of "timesteps=2" to produce a single timestep.
    """
    array = cpl.evolve(array.reshape(1, -1), timesteps=2, 
                        apply_rule=lambda n, c, t: cpl.nks_rule(n, rule_number))[-1]
    array[0] = Rbound
    array[-1] = Lbound
    return array

def evolution_with_bits(rule_number: int, initial_condition: np.ndarray, timesteps: int) -> np.ndarray:
    """Produces time-evolution using direct bit operations for cellular automaton.

    This function simulates the cellular automaton evolution using direct
    bit array manipulations, without tensor network representations.

    Note here that we do some reversal operations to obtain the same output
    convention as is produced by the tensor-network based time-evolution.

    Args:
        rule_number: The Wolfram rule number.
        initial_condition: Initial state.
        timesteps: Number of evolution steps to perform.

    Returns:
        A numpy array of shape (timesteps+1, sites) containing the full trajectory,
        including the initial condition.

    Notes:
        - Uses cellpylib for evolution steps with custom boundary condition.
    """
    Lbound = initial_condition[0]
    Rbound = initial_condition[-1]

    L = []
    L.append(initial_condition)
    array = initial_condition.copy()[::-1]
    for _ in range(timesteps):
        array = ca_step(array, rule_number, Lbound, Rbound)
        L.append(array[::-1])

    return np.array(L, dtype = dtype)


def evolution_with_tensors(rule_number: int, initial_condition: np.ndarray, timesteps: int, noise_level: float = 0.0, Qreconstruct_mps: bool = False) -> np.ndarray:
    """Produces time-evolution with tensor networks.

    Simulates the cellular automaton evolution using an MPS representation
    of the state, and a MPO representation of the mapping. Here it is possible
    to include noise to the operator, and the resulting state is measured by
    contracting the tensor network with a Pauli-Z operator observable.

    Args:
        rule_number: The Wolfram rule number.
        initial_condition: Initial state as a bit array.
        timesteps: Number of evolution steps.
        noise_level: Standard deviation of the added noise.
        Qreconstruct_mps: Reconstruct a noiseless MPS after each measure.

    Returns:
        A numpy array of shape (timesteps+1, sites) with the trajectory.

    Notes:
        - Adds random noise to each operator tensor.
        - Uses measure_observable to extract bit values from MPS.
        - Applies MPS optimization after each step.

    """
    sites = len(initial_condition)
    mps = create_mps_from_bit_array(initial_condition)
    mpo = construct_eca_mpo(rule_number, sites)
    for i in range(len(mpo)):
        mpo[i] += np.random.rand(*mpo[i].shape) * noise_level

    L = []
    L.append(initial_condition)
    for _ in range(timesteps):
        mps = mpo_mps_contraction(mpo, mps)
        mps = optimize_mps_R(optimize_mps_L(mps))
        state_array = spin_to_bit(measure_observable(mps))
        L.append(state_array)
        if Qreconstruct_mps:
            mps = create_mps_from_bit_array(np.rint(state_array).astype(int))

    return np.array(L, dtype=dtype)
