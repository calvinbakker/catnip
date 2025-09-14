from catnip.config import *

def get_contracted_state_tensors(mps: List[np.ndarray]) -> List[np.ndarray]:
    """Computes contracted state tensors by contracting each MPS tensor with itself.

    This function takes a Matrix Product State (MPS) represented as a list of tensors
    and computes the contracted state tensors. Each contracted tensor is obtained by
    contracting the physical index of a tensor with itself, effectively computing
    the density matrix elements for that site.

    The MPS structure is visualized as:
    ```
    #--1    1--#--2   1--#--2    1--#
    |          |         |          |
    0          0         0          0
    ```
    Where 0 represents the physical index and 1,2 represent bond indices.

    Args:
        mps: A list of numpy arrays representing the MPS tensors. Each tensor
            has shape (physical_dim, left_bond, right_bond) for middle tensors,
            and appropriate shapes for boundary tensors.

    Returns:
        A list of contracted state tensors, where each element is a numpy array
        resulting from the self-contraction of the corresponding MPS tensor.

    Notes:
        - For the first tensor: Contract along physical index and reshape.
        - For middle tensors: Contract, transpose to (0,2,1,3), and reshape.
        - For the last tensor: Contract along physical index and reshape.
        - The reshaping uses Fortran order ('F') to maintain column-major ordering.
    """
    contracted_state_tensors = []

    # Contract first tensor
    tensorA = mps[0]
    td = (0, 0)
    tp = ()
    rs = (-1)
    tensorB = np.tensordot(tensorA, tensorA, axes=td).reshape(rs, order='F')

    contracted_state_tensors.append(tensorB)

    # Contract middle tensors        
    for i in range(1, len(mps) - 1):
        tensorA = mps[i]
        td = (0, 0)
        tp = (0, 2, 1, 3)
        rs = (tensorA.shape[1]**2, tensorA.shape[2]**2)
        tensorB = np.tensordot(tensorA, tensorA, axes=td).transpose(tp).reshape(rs, order='F')

        contracted_state_tensors.append(tensorB)

    # Contract last tensor
    tensorA = mps[-1]
    td = (0, 0)
    tp = ()
    rs = (-1)
    tensorB =  np.tensordot(tensorA, tensorA, axes=td).reshape(rs, order='F')
    
    contracted_state_tensors.append(tensorB)

    return contracted_state_tensors


def get_lccst_and_rccst(contracted_state_tensors: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    """Computes left and right cumulative contracted state tensors and the norm.

    This function computes the cumulative contractions from left and right,
    which are used in the measurement process for efficient computation.

    The contracted_state_tensors structure is:
    ```
    #--0   0--#--1  ...  0--#
    ```
    Where 0 represents the contracted physical indices.

    Args:
        contracted_state_tensors: List of contracted state tensors from
            get_contracted_state_tensors.

    Returns:
        A tuple containing:
        - lccst: List of left cumulative contracted state tensors.
        - rccst: List of right cumulative contracted state tensors.
        - norm: The normalization factor (square root of the total contraction).

    Notes:
        - Left cumulative: Starts from left, contracting sequentially.
        - Right cumulative: Starts from right, contracting sequentially.
        - The norm is computed as sqrt of the final right cumulative element.
    """
    lccst = []  # left_cumulative_contracted_state_tensors

    # Initialize left cumulative with first two tensors
    tensorA, tensorB = contracted_state_tensors[0], contracted_state_tensors[1]
    td = (0, 0)
    tp = ()
    rs = (-1)
    tensorC = np.tensordot(tensorA, tensorB, axes=td).reshape(rs, order='F')
    lccst.append(tensorC)

    # Continue left cumulative
    for i in range(2, len(contracted_state_tensors)):
        tensorA, tensorB = lccst[-1], contracted_state_tensors[i]
        td = (0, 0)
        tp = ()
        rs = (-1)
        tensorC = np.tensordot(tensorA, tensorB, axes=td).reshape(rs, order='F')
        lccst.append(tensorC)

    rccst = []  # right_cumulative_contracted_state_tensors

    # Initialize right cumulative with last two tensors
    tensorA, tensorB = contracted_state_tensors[-1], contracted_state_tensors[-2]
    td = (0, 1)
    tp = ()
    rs = (-1)
    tensorC = np.tensordot(tensorA, tensorB, axes=td).reshape(rs, order='F')
    rccst.append(tensorC)
    # Continue right cumulative
    for i in range(2, len(contracted_state_tensors) - 1):
        tensorA, tensorB = rccst[-1], contracted_state_tensors[-1 - i]
        td = (0, 1)
        tp = ()
        rs = (-1)
        tensorC = np.tensordot(tensorA, tensorB, axes=td).reshape(rs, order='F')
        rccst.append(tensorC)
    # Final right cumulative
    tensorA, tensorB = rccst[-1], contracted_state_tensors[0]
    td = (0, 0)
    tp = ()
    rs = (-1)
    tensorC = np.tensordot(tensorA, tensorB, axes=td).reshape(rs, order='F')
    rccst.append(tensorC)

    norm = np.sqrt(rccst[-1][0])

    return lccst, rccst, norm

def spin_to_bit(spin: np.ndarray) -> np.ndarray:
    """Converts a spin state vector to its corresponding bit representation.

    Args:
        spin: A numpy array representing the spin state vector.

    Returns:
        A numpy array containing the bit representation of the spin state.
    """
    return (1 - spin) / 2

def measure_observable(mps: List[np.ndarray]) -> np.ndarray:
    """Measures the Pauli-Z observable on each site of the MPS.

    This function computes the expectation value of the Pauli-Z operator
    at each site of the Matrix Product State. The observable is hardcoded
    as the Pauli-Z operator: [[1, 0], [0, -1]].
    Due to this operator the convention of [1,0] to be the "0" state vector
    and [0,1] to be the "1" state vector, spin +1 is associated with "0"
    and spin -1 is associated with "1". 

    The measurement process involves contracting the MPS with the observable
    inserted at each site, using the precomputed cumulative tensors for efficiency.

    Args:
        mps: A list of numpy arrays representing the MPS tensors.

    Returns:
        A numpy array containing the measurement values for each site,
        normalized by the MPS norm.
    """
    contracted_state_tensors = get_contracted_state_tensors(mps)
    lccst, rccst, norm = get_lccst_and_rccst(contracted_state_tensors)

    # Pauli-Z operator
    observable = np.array(
        [[1, 0], [0, -1]], 
        dtype=dtype  
    ) 

    # Measurement array to which the values will be written to
    measurement = np.zeros(shape=len(mps), dtype=dtype)

    # Measure at first site
    tensorA, tensorB = mps[0], observable
    td = (0, 0)
    tp = (1, 0)
    rs = ()
    tensorC = np.tensordot(tensorA, tensorB, axes=td).transpose(tp)
    tensor_and_observable = tensorC
    
    tensorA, tensorB = tensor_and_observable, mps[0]
    td = (0, 0)
    tp = ()
    rs = (-1)
    tensorC = np.tensordot(tensorA, tensorB, axes=td).reshape(rs, order='F')
    tensor_and_observable_and_tensor = tensorC

    measure = np.tensordot(tensor_and_observable_and_tensor, rccst[-2], axes=td)
    measurement[0] = measure / norm

    # Measure at middle sites
    for index in range(1, len(mps) - 1):
        tensorA, tensorB = mps[index], observable
        td = (0, 0)
        tp = (2, 0, 1)
        rs = ()
        tensorC = np.tensordot(tensorA, tensorB, axes=td).transpose(tp)
        tensor_and_observable = tensorC
        
        tensorA, tensorB = tensor_and_observable, mps[index]
        td = (0, 0)
        tp = (0, 2, 1, 3)
        rs = (mps[index].shape[2] ** 2, mps[index].shape[1] ** 2)
        tensorC = np.tensordot(tensorA, tensorB, axes=td).transpose(tp).reshape(rs, order='F')
        tensor_and_observable_and_tensor = tensorC

        left_tensor = lccst[index - 1]
        right_tensor = rccst[len(rccst) - 1 - index]

        measure = np.tensordot(np.tensordot(left_tensor, tensor_and_observable_and_tensor, axes=(0, 0)), right_tensor, axes=(0, 0))
        measurement[index] = measure / norm

    # Measure at last site
    tensorA, tensorB = mps[-1], observable
    td = (0, 0)
    tp = (1, 0)
    rs = ()
    tensorC = np.tensordot(tensorA, tensorB, axes=td).transpose(tp)
    tensor_and_observable = tensorC
    
    tensorA, tensorB = tensor_and_observable, mps[-1]
    td = (0, 0)
    tp = ()
    rs = (-1)
    tensorC = np.tensordot(tensorA, tensorB, axes=td).reshape(rs, order='F')
    tensor_and_observable_and_tensor = tensorC
    
    measure = np.tensordot(lccst[-2], tensor_and_observable_and_tensor, axes=td)
    measurement[-1] = measure / norm

    # return spin_to_bit(measurement)
    return measurement
