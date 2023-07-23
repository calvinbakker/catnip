from config import hsd, num_threshold
from tensor import optimize_mpo, reduce_norm, bonds_of_mpo
from vecnum import *

def get_string(rule):
    '''
    Takes a number as input and returns the bit-string for a specific hsd (Hilbert 
    space dimension).

    Input: 
        A rule number in decimal.
    Output:
        A string of the rule number in the number-system of the hsd (hsd-ns), where 
        the lowest integer is on the RIGHT side.
    
    '''

    n = rule
    if n == 0:
        return '0'
    nums = []
    while n:
        n, r = divmod(n, hsd)
        nums.append(str(r))
    y = ''.join(reversed(nums))
    return y

def get_map(rule):
    '''
    For a given rule, a mapping is returned (extended bit-string).

    Input:
        A rule number in decimal.
    Output:
        An array of the rule number in hsd-ns, where the lowest integer in of the LEFT 
        side of the array. This is the mapping between states as will be used in the 
        time-evolution of the cellular automata.

    '''

    p = []
    for i in get_string(rule)[::-1]:
        p.append(int(i))
    for _ in range(hsd**(3)-len(p)):
        p.append(0)
    return p

def get_rule_tensor(rule):
    '''
    For a given rule, the Wolfram rule-tensor is returned.

    Input:
        A rule number in decimal.
    Output:
        A tensor reproducing the Wolfram rule-mapping.

    '''
   
    rule_map = get_map(rule)
    tensor = np.zeros(shape=tuple(7*[hsd]),order='F')
    for x in range(hsd*hsd*hsd):
        i, j, k = number_to_bit_array(x,3)
        l = rule_map[x]
        tensor[i][j][k][i][j][k][l]=1
    return tensor

def get_rule_mpo(rule_tensor):
    '''
    For a given rule-tensor, return the matrix-product representation of this tensor.

    Input:
        A Wolfram rule-tensor.
    Output:
        A matrix-product operator.

    '''

    d           = len(rule_tensor.shape)
    tensors     = []
    bonds       = []


    #     0 1 2        i j k
    #     | | | 
    #     #####--6            o
    #     | | |
    #     3 4 5        l m n
    #
    T = rule_tensor


    #
    #   i--#--j        0    2
    #      #--k             3
    #      #--m             4
    #      #--n             5
    #   l--#--o        1    6
    #
    tp = (0,3,1,2,4,5,6)
    rs = (hsd**2,hsd**(d-2))
    T = T.transpose(tp).reshape(rs,order='F')


    #
    #      #--j             1
    #      #--k             2
    #   d--#--m        0    3
    #      #--n             4
    #      #--o             5
    #
    u, s, v = np.linalg.svd(T, full_matrices=False)
    bonds.append(np.sum(s>num_threshold))
    U = u[:,:bonds[-1]]
    S = np.diag(s[:bonds[-1]])
    V = v[:bonds[-1],:]
    tensors.append(U.reshape(hsd,hsd,bonds[-1],order='F'))
    rs = [bonds[-1]]+ (d-2)*[hsd]
    T = np.tensordot(S,V,axes=(1,0)).reshape(rs,order='F')


    #   d--#--k      0    2
    #   j--#         1    
    #   m--#         3    
    #   o--#--n      5    4
    #
    tp = (0,1,3,5,2,4)
    T = T.transpose(tp).reshape((bonds[-1]*hsd*hsd*hsd,-1),order='F')
    u, s, v = np.linalg.svd(T, full_matrices=False)
    bonds.append(np.sum(s>num_threshold))
    U = u[:,:bonds[-1]]
    S = np.diag(s[:bonds[-1]])
    V = v[:bonds[-1],:]
    
    #   d--#         0
    #   j--#--d2     1     4      
    #   m--#         2 
    #   o--#         3
    #
    tp = (1,2,0,4,3)
    tensors.append(U.reshape(bonds[-2],hsd,hsd,hsd,bonds[-1],order='F').transpose(tp))

    #       #--k           1
    #   d2--#         0    
    #       #--n           2
    #
    rs = [bonds[-1]]+[hsd]*2
    T = np.tensordot(S,V,axes=(1,0)).reshape(rs,order='F')
    tp = [1,2,0]
    T = T.transpose(tp).reshape((hsd,hsd,bonds[-1]),order='F')
    tensors.append(T)
    return tensors

def test_reverse_mpo(tensors):
    '''
    Checks if the rule-tensor can be decomposed and recomposed to produce the same result.

    Input:
        A matrix-product operator
    Output:
        A tensor

    '''

    return  np.rint(np.abs(
                np.transpose(
                    np.tensordot(
                        np.tensordot(
                            tensors[0],tensors[1],axes=(2,2)
                        ), tensors[2], axes=(4,2)
                    ),(0,2,5,1,3,6,4)
                )
            ))

def get_bound_tensor():
    '''
    Produces a copying tensor needed for the implementation of the parallel procedure to obtain 
    the matrix-product operator for the Wolfram cellular automata systems.

    Input:
        A Wolfram rule-tensor.
    Output:
        A matrix-product operator.

    '''

    tensor = np.zeros(shape=(hsd,hsd,hsd),order='F')
    I = np.identity(2)
    for x in range(hsd):
        tensor[x][x]=I[x]
    return tensor

def get_wolfram_network(rule,sites):
    '''
    For a given Wolfram cellular automata rule and an amount of sites, this function return the
    matrix-product operator that produces the rule-mapping.

    Input:
        A rule-number and an amount of sites.
    Output:
        Wolfram cellular automata matrix-product operator.

    '''

    if sites <3:
        print("👽 Bigger network please...")
        return None

    rule_tensor = get_rule_tensor(rule)
    a, b, c     = get_rule_mpo(rule_tensor)
    if sites ==3:
        b = np.tensordot(b,[1,1],axes=(1,0)).transpose(0,3,1,2)
        return [a.transpose(1,0,2),b.transpose(1,0,2,3),c.transpose(1,0,2)]
    else:
        bound       = get_bound_tensor()

        A = np.tensordot(bound,a,axes=(1,0)).transpose((0,2,3,1))
        B = np.tensordot(b,a,axes=(1,0)).transpose(0,1,2,5,4,3).reshape((hsd,b.shape[2],b.shape[3]*a.shape[2],hsd,hsd),order='F').transpose(0,3,1,2,4)
        C = np.tensordot(c,B,axes=(1,0)).transpose(0,1,3,4,2,5).reshape((hsd,c.shape[2]*B.shape[2],B.shape[3],hsd,hsd),order='F').transpose(0,3,1,2,4)
        D = np.tensordot(c,b,axes=(1,0)).transpose(0,1,3,4,2,5).reshape((hsd,c.shape[2]*b.shape[2],b.shape[3],hsd,hsd),order='F').transpose(0,3,1,2,4)
        E = np.tensordot(c,bound,axes=(1,0)).transpose(0,2,1,3)

        alpha   = np.tensordot(A,[1,1],axes=(1,0)).transpose((0,2,1))
        beta    = np.tensordot(B,[1,1],axes=(1,0)).transpose((0,3,1,2))
        gamma   = np.tensordot(C,[1,1],axes=(1,0)).transpose((0,3,1,2))
        delta   = np.tensordot(D,[1,1],axes=(1,0)).transpose((0,3,1,2))
        epsilon = np.tensordot(E,[1,1],axes=(1,0)).transpose((0,2,1))

        alpha   = alpha.transpose((1,0,2))
        beta    = beta.transpose((1,0,2,3))
        gamma   = gamma.transpose((1,0,2,3))
        delta   = delta.transpose((1,0,2,3))
        epsilon = epsilon.transpose((1,0,2))

        network = [alpha,beta]+(sites-4)*[gamma]+[delta,epsilon]
        return network


# the 88 unique wolfram rules
unique_wolfram_rules = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
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

def init_wolfram(rule,sites):
    '''
    For a given Wolfram cellular automata rule and an amount of sites, this function return the
    matrix-product operator that produces the rule-mapping. The MPO is optimized and put in 
    mixed-canonical form, as well as the entanglement entropy and maximimum bond dimension.

    Input:
        A rule-number and an amount of sites.
    Output:
        Wolfram cellular automata matrix-product operator and measures.

    '''
    network     = get_wolfram_network(rule,sites)
    network, ee = optimize_mpo(network,direction='LRM')
    network     = reduce_norm(network)
    bonds       = bonds_of_mpo(network)
    return network, ee, bonds
