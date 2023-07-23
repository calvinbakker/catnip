import numpy as np
from scipy import linalg
from config import hsd, num_threshold, precision, calibration_factor
from vecnum import vector_to_bit_array
from rule import rule_matrix


def operator_to_mpo(operator):
    '''
    This function decomposes an operator (matrix) into an tensor network (matrix product operator).
    The convention is that the tensors are numbered by index as follows:

    Operator, where the vector is input at bond 0.

    |i>

     0 
     |
    ---
    |M|
    ---
     |
     1


    Tensor network

    |0>     |1>     ...  |n> 
     
     0       0            0
     |       |            |
    ---     ---          ---
    |T|-2 2-|T|-3      2-|T|
    ---     ---          ---
     |       |            |
     1       1            1


    '''
    d           = operator.shape[0]
    tensors     = []
    bonds       = []
    order       = int(np.rint(np.log(d)/np.log(hsd)-1))

    ## INITIALIZE
    T = operator.reshape(tuple(2*(order+1)*[hsd]),order='F')
    tp = [0]+[(order+1)]+list(range(1,order+1))+list(range(order+2,2*(order+1)))
    T = T.transpose(tp).reshape((hsd*hsd,-1),order='F')


    ## FIRST DECOMPOSITION
    u, s, v = np.linalg.svd(T, full_matrices=False)
    bonds.append(np.sum(s>num_threshold))
    U = u[:,:bonds[-1]]
    S = np.diag(s[:bonds[-1]])
    V = v[:bonds[-1],:]
    tensors.append(U.reshape(hsd,hsd,bonds[-1],order='F'))

    if order ==1:
        tensors.append(np.tensordot(S,V,axes=(1,0)).reshape((bonds[-1],hsd,hsd),order='F').transpose(1,2,0))
        return tensors 
    
    else:
        tp = tuple([bonds[-1]]+ (2*(order+1)-2)*[hsd])
        T = np.tensordot(S,V,axes=(1,0)).reshape(tp,order='F')
        tp = [0,1,order+1]+list(range(2,order+1))+list(range(order+2,2*(order+1)-1))
        T = T.transpose(tp).reshape((bonds[-1]*hsd*hsd,-1),order='F')

        for x in range(order-2):
            ## SECOND DECOMPOSITION
            u, s, v = np.linalg.svd(T, full_matrices=False)
            bonds.append(np.sum(s>num_threshold))
            U = u[:,:bonds[-1]]
            S = np.diag(s[:bonds[-1]])
            V = v[:bonds[-1],:]
            tensors.append(U.reshape(bonds[-2],hsd,hsd,bonds[-1],order='F').transpose((1,2,0,3)))
            
            tp = tuple([bonds[-1]]+[hsd]*(2*(order-1-x)))
            T = np.tensordot(S,V,axes=(1,0)).reshape(tp,order='F')
            tp = tuple([0,1,order-x]+list(range(2,order-x))+list(range(order+1-x,2*(order-x)-1)))
            T = T.transpose(tp).reshape((bonds[-1]*hsd*hsd,-1),order='F')


        ## FINAL DECOMPOSITION
        u, s, v = np.linalg.svd(T, full_matrices=False)
        bonds.append(np.sum(s>num_threshold))
        U = u[:,:bonds[-1]]
        S = np.diag(s[:bonds[-1]])
        V = v[:bonds[-1],:]
        tensors.append(U.reshape(bonds[-2],hsd,hsd,bonds[-1],order='F').transpose((1,2,0,3)))
        tensors.append(np.tensordot(S,V,axes=(1,0)).reshape((bonds[-1],hsd,hsd),order='F').transpose(1,2,0))

        # #################
        # tensors2 = []
        # for tensor in tensors:
        #     if len(tensor.shape)==3:
        #         tensors2.append(tensor.transpose((1,0,2)))
        #     if len(tensor.shape)==4:7
        #         tensors2.append(tensor.transpose((1,0,2,3)))
        # return tensors2

        return tensors

def mpo_to_operator(tensors,round=False):
    order = len(tensors)-1
    t1 = tensors[0]
    i  = 2
    for t2 in tensors[1:]:
        t1 =  np.tensordot(t1,t2,axes=(i,2))
        i  += 2
    t3 = t1.transpose(list(range(0,2*(order+1),2))+list(range(1,2*(order+1),2)))
    d = int(np.rint(np.sqrt(np.product(t3.shape))))
    if round:
        t3 = np.abs(np.rint(t3))
    return t3.reshape((d,d),order='F')

def v_contract_bound(A,B):
    sA = A.shape
    sB = B.shape
    a = sA[2]
    b = sB[2]
    
    '''
    right
      | 
      #--
      |
      #--
      |

    left
      | 
    --#
      |
    --#
      |

'''
    C =     np.reshape(
                np.transpose(
                    np.tensordot(
                        A,B,axes=(1,0)),
                    (0,2,1,3)), 
                (hsd,hsd,a*b), order='F')
                
    return C

def v_contract(A,B):
    '''
    Function that performs the contraction between two tensors in the vertical direction (in the conventions of this code).
    
    The following contractions are the only contractions encountered in the Cellular Automata that are used. For a more general contraction the horizontal mirror-images have to be added as well, and chirality has to be specified instead of sA and sB.

    '''
    sA = A.shape
    sB = B.shape

    '''
      | 
    --#--
      |
      #--
      |

    '''
    if len(sA)==4 and len(sB)==3:
        a1 = sA[2]
        a2 = sA[3]
        b2 = sB[2]
        C = np.transpose(
                np.reshape(
                    np.transpose(
                        np.tensordot(
                            A,B,axes=(0,1)),
                        (4,2,1,3,0)), # miss 2 en 4 switchen
                    (a2*b2,a1,hsd,hsd)
                    ),
                (2,3,1,0)
            )   

    '''
      | 
    --#
      |
    --#--
      |
     
    '''
    if len(sA)==3 and len(sB)==4:
        a1 = sA[2]
        b1 = sB[2]
        b2 = sB[3]
        C = np.transpose(
                np.reshape(
                    np.transpose(
                        np.tensordot(
                            A,B,axes=(0,1)),
                        (3,1,4,0,2)), 
                    (a1*b1,b2,hsd,hsd)
                    ),
                (3,2,0,1)
            )
    '''
      | 
    --#--
      |
    --#--
      |
     
    '''
    if len(sA)==4 and len(sB)==4:
        a1 = sA[2]
        a2 = sA[3]
        b1 = sB[2]
        b2 = sB[3]
        C = np.transpose(
                np.reshape(
                    np.transpose(
                        np.tensordot(
                            A,B,axes=(1,0)),
                        (0,1,4,2,5,3)), 
                    (hsd,a1*b1,a2*b2,hsd),order='F'
                    ),
                (0,3,1,2)
            )
    '''
      | 
    --#
      |
      #--
      |
     
    '''
    if len(sA)==3 and len(sB)==3:
        a1 = sA[2]
        b1 = sB[2]
        C = np.transpose(np.tensordot(A,B,axes=(0,1)),(2,0,1,3))
    return C

def v_contract_multiple(network):
    n = len(network)
    mpo = network[0]
    for i in range(1,n):
        mpo = v_contract(mpo,network[i])
    return mpo

def create_network(L,sites):
    order = len(L)-1
    if sites<order+1:
        tensors = L
    elif order+1<=sites<=2*order+1:
        tensors = []
        for i in range(order+1):
            tensors.append([L[i]])
        for i in range(sites-order-1):
            tensors.append([L[-1]])

        for j in range(sites-order-1):
            for i in range(order):
                tensors[j+i+1].append(L[i])
    elif 2*order+1<sites:
        tensors = []
        for i in range(order+1):
            tensors.append([L[i]])
        for i in range(2*order+1-order-1):
            tensors.append([L[-1]])

        for j in range(2*order+1-order-1):
            for i in range(order):
                tensors[j+i+1].append(L[i])

    new_tensors = []
    for tensor in tensors:
        new_tensors.append(v_contract_multiple(tensor))
    
    if 2*order+1<sites:
        c = sites-(2*order+1)
        new_tensors = new_tensors[:order]+c*[new_tensors[order]]+new_tensors[order:]
        
    return new_tensors

def product(network1,network2):
    n = len(network1)

    network3 = []
    network3.append(v_contract_bound(network1[0],network2[0]))
    for i in range(1,n-1):
        network3.append(v_contract(network1[i],network2[i]))
    network3.append(v_contract_bound(network1[n-1],network2[n-1]))
    return network3

def bonds_of_mpo(network):
    l = []
    for tensor in network:
        l.append(tensor.shape[-1])
    return l[:-1]

def zero_rounding(network):
    network_2 = []
    for tensor in network:
        tensor[np.abs(tensor) < num_threshold] = 0
        network_2.append(tensor)
    return network_2

def custom_svd(t):
    try:
        u, s, v = np.linalg.svd(t, full_matrices=False) 
    except:
        try:
            print("🤖 svd error, trying gesvd instead...")
            u, s, v = linalg.svd(t, full_matrices=False, lapack_driver='gesvd')
        except:
            print("🤖 svd error, trying zero rounding...")
            t[np.abs(t) < num_threshold] = 0
            u, s, v = np.linalg.svd(t, full_matrices=False)
    
        
    while np.max(s) > calibration_factor:
        s/=calibration_factor
        # print("🤖 refactored s... (+)")
    while np.max(s) < 1/calibration_factor:
        s*=calibration_factor
        # print("🤖 refactored s... (-)")
    
    rank = np.sum(s>num_threshold)
    U = u[:,:rank]
    S = s[:rank]
    V = v[:rank,:]

    return U, S, V, rank

def svd_mpo(A,B,direction='R'):
    try:
        sA = A.shape
        sB = B.shape
        if len(sA)==4 and len(sB)==4:
            leg1 = sA[2]
            leg2 = sB[3]
            t = np.reshape(
                    np.transpose(
                        np.tensordot(
                            A,B,axes=[len(A.shape)-1,len(B.shape)-2]),
                        (0,2,1,4,5,3)),
                    ((hsd**2)*leg1,(hsd**2)*leg2)
                )
            U, S, V, rank = custom_svd(t)
            if direction == 'R':
                V = np.tensordot(np.diag(S),V,axes=[1,0])
            elif direction == 'L':
                U = np.tensordot(U,np.diag(S),axes=[1,0])
            elif direction == 'U':
                None
            else:
                print("🤖 bleep bloop... please give me R or L to decide what to do!")  
            mpo1= np.transpose(np.reshape(U,(hsd,leg1,hsd,rank)), (0,2,1,3))
            mpo2= np.transpose(np.reshape(V,(rank,hsd,leg2,hsd)), (3,1,0,2))      


        if len(sA)==3 and len(sB)==4:
            leg2 = sB[3]
            t = np.reshape(
                    np.transpose(
                        np.tensordot(
                            A,B,axes=[len(A.shape)-1,len(B.shape)-2]),
                        (0,1,3,4,2)),
                    ((hsd**2),(hsd**2)*leg2)
                )
            U, S, V, rank = custom_svd(t)
            if direction == 'R':
                V = np.tensordot(np.diag(S),V,axes=[1,0])
            elif direction == 'L':
                U = np.tensordot(U,np.diag(S),axes=[1,0])
            elif direction == 'U':
                None
            else:
                print("🤖 bleep bloop... please give me R or L to decide what to do!")       
            mpo1= np.transpose(np.reshape(U,(hsd,hsd,rank)), (0,1,2))
            mpo2= np.transpose(np.reshape(V,(rank,hsd,leg2,hsd)), (3,1,0,2))

        if len(sA)==4 and len(sB)==3:
            leg1 = sA[2]
            t = np.reshape(
                    np.transpose(
                        np.tensordot(
                            A,B,axes=[3,2]),
                        (0,2,1,4,3)),
                    ((hsd**2)*leg1,(hsd**2))
                )
            U, S, V, rank = custom_svd(t)
            if direction == 'R':
                V = np.tensordot(np.diag(S),V,axes=[1,0])
            elif direction == 'L':
                U = np.tensordot(U,np.diag(S),axes=[1,0])
            elif direction == 'U':
                None
            else:
                print("🤖 bleep bloop... please give me R or L to decide what to do!")  
            mpo1= np.transpose(np.reshape(U,(hsd,leg1,hsd,rank)), (0,2,1,3))
            mpo2= np.transpose(np.reshape(V,(rank,hsd,hsd)), (2,1,0))      



        if len(sA)==3 and len(sB)==3:   
            t = np.tensordot(A,B,axes=[2,2]).reshape(hsd*hsd,hsd*hsd,order='F')
            U, S, V, rank = custom_svd(t)
            if direction == 'R':
                V = np.tensordot(np.diag(S),V,axes=[1,0])
            elif direction == 'L':
                U = np.tensordot(U,np.diag(S),axes=[1,0])
            elif direction == 'U':
                None
            else:
                print("🤖 bleep bloop... please give me R or L to decide what to do!")  
            mpo1= U.reshape(hsd,hsd,rank,order='F')
            mpo2= V.reshape((rank,hsd,hsd),order='F').transpose(1,2,0)       

        return mpo1,mpo2,S
    except:
        print("🤖 Returning tensors without svd...")
        print(A.shape,B.shape)
        return A, B, []

def optimize_mpo(network, direction='R'):
    # direction = ['M','L','R','LM','LR','LRM']
    n = len(network)
    
    if direction=='R':
        network_2=[]
        mpo1, mpo2, s = svd_mpo(network[0],network[1],'R')
        network_2.append(mpo1)
        for i in range(2,n):
            mpo1, mpo2, s = svd_mpo(mpo2,network[i],'R')
            network_2.append(mpo1)
        network_2.append(mpo2)
        e_e = neumann(s)
        return network_2, e_e
    
    elif direction=='M':
        if n==2:
            A = network[0]
            B = network[1]
            A, B, s = svd_mpo(A,B,'R')
            e_e = neumann(s)
            L = [A,B]
        elif n==3:
            A = network[0]
            B = network[1]
            C = network[2]
            A, B, s = svd_mpo(A,B,'R')
            B, C, s = svd_mpo(B,C,'L')
            e_e = neumann(s)
            L = [A,B,C]
        else:
            m = int(n/2)

            network_2=[]
            mpo1, A, s = svd_mpo(network[0],network[1],'R')
            network_2.append(mpo1)
            for i in range(2,m):
                mpo1, A, s = svd_mpo(A,network[i],'R')
                network_2.append(mpo1)
            
            network_3=[]
            B, mpo2, s = svd_mpo(network[n-2],network[n-1],'L')
            network_3.append(mpo2)
            for i in range(2,n-m):
                B, mpo2, s = svd_mpo(network[n-1-i],B,'L')
                network_3.append(mpo2)

            A, B, s = svd_mpo(A,B,'R')
            e_e = neumann(s)
            L = network_2+[A,B]+network_3[::-1]
        
        return L, e_e
        
    elif direction=='L':
        network_2=[]
        mpo1, mpo2, s = svd_mpo(network[n-2],network[n-1],'L')
        network_2.append(mpo2)
        for i in range(2,n):
            mpo1, mpo2, s = svd_mpo(network[n-1-i],mpo1,'L')
            network_2.append(mpo2)
        network_2.append(mpo1)
        e_e = neumann(s)
        return network_2[::-1], e_e
    
    elif direction=='LM':
        network_2=[]
        mpo1, mpo2, s = svd_mpo(network[n-2],network[n-1],'L')
        network_2.append(mpo2)
        for i in range(2,n):
            mpo1, mpo2, s = svd_mpo(network[n-1-i],mpo1,'L')
            network_2.append(mpo2)
        network_2.append(mpo1)

        network = network_2[::-1]
        m = int(n/2)
        network_2=[]
        mpo1, A, s = svd_mpo(network[0],network[1],'R')
        network_2.append(mpo1)
        for i in range(2,m):
            mpo1, A, s = svd_mpo(A,network[i],'R')
            network_2.append(mpo1)
        
        B = network[m]
        A, B, s = svd_mpo(A,B,'R')
        network_2 = network_2 +[A,B]+network[m+1:]
        e_e = neumann(s)
        return network_2, e_e

    elif direction=='LR':
        network_2=[]
        mpo1, mpo2, s = svd_mpo(network[n-2],network[n-1],'L')
        network_2.append(mpo2)
        for i in range(2,n):
            mpo1, mpo2, s = svd_mpo(network[n-1-i],mpo1,'L')
            network_2.append(mpo2)
        network_2.append(mpo1)
        
        network = network_2[::-1]
        network_2=[]
        mpo1, mpo2, s = svd_mpo(network[0],network[1],'R')
        network_2.append(mpo1)
        for i in range(2,n):
            mpo1, mpo2, s = svd_mpo(mpo2,network[i],'R')
            network_2.append(mpo1)
        network_2.append(mpo2)
        e_e = neumann(s)
        return network_2, e_e
    
    elif direction=='LRM':
        network_2=[]
        mpo1, mpo2, s = svd_mpo(network[n-2],network[n-1],'L')
        network_2.append(mpo2)
        for i in range(2,n):
            mpo1, mpo2, s = svd_mpo(network[n-1-i],mpo1,'L')
            network_2.append(mpo2)
        network_2.append(mpo1)
        
        network = network_2[::-1]
        network_2=[]
        mpo1, mpo2, s = svd_mpo(network[0],network[1],'R')
        network_2.append(mpo1)
        for i in range(2,n):
            mpo1, mpo2, s = svd_mpo(mpo2,network[i],'R')
            network_2.append(mpo1)
        network_2.append(mpo2)

        network = network_2
        m = int(n/2)
        network_3=[]
        B, mpo2, s = svd_mpo(network[n-2],network[n-1],'L')
        network_3.append(mpo2)
        for i in range(2,n-m):
            B, mpo2, s = svd_mpo(network[n-1-i],B,'L')
            network_3.append(mpo2)
        A = network[m-1]
        A, B, s = svd_mpo(A,B,'R')
        network_2 = network[:m-1] +[A,B]+network_3[::-1]
        e_e = neumann(s)
        return network_2, e_e
    
    elif direction=='LRLM':
        network_2=[]
        mpo1, mpo2, s = svd_mpo(network[n-2],network[n-1],'L')
        network_2.append(mpo2)
        for i in range(2,n):
            mpo1, mpo2, s = svd_mpo(network[n-1-i],mpo1,'L')
            network_2.append(mpo2)
        network_2.append(mpo1)
        
        network = network_2[::-1]
        network_2=[]
        mpo1, mpo2, s = svd_mpo(network[0],network[1],'R')
        network_2.append(mpo1)
        for i in range(2,n):
            mpo1, mpo2, s = svd_mpo(mpo2,network[i],'R')
            network_2.append(mpo1)
        network_2.append(mpo2)

        network = network_2
        # LR
        ###################
        # L
        
        network_2=[]
        mpo1, mpo2, s = svd_mpo(network[n-2],network[n-1],'L')
        network_2.append(mpo2)
        for i in range(2,n):
            mpo1, mpo2, s = svd_mpo(network[n-1-i],mpo1,'L')
            network_2.append(mpo2)
        network_2.append(mpo1)

        network = network_2[::-1]
        #M
        m = int(n/2)
        network_2=[]
        mpo1, A, s = svd_mpo(network[0],network[1],'R')
        network_2.append(mpo1)
        for i in range(2,m):
            mpo1, A, s = svd_mpo(A,network[i],'R')
            network_2.append(mpo1)
        
        B = network[m]
        A, B, s = svd_mpo(A,B,'R')
        network_2 = network_2 +[A,B]+network[m+1:]


        e_e = neumann(s)
        return network_2, e_e

def get_network(rule,order,sites):
    operator    = rule_matrix(rule,order)
    network     = create_network(operator_to_mpo(operator),sites)
    return network

def reduce_norm(tensors):
    for tensor in tensors:
        max = np.max(tensor)
        min = np.min(tensor)
        if max-min>2:
            tensor /= (max-min)/1.4
    return tensors

def neumann(s):
    p = (s/np.sqrt(np.sum(s**2)))
    return np.abs(-np.sum((p**2)*np.log((p**2))/np.log(hsd)))

'''
Functions related to matrix-product-states:
'''

def create_mps(array,complex=False):
    I = np.identity(hsd)
    if complex:
        I = np.cdouble(I)
    L = []
    L.append(
        I[array[0]].reshape((hsd,1))
        )
    for x in array[1:-1]:
        L.append(
            I[x].reshape((hsd,1,1))
            )
    L.append(
        I[array[-1]].reshape((hsd,1))
        )   
    return L

def mps_to_vector(mps):
    n = len(mps)
    tensor_1 = mps[0]
    i=1
    for tensor_2 in mps[1:]:
        tensor_1 = np.tensordot(tensor_1,tensor_2,axes=(i,1))
        i+=1
    return tensor_1.reshape((hsd**n),order='F')

def mps_to_array(mps):
    n = len(mps)
    tensor_1 = mps[0]
    i=1
    for tensor_2 in mps[1:]:
        tensor_1 = np.tensordot(tensor_1,tensor_2,axes=(i,1))
        i+=1
    tensor_1 = tensor_1.reshape((hsd**n),order='F')
    
    # o--1  1--o--2          o----2
    # |        |      ->    / \
    # 0        0           0   1

    
    unit_vector = (np.sum(np.rint(tensor_1)!=0)==1)
    if unit_vector:
        result = vector_to_bit_array(tensor_1,n)
    else:
        print("🦕")
        print(tensor_1)
        result = np.zeros(n)
    return result

def mps_bool_check(mps):
    return np.allclose(bonds_of_mpo(mps),1)
    
def mps_to_array_fast(mps):
    if hsd==2:
        L = [np.rint(np.abs(x.ravel())[1]) for x in mps]
    if hsd>2:
        L = [np.where(np.rint(np.abs(x.ravel()))==1)[0][0] for x in mps]
    return L

def random_mps(sites,low=1,high=5):
    mps1 = []
    bonds1 = np.random.randint(low,high,size=sites-1)
    mps1.append(
        np.random.rand(2*bonds1[0])\
            .reshape(2,bonds1[0])
        )
    for i in range(sites-2):
        mps1.append(
            np.random.rand(2*bonds1[i]*bonds1[i+1])\
                .reshape(2,bonds1[i],bonds1[i+1])
            )
    mps1.append(
        np.random.rand(2*bonds1[-1])\
            .reshape(2,bonds1[-1])
        )
    return mps1

def mpsoc(mps,mpo):
    # MPSOC = MPS-mpO Contraction
    
    n = len(mpo)
    mps_2 = []

    mps_2.append(
        np.tensordot(mps[0],mpo[0],axes=(0,1))\
            .transpose(1,2,0).reshape((
                hsd,
                mps[0].shape[1]*mpo[0].shape[2]
            ),order='F')
    )

    for i in range(1,n-1):
        mps_2.append(
            np.tensordot(mps[i],mpo[i],axes=(0,1))\
                .transpose(2,3,0,4,1).reshape((
                    hsd,
                    mps[i].shape[1]*mpo[i].shape[2],
                    mps[i].shape[2]*mpo[i].shape[3]
            ),order='F')
        )
        
    mps_2.append(
        np.tensordot(mps[-1],mpo[-1],axes=(0,1))\
            .transpose(1,2,0).reshape((
                hsd,
                mps[-1].shape[1]*mpo[-1].shape[2]
            ),order='F')
    )   

    
    #  o--0   0--o--1   0--o--1   0--o
    #  |         |         |         |
    #  #--2   3--#--4   3--#--4   2--#
    #  |         |         |         |
    #  1         2         2         1

    return mps_2

def svd_mps(A,B,direction):
    # direction = 'R' or 'L'

    sA = A.shape
    sB = B.shape
 
    #   1--#--3
    #     / \
    #    0   2

    #  1--#--2  1--#--2
    #     |        |
    #     0        0

    if len(sA)==len(sB)==3:
        leg1 = sA[1]
        leg2 = sB[2]
        t = np.tensordot(A,B,axes=(2,1))\
        .reshape((hsd*leg1,hsd*leg2),order='F')
        U, S, V, rank = custom_svd(t)
        if direction == 'R':
            A2 = U.reshape((hsd,leg1,rank),order='F')
            B2 = np.tensordot(np.diag(S),V,axes=(1,0))\
                .reshape((rank,hsd,leg2),order='F').transpose((1,0,2))
        elif direction == 'L':
            A2 = np.tensordot(U,np.diag(S),axes=(1,0))\
                .reshape((hsd,leg1,rank),order='F')
            B2 = V.reshape((rank,hsd,leg2),order='F').transpose((1,0,2))
        
        # #--1  1--#--2
        # |        |
        # 0        0
    elif (len(sA)==2) and (len(sB)==3):
        leg2 = sB[2]
        t = np.tensordot(A,B,axes=(1,1))\
        .transpose(0,1,2).reshape((hsd,hsd*leg2),order='F')
        U, S, V, rank = custom_svd(t)
        if direction == 'R':
            A2 = U.reshape((hsd,rank),order='F')
            B2 = np.tensordot(np.diag(S),V,axes=(1,0)).reshape((rank,hsd,leg2),order='F').transpose((1,0,2))
        elif direction == 'L':
            A2 = np.tensordot(U,np.diag(S),axes=(1,0)).reshape((hsd,rank),order='F')
            B2 = V.reshape((rank,hsd,leg2),order='F').transpose((1,0,2))  
    
        # 1--#--2  1--#
        #    |        |
        #    0        0
    elif (len(sA)==3) and (len(sB)==2):
        leg1 = sA[1]
        t = np.tensordot(A,B,axes=(2,1))\
        .transpose(0,1,2).reshape((hsd*leg1,hsd),order='F')
        U, S, V, rank = custom_svd(t)
        if direction == 'R':
            A2 = U.reshape((hsd,leg1,rank),order='F')
            B2 = np.tensordot(np.diag(S),V,axes=(1,0)).reshape((rank,hsd),order='F').transpose((1,0))
        elif direction == 'L':
            A2 = np.tensordot(U,np.diag(S),axes=(1,0)).reshape((hsd,leg1,rank),order='F')
            B2 = V.reshape((rank,hsd),order='F').transpose((1,0))
        
        # #--1  1--#
        # |        |
        # 0        0
    elif (len(sA)==2) and (len(sB)==2):
        t = np.tensordot(A,B,axes=(1,1))\
        .reshape((hsd,hsd),order='F')
        U, S, V, rank = custom_svd(t)
        if direction == 'R':
            A2 = U
            B2 = np.tensordot(np.diag(S),V,axes=(1,0)).transpose((1,0))
        elif direction == 'L':
            A2 = np.tensordot(U,np.diag(S),axes=(1,0))
            B2 = V.transpose((1,0))


    else:
        print("🦕")
        A2 = A
        B2 = B
    return A2, B2, S
    
def optimize_mps(mps,direction):
    n = len(mps)
    m = int(n/2)

    if direction=='M':
        if n==2:
            A = mps[0]
            B = mps[1]

            A, B, _ = svd_mps(A,B,'R')
            L = [A,B]
        elif n==3:
            A = mps[0]
            B = mps[1]
            C = mps[2]

            A, B, _ = svd_mps(A,B,'R')
            B, C, _ = svd_mps(B,C,'L')
            L = [A,B,C]
        elif n>=4:
            L1 = []
            tensor_1, A, _ = svd_mps(mps[0],mps[1],'R')
            L1.append(tensor_1)
            for i in range(2,m):
                tensor_1, A, _ = svd_mps(A,mps[i],'R')
                L1.append(tensor_1)

            L2 = []
            B, tensor_2, _ = svd_mps(mps[n-2],mps[n-1],'L')
            L2.append(tensor_2)
            for i in range(2,n-m):
                B, tensor_2, _ = svd_mps(mps[n-1-i],B,'L')
                L2.append(tensor_2)
            
            A, B, _ = svd_mps(A,B,'R')
            L = L1+[A,B]+L2[::-1]

    elif direction=='R':
        L = []
        tensor_1, tensor_2, _ = svd_mps(mps[0],mps[1],'R')
        L.append(tensor_1)
        for i in range(2,n):
            tensor_1, tensor_2, _ = svd_mps(tensor_2,mps[i],'R')
            L.append(tensor_1)
        L.append(tensor_2)

    elif direction=='L':
        L = []
        tensor_1, tensor_2, _ = svd_mps(mps[n-2],mps[n-1],'L')
        L.append(tensor_2)
        for i in range(2,n):
            tensor_1, tensor_2, _ = svd_mps(mps[n-1-i],tensor_1,'L')
            L.append(tensor_2)
        L.append(tensor_1)
        L = L[::-1]

    return L

def ee_mps(mps):
    n = len(mps)
    m = int(n/2)

    if n==2:
        A = mps[0]
        B = mps[1]

        A, B, S = svd_mps(A,B,'R')
        L = [A,B]
    elif n==3:
        A = mps[0]
        B = mps[1]
        C = mps[2]

        A, B, _ = svd_mps(A,B,'R')
        B, C, S = svd_mps(B,C,'L')
        L = [A,B,C]
    elif n>=4:
        L1 = []
        tensor_1, A, _ = svd_mps(mps[0],mps[1],'R')
        L1.append(tensor_1)
        for i in range(2,m):
            tensor_1, A, _ = svd_mps(A,mps[i],'R')
            L1.append(tensor_1)

        L2 = []
        B, tensor_2, _ = svd_mps(mps[n-2],mps[n-1],'L')
        L2.append(tensor_2)
        for i in range(2,n-m):
            B, tensor_2, _ = svd_mps(mps[n-1-i],B,'L')
            L2.append(tensor_2)
        
        A, B, S = svd_mps(A,B,'R')
        L = L1+[A,B]+L2[::-1]
    ee = neumann(S)
    return L, ee


def evolve(network_t,network,sweep):
    network_t       = product(network_t,network)
    network_t, ee   = optimize_mpo(network_t,direction=sweep)
    bonds           = bonds_of_mpo(network_t)
    return network_t, ee, bonds

def create_mps_uniform(sites,complex=False,quantum=False):
    I = np.array([1,1])
    if quantum:
        I = np.sqrt(1/2)*I
    if complex:
        I = np.cdouble(I)
    L = []
    L.append(
        I.reshape((hsd,1))
        )
    for x in range(sites-2):
        L.append(
            I.reshape((hsd,1,1))
            )
    L.append(
        I.reshape((hsd,1))
        )   
    return L

def init_directed(rule,order,sites):

    operator    = rule_matrix(rule,order)
    mpo         = operator_to_mpo(operator)
    network     = create_network(mpo,sites)
    network, ee = optimize_mpo(network,direction='LRM')
    network     = reduce_norm(network)
    bonds       = bonds_of_mpo(network)
    return network, ee, bonds   