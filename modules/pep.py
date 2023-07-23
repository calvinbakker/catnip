import numpy as np
from ncon import ncon
import sys
from tensor import operator_to_mpo, create_network


'''
Conventions
===========

For any three dimensional tensor object that is used:

     e a           4 0
     |/          
 c --#-- d     2       3
    /|         
   b f          1  5 

Any extra bonds that are introduced get higher indices (6, 7, et cetera).


leg   index   info
---   -----   --------
a   |   0   | input
b   |   1   | output
c   |   2   | horizontal left 
d   |   3   | horizontal right
e   |   4   | vertical up
f   |   5   | vertical down

left and right are connecting different words
up and down are connection letters in words

---

0 and 1 have bond dimension 2
2,3,4,5 have variable bond dimension:
- d_a01 to specify a vertical bond
- d_ab0 to specify a horizontal bond

'''

words       = 4
word_length = 3

def logic(n,shape=True):
    T = np.zeros((2,2,2,2,2,2,2),order='F')
    tensors     = []
    bonds       = []

    #      0 2 4
    #      | | |
    #   6--##### 
    #      | | |
    #      1 3 5

    i = np.identity(2)
    for b1 in range(2):
        for b2 in range(2):
            for b3 in range(2):
                if 0<=n<=15:
                    T[b1,b1,b2,b2,b3,b3]=i[int((b1 and b2) or ((not b1) and b3))]
                elif 16<=n<=31:
                    T[b1,b1,b2,b2,b3,b3]=i[int((b3 and b1) or ((not b3) and b2))]
                elif 32<=n<=47:
                    T[b1,b1,b2,b2,b3,b3]=i[int(b1 ^ b2 ^ b3)]
                elif 48<=n<=63:
                    T[b1,b1,b2,b2,b3,b3]=i[int(b2 ^ (b1 or (not b3)))]
                else:
                    print("🤖 Bleep bloop, uhm bro, dit gaat niet goed...")


    #      0        1  3
    #      |        |  |
    #   2--#--3  0--####
    #      |        |  |
    #      1        2  4

    tp = (0,1,6,2,3,4,5)
    rs = (2**3,2**4)
    T = T.transpose(tp).reshape(rs,order='F')

    u, s, v = np.linalg.svd(T, full_matrices=False) 
    bonds.append(np.sum(s>1e-10))
    U = u[:,:bonds[-1]]
    S = np.diag(s[:bonds[-1]])
    V = v[:bonds[-1],:]
    US = np.tensordot(U,S,axes=(1,0)).reshape(2,2,2,bonds[-1],order='F')
    tensors.append(US)
    rs = [bonds[-1]] + 4*[2]
    T = V.reshape(rs,order='F')

    #     0          0
    #     |          | 
    #  2--#--3    1--#
    #     |          | 
    #     1          2 


    T = T.reshape(bonds[-1]*2*2,2*2)
    u, s, v = np.linalg.svd(T, full_matrices=False)
    bonds.append(np.sum(s>1e-10))
    U = u[:,:bonds[-1]]
    S = np.diag(s[:bonds[-1]])
    V = v[:bonds[-1],:]
    US = np.tensordot(U,S,axes=(1,0))\
        .reshape(bonds[-2],2,2,bonds[-1],order='C').transpose(1,2,0,3)
    tensors.append(US)
    rs = [bonds[-1],2,2]
    T = V.reshape(rs,order='F').transpose(1,2,0)
    tensors.append(T)

    if shape:
        network = []
        network.append(id_tensor())
        network.append(tensors[0].reshape(
            list(tensors[0].shape)+[1,1]
            ,order='F'))
        network.append(tensors[1].reshape(
            list(tensors[1].shape)+[1,1]
            ,order='F'))
        network.append(tensors[2].reshape(
            list(tensors[2].shape)+[1,1,1]
            ,order='F'))
        return network  
    else:
        return tensors

def id_tensor():
    return np.identity(2).reshape((2,2,1,1,1,1),order='F')

def get_pepo_logic(word_length,iteration):
    pepo = []
    for _ in range(word_length):
        pepo.append(logic(iteration))
    return pepo

def contract_tensors(tensor_1,tensor_2):
    shape = [2,2]+list(np.array(tensor_1.shape[2:])*np.array(tensor_2.shape[2:]))
    result = ncon([tensor_1,tensor_2],(
        [-1,1,-3,-5,-7,-9],
        [1,-2,-4,-6,-8,-10]
        )).reshape(shape,order='F')
    return result

def contract_pepo(pepo_1,pepo_2):
    pepo_3 = []
    words = 4
    if len(pepo_1)==len(pepo_2):
        word_length = len(pepo_1)
    else:
        raise ValueError("🤖 Non compatible networks")
    
    pepo_3 = []
    for i in range(word_length):
        line = []
        for j in range(words):
            line.append(contract_tensors(pepo_1[i][j],pepo_2[i][j]))
        pepo_3.append(line)
    return pepo_3

def contract_pepo_multiple(L):
    pepo = L[0]
    for pepo2 in L[1:]:
        pepo = contract_pepo(pepo,pepo2)
    return pepo

def optimize_pepo_RD(pepo):
    words = 4
    word_length = len(pepo)
    
    for i in range(word_length):
        for j in range(words):
            if pepo[i][j].shape[3]!=1 and j!=(words-1):
                pepo[i][j],pepo[i][j+1] = optimize_bond(
                pepo[i][j],pepo[i][j+1]
                
                ,direction='horizontal',absorb='second')
            if pepo[i][j].shape[5]!=1 and i!=(word_length-1):
                pepo[i][j],pepo[i+1][j] = optimize_bond(
                pepo[i][j],pepo[i+1][j]
                
                ,direction='vertical',absorb='second')
    return pepo

def ADD_af(word_length):
    add = np.zeros((2,2,1,2,2,2),order='F')

    i = np.identity(2)
    for b1 in range(2):
        for b2 in range(2):
            for ri in range(2):
                b3 = (b1+b2+ri)%2
                if (b1+b2+ri)>1:
                    ro = 1
                else:
                    ro = 0
                add[b1,:,0,b2,ri,ro]=i[b3]
                add[b1,b3,0,b2,ri,:]=i[ro]
    
    add0 = add[:,:,:,:,0,:].reshape((2,2,1,2,1,2),order='F')
    L = []

    ''' 

        e a           4 0             
        |/          
    c --#-- d     2       3
       /|         
      b f          1  5 

    a = A
    b = A+F
    c = dummy
    d = F
    e = rest_in
    f = rest_out

    ''';

    L.append(add0)
    for _ in range(1,word_length):
        L.append(add)
    L[-1]=np.tensordot(L[-1],[1,1],axes=(-1,0)).reshape((2,2,1,2,2,1),order='F')
    
    I = id_tensor()
    network = []
    for l in L:
        network.append([l]+(words-1)*[I])

    return network

def ADD_akw(word_length):
    add = np.zeros((2,2,1,2,2,2),order='F')

    i = np.identity(2)
    for b1 in range(2):
        for b2 in range(2):
            for ri in range(2):
                b3 = (b1+b2+ri)%2
                if (b1+b2+ri)>1:
                    ro = 1
                else:
                    ro = 0
                add[b1,:,0,b2,ri,ro]=i[b3]
                add[b1,b3,0,b2,ri,:]=i[ro]
    
    add0 = add[:,:,:,:,0,:].reshape((2,2,1,2,1,2),order='F')
    L = []


    L.append(add0)
    for _ in range(1,word_length):
        L.append(add)
    L[-1]=np.tensordot(L[-1],[1,1],axes=(-1,0)).reshape((2,2,1,2,2,1),order='F')
    
    for i in range(word_length):
        L[i]=L[i].transpose(0,1,3,2,4,5)

    ''' 

        e a           4 0             
        |/          
    c --#-- d     2       3
       /|         
      b f          1  5 

    a = A
    b = A+KW
    c = KW
    d = dummy
    e = rest_in
    f = rest_out

    ''';

    I = id_tensor()
    network = []
    for l in L:
        network.append([l]+(words-1)*[I])
    return network

def leftrotate(word_length):
    I = np.identity(4)
    SWAP = np.array([I[0],I[2],I[1],I[3]]) 
    network = create_network(operator_to_mpo(SWAP),word_length)
    L = []
    for tensor in network[::-1]:
        if len(tensor.shape)==4:
            L.append(tensor.transpose(0,1,3,2))
        elif len(tensor.shape)==3:
            L.append(tensor)
        else:
            print("🐿️")

    L[0] = L[0].reshape((2,2,1,1,1,4),order='F')
    for i in range(1,word_length-1):
        L[i] = L[i].reshape((2,2,1,1,4,4),order='F')
    L[-1] = L[-1].reshape((2,2,1,1,4,1),order='F')

    I = id_tensor()
    network = []
    for l in L:
        network.append([l]+(words-1)*[I])
    return network

def ADD_ab(word_length):
    s1 = [2,2,1    ,2,2] # in a, out a+b, dummy,  (), ri,   ro
    s2 = [2,2    ,1,1,1] # in b, out b,    (), dummy, dummy,dummy
    p1 = np.prod(s1)
    p2 = np.prod(s2)

    add = np.zeros(tuple(s1+s2),order='F')

    i = np.identity(2)
    for b1 in range(2):
        for b2 in range(2):
            for ri in range(2):
                b3 = (b1+b2+ri)%2
                if (b1+b2+ri)>1:
                    ro = 1
                else:
                    ro = 0
                add[b1, :,0,ri,ro,b2,b2,0,0,0]=i[b3]
                add[b1,b3,0,ri, :,b2,b2,0,0,0]=i[ro]
    
    T = add.reshape((p1,p2),order='F')
    u, s, v = np.linalg.svd(T, full_matrices=False) 
    bond = np.sum(s>1e-10)
    U = u[:,:bond]
    S = np.diag(s[:bond])
    V = v[:bond,:]
    US = np.tensordot(U,S,axes=(1,0))
    tensor_1 = US.reshape([2,2,1,2,2,bond],order='F').transpose(0,1,2,5,3,4)
    tensor_2 =  V.reshape([2,2,1,1,1,bond],order='F').transpose(0,1,5,2,3,4)
    L = [tensor_1,tensor_2]

    I = id_tensor()

    network = []
    network.append([L[0][:,:,:,:,0,:].reshape((2,2,1,bond,1,2),order='F'),L[1]]+2*[I])
    for i in range(1,word_length-1):
        network.append(L+2*[I])
    network.append([np.tensordot(L[0],[1,1],axes=(5,0)).reshape((2,2,1,bond,2,1),order='F'),L[1]]+2*[I]) 

    return network

def rightrotate(word_length):
    I = np.identity(4)
    SWAP = np.array([I[0],I[2],I[1],I[3]]) 
    L = create_network(operator_to_mpo(SWAP),words)
    L[0] = L[0].reshape((2,2,1,4,1,1),order='F')
    for i in range(1,words-1):
        L[i] = L[i].reshape((2,2,4,4,1,1),order='F')
    L[-1] = L[-1].reshape((2,2,4,1,1,1),order='F')

    network = word_length*[L]
    return network



def display_bonds(network,plot=False):
    x = len(network[0])
    y = len(network)

    grid = np.zeros((3*y,3*x))


    coord = np.array([[1,0],[1,2],[0,1],[2,1]])

    for k in range(y):
        for j in range(x):
            for i,bond in enumerate(network[k][j].shape[2:]):
                grid[3*k+coord[i][0]][3*j+coord[i][1]]=bond
    grid = grid[1:-1,:-1]
    if plot:
        for i in range(word_length):
            for j in range(words):
                grid[3*i,3*j+1] = np.nan
        import matplotlib.pyplot as plt
        plt.figure(figsize=(3,3))
        plt.imshow(grid)
        plt.colorbar()
    return grid



def optimize_bond(tensor_1,tensor_2,direction,absorb,ee_return=False):
    # direction = ['horizontal','vertical'] (to the right, or to the bottom)
    # absorb = ['first','second']
    s1 = list(tensor_1.shape)
    s2 = list(tensor_2.shape)
    p1 = np.prod(s1)
    p2 = np.prod(s2)

    if direction=='horizontal':
        index = 3
        d  = tensor_1.shape[index]
        T = np.tensordot(tensor_1,tensor_2,axes=(index,index-1)).reshape((p1//d,p2//d),order='F')
        tp1 = (0,1,2,5,3,4)
        tp2 = (1,2,0,3,4,5)
    elif direction=='vertical':
        index = 5
        d = tensor_1.shape[index]
        T = np.tensordot(tensor_1,tensor_2,axes=(index,index-1)).reshape((p1//d,p2//d),order='F')
        tp1 = (0,1,2,3,4,5)
        tp2 = (1,2,3,4,0,5)

    ''' 

        e a           4 0
        |/          
    c --#-- d     2       3
        /|         
       b f          1  5 
    
    ''';
    u, s, v = np.linalg.svd(T, full_matrices=False) 
    bond = np.sum(s>1e-10)
    
    s = s[:bond]
    p = (s/np.sqrt(np.sum(s**2)))
    e_e = np.abs(-np.sum((p**2)*np.log((p**2))/np.log(2)))

    U = u[:,:bond]
    S = np.diag(s)
    V = v[:bond,:]
    if absorb=='first':
        US = np.tensordot(U,S,axes=(1,0))
        tensor_3 = US.reshape(s1[:index]+s1[index+1:]+[bond],order='F').transpose(tp1)
        tensor_4 = V.reshape([bond]+s2[:index-1]+s2[index:],order='F').transpose(tp2)
    elif absorb=='second':
        SV = np.tensordot(S,V,axes=(1,0))
        tensor_3 = U.reshape(s1[:index]+s1[index+1:]+[bond],order='F').transpose(tp1)
        tensor_4 = SV.reshape([bond]+s2[:index-1]+s2[index:],order='F').transpose(tp2)


    if ee_return:
        return tensor_3, tensor_4, e_e
    else:
        return tensor_3, tensor_4
    
def optimize_init(network):
    for j in range(word_length):
        for i in range(3):
            network[j][3-i-1],network[j][3-i] = optimize_bond(network[j][3-i-1],network[j][3-i],direction='horizontal',absorb='first')
    network[0][0],network[1][0],e_e = optimize_bond(network[0][0],network[1][0],direction='vertical',absorb='first',ee_return=True)
    return network,e_e


def pepo_to_operator2(network,d):
    from ncon import ncon

    shapes = []

    for j in range(word_length):
        for i in range(words):
            s = np.array(network[j][i].shape)
            shapes.append(s[s!=1])

    i = 0
    result = ncon([
        network[i][0].reshape(shapes[4*i+0],order='F'),
        network[i][1].reshape(shapes[4*i+1],order='F'),
        network[i][2].reshape(shapes[4*i+2],order='F'),
        network[i][3].reshape(shapes[4*i+3],order='F')],
                (
        [-1,-5,-9,1,-10],
        [-2,-6,1,2],
        [-3,-7,2,3],
        [-4,-8,3]
                ))

    # print(result.shape)

    i=1
    result2 = ncon([
        result,
        network[i][0].reshape(shapes[4*i+0],order='F'),
        network[i][1].reshape(shapes[4*i+1],order='F'),
        network[i][2].reshape(shapes[4*i+2],order='F'),
        network[i][3].reshape(shapes[4*i+3],order='F')],
                (
        [-1,-2,-3,-4,-9,-10,-11,-12,-17,1],
        [-5,-13,-18,2,1],
        [-6,-14,2,3],
        [-7,-15,3,4],
        [-8,-16,4]
                ))

    # print(result2.shape)
    # print(np.prod(result2.shape))
    operator = result2.reshape(2**8,2**8,d,order='F')
    return operator


def ee_measure(word_length,i):
    step0 = get_pepo_logic(word_length,iteration=0)
    step1 = ADD_af(word_length)
    step2 = ADD_akw(word_length)
    step3 = leftrotate(word_length)
    step4 = ADD_ab(word_length)
    step5 = rightrotate(word_length)

    L = []
    network = contract_pepo_multiple([step0,step1,step2,step3,step4,step5])
    network,e_e = optimize_init(network)
    network_t = network.copy()
    L.append(e_e)
    print("🫡")
    for _ in range(i):
        network = contract_pepo(network,network_t)
        network,e_e = optimize_init(network)
        L.append(e_e)
    return network, L
    