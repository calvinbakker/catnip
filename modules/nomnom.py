import numpy as np
from ncon import ncon
import sys
from tensor import operator_to_mpo,product
from nomnom import rightrotate

def AND_operator():
    AND = np.zeros((2,2,2,2,2),order='F')
    #   0  2
    #   |  |
    #   ####--4
    #   |  |
    #   1  3

    i = np.identity(2)
    for b1 in range(2):
        for b2 in range(2):
            AND[b1,b1,b2,b2]=i[b1 and b2]
    return AND

def S0_operator(q):
    S0 = np.zeros((2,2,2),order='F')
    #   0  
    #   |  
    #   #--2
    #   |  
    #   1  
 
    i = np.identity(2)
    for b1 in range(2):
        S0[b1,b1]=i[q]
    return S0

def XOR_operator():
    #  1   2
    #   \ /
    #    #
    #    |
    #    3
    
    XOR = np.zeros((2,2,2),order='F')
    i = np.identity(2)
    for b1 in range(2):
        for b2 in range(2):
            b3 = b1 or b2
            XOR[b1,b2]=i[b3]
    XOR[1,1]=i[0]
    return XOR

def ADD_operator():
    #  1   2
    #   \ /
    #    #
    #    |
    #    3
    
    XOR = np.zeros((2,2,2),order='F')
    i = np.identity(2)
    for b1 in range(2):
        for b2 in range(2):
            b3 = b1 or b2
            XOR[b1,b2]=i[b3]
    XOR[1,1]=i[0]
    return XOR

def S1_operator(q):
    S0 = np.zeros((2,2,2),order='F')
    #   0  
    #   |  
    #   #--2
    #   |  
    #   1  
 
    i = np.identity(2)
    for b1 in range(2):
        S0[b1,b1]=i[q]
    return S0


def NOT_operator():
    #   0  
    #   |  
    #   #
    #   |  
    #   1  
 
    NOT = np.identity(2)[::-1]
    return NOT


def COPY_operator():
    COPY = np.zeros((2,2,2),order='F')
    i = np.identity(2)
    for b in range(2):
        COPY[b,b,:]=i[b]
        COPY[b,:,b]=i[b]
    return COPY

def KW_operator(q):
    i = np.identity(2)
    return i[q]


def SHA2_operator(kw,s0,s1):
    S0 = S0_operator(s0)
    ADD = ADD_operator()
    AND = AND_operator()
    XOR = XOR_operator()
    NOT = NOT_operator()
    S1 = S1_operator(s1)
    COPY = COPY_operator()
    KW = KW_operator(kw)

    ABC = ncon([S0,AND,AND,AND,XOR,XOR,ADD],(
        [-1,1,2],
        [1,3,-2,4,5],
        [3,-4,-3,6,7],
        [4,-5,6,-6,8],
        [5,7,9],
        [9,8,10],
        [2,10,-7]
        )
    )
    # 0  1  2 
    # |  |  |
    # #######--6
    # |  |  |
    # 3  4  5


    EFG = ncon([S1,AND,NOT,AND,XOR,ADD,NOT],(    
            [-1,1,2],
            [1,3,-2,-5,4],
            [3,5],
            [5,8,-3,-6,6],
            [4,6,7],
            [2,7,-7],
            [8,-4]
            )
    )

    DH = ncon([KW,ADD,ADD,COPY,ADD,ADD],(
            [1],
            [-2,-4,2],
            [1,2,3],
            [3,4,5],
            [-3,4,-6],
            [5,-1,-5]          
            )
    )

    # print(ABC.shape)
    # print(DH.shape)
    # print(EFG.shape)

    full = ncon([ABC,DH,EFG],
                (
        [-1,-2,-3,-9,-10,-11,1],
        [1,-4,-5,2,-12,-13],
        [-6,-7,-8,-14,-15,-16,2]
                ))

    # (a,b,c,hn,dn,e,f,g)

    d = len(full.shape)//2
    step1 = full.reshape(2**d,2**d,order='F')
    network1 = operator_to_mpo(step1)


    R4 = rightrotate(4)
    R2 = rightrotate(2)

    c1 = ncon([R4[-1],R2[0]],([-1,1,-3],[1,-2,-4]))
    c2 = ncon([R4[0],R2[-1]],([-1,1,-4],[1,-2,-3]))

    network2 = R4[:-1]+[c1,c2]+R4[1:]

    SHA2 = product(network1,network2)

    return SHA2
