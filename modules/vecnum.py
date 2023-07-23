import numpy as np
from config import hsd
from rule import get_string

'''
Note about conventions:

The number 00 corresponds to the array [1,0,0,0]
The number 11 corresponds to the array [0,0,0,1]

'''


def kronecker_product(x):
    '''
    Performes the Kronecker product for a list of vectors.

    Input: 
        A list of vectors.
    Output:
        A single vector that is the Kronecker product of all the inputs
    
    '''

    m = np.kron(x[0],x[1])
    if len(x)>2:
        for i in range(len(x)-2):
            m = np.kron(m.copy(),x[i+2])
    return m

def number_to_bit_array(number,sites):
    '''
    Maps a number to a bit-array

    Input: 
        A number and an amount of sites 
    Output:
        A bit-array represenation of the number
    
    '''

    bit_string = get_string(number)[::-1]
    bit_string += (sites-len(bit_string))*"0"
    bit_string = bit_string[::-1]

    bit_array = []
    for bit in bit_string:
        bit_array.append(int(bit))
    bit_array = np.array(bit_array)
    return bit_array[::-1]

def bit_array_to_vector(bit_array):
    '''
    Takes a bit-array and maps this to a vector in Hilbert space.

    Input: 
        A bit-array.
    Output:
        A vector.
    
    '''
    
    vector = []
    for x in bit_array:
        vector.append(np.identity(hsd)[x])
    vector = kronecker_product(vector[::-1])
    return vector

def vector_to_bit_array(vector,sites):
    '''
    Takes a vector and maps it to a bit-array. This operation needs the amount of sites to be specified.

    Input: 
        A vector.
    Output:
        A bit-array.
    
    '''
    
    decomposition = []
    state = np.rint(vector)
    for _ in range(sites):
        nn = int(np.floor((np.where(state==1)[0]/len(state))*hsd))
        state = state[int(nn*len(state)/hsd):int((nn+1)*len(state)/hsd)]
        decomposition.append(nn)
    return np.array(decomposition)[::-1]

def bit_array_to_number(bit_array):
    '''
    Takes a bit-array and maps it to a decimal number.

    Input: 
        A bit-array.
    Output:
        A number.
    
    '''

    bit_string2 = ""
    for z in bit_array[::-1]:
        bit_string2+=str(z)
    return int(bit_string2,hsd)