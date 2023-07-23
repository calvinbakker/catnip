import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

from config import hsd
from rule import get_map, rule_matrix
from tensor import *
from vecnum import bit_array_to_vector, vector_to_bit_array, number_to_bit_array
import wolfram

def save_image(L,name):
    '''
    Takes trajectory L, which is a list of bit-arrays, and visualizes this as a .png image.

    Input: 
        A list of bit-arrays.
    Output:
        A .png image.
    
    '''
    colormap = ['copper','bwr',colors.ListedColormap(['maroon','turquoise','orange','ivory']),colors.ListedColormap(['white','black','yellow','blue','red'])][hsd-2]
    # colormap = colors.ListedColormap(['ivory','cornflowerblue'])
    # colormap = colors.ListedColormap(['ivory','orangered'])
    os.makedirs(f'./image',exist_ok=True)
    plt.imsave(f"./image/{name}.png",L, cmap=colormap,vmin=0,vmax=hsd-1)
    return None


# directed cellular automata


def time_evolution_bit(rule,order,array,timesteps,name):
    '''
    Produces the time-evolution of a trajectory using the algorithmic non-linear rule-map for a directed cellular automaton.
    The directed cellular automaton is specified by the rule, and the order of the class.
    Given an initial condition (array) the function performs the time-evolution for an amount of time-steps.
    The resulting trajectory is exported as an image.

    Input: 
        Directed cellular automata rule with initial condition and an amount of time-steps.
    Output:
        An image.
    
    '''

    p = get_map(rule,order)
    p2 = [list(i)[::-1] for i in itertools.product(list(range(hsd)), repeat=(order+1))]
    sites = len(array)

    L=[]
    L.append(array.copy())
    for _ in range(timesteps):
        array2 = array.copy()
        for i in range(sites-order):
            subset = []
            for j in range(order+1):
                subset.append(array2[i+j])
            array[i] = p[p2.index(subset)]
        L.append(array.copy())

    save_image(L,name)
    return None

def time_evolution_mps_array(rule,order,array,timesteps,name):
    '''
    Produces the time-evolution of a trajectory using MPS+MPO contractions for a directed cellular automaton.
    The directed cellular automaton is specified by the rule, and the order of the class.
    Given an initial condition (array) the function performs the time-evolution for an amount of time-steps.
    The resulting trajectory is exported as an image.

    Input: 
        Directed cellular automata rule with initial condition and an amount of time-steps.
    Output:
        An image.
    
    '''

    sites = len(array)

    
    operator    = rule_matrix(rule,order)
    mpo         = operator_to_mpo(operator)
    network     = create_network(mpo,sites)

    L = []
    L.append(array)
    for _ in range(timesteps):
        mps = create_mps(array)
        mps2 = mpsoc(mps,network)
        array = mps_to_array(mps2)
        L.append(array)

    save_image(L,name)
    return None

def time_evolution_mpo_matrix(rule,order,array,timesteps,name):
    '''
    Produces the time-evolution of a trajectory using a linear-algebraic rule-map for a directed cellular automaton.
    The directed cellular automaton is specified by the rule, and the order of the class.
    Given an initial condition (array) the function performs the time-evolution for an amount of time-steps.
    The resulting trajectory is exported as an image.

    Input: 
        Directed cellular automata rule with initial condition and an amount of time-steps.
    Output:
        An image.
    
    '''
    sites = len(array)

    if sites > 10:
        return None

    operator    = rule_matrix(rule,order)
    mpo         = operator_to_mpo(operator)
    network     = create_network(mpo,sites)
    network_t   = network.copy()

    vector      = bit_array_to_vector(array)

    L = []
    # 1)
    L.append(array)
    # 2)
    big_matrix = mpo_to_operator(network_t)
    L.append(vector_to_bit_array(big_matrix@vector,sites))
    # n)
    for _ in range(timesteps-1):
        network_t = product(network_t,network)
        network_t, _ = optimize_mpo(network_t,direction='M')
        big_matrix = mpo_to_operator(network_t)
        L.append(vector_to_bit_array(big_matrix@vector,sites))

    save_image(L,name)
    return None

def show_rule(rule,order):
    '''
    Given a directed cellular automata, it prints the rule-mapping.

    Input: 
        Directed cellular automata rule.
    Output:
        The rule-mapping printed to the terminal.
    
    '''
    sites       = order+1
    matrix      = rule_matrix(rule,order)
    for state in range(hsd**(order+1)):
        in_state    = number_to_bit_array(state,sites)
        in_arr      = bit_array_to_vector(in_state)
        out_arr     = matrix@in_arr
        out_state   = vector_to_bit_array(out_arr,sites)
        print(in_state)
        print([out_state[0]])
        print()
    return None

def time_evolution_mps(rule,order,array,timesteps,name):
    '''
    Experimental function

    '''

    sites = len(array)

    operator    = rule_matrix(rule,order)
    mpo         = operator_to_mpo(operator)
    network     = create_network(mpo,sites)

    mps = create_mps(array)

    L = []
    L.append(array)
    for _ in range(timesteps):
        mps = mpsoc(mps,network)
        mps = optimize_mps(optimize_mps(mps,direction='L'),direction='R')
        array = mps_to_array_fast(mps)
        L.append(array)
        

    save_image(L,name)
    return None



# Wolfram cellular automata


def time_evolution_wolfram(rule,array,timesteps,name):
    '''
    Produces the time-evolution of a trajectory using MPS+MPO for a Wolfram cellular automaton.
    The Wolfram cellular automaton is specified by the rule, and the order of the class.
    Given an initial condition (array) the function performs the time-evolution for an amount of time-steps.
    The resulting trajectory is exported as an image.

    Input: 
        Wolfram cellular automata rule with initial condition and an amount of time-steps.
    Output:
        An image.
    
    '''
    sites = len(array)

    network = wolfram.get_wolfram_network(rule,sites)
    mps = create_mps(array)

    L = []
    L.append(array)
    for _ in range(timesteps):
        mps = mpsoc(mps,network)
        mps = optimize_mps(optimize_mps(mps,direction='L'),direction='R')
        array = mps_to_array_fast(mps)
        L.append(array)
        

    save_image(L,name)
    return None

def time_evolution_wolfram_bit(rule,array,timesteps,name):
    '''
    Produces the time-evolution of a trajectory using the algorithmic non-linear rule-map for a Wolfram cellular automaton.
    The Wolfram cellular automaton is specified by the rule, and the order of the class.
    Given an initial condition (array) the function performs the time-evolution for an amount of time-steps.
    The resulting trajectory is exported as an image.

    Input: 
        Wolfram cellular automata rule with initial condition and an amount of time-steps.
    Output:
        An image.
    
    '''

    p = wolfram.get_map(rule)
    p2 = [list(i)[::-1] for i in itertools.product(list(range(2)), repeat=3)]
    sites = len(array)

    L=[]
    L.append(array.copy())
    for _ in range(timesteps):
        array2 = array.copy()
        for i in range(1,sites-2):
            subset = []
            for j in range(3):
                subset.append(array2[i+j-1])
            array[i] = p[p2.index(subset)]
        L.append(array.copy())
    save_image(L,name)
    return L

