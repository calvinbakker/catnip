from config import hsd
import numpy as np

'''
Note about conventions:

These functions correspond to the case where the rules map the most LEFT value of the input-state to the corresponding output-state.


BE CAREFUL:
The mapping from string to Hilbert space is changing the chirality of the numbers. In the arrays small-to-large will be from LEFT-to-RIGHT, while for strings it RIGHT-TO-LEFT. This way of counting corresponds with the way Wolfram counts his rules.
'''


def get_string(rule):
    '''
    Takes a number as input and returns the bit-string for a specific hsd (Hilbert space dimension).

    Input: 
        A rule number in decimal.
    Output:
        A string of the rule number in the number-system of the hsd (hsd-ns), where the lowest integer is on the RIGHT side.
    
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

def get_map(rule,order):
    '''
    For a given rule, a mapping is returned (extended bit-string).

    Input:
        A rule number in decimal.
    Output:
        An array of the rule number in hsd-ns, where the lowest integer in of the LEFT side of the array. This is the mapping between states as will be used in the time-evolution of the cellular automata.

    '''

    p = []
    for i in get_string(rule)[::-1]:
        p.append(int(i))
    for _ in range(hsd**(order+1)-len(p)):
        p.append(0)
    return p

def rule_matrix(rule,order):
    '''
    Gives the transition-matrix corresponding to the given rule.
    '''
    
    n           = hsd**(order+1)
    identity    = np.identity(n)
    p           = get_map(rule,order)

    R = []
    for i in range(n):
        R.append(identity[p[i]+hsd*int(i/hsd)])
    return np.array(R).T

def random_rule(order):
    '''
    Gives a rule-number in the valid range of rules for a given order and hsd.
    '''
    
    return np.random.randint(low=0,high=hsd**(hsd**(order+1)))

