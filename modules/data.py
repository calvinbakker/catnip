import numpy as np
import pandas as pd
import pickle
import os
import psutil


def check_RAM():
    '''
    Prints a warning if the amount of RAM used exceeds 95%
    '''

    if psutil.virtual_memory().percent > 95:
        print("🤖 bleep bloop... RAM is almost full")


# checks for exponential growth
def entropy_check(EE):
    '''
    True/False check for systems that have integer linear growth 
    '''

    if len(EE)==3:
        if np.sum(np.round(EE[:3],2)==np.round([1.,2.,3.],2))==3:
            return True
        elif np.sum(np.round(EE[:3],2)==np.round([2.,4.,6.],2))==3:
            return True
        else:
            return False
    else:
        return False

def bonds_check(bonds):
    '''
    True/False check for systems that have exponential bond growth
    '''

    if len(bonds)==3:
        if np.sum(np.round(bonds,2)==np.round([2.,4.,8.],2))==3:
            return True
        elif np.sum(np.round(bonds,2)==np.round([4.,16.,64.],2))==3:
            return True
        else:
            return False
    else:
        return False


# export functions
def export_network(network,rule,sites,timesteps,data_path):
    '''
    Exports network as a pickle file .tn
    '''

    path = data_path+f'/tensors/s{sites}_R{rule}_t{timesteps}.tn'
    with open(path, 'wb') as file:
        pickle.dump(network,file)
    return None

def export_csv(L,name,abs=True):
    '''
    Exports data as .csv file
    '''
    
    if abs:
        df = pd.DataFrame(np.abs(L)).T
    else:
        df = pd.DataFrame(np.array(L)).T
    df.columns = np.abs(df.iloc[0]).astype(int)
    df.drop(0,inplace=True)
    df.to_csv(name,index=False)
    return None

