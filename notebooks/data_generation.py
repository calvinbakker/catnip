import numpy as np
import os
import sys
import time
sys.path.append('./modules/') 
from tensor import *
from data import *
from wolfram import init_wolfram, unique_wolfram_rules	

data_path = f'./'			
			


## config
timesteps       = 20
sites           = 50 
r_list          = unique_wolfram_rules[:10]


## open files
os.makedirs(data_path,exist_ok=True)
os.makedirs(data_path+'/tensors',exist_ok=True)


## algorithm
EE_list = []
BONDS_list = []
for rule in r_list:
    print(f"👽 Wolfram R{rule}")
    start_time = time.time()
    EE = []
    BONDS = []

    network, ee, bonds = init_wolfram(rule,sites)
    EE.append(ee)
    BONDS.append(np.max(bonds))

    print(f"⌚:",bonds)
    network_t   = network.copy()
    for _ in range(timesteps):
        network_t, ee, bonds = evolve(network_t,network,'LRM')
        print(f"⌚:",bonds)
        EE.append(ee)
        BONDS.append(np.max(bonds))

        if bonds_check(BONDS):
            break
        if np.max(bonds)>1000:
            break
    EE_list.append([rule]+EE)
    BONDS_list.append([rule]+BONDS)
    # export_network(network_t,rule,sites,timesteps,data_path)
    end_time = time.time()
    print("Time: ", round(end_time - start_time), "seconds")



## export data
try:
    export_csv(EE_list, data_path+f'/s{sites}_ee.csv')
    export_csv(BONDS_list, data_path+f'/s{sites}_bonds.csv')
except:
    EE_list_2 = []
    for x in EE_list:
        l = len(x)
        if l!=timesteps+2:
            x= x+((timesteps+2)-l)*[np.nan]
        EE_list_2.append(x)

    BONDS_list_2 = []
    for x in BONDS_list:
        l = len(x)
        if l!=(timesteps+2):
            x= x+((timesteps+2)-l)*[np.nan]
        BONDS_list_2.append(x)
        
    export_csv(EE_list_2, data_path+f'/s{sites}_ee.csv')
    export_csv(BONDS_list_2, data_path+f'/s{sites}_bonds.csv')
