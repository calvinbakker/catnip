import numpy as np
import math
from typing import Tuple, List, Any

#                                 /$$$$$$  /$$          
#                                /$$__  $$|__/          
#   /$$$$$$$  /$$$$$$  /$$$$$$$ | $$  \__/ /$$  /$$$$$$ 
#  /$$_____/ /$$__  $$| $$__  $$| $$$$    | $$ /$$__  $$
# | $$      | $$  \ $$| $$  \ $$| $$_/    | $$| $$  \ $$
# | $$      | $$  | $$| $$  | $$| $$      | $$| $$  | $$
# |  $$$$$$$|  $$$$$$/| $$  | $$| $$      | $$|  $$$$$$$
#  \_______/ \______/ |__/  |__/|__/      |__/ \____  $$
#                                              /$$  \ $$
#                                             |  $$$$$$/
#                                              \______/ 


hsd = 2 # Local Hilbert space dimension (e.g., 2 for spin-1/2 systems, 3 for spin-1 systems, etc.)
num_threshold = 1e-10 # Numerical threshold used for SVD truncation
dtype = np.float64 # Default data type for tensors