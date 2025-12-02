
import numpy as np
from typing import Tuple, Dict, List, Any



# var_names = ['X1', 'X2', 'X3', 'X4']
################################
# This function is to check the output of tigramite (PCMCI and PCMCI+ and LPCMCI) to see if it contains strings I do not expect, if so, raise an error. 
################################

DEFAULT_ALLOWED = ['', '-->', '<--', 'o->', '<-o', 'o-o']

def get_allowed_strings():
    return DEFAULT_ALLOWED

def string_checker(matrix):

    if isinstance(matrix, np.ndarray):
        raise TypeError(f"Expected a numpy.ndarray when enforce_numpy=True, got {type(matrix).__name__!s}")
    
    l_matrix = list(matrix.ravel())

    invalid_strings = [sub_string for sub_string in l_matrix if sub_string not in DEFAULT_ALLOWED]

    if len(invalid_strings) == 0:
        return True
    else:
        raise ValueError(f"Invalid strings found: {invalid_strings}")
    


dict_for_score={'-->':1 , '<--':1, 'o->':2, '<-o':2, '' :0, 'o-o':3}

def get_score_dict():
    return dict_for_score

def make_categorical_causal_graph(causal_graph: np.ndarray, dict_for_score: Dict[str, int]) -> np.ndarray:
    """
    Convert a tensor of link strings (N,N,T) into a categorical tensor of integer codes
    according to the provided mapping dict_for_score.

    Parameters
    - graph_strs: np.ndarray shape (N,N,T) of link strings
    - dict_for_score: dict mapping link strings to integer codes

    Returns
    - codes: np.ndarray shape (N,N,T) dtype=int
    """

    N, N2, T = causal_graph.shape
    codes = np.zeros((N, N, T), dtype=int)

    var_names = [f'X{i+1}' for i in range(N)]

    for i in range(N):
        for j in range(N):
            for k in range(T):
                s = str(causal_graph[i, j, k]).strip()
                # print(s)
                if not s:
                    codes[i, j, k] = 0
                    continue
                print(s)
                if s not in dict_for_score.keys():
                    raise ValueError(f"Unexpected link string: {s}")
                
                elif s.find(">") != -1:
                    print(f'{var_names[i]} is causing {var_names[j]} from lag {k}')

                    source = i
                    target = j

                    codes[source, target, k] = dict_for_score[s]

                elif s.find("<") != -1:
                    print(f'{var_names[j]} is causing {var_names[i]} from lag {k}')

                    source = j
                    target = i

                    codes[source, target, k] = dict_for_score[s]

                elif s.find("o") != -1:
                    print(f'Undirected or partially directed link between {var_names[i]} and {var_names[j]} at lag {k}')
                
                    codes[i, j, k] = dict_for_score[s]


    return codes



