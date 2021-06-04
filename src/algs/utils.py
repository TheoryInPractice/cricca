
import pandas as pd
import numpy as np
import sympy, os
from collections import Counter
import itertools, random, math, re
from collections import deque 

   
def equiv_rows(u_row, a_row):
    '''
    Args: u_row : 1-D pandas array
          a_row : 1-D pandas array
    '''
    for i, val in u_row.items():
        if not equiv_ints(u_row[i], a_row[i]):
            return False
    return True


def equiv_ints(u, a):
    if u != a and a != math.inf and u != math.inf:  
        return False
    else:
        return True
        

def w_limited_guess(cart, *args, numrows=1, numcols=1, index=0, w=0):
    # used in non-backtracking versions
    P = np.array(cart.entryAt(index)).reshape(numrows,numcols)
    
    # check if compatible (w-limitness)
    compat = P.dot(P.T).max()
    
    # check for lin independence
    # reduced row echelon form
    # inds are indices of pivot cols 
    _, inds = sympy.Matrix(P).T.rref() 
    
    if compat <= w and len(inds)==len(P):
        return P
    else:
        return np.full((1,1), -1)  
    

def weighted_i_compatible(A, B, W, v, i):    
    aii = A[i, i]
    
    vW = np.dot(v, W)
    vWvT = np.dot(vW, v.T)
    
    # first condition
    if not equiv_ints(vWvT, aii):
        return False
        
    # second condition
    j = 0
    compat = True
    for Bj in B:
        if j != i and not np.all(Bj==-1):  # i!=j and Bj not null row
            Aij = A[i, j]
            vWBjT = np.dot(vW, Bj.T)
            if vWBjT != Aij:
                compat = False
        j+=1
    return compat


def construct_weight_matrix(v, vals, k):
    '''
    v: the diagonal indices to actually insert values
    vals: the weight values
    
    Inserts values into the diagonal weight matrix 
    '''
    W = np.zeros((k, k))
    
    fillcount=0
    for i in range(W.shape[0]):  # makes diagonal null values
        if v[i] == 0:
            W[i, i] = np.inf
        else:
            if vals[fillcount] == -0.0:
                W[i,i] = 0.0
            else: 
                W[i,i] = vals[fillcount]
            fillcount+=1
    return W


def is_lin_indep(P, Pb):    
    # check for lin indep w. previous rows, returns T/F + rank
    P.append(Pb)
    _, inds = sympy.Matrix(P).T.rref() 
    psize = len(P)
    P.pop()
    
    if len(inds)==psize:
        return True, len(inds)
    else:
        return False, len(inds)
    

def remove_constraints(model, count):
    # removes the last count number of constraints added to model
    for c in range(count):
        constraint = model.getConstrs()[-1-c]
        model.remove(constraint)
    model.update()


#----------------------------------------------------- ip find weights
class Weights:
    def __init__(self, k):
        self.W_0 = construct_weight_matrix([1]*k, [-1]*k, k) # init weight matrix
        self.w_deque = deque()
        self.w_deque.append(self.W_0)
        
        
    def extend(self, ws):
        self.w_deque.extendleft(ws)
    
    def appendleft(self, w):
        self.w_deque.appendleft(w)
    
    def print_ws(self):
        for w in self.w_deque:
            print(w)

def get_one_rnd_perm(k_total, k_distinct):
    # used in datagen.py
    # finds combs sum to A, then finds all permutations of each comb 
    if k_distinct==1:
        return [k_total] 
    
    combos = combo(k_total, k_distinct)
    
    valid_combos = []
    for comb in combos:
        if len(list(comb.elements())) == k_distinct:
            valid_combos.append(comb)
    
    number_potential = len(valid_combos)
    
    rnd_perm_ind = random.randint(0, number_potential-1) 
    
    clique_weights = list(valid_combos[rnd_perm_ind].elements())
        
    random.shuffle(clique_weights)
    
    return clique_weights


def combo(total, n):
    '''
    Find all combinations of n integers from 0 to total that sum to total
    '''
    weights = range(0,total+1)
    
    # initialize list of list for potential weights
    pws = [[Counter()]] + [[] for _ in range(total)]   # A/maxweight+1 time
    
    # this is pretty close to being O(n^2)
    for weight in weights:  # {0, 1, 2, 3, ..., total} A/maxweight time
        for i in range(weight, total + 1): # A/maxweight+1 time 
                        
            # increment pws at index i and add a weight
            for pw in pws[i-weight]:
                
                # prevents combos w. too many elements
                if len(list(pw.elements()))+1<=n:          
                    pws[i] += [pw + Counter({weight: 1})]
             
    return pws[total]


def get_combos(sum_to, num_ints):
    if num_ints==1:
        return [Counter({int(sum_to) : 1})]
    
    combos = combo(int(sum_to), int(num_ints))
    return combos


def update(Aij, W, v, Bj, w_deque):
    '''
    Returns list of new weights that are compatible
    '''    
    vW = np.dot(v, W)
    vWBjT = np.dot(vW, Bj.T)
    
    # gets the number of weights to find at this iteration
    # also the sum given the past weights
    num_ints=0
    ind=0
    sum_to = Aij
    
    for l in vW:
        if l==-1 and Bj[ind]==1:
            num_ints+=1
        elif l > 0 and Bj[ind] == 1:
            sum_to-=l     
        ind+=1
   
    # only find permuations when we need to 
    if sum_to > 0 and num_ints > 0:
        combs = get_combos(sum_to, num_ints)

        # --Now, update W
        for comb in combs:
            combination = list(comb.elements()) 

            # specific combo has proper # of ints
            if len(combination) == num_ints: 
            
                W_copy = np.copy(W) 
                numfilled = 0
                v_ind = 0
                
                # insert new weights in correct pos. in W
                for v_val in v:             
                    if vW[v_ind] == -1 and v_val == 1 and Bj[v_ind]==1:
                        W_copy[v_ind, v_ind] = combination[numfilled]
                        numfilled+=1
                    v_ind+=1
                    
                w_deque.appendleft(W_copy)
    else:
        # if we dont to update weights, checks for compatiblity
        # this check is necessary
        if equiv_ints(vWBjT, Aij):
            w_deque.appendleft(W)


 
 

