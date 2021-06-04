'''
    Binary symmetric weighted decomp w. diagonal wildcards.
    Solves for the weights using an integer partitioning method
    only on basis rows. Finds basis by backtracking
    
    (ipart algorithm)
'''  

import numpy as np
import networkx as nx
import itertools, math, time, random
from collections import deque 

import algs.utils as utils


class Bookeeper:
    def __init__(self, k, max_val):
        self.max_val = max_val
        self.k = k
        self.b = 0
        self.i = 0
        
        # for extending basis
        self.cart_V = utils.LazyCartesianProduct(k)
        
        # each index represents a basis row, the value represents the index in cartV of 
        # current pattern, each time we need a new pattern row, the index is incremented 
        self.curr_basis_pattern = [0]*max_val  
        
        # tracks the indices in B that are filled w. basis rows
        self.basis_inds = [-1]*max_val  
            
        # holds num constraints added per basis row to model   
        self.all_weights_checkpts = [0]*max_val
        
    def get_next_patt_ind(self):
        # gets next index thats not linearly dependent 
        while True:
            if self.curr_basis_pattern[self.b]>=2**self.k:
                return -1
            
            ind = self.curr_basis_pattern[self.b]
            self.curr_basis_pattern[self.b]+=1
        
            if ind != 0:  
                return ind

    def get_next_patt_row(self):
        ind = self.get_next_patt_ind()
    
        if ind == -1:
            return [-1]
        else:
            Pb = self.cart_V.entryAt(ind) # get potential basis row 
            return Pb
    
    def backtrack(self, B):
        # holds weights deque before trying to insert basis at index i
        if self.b-1==-1:
            return False
        
        self.all_weights_checkpts[self.b-1]=0
        self.curr_basis_pattern[self.b]=0
                    
        self.b-=1
        self.i = self.basis_inds[self.b]
        self.basis_inds[self.b]=-1
        B[self.i] = [-1]*self.k   
        return True
     
     
def i_compatible_findweights_driver(all_weights, A, B, v, row_i, k, is_basis):
    '''
    Loops through each partially filled (or fully filled) weight matrix
    if it's trying to insert a basis row (is_basis is T) send current W to 
    compute rest of weights
        if the deque returned by icomp findweights is empty, then no weight matrices
        could be found that are compatible, add W to the didnt work deque
        
    if not, just check it W is icompat 
        if not, add to didnt work, if it is, add it to all_weights
    '''
    w_iter = 0
    icomp = False
    dq_size = len(all_weights.w_deque)
    didnt_work = deque()
    
    if is_basis:
        while w_iter<dq_size:
            W = all_weights.w_deque.pop()
            
            # returns updated list of weights based on W
            temp_dq = i_compatible_findweights(A, B, v, row_i, k, W)
            
            # if at least one W is icomp, then return icomp true
            if len(temp_dq)>0: 
                icomp = True
                all_weights.extend(temp_dq) # adds multiple to left
            else:
                didnt_work.append(W)

            w_iter+=1
    
    # NOTE only evaluate one weight matrix during extend basis insertions
    if not is_basis:
        W = all_weights.w_deque.pop()
        icomp = utils.weighted_i_compatible(A, B, W, v, row_i)
        
        if icomp: 
            all_weights.appendleft(W) # adds to left
        else:
            didnt_work.append(W)    
    
    if not icomp:
        all_weights.w_deque = didnt_work
    
    return icomp


def i_compatible_findweights(A, B, v, i, k, W):
    temp_dq = deque()
    
    Ai = A[i]
    Aii = Ai[i]
    
    vW = np.dot(v, W)
    vWvT = np.dot(vW, v.T)
        
    compat = False
    W_copy = np.copy(W)
    
    #---------------first condition
    if Aii != math.inf:  
        # these W's are i-compatible at Aii for v at index i, given W
        # check i==i case, (B_i * W) * B_i.T  = A_ii, B_i is our current guess v
        utils.update(Aii, W_copy, v, v, temp_dq)
    else: 
        if utils.equiv_ints(vWvT, Aii):
            temp_dq.appendleft(W_copy)
                
    #---------------second condition
    # now, check that v works for the non-null rows of B        
    j=0
    for Bj in B:
        if j != i and not np.all(Bj==-1):  # i!=j and Bj not null row
            Aij = A[i, j]
            
            w_iter=0
            dq_size=len(temp_dq)
                        
            # for all current Ws check if compat w. Bj, update if necessary
            while w_iter<dq_size:
                w=temp_dq.pop()
                utils.update(Aij, w, v, Bj, temp_dq)  # will readadd w if compat
                w_iter+=1
        j+=1
    
    return temp_dq
 
 
def extend_basis(A, B, cart_V, k, all_weights):   
    # need copy since driver adds and removes Ws
    temp_dq = all_weights.w_deque.copy() 
    
    row_i = 0
    for Bi in B:
        filled = False
        
        if np.all(Bi == -1):            
            for j in range(0,2**k):
                v = np.array(cart_V.entryAt(j))
                
                # false param since we're not finding basis row
                icomp = i_compatible_findweights_driver(all_weights, A, B, 
                                                        v, row_i, k, False)
                
                if icomp: 
                    B[row_i] = v
                    filled=True
                    break
                
            if not filled:
                all_weights.w_deque = temp_dq
                return B, row_i
        row_i+=1
    
    return B, A.shape[0]


def get_icomp_basis_row(A, B, book, all_weights):
    '''
    Gets the next basis row that is i-compatible.
    '''    
    temp_dq = all_weights.w_deque.copy()
    
    icomp=False
    while not icomp:  # get basis row until icomp
        Pb = book.get_next_patt_row()
        #print(book.curr_basis_pattern, Pb)
        
        if Pb[0] == -1:            
            # reset all_weights to before inserting basis book.b-1 since
            # we're going back to rethink book.b-1
            if book.b-1<0: # could not find a soln
                return [-1]
            
            all_weights.w_deque = book.all_weights_checkpts[book.b-1]

            temp_dq = all_weights.w_deque.copy() # reset temp_dq too
            status = book.backtrack(B)
        else:
            icmp = i_compatible_findweights_driver(all_weights, A, B, 
                                                   np.array(Pb), book.i, book.k, True)
                
            # note: dont check for lin indep (in basis rows only case)
            # nor w-lim in weighted case
            if icmp: # inserting basis row                
                book.basis_inds[book.b] = book.i
                
                # at b ind insert what all_weights was before inserting b basis row
                book.all_weights_checkpts[book.b] = temp_dq 
                
                icomp=True
                        
    return Pb  # returning icomp basis row 


def BSWD_DW(A, k):
    n = A.shape[0]
    if 2*k>n:
        max_val = n
    else:
        max_val = 2*k
    
    book = Bookeeper(k, max_val)
   
    all_weights = utils.Weights(k)
    
    B_tilde = np.full((n, k), -1)   # nxk 'null' rows
    backtracked=False
    
    # book.b: our current basis row [0, 1, ..., maxval-1]
    # book.i: the index of B we are trying to insert basis row
    while book.b < max_val:
        if backtracked:
            book.b-=1
        
        Pb = get_icomp_basis_row(A, B_tilde, book, all_weights)
        
        if Pb[0]==-1: # failed
            break
        
        B_tilde[book.i] = Pb  
        
        B_tilde_copy = np.copy(B_tilde)    
        
        B, i, = extend_basis(A, B_tilde_copy, book.cart_V, k, all_weights)
        book.i = i
        
        if book.i == n:
            return B, all_weights.w_deque.pop()
        
        # backtracking not finished yet but while loop is about to quit
        backtracked=False
        if book.b+1 == max_val and not all(val == (2**k-1) for val in book.curr_basis_pattern):
            book.i = book.basis_inds[book.b]
            book.backtrack(B_tilde)
            backtracked=True
        
        book.b+=1

    return np.full((1,1), -1), np.full((1,1), -1)



