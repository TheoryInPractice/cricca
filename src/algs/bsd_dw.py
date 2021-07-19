'''
    Binary symmetric decomposition w. diagonal wildcards
    Original algorithm from [Feldmann, Issac, Rai '20]
    Computes basis by backtracking
'''    


import random, math
import numpy as np
import sympy

import algs.utils as utils


class Bookeeper:
    def __init__(self, k):
        self.k = k
        self.b = 0
        self.i = 0
        
        # takes k access time, not very much space
        self.cart_V = utils.PatternRows(k)
        
        # each index represents a basis row, the value represents the 
        # index in cartV of the next pattern, each time we need a new 
        # pattern row, current is returned then index is incremented  
        self.next_basis_pattern = [0]*k  
        
        # tracks the indices in B that are filled w. basis rows
        self.basis_inds = [-1]*k  
            
        # holds num lin dep rows added per basis row 
        self.num_added = [0]*k    
        
        # holds the indices in cartV of the lin dependent rows
        self.lin_depen_inds = []
        
        self.P = []
        
    
    def get_next_patt_ind(self):
        # gets next index thats not linearly dependent 
        while True:
            if self.next_basis_pattern[self.b]>=2**self.k:
                return -1
            
            ind = self.next_basis_pattern[self.b]
            self.next_basis_pattern[self.b]+=1
        
            # ind = 0 -> [0, ..., 0] so we don't want this
            if ind not in self.lin_depen_inds and ind != 0:
                return ind


    def get_next_patt_row(self):
        ind = self.get_next_patt_ind()
    
        if ind == -1:
            return [-1], ind
        else:
            Pb = self.cart_V.entryAt(ind) # get potential basis row  
            return Pb, ind
    
    
    def backtrack(self, B):        
        for j in range(self.num_added[self.b]):  
            self.lin_depen_inds.pop()
                            
        self.num_added[self.b]=0
        self.next_basis_pattern[self.b]=0
        self.basis_inds[self.b]=-1
        B[self.i] = [-1]*self.k   
        
        while len(self.P) >= self.b:
            if len(self.P) == 0:
                return False
            self.P.pop()
            
        self.b-=1
        self.i = self.basis_inds[self.b]
        self.basis_inds[self.b]=-1
        B[self.i] = [-1]*self.k   
        
        return True


def i_compatible(A, B, v, i):
    aii = A[i, i]
    vvt = np.dot(v, v.T)

    # first condition
    if not utils.equiv_ints(vvt, aii):
        return False
        
    # second condition
    j = 0
    compat = True
    for Bj in B:
        if j != i and not np.all(Bj==-1):  # i!=j and Bj not null row
            Aij = A[i, j]
            vtBjt = np.dot(v.T, Bj.T)
            if vtBjt != Aij:    
                compat = False
        j+=1
    return compat
    

def extend_basis(A, B, cart_V, k):
    row_i = 0
    
    for Bi in B:
        filled = False
        
        if np.all(Bi == -1):
            for j in range(0,2**k):
                v = np.array(cart_V.entryAt(j))
                if(i_compatible(A, B, v, row_i)):
                    B[row_i] = v
                    filled=True                 
                    break     # breaks out of this inner loop
                
            if not filled:
                return B, row_i
        row_i+=1
    
    return B, A.shape[0] 


def get_icomp_basis_row(A, B, book, w):
    '''
    Gets the next basis row that is w-limited, i-compatible, 
    and linearly indep.
    '''    
    icomp=False
    while not icomp:  # get basis row until compatible
        Pb, Pb_ind = book.get_next_patt_row()
                
        if Pb[0] == -1:
            status = book.backtrack(B)
            
            if status == False: # backtracking failed
                return [-1]
        else:
            book.P.append(Pb)
        
            # checks for w-limited
            compat = np.array(book.P).dot(np.array(book.P).T).max()
            
            # check for lin indep w. previous rows
            _, inds = sympy.Matrix(book.P).T.rref() 
            psize = len(book.P)
            
            book.P.pop() # will actually add to P when inserting row in B
        
            icmp = i_compatible(A, B, np.array(Pb), book.i)
            
            # icompatible + lin. indep + w-lim -- add Pb to basis
            if icmp and len(inds)==psize and compat <= w:
                icomp=True
                book.basis_inds[book.b] = book.i  # inserting basis row
                book.P.append(Pb)
                
            if not len(inds)==psize:  # not lin indep
                book.lin_depen_inds.append(Pb_ind)
                book.num_added[book.b]+=1
                
    return Pb  # returning icomp basis row 


def BSD_DW(A, k, w):
    n = A.shape[0]
    book = Bookeeper(k)
    B_tilde = np.full((n, k), -1)   # nxk 'null' rows
    backtracked=False
    
    # book.b: our current basis row [0, 1, ..., k-1]
    # book.i: the index of B we are trying to insert basis row
    while book.b < k:         
        if backtracked:
            book.b-=1
        
        Pb = get_icomp_basis_row(A, B_tilde, book, w)
        
        if Pb[0]==-1: # failed
            break
        
        B_tilde[book.i] = Pb
        
        B_tilde_copy = np.copy(B_tilde) 
        
        B, i = extend_basis(A, B_tilde_copy, book.cart_V, k)
        book.i = i
        
        if book.i == n:
            return B 
                
        # backtracking not finished yet but while loop is about to quit
        backtracked=False
        if book.b+1 == k and not all(val == (2**k-1) for val in book.next_basis_pattern):
            book.i = book.basis_inds[book.b]
            book.backtrack(B_tilde)
            backtracked=True
            
        book.b+=1
        
    return np.full((1,1), -1)    


            
   
