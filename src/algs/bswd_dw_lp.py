'''
    Binary Symmetric Weighted Decomposition w. Diagonal Wildcards
    Finds the weights by solving a LP (only on basis rows)
    Computes basis by backtracking
    
    (lp algorithm)
'''   

import random, math
import numpy as np

import gurobipy as gp
from gurobipy import GRB

import algs.utils as utils


class Bookeeper:
    def __init__(self, k, max_val):
        self.max_val = max_val
        self.k = k
        self.b = 0
        self.i = 0
        
        # for extending basis -- gives each pattern row 
        self.cart_V = utils.LazyCartesianProduct(k)
        
        # each index represents a basis row, the value represents the 
        # index in cartV of the next pattern, each time we need a new 
        # pattern row, current is returned then index is incremented 
        self.next_basis_pattern = [0]*max_val  
        
        # tracks the indices in B that are filled w. basis rows
        self.basis_inds = [-1]*max_val  
            
        # holds num constraints added to insert the basis row at each index
        self.cnstrs_added = [0]*max_val    
        
        # at index i, holds the weight matrix found before inserting basis row i
        self.Ws = [0]*max_val
                
    
    def get_next_patt_ind(self):
        # returns the index of the desired pattern row, increments
        while True:
            if self.next_basis_pattern[self.b]>=2**self.k:
                return -1
            
            ind = self.next_basis_pattern[self.b]
            self.next_basis_pattern[self.b]+=1
        
            # ind = 0 -> [0, ..., 0] so we don't want this
            if ind != 0:  
                return ind


    def get_next_patt_row(self):
        ind = self.get_next_patt_ind()
    
        if ind == -1:
            return [-1]
        else:
            Pb = self.cart_V.entryAt(ind) # get potential basis row 
            return Pb
    
    
    def backtrack(self, B, model):
        '''
        called when self.b == 2**self.k, meaning
        there are no more unique pattern rows to try
        given the previous pattern rows, so backtrack 
        '''    
        if self.b-1==-1:
            return False
        
        self.next_basis_pattern[self.b]=0
        self.Ws[self.b]=0  # reset weights before inserting b
        
        self.b-=1  # go back to rethink last pattern row
        
        self.i = self.basis_inds[self.b]
        
        # remove constraints that allowed inserting of new self.b
        utils.remove_constraints(model, self.cnstrs_added[self.b])
        self.cnstrs_added[self.b]=0
        self.basis_inds[self.b] =- 1
        B[self.i] = [-1]*self.k   
        
        return True
        

def optimize_model(model, book, W): 
    # Returns True is model is feasible+bounded and updates W
    # False otherwise
    
    # Solve model
    model.optimize()
    status = model.status
    
    # Get the variables
    # inserts weights in W diagonals
    if status == GRB.Status.OPTIMAL: 
        ind = 0
        for v in model.getVars():
            W[ind, ind]=v.x
            ind+=1
    elif status == GRB.Status.INF_OR_UNBD or status==GRB.Status.INFEASIBLE:
        return False
    
    return True


def add_model_constraints(A, B, Pi, book, dv_w, model):
    '''
    book.i: current index of B we are attempting to insert a basis row (Pi).
    book.basis_inds: the row indices of the previously added basis rows
    
    returns: number of constraints added at this one step.
    '''
    #---------- Add the new constraints to model and solve
    # diagonal constraints: 
    # Aii = Pi1*W1*Pi1 + Pi2*W2*Pi2 + ... + Pik*Wk*Pik
    count=0
    if A[book.i, book.i] != math.inf:
        aii = 0
        for q in range(book.k):
            aii += Pi[q] * dv_w[q] * Pi[q]
        
        cst_name = "{}constraint_{}".format(book.i,count)
        model.addConstr(aii == A[book.i,book.i], name=cst_name)
        count+=1
            
    # non-diagonal constraints: 
    # Aij = Pi1*W1*Bj1.T + Pi2*W2*Bj2.T + ... + Pik*Wk*Bjk.T
    for j in book.basis_inds:
        if j != -1 and A[book.i,j] != math.inf:  
            aij = 0
            for q in range(book.k):
                aij += Pi[q] * dv_w[q] * B[j, q].T
                        
            cst_name = "{}constraint_{}".format(book.i,count)
            model.addConstr(aij == A[book.i,j], name=cst_name)
            count+=1

    model.update()
    return count

     
def extend_basis(A, B, W, cart_V, k):
    row_i = 0
    
    for Bi in B:
        filled = False
        
        if np.all(Bi == -1):
            for j in range(0,2**k):
                v = np.array(cart_V.entryAt(j))
                if(utils.weighted_i_compatible(A, B, W, v, row_i)):
                    B[row_i] = v
                    filled=True
                    break
                
            if not filled:
                return B, row_i
        row_i+=1
    
    return B, A.shape[0] 


def get_icomp_basis_row(A, B, W, book, model, dv_w):
    # Gets the next basis row that is i-compatible.
    icomp=False
    while not icomp:  # get basis row until icomp
        Pb = book.get_next_patt_row()
        
        if Pb[0] == -1:
            status = book.backtrack(B, model)
            if status is False:
                return [-1]
            
            W = book.Ws[book.b] # resets W to the W before inserting current book.b 
        else:
            count=0         
            count = add_model_constraints(A, B, Pb, book, dv_w, model)
            feasible = optimize_model(model, book, W)  # updates W if feasible
                
            if not feasible:
                utils.remove_constraints(model, count)
                continue
            
            icomp=True
            book.cnstrs_added[book.b]+=count
            book.basis_inds[book.b] = book.i
            
    return Pb  # returning icomp basis row 
    
    
def BSWD_DW(A, k):
    n=A.shape[0]
    
    if 2*k>n:
        max_val = n
    else:
        max_val = 2*k
    
    book = Bookeeper(k, max_val)
    
    #---------- Create a new model
    model = gp.Model("weights")
    model.Params.LogToConsole = 0  # dont print to console
    
    #---------- Add Decision Variables [w0, w1, ..., wk] 1-D array
    dv_w = model.addMVar(shape=(k), vtype=GRB.INTEGER, name="w")
    model.addConstr(dv_w >= 0)  #non-negativity constraint

    #---------- Set objective function
    model.setObjective(dv_w.sum(), GRB.MINIMIZE)
    model.update()
    
    W = utils.construct_weight_matrix([1]*k, [0]*k, k) # init weight matrix

    B_tilde = np.full((n, k), -1)   # nxk 'null' rows
    backtracked=False    
    
    # book.b: our current basis row [0, 1, ..., maxval-1]
    # book.i: the index of B we are trying to insert basis row
    while book.b < max_val: 
        if backtracked:
            book.b-=1
            
        book.Ws[book.b] = np.copy(W)  # W before inserting book.b basis row
        
        Pb = get_icomp_basis_row(A, B_tilde, W, book, model, dv_w)
            
        if Pb[0]==-1: # failed
            return np.full((1,1), -1), np.full((1,1), -1)  # bsd failed
        
        B_tilde[book.i] = Pb  
        
        B_tilde_copy = np.copy(B_tilde) 
        
        B, i = extend_basis(A, B_tilde_copy, W, book.cart_V, k)
        book.i = i
        
        if book.i == n:
            return B, W  # bsd found soln
        
        # backtracking not finished yet but while loop is about to quit
        backtracked=False
        if book.b+1 == max_val and not all(val == (2**k-1) for val in book.next_basis_pattern):
            book.i = book.basis_inds[book.b]
            book.backtrack(B_tilde, model)
            backtracked=True
            
        book.b+=1
        
    return np.full((1,1), -1), np.full((1,1), -1)  # bsd failed


            
   
