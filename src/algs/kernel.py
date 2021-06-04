'''
    Kernel reduction rules 
'''

import numpy as np
import algs.utils as utils

def num_blocks(A):
    '''
    Args: A : 2-D pandas adjacency list
    '''    
    # reduction rule 1
    blocks=[]
    inds = list(A.index.values)
    added = {}
    for ind in inds:
        added[ind]=False
    count=0
        
    for i in inds:
        block=set()
        if added[i]==False:  # current vert i has not been added to a block yet
            block.add(i)
            added[i]=True
            
            # try to find all of i's equiv rows
            for j in inds[count:]:
                if i != j and added[j]==False: # if j has not been added yet to a block    
                    if utils.equiv_rows(A.loc[i, :], A.loc[j, :]) and A.loc[i, j]>=1:
                        block.add(j)
                        added[j]=True
            blocks.append(block)
        count+=1
         
    return blocks


def remove_arbitrary(A, blocks, k):
    '''
    Args: A : 2-D pandas adjacency list
    '''    
    # reduction rule 2
    toremove = []
    b_index=0
        
    for block in blocks:
        if len(block) > 2**k:   
            v = block.pop()
            u = block.pop()
            A.loc[v,v] = A.loc[u,v]
            
            toremove.append(u)            

            for i in block:
                toremove.append(i)
  
            # replace block w. block containing representative vertex
            newblock = set()
            newblock.add(v)
            blocks[b_index]=newblock
        
        b_index+=1
    
    A_red = A.drop(toremove, axis=1)
    A_red = A_red.drop(toremove, axis=0)
        
    return A_red, toremove


def reduction_rules(A, k):
    '''
    
    '''
    blocks = num_blocks(A)

    if len(blocks) > 2**k: 
        return A, [-1]  # NO instance
    
    A_red, removed_vertices = remove_arbitrary(A, blocks, k)
    
    return A_red, removed_vertices


