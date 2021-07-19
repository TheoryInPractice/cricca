
import matplotlib.pyplot as plt
from collections import deque
import networkx as nx
import numpy as np
import pickle, copy, time, argparse, os

import algs.utils as utils
import algs.utils_misc as utils_misc


def update_output_vals(G, output, dat, k_distinct, time_proc):
    # updates the pkl output values after removing vertices
    clique_verts = output['clique_vertices']
    rem_inds = []
    
    # get inds of disconnected cliques 
    for rem_clique in dat['disconnected_clqs']:
        i=0
        for true_clique in clique_verts:            
            if sorted(list(rem_clique))==true_clique:
                rem_inds.append(i)
            i+=1
        
    # get single clique indices that were removed
    for rem_clique in dat['single_clqs']:
        i=0
        for true_clique in clique_verts:
            if sorted(list(rem_clique))==true_clique:
                rem_inds.append(i)
            i+=1
    
    # update output
    upd_output = copy.deepcopy(output)  
    
    rem_inds = sorted(rem_inds)    
    rem_inds = rem_inds[::-1] 
    for i in rem_inds:
        del upd_output['clique_weights'][i]
        del upd_output['clique_vertices'][i]

    k_distinct_after=len(upd_output['clique_vertices'])    
    
    winf = None
    if G.order()>0:
        winf =  np.linalg.norm(nx.to_numpy_matrix(G), np.inf)
    
    upd_output['n'] = G.order()
    upd_output['m'] = G.number_of_edges()
    upd_output['kdistinct'] = k_distinct_after
    upd_output['ktotal'] = sum(upd_output['clique_weights'])
    upd_output['max_eweight'] = utils_misc.get_max_edgeweight(G)
    upd_output['w_inf'] = winf
    upd_output['preprocess_time'] = time_proc
    
    assert len(upd_output['clique_weights']) == upd_output['kdistinct']
    assert len(upd_output['clique_vertices']) == upd_output['kdistinct']
    
    return upd_output
    

def get_removal_vert_set(G, found_rem_cliques):
    removal_verts = set()
    disc_edges = set() 
    i=0
    
    for cliq in found_rem_cliques:
        for v in cliq:
            remove=True
            for u in G.neighbors(v):
                if u not in cliq:
                    contain=False
                    j=0
                    for cliq2 in found_rem_cliques:
                        if i!=j:
                            if u in cliq2 and v in cliq2:
                                contain=True
                        j+=1
                    if not contain:
                        remove=False 
            if remove:
                removal_verts.add(v)
        i+=1
        
    # now actually delete the proper vertices from the graph
    G.remove_nodes_from(removal_verts)
    
    # go through an make sure to remove all edges in each cliq
    for cliq in found_rem_cliques:
        for v in cliq:
            for u in cliq:
                if u!=v and G.has_edge(u,v):
                    G.remove_edge(u, v)
    
    return removal_verts


##################################################################


def find_discon_cliques(G):
    '''
    finds and removes all disconnected cliques
    '''
    n = G.order()
    visited=set()
    found_rem_cliques=[]
    removal=[]
    
    for v,outer_d in G.nodes(data=True):
        if v not in visited:        
            visited.add(v)
            
            # finds if the current component is a disconnected clique
            clique_verts, is_clique = is_clique_component(G, v, visited)

            if is_clique:
                found_rem_cliques.append(clique_verts)
                
                for v in clique_verts:
                    removal.append(v)
                                
    G.remove_nodes_from(removal)
        
    return found_rem_cliques
                

def is_clique_component(G, v, visited):
    # gets the current component and checks if it's detached clique
    comp_verts=set()
    comp_verts.add(v)
    q = deque()
    q.append(v)
    
    is_clique=True
    comp_weight=None
    deg = G.degree[v]
 
    while q:
        v = q.popleft()
        
        for u in G.neighbors(v):
            edge_weight = G.get_edge_data(v, u)
            
            # check if vertex degrees are equal
            if deg != G.degree[u]:
                is_clique=False
            
            # check if edge weights are the same
            if comp_weight is None:
                comp_weight=edge_weight
            elif comp_weight != edge_weight:
                is_clique=False
            
            if u not in visited:
                visited.add(u)
                comp_verts.add(u)
                q.append(u)
            
    if len(comp_verts)-1 != deg:
        is_clique=False
    
    return comp_verts, is_clique


def find_single_overlap(G):
    '''    
    start_v_visited: holds vertices already considered as the starting vert 
        
    '''    
    n = G.order()
    start_v_visited=set()
    clique_verts_visited=set()
    found_rem_cliques=[]
    
    # sort vertices by smallest degree to largest
    vertex_order = sorted(G.degree, key=lambda x: x[1], reverse=True)
    
    for vertex in vertex_order:
        v = vertex[0]
        if v not in start_v_visited:        
            start_v_visited.add(v)
        
            # start search with this vertex: all adjacent edge weights are unif 
            is_clique, curr_clique = is_single_overlap(G, v, start_v_visited, 
                                                       clique_verts_visited)    
            if is_clique:
                found_rem_cliques.append(curr_clique)
    
    removal_verts = get_removal_vert_set(G, found_rem_cliques)
    
    return removal_verts, found_rem_cliques
         
         
def is_single_overlap(G, v_init, start_v_visited, clique_verts_visited):    
    ''' 
    computes if clique via bfs for overlapping cliques on one vertex
    
    - input vertex v adjacent edges all have same weights
    - from v, insert all nbrs into visited & queue
    - if any nbr has diff degree, add it to the set diff_deg_verts
    - loop over the queue
    - for each vertex, check all nrbing edge weights (for nbrs in the same clique) 
        if not the same, then the clique contains an overlapping edge, meaning it shouldnt
        get removed
    - 
    '''
    q = deque()
    q.append(v_init)
    
    comp_weight=None
    deg = G.degree[v_init]
    is_clique=True
    
    curr_clique=set()
    curr_clique.add(v_init)
    diff_deg_verts=set()
    
    # add all nbrs of v_init to queue and visited + check edge weights
    for u in G.neighbors(v_init):               # E
        if u not in clique_verts_visited:
            edge_weight = G.get_edge_data(v_init, u)
            
            # check if edge weights are the same
            if comp_weight is None:
                comp_weight=edge_weight
            elif comp_weight != edge_weight:
                is_clique=False
                break
            
            # check degree of each vertex
            if G.degree[u]!=deg:
                diff_deg_verts.add(u)
            
            curr_clique.add(u)
            q.append(u)
    
    if is_clique:
        # check all nbrs of start vertex
        while q:              # E
            v = q.popleft()
            
            # first check if proper clique
            for u in curr_clique:  #  E
                if u != v:
                    if G.has_edge(u, v):
                        if G.get_edge_data(v, u) != comp_weight:
                            is_clique=False
                            break
                    else:
                        is_clique=False
                        break
                        
    if is_clique:
        clique_verts_visited.update(curr_clique)
        start_v_visited.update(curr_clique)
    
    for v in diff_deg_verts:
        if v in clique_verts_visited:
            clique_verts_visited.remove(v)
    
    return is_clique, curr_clique


def preprocess(G, k_distinct):
    '''
    finds components of graph
     
    first: test if component is a detached clique via modified bfs,
            if yes, remove and decrease k value
    
    second: run modified bfs to check for cliques overlapping on only one vertex
    
    Convert a weighted adjacency matrix into an ordinary adjacency.
        
    '''
    print('\npreprocessing, k=', k_distinct)
    dat = {}
    
    disconnected_clqs = find_discon_cliques(G)    
    removal_verts, single_clqs = find_single_overlap(G)
    
    dat['num_discon_rem'] = len(disconnected_clqs)
    dat['disconnected_clqs'] = disconnected_clqs
    dat['num_single_rem'] = len(single_clqs)
    dat['single_removal_verts'] = removal_verts
    dat['single_clqs'] = single_clqs
    dat['kdistinct_postproc'] = k_distinct-len(disconnected_clqs)-len(single_clqs)
        
    return dat 
    
    
    
    
