'''
generate_LV_data: 
    - total genes: 6750, known lvs: 690, unique lvs: 988
    
    Reads in LV multiplier data
    samples 20% of the input k (distinct cliques) randomly from all LVS (~7000 or so)
    samples 80% of k from known LVs to be associated with genes
    subsets all genes associated w. lvs ~690
        (each lv has a column with all genes, but each gene has different correlation)
    thresholds these to only take genes with correlation larger 
    computes each clique weight by linearly scaling weights and taking average of 
        all gene correlation values in each lv
    
    - dont input max_weight since this is determined by the data
    - each time this function is called inputs a fileid - graphs with the same 
        fileid have the same subset of lvs sizes are varied due to different 
        thresholds and scale factors 


generate_TF_data:
    - creates gene-gene networks, each clique represents a random transcription factor
    - unique genes:2650,  unique tfs: 345

    randomly samples k numbers in the range of 0-number of unique tfs-these 
    are our random tfs subsets the genes that are associated w. each tf
    generates random weights
    generates graphs
    
    - each time this function is called inputs a fileid - graphs with 
        the same fileid have the same subset of tfs
        multiple graphs are generate each with different (random) clique assignment
'''

import networkx as nx
import pandas as pd
import numpy as np
import argparse, random, os, pickle, time, math, sys

from os.path import dirname,realpath
sys.path.insert(0,dirname(realpath(__file__))[:-10])

import algs.utils_misc as utils_misc


def create_graph_name(f_id, typ, seed, k, ktotal, n, scalefac, threshold=None, witer=None):
    fname = str(f_id)
    fname+='_'+typ
    fname+='_seed'+str(seed)
    fname+='_scalefac'+str(scalefac)
    fname+='_kdistinct'+str(k)
    
    fname+='_ktotal'+str(ktotal)
    fname += '_n'+str(n)
    
    if typ=='TF':
        fname+='_witer'+str(witer)
    elif typ=='LV':
        fname+='_threshold'+str(threshold).split('.')[1]
    
    fname+='.txt'
    
    return fname


def save_graph(dir_name, fname, G, clique_weights, clique_vertices, n, kdistinct, ktotal, max_eweight):
    '''
    
    '''
    print(fname, end='')
    
    # save graph 
    nx.write_edgelist(G, dir_name+fname, data=['weight'])
    
    # save pickle info
    ground_truth = {'clique_weights' : clique_weights,
                    'clique_vertices' : clique_vertices, 
                    'n' : n,
                    'm' : G.number_of_edges(),
                    'kdistinct' : kdistinct,
                    'ktotal' : ktotal,
                    'max_eweight' : max_eweight}
        
    with open(dir_name+fname[0:-4]+'.pkl', 'wb') as f:
        pickle.dump(ground_truth, f)


def gen_psuedo_powerlaw_weights(k, max_weight):
    '''
    Generates weights that are similar to some power law distribution
    Does this by creating three windows, one large, one small, one smaller
    '''
    # set the windows - clique weights are within these ranges
    win1_upper = max_weight
    win1_lower = max_weight*0.9
    
    win2_upper = max_weight*0.28
    win2_lower = max_weight*0.2
    
    win3_upper = max_weight*0.14
    win3_lower = 1
    
    if win2_lower < 1:
        win2_lower = 1
    if win1_lower < 1:
        win1_lower = 1
    if win3_upper < 1:
        win3_upper = 1
    if win2_upper < 1:
        win2_upper = 1
    
    # windows to randomly selects weight values from
    win_1 = range(int(win1_lower), int(win1_upper)+1)          
    win_2 = range(int(win2_lower), int(win2_upper)+1)
    win_3 = range(int(win3_lower), int(win3_upper)+1)           
    
    # to get power law, there should be at least one with true max weight
    weights = [max_weight]  
    
    while len(weights) < k:
        # toss uniform coin, whichever range it falls in, 
        # sample either large, med, small weight
        toss = np.random.uniform(0, 1)
        
        w=None
        if toss <= 0.10:  # few samples from large window
            w = random.choices(win_1, k=1)[0]
        elif 0.10 < toss <= 0.25: # a few samples from med window
            w = random.choices(win_2, k=1)[0]
        elif 0.25 < toss < 1.0: # lots of samples from small window
            w = random.choices(win_3, k=1)[0]
        else:
            print('Error: incorrect windows')
        weights.append(w)
        
    return weights


def get_tf_gene_subset(dat, k):
    '''
    Randomly samples k tfs, and returns corresponding gene sets
    '''
    gene_TF_dat = dat.iloc[:, 0:2]   # gene + tf columns
    
    GENES = gene_TF_dat.iloc[:, 0]
    TFs = gene_TF_dat.iloc[:, 1]
    
    n_unique_genes = GENES.nunique()
    unique_genes = GENES.unique()
    
    n_unique_tfs = TFs.nunique()
    unique_tfs = TFs.unique()
              
    #------- 1. sample k random tfs     
    tfs = []
    TF_genes = []
    rnd_inds = []
    
    # randomly sample one tf at a time with at least 2 genes
    while len(tfs) < k:        
        rnd_ind = random.choice(range(0, n_unique_tfs))
        
        if rnd_ind not in rnd_inds:  # dont want repeats
            rnd_inds.append(rnd_ind)
            tf = unique_tfs[rnd_ind]
                    
            genes = gene_TF_dat[gene_TF_dat["TF (OFFICIAL_TF_CODING_GENE_NAME)"]==tf]
            
            # convert to set to prevent duplicates
            genes = set(genes["Target gene (OFFICIAL_GENE_NAME)"].to_list())

            # ensures that each clique has >= 2 vertices
            if len(genes) > 1:
                print('tf=', tf, ' num genes=', len(genes))
                TF_genes.append(genes)
                tfs.append(tf)
    return tfs, TF_genes


def get_lv_gene_subset(z_tsv, sum_tsv, k):
    '''
    Randomly samples k lvs, 20% from the entire set of lvs, and 80% from lvs know 
    to be associated with pathways.
    '''
    n_lvs = z_tsv.shape[1]
        
    # sample k random lvs     
    #   sample 20% randomly
    lv_labels = random.sample(range(1, n_lvs), int(k*0.20))
    
    #   sample 80% from known lvs   
    known_lvs = sum_tsv.iloc[:, 1]
        
    know_lvs_list = known_lvs.unique()
    while len(lv_labels) < k: 
        rnd = random.choice(range(0, known_lvs.nunique()))  # sample one value
        
        if know_lvs_list[rnd] not in lv_labels:
            lv_labels.append(know_lvs_list[rnd])
    
    LVs = z_tsv.iloc[:, lv_labels]  # lv columns (~)
    genes = z_tsv.iloc[:, 0]        # gene label col
    
    return LVs, genes, lv_labels



def create_graphs(geneset, clique_weights):
    '''
    Args:
        
    Output: 
        
    '''
    G = nx.Graph()
        
    # create integer vertex labels for each gene    
    vertex_labels = {}
    curr_label = 0
    for genes in geneset:        
        for gene in genes:
            if gene not in vertex_labels:
                vertex_labels.update({gene : curr_label})
                curr_label+=1
    
    i=0
    clique_vertices = []
    
    # for each of the chosen lv - create its corresponding clique
    for genes in geneset:
        j=0
        curr_clique_verts = set()
        genes = list(genes) # convert back to list, to allow indexing
        
        for gene1 in genes:
            for gene2 in genes[j:]:
                if gene1 != gene2:
                    u = vertex_labels.get(gene1)
                    v = vertex_labels.get(gene2)
                                        
                    curr_clique_verts.add(u)
                    curr_clique_verts.add(v)
                    
                    # add edge to the current clique
                    edge = {u, v}
                    
                    if G.has_edge(u,v):
                        # increase edge weight by clique weight if exists
                        G[u][v]['weight'] = G[u][v]['weight']+clique_weights[i]  
                    else:
                        G.add_edge(u, v, weight=clique_weights[i])
            j+=1
        i+=1
        clique_vertices.append(sorted(curr_clique_verts))
        
    return G, clique_vertices, vertex_labels


def generate_LV_data(f_id, kdistinct, dir_name, seed, threshold, scale_fac, LVs, genes, lv_labels):
    '''
    
    '''
    # subset genes for each LV
    LV_genes = []
    LV_vals = []
    for (name, dat) in LVs.iteritems(): 
        i = 0
        lv_gene = set()  # set to prevent duplicates    
        lv_val = []
        for prob in dat:
            if abs(float(prob)) >= threshold:
                lv_gene.add(genes[i])
                lv_val.append(prob)
            i+=1
        LV_genes.append(lv_gene)
        LV_vals.append(lv_val)
    
    
    # generate clique weights
    unscaled_clique_ws = []
    scaled_clique_ws = []
    
    for lv_val in LV_vals:
        # since indiv vals are small, linearly scale to get larger integers
        unscaled_clique_ws.append(sum(lv_val)/len(lv_val))
        w = math.ceil((scale_fac*sum(lv_val))/len(lv_val))  
        
        if w<1: # dont want weights less than one
            w=1
        scaled_clique_ws.append(w)
    
    # given clique weights, create graph
    ktotal = sum(scaled_clique_ws)
    
    start = time.time()
    G, clique_vertices, vertex_labels = create_graphs(LV_genes, scaled_clique_ws)   
    end = time.time()
    
    n = G.number_of_nodes()
        
    fname = create_graph_name(f_id, 'LV', seed, kdistinct, ktotal, 
                              n, scale_fac, threshold=threshold)
    
    max_eweight = utils_misc.get_max_edgeweight(G)
    
    save_graph(dir_name, fname, G, scaled_clique_ws, 
               clique_vertices, n, kdistinct, 
               ktotal, max_eweight)
        
    print(' time:', end - start)
    

def generate_TF_data(f_id, kdistinct, mw, dir_name, seed, wi, clique_weights, TFs, TF_genes):
    '''
    
    '''     
    ktotal = sum(clique_weights)

    start = time.time()
    G, clique_vertices, vertex_labels = create_graphs(TF_genes, clique_weights)    
    end = time.time()
    
    n = G.number_of_nodes()
    
    fname = create_graph_name(f_id, 'TF', seed, kdistinct, ktotal,
                              n, mw, witer=wi)
    
    max_eweight = utils_misc.get_max_edgeweight(G)
    
    save_graph(dir_name, fname, G, clique_weights, 
               clique_vertices, n, kdistinct, 
               ktotal, max_eweight)
    
    print(' time:', end - start)
    



