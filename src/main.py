
'''
    Script to run wecp, ipart, or lp on single graph files. 
    Accepted graph file formats: .csv and .txt edge lists
'''

import argparse, time
import networkx as nx
import numpy as np

import algs.utils as utils
import algs.utils_misc as utils_misc

import algs.preprocess as preprocess
import algs.kernel as kernel

# different alg versions
import algs.bsd_dw as bsd_dw
import algs.bswd_dw_lp as bswd_dw_lp
import algs.bswd_dw_ip as bswd_dw_ip


def read_data(file_name):
    '''
    if file format is .csv, reads in each line
    if file format is .txt, reads graph directly
    '''

    G = nx.Graph()
    if file_name[-4:] == '.csv':
        with open(file_name) as fn:
            lines = fn.readlines()
            
            for line in lines:
                ln = line.split(',')
                
                if len(ln) == 3:
                    try:
                        u = int(ln[0])
                        v = int(ln[1])
                        w = int(ln[2])
                        
                        G.add_edge(u, v, weight=w)
                        
                    except:
                        pass
                else:
                    print("Error reading data file")
                
    elif file_name[-4:] == '.txt':
        G = nx.read_weighted_edgelist(file_name, nodetype=int)
    else:
        print('Error: incorrect file format')
        
    return G
        

def run_preprocess(G, k_distinct):
    '''
    runs preprocessor to remove disjoint cliques/single overlapping cliques
    '''
    n_pre = G.order()
    start = time.time()
    dat = preprocess.preprocess(G, k_distinct)
    end = time.time()   
    time_proc = end - start        
    
    n_post = G.order()
    k_post_pp = dat['kdistinct_postproc']
    
    print('preprocess time: {}, n_pre={}, n_post={}'.format(time_proc, n_pre, n_post))
    
    return G, k_post_pp
    

def run_kernel(A, G, k):
    '''
    kernelizes graph
    '''
    n_pre = G.order()
    
    # reduction rules    
    start = time.time()
    A_kernel, removal_vertices = kernel.reduction_rules(A, k)
    end = time.time()   
    time_kernel = end - start
    
    G.remove_nodes_from(removal_vertices)
    n_post = G.order()
    
    print('kernel time: {}, n_pre={}, n_post={}'.format(time_kernel, n_pre, n_post))
    return A_kernel


def run_bsd(A, G, k_input, alg_version):
    '''
    runs either wecp, ipart, or lp decomposition algorithms
    '''
    W=None
    start = time.time()
    if alg_version == 'wecp':  
        # original BSD_DW (Feldman et al. 2020)
        if G.order()>0:
            winf =  np.linalg.norm(nx.to_numpy_matrix(G), np.inf)
        
        B = bsd_dw.BSD_DW(A, k_input, winf) 
    elif alg_version == 'lp':  
        # linear programming version
        B, W = bswd_dw_lp.BSWD_DW(A, k_input)
    elif alg_version == 'ipart': 
        # integer partitioning version
        B, W = bswd_dw_ip.BSWD_DW(A, k_input)
    else:
        print('Error: incorrect algorithm version. Options: "wecp", "lp", "ipart"')
            
    end = time.time()
    time_bsd = end-start
    print('Decomposition time: ', time_bsd)
    
    return B, W


def run(alg_version, G, k, to_preprocess, to_kernelize):
    
    if to_preprocess:
        G, k = run_preprocess(G, k)
    
    if to_kernelize:
        A_wc = utils_misc.get_wildcard_adjacency(G)
        A = run_kernel(A_wc, G, k)
        
    A = A.to_numpy()
    B, W = run_bsd(A, G, k, alg_version)
    
    print('\nClique Membership Matrix B:\n', B)
    print('Weight Matrix W:\n')
    
    if np.all(np.equal(B, -1)):  
        print("Decomposition Failed")
    else:
        print("Decomposition Passed")
        
        if alg_version=='wecp':
            A_prime = np.dot(B, B.T)
        else:
            A_prime = np.dot(np.dot(B, W), B.T)
        A_masked = np.ma.masked_array(A, A==np.inf)  
        reconstructs = np.all(A_masked.astype(int)==A_prime)
        print("Reconstructs solution? ", reconstructs)
    

def main():
    parser = argparse.ArgumentParser()
   
    #----------------------------------- required args
    parser.add_argument('-a', '--algorithm', type=str,
        help="options: wecp, ipart, lp", required=True)
    
    parser.add_argument('-g', '--graph_filename', type=str,
        help="enter .txt edge list data file", required=True)
    
    parser.add_argument('-k', '--parameter', type=int,
        help="enter integer parameter value", required=True)
    
    #----------------------------------- nonrequired args
    parser.add_argument('-p', '--preprocess', type=bool, default=False,
        help="preprocess? True or False", required=False)
    parser.add_argument('-n', '--kernelize', type=bool, default=True,
        help="kernelize? True or False", required=False)
    
    args = vars(parser.parse_args())
    
    algorithm = args['algorithm']
    graph_filename = args['graph_filename']
    k = args['parameter']
    
    to_preprocess = args['preprocess']
    to_kernelize = args['kernelize']
    
    print("Filename : ", graph_filename)
    G = read_data(graph_filename)
    
    run(algorithm, G, k, to_preprocess, to_kernelize)
    

if __name__=="__main__":
    main()
