
import numpy as np
import pandas as pd
import networkx as nx
from multiprocessing import Process, Manager # for mem usage 

import math, os, time, random, argparse, signal, pickle, re, copy
import pprint
from resource import *

import algs.utils as utils
import algs.utils_misc as utils_misc
import algs.kernel as kernel

import algs.bsd_dw as bsd_dw
import algs.bswd_dw_lp as bswd_dw_lp
import algs.bswd_dw_ip as bswd_dw_ip



def get_num_unrun_files(files, k_distinct_max, pp_dirname, typ):
    '''
    Returns the number of files in the dir that need running
    '''
    torun_count=0
    
    for fname in files:
        fn = fname.split('/')[-1]
        ppfname = pp_dirname+fn  # post_preprocessing filename
        pklfname = fname
        
        # get post_preprocessing pkl file info
        with open(ppfname, 'rb') as infile: 
            pp_output = pickle.load(infile)
            
        with open(pklfname, 'rb') as infile: 
            kernel_output = pickle.load(infile)
            
        witer=None 
        if 'witer' in fname:
            witer = int(utils_misc.get_fname_value(fname, 'witer'))
        
        k_distinct = pp_output['post_preprocessing']['kdistinct']
        if k_distinct>k_distinct_max:
            continue
        
        if witer is not None:
            if witer!=0:
                continue
            
        # only run mem tests on max weight scale
        if typ=='tf':
            wscale_val = int(utils_misc.get_fname_value(fname, 'scalefac'))
            if wscale_val!=16:
                continue
        elif typ=='lv':
            wscale_val = int(utils_misc.get_fname_value(fname, 'scalefac'))
            if wscale_val!=4:
                continue
            
        torun_count+=1
                
    return torun_count


def get_final_clique_sets(B, vertex_indices, k_input):
    '''
    
    '''
    cliques = [[] for i in range(k_input)]
    
    i=0
    for row in B:
        j=0
        for c in row:
            if c==1:
                cliques[j].append(vertex_indices[i])
            j+=1
        i+=1
    return cliques


def compare_found_cliques(computed_cliqs, groundtruth_cliqs):
    '''
    Compares ground truth cliques to the computed cliques via decomp
    '''
    counted = []
    found_count=0
    for cliq in computed_cliqs:
        for gtcliq in groundtruth_cliqs:
            if sorted(cliq)==gtcliq and sorted(cliq) not in counted:
                counted.append(sorted(cliq))
                found_count+=1
    
    return found_count


def print_output_status(passed_bsd, time_bsd, reconstructs, found_cliq_fraction):
    CGREEN = '\33[42m'
    OKBLUE='\033[94m'
    CRED = '\033[91m'
    
    END = '\033[0m'
    
    if passed_bsd=='PASSED':
        col = CGREEN
    elif passed_bsd=='TIMEOUT':
        col = OKBLUE
    elif passed_bsd=='FAILED':
        col = CRED
    
    print(col+'END DECOMP: passed_bsd? '+passed_bsd+' time: '+str(time_bsd)+' reconstructs? '+str(reconstructs)+' found cliq frac: '+str(found_cliq_fraction)+END)
        

def run_bsd(out, A, k_input, alg_version, kernel_output, timeout, vertex_indices, groundtruth_cliqs, winf=None):
    '''

    '''
    print('\nRunning decomp:', alg_version, ' kinput=', k_input)
    print('** os.pid=', os.getppid())
    mem_usage=None
    
    #---------------------------------------------- run bsd 
    tout = False
    B=np.full((1,1), -1)
    W=None
    
    # for the timeout
    signal.signal(signal.SIGALRM, utils_misc.handler) # WARNING only on unix?
    signal.alarm(timeout)
    start = time.time()
    try:
        if alg_version == 'bsd_dw':  
            # original BSD_DW
            B = bsd_dw.BSD_DW(A, k_input, winf) 
        elif alg_version == 'bswd_dw_lp':  
            # LP-weights
            B, W = bswd_dw_lp.BSWD_DW(A, k_input)
        elif alg_version == 'bswd_dw_ip': 
            # int. part. weights (basis only)
            B, W = bswd_dw_ip.BSWD_DW(A, k_input)
    except utils_misc.TimeOutException as ex:
        print(ex)
        tout = True
    end = time.time()
    time_bsd = end-start
    signal.alarm(0)
    #----------------------------------------------

    reconstructs=None
    passed_bsd=None
    computed_cliques=None
    found_cliq_fraction=None
    
    if np.all(np.equal(B, -1)):    
        reconstructs=False
        if not tout:
            passed_bsd='FAILED'   # BSD outputs a no answer
        elif tout:
            passed_bsd='TIMEOUT'  # BSD_DW timed out
    else:
        passed_bsd='PASSED'       # BSD_DW found solution
            
        if alg_version=='bsd_dw':
            A_prime = np.dot(B, B.T)
        else:
            A_prime = np.dot(np.dot(B, W), B.T)
        A_masked = np.ma.masked_array(A, A==np.inf)  
        reconstructs = np.all(A_masked.astype(int)==A_prime)
        computed_cliques = get_final_clique_sets(B, vertex_indices, k_input)

        found_count = compare_found_cliques(computed_cliques, groundtruth_cliqs)
        found_cliq_fraction = found_count/len(groundtruth_cliqs)
        
    print_output_status(passed_bsd, time_bsd, reconstructs, found_cliq_fraction)
    
    rusage_denom = 1024
    mem_usage = getrusage(RUSAGE_SELF).ru_maxrss #/ rusage_denom
    print(mem_usage)
    
    o = {'B' : B,
           'W' : W,
           'kinput' : k_input,
           'found_cliq_fraction' : found_cliq_fraction,
           'alg_version' : alg_version,
           'time_bsd' : time_bsd,
           'passed_bsd' : passed_bsd,
           'reconstructs' : reconstructs,
           'vertex_indices' : vertex_indices,  
           'computed_cliques' : computed_cliques,
           'mem_usage' : mem_usage}
            
    out[alg_version]=o


def run_distinct_memexp(kernel_output):
    '''
    
    '''
    run_ipart=False
    run_lp=False
        
    if kernel_output['true_distinct_kernel'] is not None:
        if kernel_output['true_distinct_kernel']['passed_kernel']==True:
            
            run_lp=True 
            run_ipart=True
    
    return run_ipart, run_lp


def run(pp_dirname, kern_dirname, out_dirname, first_seed, last_seed, k_distinct_max, timeout, typ):
    '''
    
    '''
    print('kernel in dirname:             ', kern_dirname)
    print('post_preprocessing in dirname: ', pp_dirname)
    print('out_dirname:                   ', out_dirname)
    
    if not os.path.exists(out_dirname):
        os.makedirs(out_dirname)
        
    kernel_files = utils_misc.get_files(kern_dirname, '.pkl')
    files = utils_misc.filter_files_seed(kernel_files, first_seed, last_seed)

    # already processed graphs
    prerun_files = utils_misc.get_files(out_dirname, '.pkl') 
    
    torun_count = get_num_unrun_files(files, k_distinct_max, pp_dirname, typ)
    count=0
    ran_count=0
    
    for fname in files: 
        count+=1
        
        # if graph has already been ran 
        partial_decompdat=None
        decomp_fn = out_dirname+fname.split('/')[-1]
        if decomp_fn in prerun_files:
            with open(decomp_fn, 'rb') as infile: 
                partial_decompdat = pickle.load(infile)
                
        # get kernel pkl file info
        pklfname = fname
        with open(pklfname, 'rb') as infile: 
            kernel_output = pickle.load(infile)
            
        # get post_preprocessing pkl file info
        ppfname = pp_dirname+fname.split('/')[-1]  # post_preprocessing filename
        with open(ppfname, 'rb') as infile: 
            pp_output = pickle.load(infile)
                
        # collect + write out data  
        decomp_data=None
        if partial_decompdat is None:
            decomp_data = {
                'timeout' : timeout,
                'bsd_dw'     : {'true_total'    : None},
                'bswd_dw_lp' : {'true_total'    : None,
                                'true_distinct' : None,
                                'guesses'       :  {-0.6 : None, 
                                                    -0.4 : None,
                                                    -0.2 : None, 
                                                    0.2 : None, 
                                                    0.4 : None,
                                                    0.6 : None }},
                'bswd_dw_ip' : {'true_total'    : None,
                                'true_distinct' : None,
                                'guesses'       :  {-0.6 : None, 
                                                    -0.4 : None,
                                                    -0.2 : None, 
                                                    0.2 : None, 
                                                    0.4 : None,
                                                    0.6 : None }}}
        else:
            decomp_data = partial_decompdat['decomp_data']
                
        witer=None 
        if 'witer' in fname:
            witer = int(utils_misc.get_fname_value(fname, 'witer'))
        
        k_total = pp_output['post_preprocessing']['ktotal']
        k_distinct = pp_output['post_preprocessing']['kdistinct']
        
        if k_distinct>k_distinct_max:
            continue
        
        if witer is not None:
            if witer!=0:
                continue
            
        # NOTE only run mem tests on max weight scale
        if typ=='tf':
            wscale_val = int(utils_misc.get_fname_value(fname, 'scalefac'))
            if wscale_val!=16:
                continue
        elif typ=='lv':
            wscale_val = int(utils_misc.get_fname_value(fname, 'scalefac'))
            if wscale_val!=4:
                continue
        
        print('\n____________________________________________ {}/{} : {}'.format(count, 
                                                                       len(files),
                                                                       fname))
        print('{}/{} memory exp'.format(ran_count, torun_count))
        print('ktotal ', k_total, ' kdistinct', k_distinct)
        
        #------------------------------------------------- run true distinct
        run_ipart, run_lp = run_distinct_memexp(kernel_output)
        
        if run_lp or run_ipart:
            '''
            run ip and lp versions w. true_distinct
            '''
            print('\n---------------------run true distinct: mem tests')
            print('main os pid=', os.getppid())
            ran_count+=1
            
            A = kernel_output['true_distinct_kernel']['A_kernel']
            vertex_indices = list(A.index.values)
            groundt_cliqs = kernel_output['true_distinct_kernel']['clique_vertices']
            k_distinct = kernel_output['true_distinct_kernel']['kinput']
            A = A.to_numpy()
        else:
            print('**** not running')
            ran_count+=1
        
        manager = Manager()
        out = manager.dict()
        
        if run_lp:
            decomp_data['bswd_dw_lp']['true_distinct']=None
            p = Process(target=run_bsd, args=(out,
                                              A, k_distinct, 'bswd_dw_lp',
                                                kernel_output, timeout, 
                                                vertex_indices, groundt_cliqs))
            p.start()
            p.join()
            p.close()
            decomp_data['bswd_dw_lp']['true_distinct']=out['bswd_dw_lp']
                    
        if run_ipart:
            decomp_data['bswd_dw_ip']['true_distinct']=None
            p = Process(target=run_bsd, args=(out,
                                              A, k_distinct, 'bswd_dw_ip',
                                                kernel_output, timeout, 
                                                vertex_indices, groundt_cliqs))
            p.start()
            p.join()
            p.close()
            decomp_data['bswd_dw_ip']['true_distinct']=out['bswd_dw_ip']

        #------------------------------------------------- final output data
        if partial_decompdat is None:
            decomp_output = { 
                'decomp_data' : decomp_data
                }
        else:
            decomp_output = partial_decompdat

        with open(decomp_fn, 'wb') as f:    
            pickle.dump(decomp_output, f)
            

def main():
    parser = argparse.ArgumentParser()
   
    #----------------------------------- required args
    parser.add_argument('-f', '--first_seed', type=int,
        help="enter first seed value in seed range", required=True)
    parser.add_argument('-l', '--last_seed', type=int,
        help="enter last seed value in seed range", required=True)
 
    parser.add_argument('-o', '--timeout', type=int, default=3600, # 1 hr default
        help="stop BSD after timeout number of seconds", required=False)    
    
    args = vars(parser.parse_args())
    
    first_seed = args.get('first_seed')
    last_seed = args.get('last_seed')
    timeout = args.get('timeout')
    
    k_distinct_max = 5
    
    tf_kern_dirname = 'data/kernels/tf/'
    lv_kern_dirname = 'data/kernels/lv/'
    
    tf_pp_dirname = 'data/post_preprocessing/tf/'
    lv_pp_dirname = 'data/post_preprocessing/lv/'
    
    tf_out_dirname = 'data/finaldata/tf/'
    lv_out_dirname = 'data/finaldata/lv/'
    
    # run bsd on tf graphs
    print('\nStarting bsd algorithms on tf kernels')
    run(tf_pp_dirname, tf_kern_dirname, tf_out_dirname, 
        first_seed, last_seed, k_distinct_max, timeout, 'tf')
    
    # run bsd on lv graphs
    print('\n\nStarting bsd algorithms on lv kernels')  
    run(lv_pp_dirname, lv_kern_dirname, lv_out_dirname, 
        first_seed, last_seed, k_distinct_max, timeout, 'lv')


if __name__=="__main__":
    main()
    
    
